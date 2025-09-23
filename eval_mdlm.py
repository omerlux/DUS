"""
DUS: Dilated Unmasking Scheduler for Masked Diffusion Language Models.

This module implements the Dilated Unmasking Scheduler (DUS) for masked diffusion
language models, supporting LLaDA, Dream, DiffuCoder, and compatible model architectures.
It provides advanced scheduling strategies for improved text generation.
"""

import accelerate
import gc
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from generate import generate, generate_scheduled


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_shift_fix(logits: torch.Tensor) -> torch.Tensor:
    """
    Fix logits for Dream/DiffuLlama models to match expected output format.
    
    This function adjusts the logits tensor by shifting dimensions to align
    with the expected token prediction format.
    
    Args:
        logits: Input logits tensor of shape (batch_size, seq_len, vocab_size)
        
    Returns:
        Adjusted logits tensor with proper alignment
    """
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def wrap_logits_shift_model_forward(model: torch.nn.Module, fix_fn: callable) -> None:
    """
    Wrap model forward function to apply logits fixing.
    
    This function modifies the model's forward method to automatically
    apply logits correction for compatible model architectures.
    
    Args:
        model: PyTorch model to wrap
        fix_fn: Function to apply to logits for correction
    """
    orig_forward = model.forward
    
    def new_forward(*args, **kwargs):
        output = orig_forward(*args, **kwargs)
        if hasattr(output, "logits"):
            output.logits = fix_fn(output.logits)
        return output
    
    model.forward = new_forward


@register_model("mdlm_dist")
class MDLMEvalHarness(LM):
    """
    Masked Diffusion Language Model evaluation harness with Dilated Unmasking Scheduler.
    
    This class provides evaluation capabilities for masked diffusion language models
    using the Dilated Unmasking Scheduler (DUS) for improved text generation.
    Supports various model architectures including LLaDA,Dream, DiffuCoder, and compatible models.
    """
    
    def __init__(
        self,
        model_path: str = '',
        mask_id: int = 126336,
        max_length: int = 4096,
        batch_size: int = 32,
        mc_num: int = 128,
        is_check_greedy: bool = True,
        cfg: float = 0.,
        steps: int = 1024,
        gen_length: int = 1024,
        block_length: int = 1024,
        remasking: str = 'low_confidence',
        device: str = "cuda",
        **kwargs,
    ):
        """
        Initialize the MDLM evaluation harness with DUS support.
        
        Args:
            model_path: Path to the model (Dream, DiffuCoder, or other compatible models)
            mask_id: Token ID for the [MASK] token (default: 126336)
            max_length: Maximum sequence length for processing
            batch_size: Mini batch size for evaluation
            mc_num: Number of Monte Carlo estimation iterations
            is_check_greedy: Whether to perform suffix greedy prediction verification
            cfg: Classifier-free guidance scale (0.0 = no guidance)
            steps: Number of diffusion steps for generation
            gen_length: Length of text to generate
            block_length: Block length for semi-autoregressive generation
            remasking: Strategy for remasking ('low_confidence', 'high_entropy', etc.)
            device: Device for computation ('cuda' or 'cpu')
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        # Initialize accelerator for distributed training
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        # Set up model loading arguments
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        # Extract model name for identification
        try:
            self.model_name = model_path.split('/')[-1].split('-')[0].lower()
        except IndexError:
            self.model_name = model_path.lower()
        
        # Load model based on architecture
        if 'dream' in model_path.lower() or 'diffucoder' in model_path.lower():
            # Dream or DiffuCoder models require logits shifting
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto" if torch.cuda.device_count() > 1 else None, 
                **model_kwargs
            )
            wrap_logits_shift_model_forward(self.model, logits_shift_fix)
        else:
            # Default model loading
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto" if torch.cuda.device_count() > 1 else None, 
                **model_kwargs
            )
        
        self.model.eval()

        # Set up devices and distributed training
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif torch.cuda.device_count() == 1: 
            self.model = self.model.to(device)

        # Initialize tokenizer
        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Set evaluation parameters
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        # Set generation parameters
        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def rank(self) -> int:
        """Get the rank of the current process in distributed training."""
        return self._rank
    
    @property
    def world_size(self) -> int:
        """Get the total number of processes in distributed training."""
        return self._world_size

    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process to add noise to the batch.
        
        This method simulates the forward diffusion process by randomly masking
        tokens in the target sequence while preserving the prompt.
        
        Args:
            batch: Input token batch of shape (batch_size, seq_len)
            prompt_index: Boolean tensor indicating prompt positions
            
        Returns:
            Tuple of (noisy_batch, noise_level) where:
                - noisy_batch: Batch with randomly masked tokens
                - noise_level: Noise level applied to each position
        """
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch: torch.Tensor, prompt_index: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for the input batch with optional classifier-free guidance.
        
        Args:
            batch: Input token batch of shape (batch_size, seq_len)
            prompt_index: Boolean tensor indicating prompt positions
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = logits + self.cfg * (logits - un_logits)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute log-likelihood of target given prefix using Monte Carlo estimation.
        
        This method estimates the likelihood by applying the forward diffusion process
        multiple times and averaging the resulting losses.
        
        Args:
            prefix: Prefix token sequence
            target: Target token sequence to evaluate
            
        Returns:
            Estimated log-likelihood as a float
        """
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(
                logits[mask_indices], 
                seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix: torch.Tensor, target: torch.Tensor) -> bool:
        """
        Check if target can be recovered through greedy decoding from prefix.
        
        This method performs greedy autoregressive generation and checks if the
        resulting sequence matches the target exactly.
        
        Args:
            prefix: Prefix token sequence
            target: Target sequence to verify
            
        Returns:
            True if greedy decoding produces the exact target, False otherwise
        """
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), 
            self.mask_id, 
            device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(
                p, dim=-1, index=torch.unsqueeze(x0, -1)
            ).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
            
        correct = target == seq[0, len(prefix):]
        return torch.all(correct).item()

    def _encode_pair(self, context: str, continuation: str) -> tuple[list, list]:
        """
        Encode context-continuation pair into token sequences.
        
        Properly handles whitespace at context boundaries to ensure correct
        tokenization of the combined sequence.
        
        Args:
            context: Context string
            continuation: Continuation string
            
        Returns:
            Tuple of (context_tokens, continuation_tokens)
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list) -> list:
        """
        Compute log-likelihoods for a batch of requests.
        
        This method processes multiple prefix-target pairs and returns their
        log-likelihoods along with greedy prediction verification scores.
        
        Args:
            requests: List of request objects with prefix and target text
            
        Returns:
            List of tuples (log_likelihood, greedy_score) for each request
        """
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096, "Sequence length exceeds maximum limit"

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                
        torch.cuda.empty_cache()
        return out

    def generate_until(self, requests: list) -> list:
        """
        Generate text until stop tokens are encountered.
        
        This method performs text generation using either the standard generate()
        function or the scheduled generate_scheduled() function based on configuration.
        
        Args:
            requests: List of generation requests with prompts and stop conditions
            
        Returns:
            List of generated text strings
        """
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [
            {"question": req.args[0], "until": req.args[1]['until']} 
            for req in requests
        ]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for i, elem in tqdm(enumerate(ds), desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
            new_stop_tokens = stop_tokens + ['<|endoftext|>']

            # Choose generation method based on configuration
            if getattr(self, 'generation', None) == 'mdlm_scheduled':
                generated_answer, _ = generate_scheduled(
                    self.model, prompt, 
                    steps=self.steps, 
                    gen_length=self.gen_length, 
                    block_length=self.block_length, 
                    temperature=0, 
                    cfg_scale=self.cfg, 
                    remasking=self.remasking, 
                    mask_id=self.mask_id, 
                    stop_tokens=new_stop_tokens, 
                    tokenizer=self.tokenizer, 
                    confidence_threshold=getattr(self, 'confidence_threshold', 0.0), 
                    scheduler=getattr(self, 'scheduler', 'binary'), 
                    base=getattr(self, 'base', 2), 
                    base_skip=getattr(self, 'base_skip', 1)
                )
            else:
                # Default: use generate() function
                generated_answer, _ = generate(
                    self.model, prompt, 
                    steps=self.steps, 
                    gen_length=self.gen_length, 
                    block_length=self.block_length, 
                    temperature=0, 
                    cfg_scale=self.cfg, 
                    remasking=self.remasking, 
                    mask_id=self.mask_id, 
                    stop_tokens=new_stop_tokens, 
                    tokenizer=self.tokenizer,
                    n_transfer=getattr(self, 'n_transfer', 'fixed'), 
                    base=getattr(self, 'base', 2)
                )
                        
            # Decode and clean generated text
            generated_answer = self.tokenizer.decode(
                generated_answer[0][prompt.shape[1]:], 
                skip_special_tokens=False
            )
            
            # Apply stop token truncation
            for stop_seq in new_stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # Remove special tokens for final output
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, 
                skip_special_tokens=True
            )
            out.append(generated_answer)

        return out


if __name__ == "__main__":
    set_seed(1234)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    print(f'DUS: Dilated Unmasking Scheduler for MDLM\n\tArguments: {sys.argv[1:]}')
    cli_evaluate()
