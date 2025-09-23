"""
Core generation functions for masked diffusion language models.

This module contains the essential generation algorithms for MDLM including
confidence-based remasking and scheduled generation strategies.
"""

import gc
import re
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Gumbel noise for categorical distribution sampling.
    
    The Gumbel max is a method for sampling from categorical distributions.
    Higher temperature values increase randomness in the sampling process.
    
    Args:
        logits: Input logits tensor of shape (..., vocab_size)
        temperature: Temperature parameter controlling randomness (0 = no noise)
        
    Returns:
        Logits with Gumbel noise applied
    """
    if temperature == 0:
        return logits
    
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Precompute the number of tokens to transition at each generation step.
    
    This function ensures uniform distribution of token unmasking across
    all generation steps within each block.
    
    Args:
        mask_index: Boolean tensor indicating masked positions
        steps: Number of generation steps
        
    Returns:
        Tensor of shape (batch_size, steps) with token counts per step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_mask_by_confidence(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Calculate token confidence based on predicted token probabilities.
    
    Confidence is defined as the probability assigned to the predicted token.
    Higher confidence indicates the model is more certain about the prediction.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        x0: Predicted tokens of shape (batch_size, seq_len)
        
    Returns:
        Confidence scores for each position
    """
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )
    return x0_p


def get_mask_by_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy-based uncertainty for each token position.
    
    Entropy measures the uncertainty in the model's prediction.
    Higher entropy indicates less certain predictions.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        
    Returns:
        Entropy values for each position
    """
    p = F.softmax(logits.to(torch.float64), dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    return entropy

def check_stop_generation(
    x: torch.Tensor, 
    stop_tokens: List[int], 
    stop_on_eos: bool, 
    mask_id: int, 
    prompt_len: int
) -> bool:
    """
    Check if generation should stop based on stop tokens or EOS conditions.
    
    Args:
        x: Current token sequence
        stop_tokens: List of token IDs that should trigger stopping
        stop_on_eos: Whether to enable early stopping
        mask_id: ID of the mask token
        prompt_len: Length of the input prompt
        
    Returns:
        True if generation should stop, False otherwise
    """
    if not stop_on_eos:
        return False
    
    if stop_tokens:
        generated_part = x[:, prompt_len:]
        for stop_token in stop_tokens:
            if stop_token in generated_part.flatten():
                return True
    
    return False


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    alternate_primary: str = 'low_confidence',
    alternate_method: str = 'random',
    alternate_rate: int = 2,
    stop_on_eos: bool = True,
    stop_tokens: Optional[List[str]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    n_transfer: str = 'fixed',
    base: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate text using masked diffusion language model.
    
    This function implements the core generation algorithm for MDLMs using
    confidence-based or entropy-based remasking strategies within blocks.
    
    Args:
        model: The masked language model for prediction
        prompt: Input prompt tensor of shape (1, prompt_len)
        steps: Number of sampling steps per block
        gen_length: Total length of text to generate
        block_length: Length of each generation block
        temperature: Sampling temperature (0 = deterministic)
        cfg_scale: Classifier-free guidance scale (0 = no guidance)
        remasking: Strategy for token remasking ('low_confidence', 'high_entropy', 'random')
        mask_id: Token ID for the [MASK] token
        alternate_primary: Primary remasking method for alternating strategy
        alternate_method: Secondary remasking method for alternating strategy
        alternate_rate: Rate of alternation between remasking methods
        stop_on_eos: Whether to stop generation early on EOS tokens
        stop_tokens: List of strings that trigger early stopping
        tokenizer: Tokenizer for processing stop tokens
        n_transfer: Token transfer strategy ('fixed' or 'inc_dilated')
        base: Base value for dilated token transfer
        
    Returns:
        Tuple of (generated_sequence, unmasking_stages) where:
            - generated_sequence: Complete sequence including prompt
            - unmasking_stages: Tensor tracking when each token was unmasked
    """
    if stop_tokens is None:
        stop_tokens = []
        
    if tokenizer and stop_tokens:
        stop_tokens = tokenizer(stop_tokens, add_special_tokens=False)["input_ids"]
    else:
        stop_on_eos = False
    
    # Initialize generation tensors
    x = torch.full(
        (1, prompt.shape[1] + gen_length), 
        mask_id, 
        dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    unmasking_stage = torch.zeros_like(x, dtype=torch.long, device='cpu')
    unmasking_stage[:, :prompt.shape[1]] = -1

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    # Validate block configuration
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps = steps // num_blocks

    total_steps = 0
    stop_generation = False
    for num_block in range(num_blocks):
        if stop_generation:
            break
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        if n_transfer == 'fixed':
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        elif n_transfer == 'inc_dilated':
            num_transfer_tokens = dilated_unmask_levels(0, block_length - 1, base=2)
            num_transfer_tokens = merge_last_level(num_transfer_tokens)
            # get only the length of the levels
            num_transfer_tokens = [len(level) for level in num_transfer_tokens]
            steps = len(num_transfer_tokens)

        for i in range(steps):
            if stop_generation:
                break
            
            mask_index = (x == mask_id)
            if not mask_index.any():
                break

            with torch.no_grad():
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    combined_x = torch.cat([x, un_x], dim=0)
                    logits = model(combined_x).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = logits + cfg_scale * (logits - un_logits)
                else:
                    logits = model(x).logits

            # Clean cache
            gc.collect()
            torch.cuda.empty_cache()

            if temperature == 0:
                x0 = torch.argmax(logits, dim=-1)
            else:
                logits = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits, dim=-1)

            # Apply remasking strategy
            if remasking == 'low_confidence':
                x0_p = get_mask_by_confidence(logits, x0)
            elif remasking == 'high_entropy':
                x0_p = get_mask_by_entropy(logits)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                x0_p = get_mask_by_confidence(logits, x0)

            # Don't unmask outside current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if n_transfer == 'fixed':
                    k = num_transfer_tokens[j, i]
                else:
                    k = num_transfer_tokens[i]
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            unmasking_stage[transfer_index] = total_steps
            total_steps += 1

            # Check for early stopping
            if check_stop_generation(x, stop_tokens, stop_on_eos, mask_id, prompt_len):
                stop_generation = True
                break
    
    return x, unmasking_stage


def dilated_unmask_levels(start: int, end: int, base: int = 2, skip_exp: int = 1) -> List[List[int]]:
    """
    Generate coarse-to-fine (dilated) unmasking schedule over [start..end].
    
    Creates multiple levels of positions to unmask, starting with sparse
    positions at regular intervals and progressively filling in gaps.
    
    Args:
        start: Starting position (inclusive)
        end: Ending position (inclusive)
        base: Dilation base factor (must be >= 1)
        skip_exp: Exponent for initial stride calculation (must be >= 1)
        
    Returns:
        List of lists, where each inner list contains positions to unmask
        at that level
        
    Raises:
        ValueError: If base or skip_exp is less than 1
    """
    if base < 1 or skip_exp < 1:
        raise ValueError("base and skip_exp must be >= 1")
    if base == 1:
        return [list(range(start, end + 1))]

    length = end - start + 1
    stride = length // (base ** skip_exp)
    levels = []
    revealed = set()

    # Dilated rounds
    while stride >= 1:
        this_round = [
            i
            for i in range(start, end + 1)
            if (i - start) % stride == 0 and i not in revealed
        ]
        if this_round:
            levels.append(this_round)
            revealed.update(this_round)
        stride //= base

    # Final round – reveal any leftovers
    remainder = [i for i in range(start, end + 1) if i not in revealed]
    if remainder:
        levels.append(remainder)

    return levels


def binary_search_levels(start: int, end: int, level: int = 0, out: Optional[List[List[int]]] = None) -> List[List[int]]:
    """
    Generate binary search unmasking schedule over [start..end].
    
    Creates a hierarchical schedule where positions are revealed in
    binary search order, with midpoints revealed first at each level.
    
    Args:
        start: Starting position (inclusive)
        end: Ending position (inclusive)
        level: Current recursion level (used internally)
        out: Output list to accumulate results (used internally)
        
    Returns:
        List of lists, where each inner list contains positions to unmask
        at that level
    """
    if out is None:
        out = []
    if start > end:
        return out
    if len(out) <= level:
        out.append([])
    mid = (start + end) // 2
    out[level].append(mid)
    binary_search_levels(start, mid - 1, level + 1, out)
    binary_search_levels(mid + 1, end, level + 1, out)
    return out


def merge_last_level(levels: List[List[int]]) -> List[List[int]]:
    """
    Merge the last level with the previous one if it's significantly smaller.
    
    This optimization prevents having a tiny final level by merging it
    with the previous level when the last level has fewer elements.
    
    Args:
        levels: List of unmasking levels
        
    Returns:
        Modified levels list with potentially merged final level
    """
    if len(levels) >= 2 and len(levels[-1]) < len(levels[-2]):
        # Merge last level into previous
        levels[-2].extend(levels[-1])
        levels.pop()
    return levels


@torch.no_grad()
def generate_scheduled(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    stop_on_eos: bool = True,
    stop_tokens: Optional[List[str]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    confidence_threshold: float = 0.3,
    scheduler: str = 'binary',
    base: int = 2,
    base_skip: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate text using structured unmasking scheduler.
    
    This function implements scheduled generation where tokens are revealed
    according to a predefined schedule (binary search or dilated) rather than
    confidence-based selection. Supports remasking based on confidence thresholds.
    
    Args:
        model: The masked language model for prediction
        prompt: Input prompt tensor of shape (1, prompt_len)
        steps: Number of sampling steps (for compatibility, not used in scheduling)
        gen_length: Total length of text to generate
        block_length: Length of each generation block
        temperature: Sampling temperature (0 = deterministic)
        cfg_scale: Classifier-free guidance scale (0 = no guidance)
        remasking: Strategy for token remasking ('low_confidence', 'high_entropy', 'none')
        mask_id: Token ID for the [MASK] token
        stop_on_eos: Whether to stop generation early on EOS tokens
        stop_tokens: List of strings that trigger early stopping
        tokenizer: Tokenizer for processing stop tokens
        confidence_threshold: Threshold below which tokens are remasked
        scheduler: Scheduling method ('binary' for binary search, 'dilated' for coarse-to-fine)
        base: Base value for dilated unmasking levels
        base_skip: Exponent for initial division in dilated scheduling
        
    Returns:
        Tuple of (generated_sequence, unmasking_stages) where:
            - generated_sequence: Complete sequence including prompt
            - unmasking_stages: Tensor tracking when each token was unmasked
    """
    if stop_tokens is None:
        stop_tokens = []
        
    if tokenizer and stop_tokens:
        stop_tokens = tokenizer(stop_tokens, add_special_tokens=False)["input_ids"]
    else:
        stop_on_eos = False

    device = model.device
    prompt_len = prompt.shape[1]
    
    # Initialize generation tensors
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone().to(device)
    
    unmasking_stage = torch.zeros_like(x, dtype=torch.long, device='cpu')
    unmasking_stage[:, :prompt_len] = -1

    # Validate block configuration
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    total_steps = 0
    stop_generation = False
    
    for num_block in range(num_blocks):
        if stop_generation:
            break
        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length - 1
        
        if scheduler == 'dilated':
            schedule = dilated_unmask_levels(block_start, block_end, base=base, skip_exp=base_skip)
        elif scheduler == 'binary':
            schedule = binary_search_levels(block_start, block_end)
        else:
            raise NotImplementedError(f"Unknown schedule: {scheduler}")
        
        schedule = merge_last_level(schedule)
        
        for step_idx, step_indices in enumerate(schedule):
            if stop_generation:
                break

            # Generate predictions for all masked positions
            mask_index = (x == mask_id)
            if not mask_index.any():
                break

            with torch.no_grad():
                if cfg_scale > 0.:
                    prompt_index = (x != mask_id)
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    combined_x = torch.cat([x, un_x], dim=0)
                    logits = model(combined_x).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = logits + cfg_scale * (logits - un_logits)
                else:
                    logits = model(x).logits

            # Clean cache
            gc.collect()
            torch.cuda.empty_cache()

            if temperature == 0:
                x0 = torch.argmax(logits, dim=-1)
            else:
                logits = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits, dim=-1)

            # Update positions scheduled for this step
            for pos in step_indices:
                if x[0, pos] == mask_id:  # Only update if still masked
                    x[0, pos] = x0[0, pos]
                    unmasking_stage[0, pos] = total_steps
            
            total_steps += 1

            # Apply remasking if specified
            if remasking != 'none' and step_idx < len(schedule) - 1:  # Don't remask in final step
                if remasking == 'low_confidence':
                    confidence = get_mask_by_confidence(logits, x0)
                elif remasking == 'high_entropy':
                    confidence = get_mask_by_entropy(logits)
                else:
                    continue  # Skip remasking

                # Remask low-confidence predictions in current block
                for pos in step_indices:
                    if pos < confidence.shape[1] and confidence[0, pos] < confidence_threshold:
                        x[0, pos] = mask_id
                        unmasking_stage[0, pos] = -1

            # Check for early stopping
            if check_stop_generation(x, stop_tokens, stop_on_eos, mask_id, prompt_len):
                stop_generation = True
                break

    return x, unmasking_stage
