# DUS: Dilated Unmasking Scheduler

**Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models**

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.19037-red)](https://arxiv.org/abs/2506.19037)
[![Website](https://img.shields.io/badge/Website-Official-blue)](https://omerlux.github.io/DUS-for-MDLMs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

This is the official implementation of the Dilated Unmasking Scheduler (DUS) paper for masked diffusion language models (MDLMs). 

<div align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/models%20plot.png" alt="DUS Performance Overview" width="600"/>
</div>

## Quick Start

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation
```bash
git clone https://github.com/omerlux/DUS.git
cd DUS
pip install -r requirements.txt
```

## Table of Contents
- [Quick Start](#quick-start)
- [Abstract](#abstract)
- [Method Overview](#method-overview)
- [Results](#results)
- [Files Included](#files-included)
- [Supported Models](#supported-models)
- [Usage](#usage)
- [Key Functions](#key-functions)
- [Citation](#citation)
- [License](#license)

## Abstract

Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. We propose the **Dilated Unmasking Scheduler (DUS)**, an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasks them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. Across math (GSM8K, MATH500), code (HumanEval, MBPP), general-knowledge (BBH, MMLU-Pro), and instruction following (IFEval) benchmarks, DUS outperforms confidence-based planners and turns the diffusion-specific quality-speed trade-off into a deterministic, predictable speedup set by the block size $B$, yielding up to $5.8\times$ wall-clock speedup over token-by-token MDLM decoding without modifying the underlying denoiser. Applied as a drop-in post-filter, dilated spacing also improves adaptive samplers.


## Method Overview

<div align="center">
<table>
<tr>
<td align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/llada-base_gsm8k_452_dus.png" alt="DUS Scheduler" width="400"/>
<br>
<b>DUS (Dilated Unmasking Scheduler)</b>
</td>
<td align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/llada-base_gsm8k_452_conf.png" alt="Confidence-based Scheduler" width="400"/>
<br>
<b>Confidence-based Scheduler</b>
</td>
</tr>
</table>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/generation_bar.png" alt="Generation Process" width="200"/>
<br>
<i>Generation process visualization: blue indicates the start of unmasking, different colors show tokens unmasked at different stages</i>
</div>

### Why Discrete Diffusion for LLMs?

Modern large‐scale LLMs almost universally use autoregressive (AR) decoding—predicting one token at a time in strict left-to-right order. While AR yields high local fidelity, it is subject to error accumulation and enforces $G$ sequential denoiser calls for a length-$G$ output, under-utilizing today's massively parallel hardware.

By contrast, masked diffusion treats the entire sequence as a latent "noisy" mask and gradually unmasks tokens over a small number of denoising passes. In principle this supports any-order token revelations and fully parallel updates—trading off the number of passes (and thus latency) against generation fidelity.

### The Problem with Existing Planners

Almost all existing diffusion samplers collapse back to AR speed and quality by unmasking one token per step, using denoiser confidence or entropy to pick the next index. In effect, the denoiser becomes an implicit planner, but it still:

  - Ignores interactions between multiple tokens unmasked in the same step
  - Fails to account for how revealing $x_i$ would change the uncertainty of $x_j$ if both are revealed together

As soon as you try to unmask more than one token at once, quality plummets.

### Our Dilated Unmasking Scheduler (DUS)

DUS is a model-agnostic, planner-model-free inference scheduler that requires no extra training or changes to the denoiser.

#### 1. Dilated Partitioning
  - Let $G$ be the sequence length. Set $K=\lceil\log G\rceil$ and $\{C_1,...,C_K\}$ be a partition of the $G$ positions into $K$ non-adjacent groups
  - Each candidate group $C_k$ has on average $G/K$ tokens with minimal pairwise dependencies

#### 2. Parallel Unmasking
For $k=1,...,K$:
  1. Unmask all tokens in $C_k$ simultaneously
  2. Run one pass of the denoiser over the full sequence

#### 3. Entropy-Bound Justification
Under a one-order fast-mixing Markov chain on token positions, non-adjacent tokens exhibit negligible mutual information, so:

$$H(x_{C_k}|s_t) \approx \sum_{i \in C_k} H(x_i|s_t)$$

Hence, grouping non-adjacent tokens in each $C_k$ controls the maximum quality loss per unmasking step.

#### 4. Speed–Quality Trade-off
  - **AR baseline:** $G$ denoiser calls (one per token)
  - **DUS:** $\lceil\log_a B\rceil$ calls per block of size $B$ $\Rightarrow$ $(G/B)\log_a B$ total NFE
  - **Empirical result:** up to $5.8\times$ wall-clock speedup over token-by-token MDLM decoding on LLaDA-8B ($B{=}32$); $5.6\times$ vs autoregressive Llama-3-8B baseline (RTX 6000 Ada). Up to 27% accuracy gain over self-confidence on math and code benchmarks.

By explicitly managing the number of unmasking steps via a dilated schedule, DUS turns the diffusion-specific quality-speed trade-off into a deterministic, predictable speedup set by the block size $B$.

## Adaptive Baselines

In addition to the DUS scheduler, this repository ships standalone implementations of two adaptive token-selection samplers from the literature, which we use as baselines in the paper's 5-model x 4-benchmark sweep (Appendix B.7):

- **EB-Sampler** (`generate_eb_sampler`) - entropy-bounded unmasking. Selects the largest prefix of low-entropy masked positions whose cumulative entropy stays within the budget $\gamma$ (in bits). Reference: Ben-Hamu et al., 2025 - *Accelerated sampling from masked diffusion models via entropy bounded unmasking*.
- **CB-Sampler** (`generate_cb_sampler`) - confidence-bounded unmasking. Unmasks every masked position whose top-token confidence exceeds threshold $\tau$; always unmasks the maximum-confidence token to guarantee progress. Reference: Wu et al., 2025 - *Fast-dLLM*.

DUS sits at the lowest-NFE end of the speed-quality plot; EB / CB reach higher accuracy at higher NFE on most tasks, with task-dependent ordering. See Figure 4 in the paper for the full sweep.

## Hybrid Post-Filter (Sec 4.4)

Section 4.4 of the paper introduces a *drop-in* dilated-spacing filter that complements EB-Sampler and CB-Sampler. After the base sampler selects its candidate positions (sorted by score), the filter accepts each candidate only if it is at least `min_gap` away from every already-accepted position; rejected candidates stay masked and are reconsidered at the next denoising step. The minimum gap is adaptive:

$$
\textit{gap} = \max\!\left(2,\ \left\lfloor \frac{M_\text{rem}\cdot g_0}{B} \right\rfloor\right),
$$

where $M_\text{rem}$ is the count of still-masked positions in the current block of size $B$, and $g_0$ (`start_stride`) is the initial gap when the block is fully masked. Larger $g_0$ enforces sparser early unmasking; as the block fills, the gap relaxes toward 1, recovering the underlying sampler.

The filter ships as `apply_spacing_filter` (pure function) plus two wrapped samplers:

- `generate_eb_sampler_spaced` - EB selection followed by the spacing filter.
- `generate_cb_sampler_spaced` - CB selection followed by the spacing filter.

At $B=32$ and $g_0=8$ on LLaDA-Instruct-8B / HumanEval, the post-filter improves an aggressive base setting by **+13.4 pass@1** for EB ($\gamma=2$) and **+12.2 pass@1** for CB ($\tau=0.5$). The dilated-spacing principle is complementary to score-based selection.

## Results

  - **Wall-clock speedup:** Up to $5.8\times$ over token-by-token MDLM decoding on LLaDA-8B at $B=32$; $5.6\times$ vs autoregressive Llama-3-8B (RTX 6000 Ada).
  - **NFE speedup:** $(G/B)\log_a B$ total denoiser calls vs $G$ for token-by-token, with $B \in \{8, 16, 32, 64\}$ giving $2.7\times$ to $10.7\times$ fewer denoiser calls.
  - **Accuracy:** Up to **+27%** over self-confidence on math (GSM8K, MATH500) and code (HumanEval, MBPP) benchmarks.
  - **Adaptive baselines:** EB-Sampler and CB-Sampler reach higher accuracy at higher NFE; DUS provides a predictable, deterministic speedup at the low-NFE end. See Figure 4 in the paper for the full 5-model x 4-benchmark sweep.
  - **Hybrid post-filter:** Dilated-spacing on top of EB / CB improves their accuracy at matched or modestly higher NFE; up to +13.4 pass@1 on HumanEval (EB, $\gamma=2$) and +12.2 (CB, $\tau=0.5$). See Sec 4.4.
  - **Benchmarks:** Evaluated on GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro, IFEval; models include LLaDA-8B, Dream-7B, DiffuCoder-7B (Base + Instruct).

<div align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/bench-mat-code.png" alt="Benchmark Results" width="800"/>
</div>

For detailed results and visualizations, see our [paper](https://arxiv.org/abs/2506.19037) and [website](https://omerlux.github.io/DUS-for-MDLMs/).

## Files Included

  - `eval_mdlm.py`: Evaluation harness for masked diffusion language models
  - `generate.py`: Core generation functions with dilated unmasking scheduler
  - `requirements.txt`: Required dependencies

## Supported Models

This implementation supports the following masked diffusion language models:
  - **LLaDA-8B** (`GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct`)
  - **Dream-7B** (`Dream-org/Dream-v0-Base-7B`, `Dream-org/Dream-v0-Instruct-7B`)
  - **DiffuCoder-7B** (`apple/DiffuCoder-7B-Base`, `apple/DiffuCoder-7B-Instruct`)

The paper reports results on LLaDA-8B (Base + Instruct), Dream-7B (Instruct only - Dream-Base underperforms in our setup), and DiffuCoder-7B (Base + Instruct).

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run evaluation with confidence-based scheduler:
   ```bash
   python eval_mdlm.py --tasks gsm8k --model mdlm_dist --model_args "model_path=GSAI-ML/LLaDA-8B-Base,gen_length=256,steps=64,block_length=32,remasking=low_confidence"
   ```

3. Run evaluation with dilated unmasking scheduler (DUS):
   ```bash
   python eval_mdlm.py --tasks gsm8k --model mdlm_dist --model_args "model_path=GSAI-ML/LLaDA-8B-Base,gen_length=256,block_length=32,generation=mdlm_scheduled,scheduler=dilated,base=2,base_skip=1,confidence_threshold=0.3,remasking=low_confidence"
   ```

4. Use the adaptive samplers (EB / CB) and the hybrid post-filter directly from Python:

   ```python
   import torch
   from transformers import AutoModel, AutoTokenizer
   from generate import (
       generate_eb_sampler,
       generate_cb_sampler,
       generate_eb_sampler_spaced,
       generate_cb_sampler_spaced,
   )

   tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
   model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True,
                                     torch_dtype=torch.bfloat16).cuda().eval()
   prompt = tokenizer('What is 12 * 7?', return_tensors='pt').input_ids.cuda()

   # EB-Sampler (entropy-bounded)
   x, stage = generate_eb_sampler(model, prompt, gen_length=128, block_length=32, gamma=1.0)

   # CB-Sampler (confidence-bounded)
   x, stage = generate_cb_sampler(model, prompt, gen_length=128, block_length=32, tau=0.7)

   # Hybrid: EB / CB + dilated-spacing post-filter (Sec 4.4)
   x, stage = generate_eb_sampler_spaced(model, prompt, gen_length=128, block_length=32,
                                         gamma=2.0, start_stride=8)
   x, stage = generate_cb_sampler_spaced(model, prompt, gen_length=128, block_length=32,
                                         tau=0.5, start_stride=8)
   ```

5. Or use VS Code debugger with the provided launch configuration.

## Key Functions

  - `generate()`: Confidence-based generation with various remasking strategies (traditional approach)
  - `generate_scheduled()`: Generation with dilated unmasking scheduler (DUS) - our proposed method
  - `generate_eb_sampler()`: EB-Sampler - entropy-bounded adaptive unmasking (Ben-Hamu et al., 2025)
  - `generate_cb_sampler()`: CB-Sampler - confidence-bounded adaptive unmasking (Wu et al., 2025, Fast-dLLM)
  - `generate_eb_sampler_spaced()`: EB-Sampler with the dilated-spacing post-filter (Sec 4.4 hybrid)
  - `generate_cb_sampler_spaced()`: CB-Sampler with the dilated-spacing post-filter (Sec 4.4 hybrid)
  - `apply_spacing_filter()`: Pure-Python dilated-spacing post-filter used by the spaced variants

## Citation

If you use this code in your research, please cite:

```bibtex
@article{luxembourg2025plan,
    title   = {Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models},
    author  = {Luxembourg, Omer and Permuter, Haim and Nachmani, Eliya},
    journal = {arXiv preprint arXiv:2506.19037},
    year    = {2025},
    note    = {Accepted at the International Conference on Machine Learning (ICML), 2026}
}
```

## Acknowledgments

We thank the authors of [LLaDA](https://github.com/ML-GSAI/LLaDA) for their open-source implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
