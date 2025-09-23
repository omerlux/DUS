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

## Quick Start

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation
```bash
git clone https://github.com/omerlux/DUS-for-MDLMs.git
cd DUS-for-MDLMs
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

Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. We propose the **Dilated Unmasking Scheduler (DUS)**, an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasks them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. Across math (GSM8K, MATH500), code (HumanEval, MBPP) and general‐knowledge benchmarks (BBH, MMLU-Pro), DUS outperforms confidence‐based planners—without modifying the underlying denoiser, and reveals the true speed-quality frontier of MDLMs.

<div align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/models%20plot.png" alt="DUS Performance Overview" width="600"/>
</div>

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
<i>Generation process visualization: tokens are unmasked progressively from left to right</i>
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
  - **DUS:** $K=\lceil\log G\rceil$ calls $\Rightarrow$ $\approx G/\log G \times$ speedup
  - **Empirical result:** DUS preserves up to 25% of quality even at 5×–10× fewer passes

By explicitly managing the number of unmasking steps via a dilated schedule, DUS uncovers the true speed–quality frontier of masked diffusion LMs—delivering sub-linear, any-order generation at practical fidelity.

## Results

DUS achieves significant speedups while maintaining competitive performance:

  - **Speed:** Up to 5×-10× faster than autoregressive generation
  - **Quality:** Preserves up to 75% of original performance quality
  - **Efficiency:** Requires only ⌈log G⌉ denoising steps for sequence length G
  - **Benchmarks:** Evaluated on GSM8K, MATH500, HumanEval, MBPP, BBH, and MMLU-Pro

<div align="center">
<img src="https://raw.githubusercontent.com/omerlux/DUS-for-MDLMs/main/assets/imgs/bench-mat-code.png" alt="Benchmark Results" width="800"/>
</div>

For detailed results and visualizations, see our [paper](https://arxiv.org/abs/2506.19037) and [website](https://omerlux.github.io/DUS-for-MDLMs/).

## Files Included

  - `eval_mdlm.py`: Evaluation harness for masked diffusion language models
  - `generate.py`: Core generation functions with dilated unmasking scheduler
  - `requirements.txt`: Required dependencies

## Supported Models

This implementation supports:
  - Dream models (`Dream-org/Dream-v0-Base-7B`, `Dream-org/Dream-v0-Instruct-7B`)
  - DiffuCoder models (`apple/DiffuCoder-7B-Base`, `apple/DiffuCoder-7B-Instruct`)
  - LLaDA models (`GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct`)
  - Other compatible masked language models

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

4. Or use VS Code debugger with the provided launch configuration.

## Key Functions

  - `generate()`: Confidence-based generation with various remasking strategies (traditional approach)
  - `generate_scheduled()`: Generation with dilated unmasking scheduler (DUS) - our proposed method

## Citation

If you use this code in your research, please cite:

```bibtex
@article{luxembourg2025plan,
    title={Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models},
    author={Luxembourg, Omer and Permuter, Haim and Nachmani, Eliya},
    journal={arXiv preprint arXiv:2506.19037},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
