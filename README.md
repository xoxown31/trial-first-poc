# Trial-First: Execution-Driven Failure Analysis for Code Generation with Small Language Models

**KCC 2026** | [Project Page](https://xoxown31.github.io/trial-first/)

## Overview

Trial-First is a training-free, test-time code generation method for small language models (SLMs). Instead of relying on chain-of-thought reasoning or naive repair, it:

1. Generates code and **executes** it immediately
2. On failure, analyzes the error into a **numbered checklist** of likely mistakes
3. **Resets context** and retries with the checklist as guidance

On HumanEval with qwen2.5-coder 1.5B, Trial-First achieves **68.3% pass@1** (k=2), outperforming Baseline (58.5%), CoT (64.0%), and Self-Repair (62.2%).

## Results

| Method | pass@1 (%) | Avg Tokens | Δacc (%) | Efficiency |
|---|---|---|---|---|
| Baseline | 58.5 | 242.4 | — | — |
| CoT | 64.0 | 704.3 | +5.5 | 1.19 |
| Self-Repair | 62.2 | 465.3 | +3.7 | 1.64 |
| **Trial-First (Ours)** | **68.3** | 720.2 | **+9.8** | **2.04** |

*Efficiency = Δacc(%) / ΔToken × 100, k=2*

## Setup

```bash
pip install -r requirements.txt
# Requires Ollama running locally with qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:1.5b
```

## Run

```bash
# Run all methods (164 HumanEval problems)
zsh run_all.sh

# Analyze results
python analyze.py

# Generate figures
python plot.py
```

## Citation

```bibtex
@inproceedings{park2026trialfirst,
  title     = {Trial-First: Execution-Driven Failure Analysis for Code Generation
               with Small Language Models},
  author    = {Park, Taejoo},
  booktitle = {Proceedings of KCC},
  year      = {2026}
}
```
