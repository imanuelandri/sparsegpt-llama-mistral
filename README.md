# SparseGPT Reproduction — Llama-2-7B & Mistral-7B

Reproduction of **SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot** (Frantar & Alistarh, 2023) applied to Llama-2-7B and Mistral-7B-v0.1.

> Presented for **CAP6614 Current Topics in Machine Learning** — Team 3  
> Suhith Gowda D.R., Andri W, Harshini M, Mahshid R

---

## Overview

The original SparseGPT paper validated one-shot unstructured pruning on OPT and BLOOM models. This project reproduces the key experiments on two modern 7B models:

- **Llama-2-7B** — sparsity sweep: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%
- **Mistral-7B-v0.1** — sparsity sweep: 0%, 20%, 40%, 50%, 60%, 70%

Both models are evaluated on:
- **Perplexity** — WikiText-2
- **Zero-shot accuracy** — HellaSwag, WinoGrande, PIQA, ARC-Easy, ARC-Challenge
- **Joint pruning + quantization** — fp16 vs int4 at selected sparsity levels

---

## Requirements

- Google Colab with **A100 GPU** (80GB)
- HuggingFace account with access to:
  - `meta-llama/Llama-2-7b-hf`
  - `mistralai/Mistral-7B-v0.1`
- Google Drive (~15GB free space per model)
- HuggingFace token stored as Colab secret: `HF_TOKEN`

---

## Setup

### Python packages
Cell 1 installs the following automatically:
```
transformers==4.37.2
accelerate==0.30.0
datasets==2.16.1
sentencepiece
safetensors
```

> ⚠️ **Important:** The zero-shot evaluation cell upgrades transformers to `4.40.0` for tokenizer compatibility (required for Mistral; also applies to Llama). After the zero-shot cell finishes, you must restart the runtime and re-run Cell 1 before running the plot cell.

---

## How to Run

### Llama-2-7B (`CAP6614_Clean_Llama.ipynb`)

| Cell | Description |
|------|-------------|
| Cell 1 | Setup — install packages, mount Drive, HF login |
| Cell 2 | Prune Llama fp16 — full sparsity sweep (0–70%) |
| Cell 3 | Sparse + Quantized — int4 sweep (0–70%) |
| Cell 4 | Extract and plot perplexity |
| Cell 5 | Pruning vs Quantization plot |
| Cell 6 | Zero-shot evaluation ⚠️ restart runtime after this |
| Cell 7 | Zero-shot plot (fp16 vs int4) |
| Cell 8 | Auto disconnect |

### Mistral-7B (`CAP6614_Clean_Mistral.ipynb`)

| Cell | Description |
|------|-------------|
| Cell 1 | Setup — install packages, mount Drive, HF login |
| Cell 2 | Prune Mistral fp16 — selected sparsity sweep |
| Cell 3 | Sparse + Quantized — int4 sweep |
| Cell 4 | Extract and plot perplexity |
| Cell 5 | Pruning vs Quantization plot |
| Cell 6 | Zero-shot evaluation ⚠️ restart runtime after this |
| Cell 7 | Zero-shot plot |
| Cell 8 | Auto disconnect |

### Runtime flow
```
Run Cell 1 → Run Cells 2-6
→ Restart Runtime
→ Re-run Cell 1
→ Run Cell 7 onwards
```

---

## Results

### Perplexity (WikiText-2) — fp16

| Sparsity | Llama-2-7B | Mistral-7B |
|----------|-----------|------------|
| 0%       | 5.47      | 5.25       |
| 10%      | 5.49      | — ¹        |
| 20%      | 5.58      | 5.28       |
| 30%      | 5.71      | — ¹        |
| 40%      | 5.94      | 5.50       |
| 50%      | 6.50      | 5.95       |
| 60%      | 8.37      | 7.46       |
| 70%      | 17.12     | 13.32      |

> ¹ Mistral was evaluated at 0%, 20%, 40%, 50%, 60%, 70% due to compute constraints. 10% and 30% were not run.

### Perplexity — int4 (pruning + quantization)

| Sparsity | Llama-2-7B | Mistral-7B |
|----------|-----------|------------|
| 0%       | 5.47      | 5.25       |
| 50%      | 6.97      | 6.27       |
| 70%      | 20.79     | 15.55      |

### Zero-Shot Average Accuracy — fp16

| Sparsity | Llama-2-7B | Mistral-7B |
|----------|-----------|------------|
| 0%       | 69.0%     | 74.1%      |
| 20%      | 68.9%     | 73.9%      |
| 40%      | 67.2%     | 72.0%      |
| 50%      | 64.9%     | 69.4%      |
| 60%      | 58.3%     | 62.3%      |
| 70%      | 43.4%     | 46.3%      |

---

## Key Findings

1. **50% sparsity is the sweet spot** — both models maintain near-baseline perplexity and zero-shot accuracy up to 50% sparsity with zero retraining.
2. **Mistral is more robust than Llama** — Mistral consistently outperforms Llama at every sparsity level on both metrics.
3. **int4 quantization is nearly free** — dense int4 matches dense fp16 exactly. At 50% sparse int4, the extra cost over fp16 is less than 0.5 perplexity points.
4. **Sharp degradation beyond 50%** — both models degrade steeply at 60% and 70% sparsity.
5. **Commonsense reasoning degrades fastest** — HellaSwag shows the steepest drop across sparsity levels compared to WinoGrande and PIQA.

---

## References

- Frantar, E., & Alistarh, D. (2023). [SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot](https://arxiv.org/abs/2301.00774). ICML 2023.
- [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) — official SparseGPT implementation
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — zero-shot evaluation framework

---

## Acknowledgements
- Claude (Anthropic) — AI assistant used for code implementation and debugging
