# Intermediate Files Reference

All intermediate files produced by the 1.5B pipeline, organized by phase.
These same files will also be produced for 7B (just replace `1.5B` with `7B` in paths).

---

## Phase 1: Data Preparation

### Length Bias Dataset
| File | Format | Shape/Size | Description |
|------|--------|------------|-------------|
| `dataset/length_dataset/train/` | HF Dataset | ~15000 samples | Tokenized training set with flipped labels |
| `dataset/length_dataset/test/` | HF Dataset | ~6121 samples | Tokenized validation set |
| `dataset/length_dataset/flipped_indices.npy` | int64 array | (~984,) | Indices of corrupted (label-flipped) training samples |
| `dataset/length_dataset/concise_indices.npy` | int64 array | (~2629,) | Validation indices where chosen is shorter (Concise subset) |
| `dataset/length_dataset/verbose_indices.npy` | int64 array | (~3492,) | Validation indices where chosen is longer (Verbose subset) |

### Sycophancy Bias Dataset
| File | Format | Shape/Size | Description |
|------|--------|------------|-------------|
| `dataset/sycophancy_dataset/train/` | HF Dataset | ~15000 samples | Tokenized training set with flipped labels |
| `dataset/sycophancy_dataset/test/` | HF Dataset | ~1071 samples | Tokenized validation set |
| `dataset/sycophancy_dataset/flipped_indices.npy` | int64 array | (~625,) | Indices of corrupted training samples |
| `dataset/sycophancy_dataset/less_syco_indices.npy` | int64 array | (~171,) | Validation indices: Less Sycophantic subset |
| `dataset/sycophancy_dataset/more_syco_indices.npy` | int64 array | (~150,) | Validation indices: More Sycophantic subset |

### HelpSteer2 (Labeling Strategy Oversight)
| File | Format | Shape/Size | Description |
|------|--------|------------|-------------|
| `dataset/helpsteer2/D_A/` | HF Dataset | ~400 samples | Alice's validation set (expert labels) |
| `dataset/helpsteer2/D_B_1/` .. `D_B_5/` | HF Dataset | ~8000 each | Bob's training sets (5 different weight vectors) |
| `dataset/helpsteer2/w_alice.npy` | float64 array | (4,) | Alice's weight vector [1.04, 0.46, 0.47, -0.33] |
| `dataset/helpsteer2/w_bob_1.npy` .. `w_bob_5.npy` | float64 array | (4,) each | Bob's 5 weight vectors |

### Dataset Sample Columns
Each tokenized sample in the HF datasets contains:
- `input_ids_chosen`: List[int], length=1024 (padded)
- `attention_mask_chosen`: List[int], length=1024
- `input_ids_rejected`: List[int], length=1024 (padded)
- `attention_mask_rejected`: List[int], length=1024
- `flipped`: bool (train only) — whether this sample's label was flipped

---

## Phase 2: Reward Model Training

| File | Format | Description |
|------|--------|-------------|
| `logs/Qwen2.5-1.5B_length/adapter_config.json` | JSON | LoRA configuration (r, alpha, target_modules, base_model_name) |
| `logs/Qwen2.5-1.5B_length/adapter_model.safetensors` | safetensors | LoRA adapter weights (lora_A, lora_B, score layer) |
| `logs/Qwen2.5-1.5B_length/training_args.bin` | pickle | RewardConfig training arguments |
| `logs/Qwen2.5-1.5B_length/trainer_state.json` | JSON | Training loss/eval metrics per step |
| `logs/Qwen2.5-1.5B_sycophancy/` | same structure | Sycophancy bias model checkpoint |

### Key values in `trainer_state.json`:
- `log_history[*].loss` — training loss per logging step
- `log_history[*].eval_accuracy` — eval accuracy per epoch
- `best_metric` — best eval accuracy achieved

---

## Phase 3A: Gradient Caching (OPORP Compressed)

| File | Format | Shape | Size | Description |
|------|--------|-------|------|-------------|
| `logs/Qwen2.5-1.5B_length/rapid_grad_train.pt` | dict{K: List[Tensor]} | {65536: [~15000 x (65536,)]} | ~1.9GB | Compressed training gradients |
| `logs/Qwen2.5-1.5B_length/rapid_grad_val.pt` | dict{K: List[Tensor]} | {65536: [~6121 x (65536,)]} | ~0.8GB | Compressed validation gradients |
| `logs/Qwen2.5-1.5B_length/RapidGrad_D*_n100_seed42.obj` | pickle | — | ~few MB | Cached OPORP permutation matrices (reusable) |
| `logs/Qwen2.5-1.5B_sycophancy/rapid_grad_train.pt` | same | {65536: [~15000 x (65536,)]} | ~1.9GB | Sycophancy training gradients |
| `logs/Qwen2.5-1.5B_sycophancy/rapid_grad_val.pt` | same | {65536: [~1071 x (65536,)]} | ~0.1GB | Sycophancy validation gradients |

### How to load:
```python
import torch
grads = torch.load("logs/Qwen2.5-1.5B_length/rapid_grad_train.pt", weights_only=False)
train_grads = grads[65536]  # List of 15000 tensors, each shape (65536,)
```

---

## Phase 3B: Influence Scores

| File | Format | Shape | Description |
|------|--------|-------|-------------|
| `logs/Qwen2.5-1.5B_length/influence_concise.npy` | float array | (~15000,) | Influence scores using Concise validation set |
| `logs/Qwen2.5-1.5B_length/influence_verbose.npy` | float array | (~15000,) | Influence scores using Verbose validation set |
| `logs/Qwen2.5-1.5B_length/influence_full.npy` | float array | (~15000,) | Influence scores using full validation set |
| `logs/Qwen2.5-1.5B_length/influence_results.json` | JSON | — | AUC summary for all validation subsets |
| `logs/Qwen2.5-1.5B_sycophancy/influence_less_sycophantic.npy` | float array | (~15000,) | Influence scores using Less Sycophantic validation |
| `logs/Qwen2.5-1.5B_sycophancy/influence_more_sycophantic.npy` | float array | (~15000,) | Influence scores using More Sycophantic validation |
| `logs/Qwen2.5-1.5B_sycophancy/influence_full.npy` | float array | (~15000,) | Influence scores using full validation set |
| `logs/Qwen2.5-1.5B_sycophancy/influence_results.json` | JSON | — | AUC summary |

### How to load:
```python
import numpy as np, json
influence = np.load("logs/Qwen2.5-1.5B_length/influence_concise.npy")
flipped = np.load("dataset/length_dataset/flipped_indices.npy")

with open("logs/Qwen2.5-1.5B_length/influence_results.json") as f:
    results = json.load(f)
print(results)
# {"Concise": {"auc": 0.xx, "n_val": 2629}, "Verbose": {"auc": 0.xx, ...}, ...}
```

### Influence score interpretation:
- **Positive** value = negatively-contributing (harms validation performance, likely biased)
- **Negative** value = positively-contributing (helps validation performance)
- Flipped samples should have higher (more positive) scores on average

---

## Transfer to 5090 for 7B

The `dataset/` directory is **shared** between 1.5B and 7B (same tokenizer family).
Copy these to the 5090 machine:
```bash
scp -r dataset/ user@5090-machine:/path/to/IntroLLM/
scp -r logs/Qwen2.5-1.5B_*/ user@5090-machine:/path/to/IntroLLM/logs/  # optional, for comparison
```

Then run 7B on 5090:
```bash
bash scripts/run_training.sh 7B
bash scripts/run_gradient_caching.sh 7B
bash scripts/run_influence.sh 7B
```
