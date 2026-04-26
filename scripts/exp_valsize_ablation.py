"""
Experiment B1: Validation Set Size Ablation.

Shows how influence function AUC changes as the validation set shrinks.
Reuses cached gradient files — no GPU required.

Memory-efficient: pre-computes the train gradient matrix and GxG^T once
per model, then only varies the val-grad averaging across fractions.

Results saved to logs/valsize_ablation.json

Usage: python scripts/exp_valsize_ablation.py
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

FRACTIONS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
N_SEEDS = 5
K = 65536

MODELS = [
    {
        "name": "Qwen2.5-1.5B",
        "log_dir": "logs/Qwen2.5-1.5B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Qwen3-1.7B",
        "log_dir": "logs/Qwen3-1.7B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Llama3.2-1B",
        "log_dir": "logs/Llama3.2-1B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Llama3.2-3B",
        "log_dir": "logs/Llama3.2-3B_length",
        "data_dir": "dataset/length_dataset",
    },
]


def datainf_precomputed(G_train, GGT, lam, val_grad):
    """DataInf with precomputed G_train and GxG^T — avoids re-stacking.

    Args:
        G_train:   (n_train, K) float32 tensor
        GGT:       (n_train, n_train) float32 tensor = G_train @ G_train.T
        lam:       scalar regularization
        val_grad:  (K,) float32 tensor, averaged validation gradient

    Returns:
        (n_train,) numpy array of influence scores
    """
    n_train = G_train.shape[0]
    val_grad_dots = G_train @ val_grad          # (n_train,)
    rapidinf = -1.0 / lam * val_grad_dots

    one_over_lam    = 1.0 / lam
    one_over_lam_n  = one_over_lam / n_train
    lam_plus_diag   = lam + GGT.diag()          # (n_train,)

    for k in range(n_train):
        rapidinf[k] += torch.sum(
            one_over_lam_n * (GGT[:, k] * val_grad_dots) / lam_plus_diag
        )
    return rapidinf.numpy()


def run_ablation(cfg: dict) -> list | None:
    log_dir = cfg["log_dir"]
    data_dir = cfg["data_dir"]

    train_path = f"{log_dir}/rapid_grad_train.pt"
    val_path   = f"{log_dir}/rapid_grad_val.pt"

    for p in [train_path, val_path]:
        if not Path(p).exists():
            print(f"  Skipping {cfg['name']}: missing {p}")
            return None

    print(f"  Loading gradient caches (may take ~30 sec)...")
    train_grads_raw = torch.load(train_path, weights_only=False)[K]
    val_grads_raw   = torch.load(val_path,   weights_only=False)[K]
    flipped         = np.load(f"{data_dir}/flipped_indices.npy")
    concise_idx     = np.load(f"{data_dir}/concise_indices.npy").tolist()
    concise_set     = set(concise_idx)
    n_concise       = len(concise_idx)

    print(f"  Train={len(train_grads_raw)}, Val={len(val_grads_raw)}, Concise={n_concise}")

    # Pre-stack training gradients and compute GxG^T once
    print("  Pre-stacking train gradients and computing GxG^T...")
    G_train = torch.stack(train_grads_raw).float()    # (n_train, K)
    del train_grads_raw
    gc.collect()

    GGT = G_train @ G_train.t()                       # (n_train, n_train)

    # lambda = 0.1/n * sum_i mean(grad_i^2) = 0.1 * mean of all squared elements
    lam = torch.tensor(0.1 * float(torch.mean(G_train ** 2)))

    # Pre-index val gradients by concise index
    val_grads_concise = {i: val_grads_raw[i].float() for i in concise_idx}
    del val_grads_raw
    gc.collect()
    print(f"  Lambda={lam:.6f}, GxG^T computed. Starting ablation...")

    # Ground-truth labels
    flipped_set = set(flipped.tolist())
    true_labels = np.array([1 if i in flipped_set else 0 for i in range(G_train.shape[0])])

    records = []
    for frac in FRACTIONS:
        n_subset = max(1, int(n_concise * frac))
        aucs = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            subset_idx = rng.choice(concise_idx, size=n_subset, replace=False).tolist()

            # Average val gradient over subset
            val_grad = torch.zeros(K)
            for i in subset_idx:
                val_grad += val_grads_concise[i]
            val_grad /= n_subset

            influence = datainf_precomputed(G_train, GGT, lam, val_grad)
            auc = roc_auc_score(true_labels, influence)
            aucs.append(float(auc))

        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        print(f"    frac={frac:.2f}  n={n_subset:5d}  AUC={mean_auc:.4f} ± {std_auc:.4f}")
        records.append({
            "fraction": frac,
            "n_val":    n_subset,
            "auc_mean": mean_auc,
            "auc_std":  std_auc,
            "auc_per_seed": aucs,
        })

    # Explicit cleanup before next model
    del G_train, GGT, val_grads_concise
    gc.collect()
    torch.cuda.empty_cache()

    return records


def main():
    print("=" * 60)
    print("Experiment B1: Validation Set Size Ablation")
    print("=" * 60)

    all_results = {}
    for cfg in MODELS:
        print(f"\n--- {cfg['name']} ---")
        records = run_ablation(cfg)
        if records:
            all_results[cfg["name"]] = records
            # Save incrementally so partial results survive a crash
            out_path = "logs/valsize_ablation.json"
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {out_path}")

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY: Mean AUC vs Validation Set Fraction")
    print("=" * 75)
    header = f"{'Model':18s} | " + " | ".join(f"f={f:.2f}" for f in FRACTIONS)
    print(header)
    print("-" * len(header))
    for model_name, records in all_results.items():
        aucs = [f"{r['auc_mean']:.4f}" for r in records]
        print(f"{model_name:18s} | " + " | ".join(aucs))


if __name__ == "__main__":
    main()
