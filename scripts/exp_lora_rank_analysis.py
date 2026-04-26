"""
Analyze LoRA rank ablation results.
Computes gradient separability and effective rank for each LoRA rank config,
and generates a summary plot.

Usage: python scripts/exp_lora_rank_analysis.py
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_rank(rank, bias_type="length"):
    model_dir = f"logs/Qwen2.5-1.5B_r{rank}_{bias_type}"
    data_dir = f"dataset/{bias_type}_dataset"

    if not Path(f"{model_dir}/rapid_grad_train.pt").exists():
        return None

    # Load
    train_grads = torch.load(f"{model_dir}/rapid_grad_train.pt", weights_only=False)[65536]
    flipped = np.load(f"{data_dir}/flipped_indices.npy")
    flipped_set = set(flipped.tolist())

    # Influence AUC
    with open(f"{model_dir}/influence_results.json") as f:
        results = json.load(f)
    auc_concise = results["Concise"]["auc"]

    # Gradient separability (sample 500)
    n_sample = min(500, len(train_grads))
    np.random.seed(42)
    indices = np.random.choice(len(train_grads), n_sample, replace=False)
    sampled_grads = torch.stack([train_grads[i] for i in indices])
    sampled_labels = np.array([1 if i in flipped_set else 0 for i in indices])

    norms = torch.norm(sampled_grads, dim=1, keepdim=True)
    normed = sampled_grads / (norms + 1e-10)
    cos_sim = torch.mm(normed, normed.t()).numpy()

    flipped_mask = sampled_labels == 1
    clean_mask = sampled_labels == 0

    within_flipped = cos_sim[np.ix_(flipped_mask, flipped_mask)]
    between = cos_sim[np.ix_(flipped_mask, clean_mask)]
    np.fill_diagonal(within_flipped, np.nan)

    separation = np.nanmean(within_flipped) - np.nanmean(between)

    # Effective rank (sample 500)
    G = sampled_grads.float()
    GGT = G @ G.t()
    eigenvalues = torch.linalg.eigvalsh(GGT).flip(0)
    eigenvalues = eigenvalues[eigenvalues > 0]
    p = eigenvalues / eigenvalues.sum()
    entropy = -(p * torch.log(p + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # LoRA param count
    try:
        with open(f"{model_dir}/adapter_config.json") as f:
            cfg = json.load(f)
        r = cfg.get("r", rank)
    except:
        r = rank
    lora_params = 7 * 2 * 1536 * r * 28 + 1536  # rough estimate

    # Training eval accuracy
    import glob
    state_files = glob.glob(f"{model_dir}/checkpoint-*/trainer_state.json")
    best_acc = 0
    if state_files:
        with open(sorted(state_files)[-1]) as f:
            state = json.load(f)
        evals = [e for e in state["log_history"] if "eval_accuracy" in e]
        if evals:
            best_acc = max(e["eval_accuracy"] for e in evals)

    return {
        "rank": rank,
        "lora_params_M": lora_params / 1e6,
        "auc_concise": auc_concise,
        "separation": separation,
        "effective_rank": effective_rank,
        "best_eval_acc": best_acc,
    }


def main():
    ranks = [4, 8, 16, 32, 64]

    print("=" * 70)
    print("LoRA Rank Ablation Analysis")
    print("=" * 70)

    results = []
    for r in ranks:
        print(f"  Analyzing r={r}...", end=" ", flush=True)
        res = analyze_rank(r)
        if res:
            results.append(res)
            print(f"AUC={res['auc_concise']:.4f}, Sep={res['separation']:.6f}, "
                  f"EffRank={res['effective_rank']:.1f}, Acc={res['best_eval_acc']:.4f}")
        else:
            print("not found")

    if len(results) < 2:
        print("Need at least 2 completed ranks to plot. Run more experiments first.")
        return

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Rank':>5s} | {'LoRA Params':>12s} | {'Eval Acc':>9s} | {'AUC':>8s} | "
          f"{'Separation':>11s} | {'Eff Rank':>9s}")
    print("-" * 90)
    for r in results:
        print(f"r={r['rank']:<4d} | {r['lora_params_M']:>10.1f}M | {r['best_eval_acc']:>8.4f} | "
              f"{r['auc_concise']:>7.4f} | {r['separation']:>10.6f} | {r['effective_rank']:>8.1f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    rs = [r["rank"] for r in results]
    aucs = [r["auc_concise"] for r in results]
    seps = [r["separation"] for r in results]
    effs = [r["effective_rank"] for r in results]
    accs = [r["best_eval_acc"] for r in results]
    params = [r["lora_params_M"] for r in results]

    # AUC vs Rank
    axes[0, 0].plot(rs, aucs, "o-", color="#1f77b4", lw=2, markersize=8)
    axes[0, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[0, 0].axhline(y=0.6922, color="green", linestyle=":", alpha=0.7, label="Probe AUC (0.69)")
    axes[0, 0].set_xlabel("LoRA Rank (r)")
    axes[0, 0].set_ylabel("Influence AUC (Concise)")
    axes[0, 0].set_title("AUC vs LoRA Rank")
    axes[0, 0].legend()
    axes[0, 0].set_xscale("log", base=2)

    # Separation vs Rank
    axes[0, 1].plot(rs, seps, "s-", color="#ff7f0e", lw=2, markersize=8)
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("LoRA Rank (r)")
    axes[0, 1].set_ylabel("Gradient Separation Gap")
    axes[0, 1].set_title("Gradient Separability vs LoRA Rank")
    axes[0, 1].set_xscale("log", base=2)

    # AUC vs LoRA Params
    axes[1, 0].plot(params, aucs, "D-", color="#2ca02c", lw=2, markersize=8)
    axes[1, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("LoRA Parameters (M)")
    axes[1, 0].set_ylabel("Influence AUC (Concise)")
    axes[1, 0].set_title("AUC vs LoRA Parameter Count")

    # Eval Accuracy vs Rank
    axes[1, 1].plot(rs, accs, "^-", color="#d62728", lw=2, markersize=8)
    axes[1, 1].set_xlabel("LoRA Rank (r)")
    axes[1, 1].set_ylabel("Best Eval Accuracy")
    axes[1, 1].set_title("Training Quality vs LoRA Rank")
    axes[1, 1].set_xscale("log", base=2)

    plt.suptitle("LoRA Rank Ablation on Qwen2.5-1.5B (Length Bias)", fontsize=14)
    plt.tight_layout()
    plt.savefig("logs/lora_rank_ablation.png", dpi=150)
    print(f"\nPlot saved to logs/lora_rank_ablation.png")


if __name__ == "__main__":
    main()
