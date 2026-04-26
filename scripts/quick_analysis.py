"""
Quick diagnostic experiments to understand WHY model size affects influence functions.

Tests 3 hypotheses:
1. Gradient separability: Are flipped/clean gradients distinguishable?
2. Reward margin: Does the model assign different margins to flipped vs clean?
3. Gradient norm distribution: Do flipped samples have distinctive gradient patterns?

Usage:
    python scripts/quick_analysis.py
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_influence_data(model_name, bias_type):
    """Load gradients, influence scores, and flipped indices for a model."""
    model_dir = f"logs/{model_name}_{bias_type}"
    data_dir = f"dataset/{bias_type}_dataset"

    grads_train = torch.load(f"{model_dir}/rapid_grad_train.pt", weights_only=False)[65536]
    grads_val = torch.load(f"{model_dir}/rapid_grad_val.pt", weights_only=False)[65536]
    flipped = np.load(f"{data_dir}/flipped_indices.npy")

    # Load influence scores
    if bias_type == "length":
        influence = np.load(f"{model_dir}/influence_concise.npy")
    else:
        influence = np.load(f"{model_dir}/influence_less_sycophantic.npy")

    return grads_train, grads_val, flipped, influence


def experiment_1_gradient_separability(models, bias_type="length"):
    """Are gradients of flipped samples distinguishable from clean ones?

    Measures: cosine similarity within-class vs between-class.
    If small models can't distinguish, within ≈ between.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Gradient Separability")
    print("=" * 60)

    results = {}
    for model_name, label in models:
        grads_train, _, flipped, _ = load_influence_data(model_name, bias_type)
        flipped_set = set(flipped.tolist())

        # Sample for speed
        n_sample = min(500, len(grads_train))
        indices = np.random.choice(len(grads_train), n_sample, replace=False)

        sampled_grads = torch.stack([grads_train[i] for i in indices])
        sampled_labels = np.array([1 if i in flipped_set else 0 for i in indices])

        # Normalize
        norms = torch.norm(sampled_grads, dim=1, keepdim=True)
        normed = sampled_grads / (norms + 1e-10)

        # Cosine similarity matrix
        cos_sim = torch.mm(normed, normed.t()).numpy()

        # Within flipped, within clean, between
        flipped_mask = sampled_labels == 1
        clean_mask = sampled_labels == 0

        within_flipped = cos_sim[np.ix_(flipped_mask, flipped_mask)]
        within_clean = cos_sim[np.ix_(clean_mask, clean_mask)]
        between = cos_sim[np.ix_(flipped_mask, clean_mask)]

        # Remove diagonal for within-class
        np.fill_diagonal(within_flipped, np.nan)
        np.fill_diagonal(within_clean, np.nan)

        results[label] = {
            "within_flipped": np.nanmean(within_flipped),
            "within_clean": np.nanmean(within_clean),
            "between": np.nanmean(between),
            "separation": np.nanmean(within_flipped) - np.nanmean(between),
        }
        print(f"\n  {label}:")
        print(f"    Within flipped:  {results[label]['within_flipped']:.6f}")
        print(f"    Within clean:    {results[label]['within_clean']:.6f}")
        print(f"    Between:         {results[label]['between']:.6f}")
        print(f"    Separation gap:  {results[label]['separation']:.6f}")

    return results


def experiment_2_gradient_norms(models, bias_type="length"):
    """Do flipped samples have different gradient norms?

    If the model "struggles" more with biased samples, their gradients
    might have higher norms (larger loss, steeper gradient).
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Gradient Norm Distribution")
    print("=" * 60)

    results = {}
    for model_name, label in models:
        grads_train, _, flipped, _ = load_influence_data(model_name, bias_type)
        flipped_set = set(flipped.tolist())

        norms = torch.tensor([torch.norm(g).item() for g in grads_train])
        flipped_norms = norms[[i for i in range(len(norms)) if i in flipped_set]]
        clean_norms = norms[[i for i in range(len(norms)) if i not in flipped_set]]

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((flipped_norms.std()**2 + clean_norms.std()**2) / 2)
        cohens_d = (flipped_norms.mean() - clean_norms.mean()) / pooled_std

        results[label] = {
            "flipped_norm_mean": flipped_norms.mean().item(),
            "clean_norm_mean": clean_norms.mean().item(),
            "cohens_d": cohens_d.item(),
        }
        print(f"\n  {label}:")
        print(f"    Flipped grad norm: {results[label]['flipped_norm_mean']:.2f}")
        print(f"    Clean grad norm:   {results[label]['clean_norm_mean']:.2f}")
        print(f"    Cohen's d:         {results[label]['cohens_d']:.4f}")

    return results


def experiment_3_influence_distribution(models, bias_type="length"):
    """Visualize influence score distributions for flipped vs clean."""
    print("\n" + "=" * 60)
    print("Experiment 3: Influence Score Distribution")
    print("=" * 60)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (model_name, label) in zip(axes, models):
        _, _, flipped, influence = load_influence_data(model_name, bias_type)
        flipped_set = set(flipped.tolist())

        inf_flipped = [influence[i] for i in range(len(influence)) if i in flipped_set]
        inf_clean = [influence[i] for i in range(len(influence)) if i not in flipped_set]

        # KL divergence approximation via histogram overlap
        bins = np.linspace(
            min(np.percentile(inf_clean, 1), np.percentile(inf_flipped, 1)),
            max(np.percentile(inf_clean, 99), np.percentile(inf_flipped, 99)),
            50,
        )
        h_flipped, _ = np.histogram(inf_flipped, bins=bins, density=True)
        h_clean, _ = np.histogram(inf_clean, bins=bins, density=True)

        # Overlap coefficient
        h_flipped_n = h_flipped / (h_flipped.sum() + 1e-10)
        h_clean_n = h_clean / (h_clean.sum() + 1e-10)
        overlap = np.sum(np.minimum(h_flipped_n, h_clean_n))

        ax.hist(inf_clean, bins=50, alpha=0.5, density=True, label="Clean", color="blue")
        ax.hist(inf_flipped, bins=50, alpha=0.5, density=True, label="Flipped", color="red")
        ax.set_title(f"{label}\nOverlap={overlap:.3f}")
        ax.set_xlabel("Influence Score")
        ax.legend()

        print(f"\n  {label}:")
        print(f"    Distribution overlap: {overlap:.4f} (1.0=identical, 0.0=separated)")
        print(f"    Flipped mean: {np.mean(inf_flipped):.2f}, Clean mean: {np.mean(inf_clean):.2f}")

    plt.tight_layout()
    plt.savefig("logs/influence_distribution_comparison.png", dpi=150)
    print(f"\n  Plot saved to logs/influence_distribution_comparison.png")


def experiment_4_effective_rank(models, bias_type="length"):
    """Measure the effective dimensionality of gradient space.

    Small models may have "collapsed" gradient spaces where all gradients
    point in similar directions, making it impossible to distinguish samples.
    Uses singular value spectrum of the gradient matrix.
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Effective Gradient Dimensionality")
    print("=" * 60)

    for model_name, label in models:
        grads_train, _, _, _ = load_influence_data(model_name, bias_type)

        # Sample for speed (SVD on full matrix is expensive)
        n_sample = min(1000, len(grads_train))
        indices = np.random.choice(len(grads_train), n_sample, replace=False)
        G = torch.stack([grads_train[i] for i in indices]).float()

        # Compute SVD (on the smaller dimension)
        # G is (n_sample, 65536), compute G @ G^T which is (n_sample, n_sample)
        GGT = G @ G.t()
        eigenvalues = torch.linalg.eigvalsh(GGT)
        eigenvalues = eigenvalues.flip(0)  # descending
        eigenvalues = eigenvalues[eigenvalues > 0]

        # Effective rank: exp(entropy of normalized eigenvalues)
        p = eigenvalues / eigenvalues.sum()
        entropy = -(p * torch.log(p + 1e-10)).sum()
        effective_rank = torch.exp(entropy).item()

        # How much variance is in top-k components
        cumvar = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
        var_top10 = cumvar[min(9, len(cumvar) - 1)].item()
        var_top50 = cumvar[min(49, len(cumvar) - 1)].item()

        print(f"\n  {label}:")
        print(f"    Effective rank:     {effective_rank:.1f} / {n_sample}")
        print(f"    Var in top-10 dims: {var_top10:.4f}")
        print(f"    Var in top-50 dims: {var_top50:.4f}")
        print(f"    Top eigenvalue:     {eigenvalues[0]:.2f}")


if __name__ == "__main__":
    np.random.seed(42)

    models = [
        ("Qwen2.5-1.5B", "Qwen2.5-1.5B (1.5B)"),
        ("Qwen3-1.7B", "Qwen3-1.7B (1.7B)"),
        ("Llama3.2-3B", "Llama-3.2-3B (3.2B)"),
    ]

    # Check which models have data
    available = []
    for model_name, label in models:
        if Path(f"logs/{model_name}_length/rapid_grad_train.pt").exists():
            available.append((model_name, label))
        else:
            print(f"Skipping {label}: no gradient data")

    print(f"Running analysis on {len(available)} models\n")

    experiment_1_gradient_separability(available, "length")
    experiment_2_gradient_norms(available, "length")
    experiment_3_influence_distribution(available, "length")
    experiment_4_effective_rank(available, "length")

    print("\n" + "=" * 60)
    print("DONE. Results saved to logs/")
    print("=" * 60)
