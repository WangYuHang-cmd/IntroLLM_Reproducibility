"""
Plotting utilities for bias detection experiments.

Generates ROC curves (Figure 2), Precision-Recall curves (Figure 8),
runtime comparison (Figure 3), and labeling strategy results (Figure 5).
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curves(results_dict, title="ROC Curve", llm_points=None, save_path=None):
    """Plot ROC curves for multiple methods.

    Args:
        results_dict: Dict mapping method_name -> metrics dict (from compute_all_metrics).
        title: Plot title.
        llm_points: List of (fpr, tpr, label) tuples for LLM baselines (single points).
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["x", "+", "^", "s"]

    for i, (name, metrics) in enumerate(results_dict.items()):
        ax.plot(
            metrics["fpr"], metrics["tpr"],
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            lw=2,
            label=f"{name} (AUC={metrics['auc']:.3f})",
        )

    if llm_points:
        for j, (fpr, tpr, label) in enumerate(llm_points):
            ax.scatter(fpr, tpr, s=100, marker=markers[j % len(markers)],
                      zorder=5, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_pr_curves(results_dict, title="Precision-Recall Curve",
                   llm_points=None, save_path=None):
    """Plot Precision-Recall curves for multiple methods."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, (name, metrics) in enumerate(results_dict.items()):
        ax.plot(
            metrics["recall"], metrics["precision"],
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            lw=2,
            label=f"{name} (AP={metrics['ap']:.3f})",
        )

    if llm_points:
        markers = ["x", "+"]
        for j, (recall, precision, label) in enumerate(llm_points):
            ax.scatter(recall, precision, s=100, marker=markers[j % len(markers)],
                      zorder=5, label=label)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ablation_val_size(sizes, auc_means, auc_stds=None, title="Validation Set Size Ablation",
                           save_path=None):
    """Plot AUC vs. validation set size (Figure 12)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(sizes, auc_means, "o-", color="#1f77b4", lw=2)
    if auc_stds is not None:
        ax.fill_between(sizes,
                       np.array(auc_means) - np.array(auc_stds),
                       np.array(auc_means) + np.array(auc_stds),
                       alpha=0.2, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("Validation Set Size", fontsize=12)
    ax.set_ylabel("Average AUC", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_labeling_strategy_results(metrics_dict, save_path=None):
    """Plot labeling strategy oversight results (Figure 5).

    Args:
        metrics_dict: Dict with keys 'Initial', 'Influence', 'Mahalanobis', 'KNN',
                      each mapping to {'label_acc': float, 'rm_acc': float, 'cos_sim': float}.
    """
    methods = list(metrics_dict.keys())
    label_acc = [metrics_dict[m]["label_acc"] for m in methods]
    rm_acc = [metrics_dict[m]["rm_acc"] for m in methods]
    cos_sim = [metrics_dict[m]["cos_sim"] for m in methods]

    x = np.arange(3)
    width = 0.2
    n_methods = len(methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [2, 1]})

    colors = ["#808080", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x[0] + offset, label_acc[i], width, color=colors[i], label=method)
        ax1.bar(x[1] + offset, rm_acc[i], width, color=colors[i])

    ax1.set_xticks(x[:2])
    ax1.set_xticklabels(["Label Acc.", "RM Acc."])
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * width
        ax2.bar(offset, cos_sim[i], width, color=colors[i])

    ax2.set_xticks([])
    ax2.set_xlabel("Cos. Sim.")
    ax2.set_ylabel("Cosine Similarity")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
