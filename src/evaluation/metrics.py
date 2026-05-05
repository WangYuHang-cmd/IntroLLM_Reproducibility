"""
Evaluation metrics for bias detection experiments.

Computes AUC, Average Precision, TNR@TPR=0.80, and generates ROC/PR curves.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_all_metrics(scores, flipped_indices, n_total):
    """Compute all evaluation metrics for a bias detection method.

    Args:
        scores: Array of detection scores (higher = more likely biased).
        flipped_indices: Array/list of truly-flipped sample indices.
        n_total: Total number of training samples.

    Returns:
        Dict with AUC, AP, TNR80, fpr, tpr arrays.
    """
    true_labels = np.zeros(n_total)
    for i in flipped_indices:
        true_labels[i] = 1

    # ROC
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    # Average Precision
    ap = average_precision_score(true_labels, scores)

    # TNR at TPR=0.80
    tnr80 = compute_tnr_at_tpr(fpr, tpr, target_tpr=0.80)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, scores)

    return {
        "auc": roc_auc,
        "ap": ap,
        "tnr80": tnr80,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def compute_tnr_at_tpr(fpr, tpr, target_tpr=0.80):
    """Compute TNR (1 - FPR) at a given TPR threshold."""
    # Find the FPR at the closest TPR >= target_tpr
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 0.0
    # Use the first (smallest FPR) point where TPR >= target
    return 1.0 - fpr[idx[0]]
