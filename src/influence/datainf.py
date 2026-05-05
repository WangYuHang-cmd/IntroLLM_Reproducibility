"""
DataInf influence function estimation on compressed gradients.

Implements the closed-form inverse-Hessian approximation from Kwon et al. (2024),
operating on OPORP-compressed gradient vectors for efficiency.

The influence score I_val(d_i) measures how much training sample d_i affects the
validation loss. Positive values = harmful (negatively-contributing),
negative values = helpful (positively-contributing).
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


def rapid_datainf(rapid_grad_train, rapid_grad_val, indices):
    """Compute DataInf influence scores using compressed gradients.

    Args:
        rapid_grad_train: List of compressed training gradient tensors.
        rapid_grad_val: List of compressed validation gradient tensors.
        indices: Indices into rapid_grad_val to use as the targeted validation set
                 (e.g., Concise indices for length bias detection).

    Returns:
        List of influence scores, one per training sample.
    """
    n_train = len(rapid_grad_train)

    # Compute regularization lambda (DataInf)
    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam

    # Average validation gradient over the targeted subset
    val_grad = torch.zeros_like(rapid_grad_val[0])
    for i, grad in enumerate(rapid_grad_val):
        if i in indices:
            val_grad += grad
    val_grad /= len(indices)

    # Stack training gradients: (n_train, D)
    train_grads = torch.stack(rapid_grad_train)
    # G @ G^T: pairwise dot products between training gradients
    train_grads_dots = torch.matmul(train_grads, train_grads.t())
    # G @ v: dot products between training gradients and avg validation gradient
    val_grad_dots = torch.matmul(train_grads, val_grad.t())

    # DataInf closed-form: first term
    rapidinf = -1 / lam * val_grad_dots

    # DataInf closed-form: second term (Woodbury correction)
    one_over_lam = 1 / lam
    one_over_lam_n_train = one_over_lam / n_train
    lam_plus_diag = lam + train_grads_dots.diag()

    for k in range(n_train):
        rapidinf[k] += torch.sum(
            one_over_lam_n_train * (train_grads_dots[:, k] * val_grad_dots) / lam_plus_diag
        )

    return rapidinf.tolist()


def rapid_tracin(rapid_grad_train, rapid_grad_val, indices):
    """TracIn-style influence (no inverse Hessian, just gradient similarity)."""
    n_train = len(rapid_grad_train)

    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam

    val_grad = torch.zeros_like(rapid_grad_val[0])
    for i, grad in enumerate(rapid_grad_val):
        if i in indices:
            val_grad += grad
    val_grad /= len(indices)

    train_grads = torch.stack(rapid_grad_train)
    val_grad_dots = torch.matmul(train_grads, val_grad.t())

    rapidinf = -1 / lam * val_grad_dots
    return rapidinf.tolist()


def rapid_selfinf(rapid_grad_train):
    """Self-influence: diagonal of G @ G^T."""
    train_grads = torch.stack(rapid_grad_train)
    train_grads_dots = torch.matmul(train_grads, train_grads.t())
    return train_grads_dots.diag().tolist()


def get_roc_auc(influence, flipped_indices):
    """Compute ROC AUC for bias detection.

    Args:
        influence: List/array of influence scores per training sample.
        flipped_indices: Indices of truly-flipped (biased) samples.

    Returns:
        (auc_score, fpr_array, tpr_array)
    """
    total_points = len(influence)
    true_conditions = np.zeros(total_points)
    for i in flipped_indices:
        true_conditions[i] = 1
    fpr, tpr, _ = roc_curve(true_conditions, influence)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def calculate_roc_auc_bootstrap(rapid_grad_train, rapid_grad_val,
                                 flipped_indices, N=100, S=0):
    """Bootstrap AUC estimation by subsampling validation sets.

    Args:
        rapid_grad_train: List of compressed training gradient tensors.
        rapid_grad_val: List of compressed validation gradient tensors.
        flipped_indices: Indices of truly-flipped training samples.
        N: Number of bootstrap rounds.
        S: Subset size per round (default: half of validation set).

    Returns:
        List of N AUC values.
    """
    import random
    random.seed(42)

    n_train = len(rapid_grad_train)
    n_val = len(rapid_grad_val)
    if S == 0:
        S = n_val // 2

    # True labels
    true_conditions = np.zeros(n_train)
    for i in flipped_indices:
        true_conditions[i] = 1

    # Precompute reusable matrices
    lam = 0
    for grad in rapid_grad_train:
        lam += torch.mean(grad**2)
    lam = 0.1 / n_train * lam

    train_grads = torch.stack(rapid_grad_train)
    train_grads_dots = torch.matmul(train_grads, train_grads.t())
    one_over_lam = 1 / lam
    one_over_lam_n_train = one_over_lam / n_train
    lam_plus_diag = lam + train_grads_dots.diag()

    auc_list = []
    for _ in tqdm(range(N), desc="Bootstrap AUC"):
        indices = random.sample(range(n_val), S)

        val_grad = torch.zeros_like(rapid_grad_val[0])
        for i in indices:
            val_grad += rapid_grad_val[i]
        val_grad /= len(indices)

        val_grad_dots = torch.matmul(train_grads, val_grad.t())
        rapidinf = -1 / lam * val_grad_dots
        for k in range(n_train):
            rapidinf[k] += torch.sum(
                one_over_lam_n_train * (train_grads_dots[:, k] * val_grad_dots) / lam_plus_diag
            )

        fpr, tpr, _ = roc_curve(true_conditions, rapidinf.numpy())
        auc_list.append(auc(fpr, tpr))

    return auc_list
