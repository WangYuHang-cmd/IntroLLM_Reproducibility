"""
Algorithm 1: Bob weight update using influence functions and SVM.

Given:
- A reward model trained on Bob's labels (D_B)
- Influence scores computed w.r.t. Alice's validation set (D_A)
- Sub-objective score differences for each training sample

The algorithm:
1. Partition D_B into "likely mislabeled" (high influence) and "correctly labeled" (low influence)
   using the median of influence scores as threshold
2. Compute score differences r_i = r^(z_i) - r^(1-z_i) for each sub-objective
3. Train a linear SVM on {(r_i, t_i)} where t_i indicates mislabeling
4. Use SVM weights as Bob's updated weight vector w_SVM

Reference: Min et al. (2025), Section 5.2 and Appendix F.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


def partition_by_influence(influence_scores, threshold=None):
    """Partition training samples by influence score.

    Args:
        influence_scores: Array of influence scores per training sample.
        threshold: Threshold for partitioning. Default: median (50:50 split).

    Returns:
        (mislabeled_mask, threshold): Boolean mask where True = likely mislabeled.
    """
    if threshold is None:
        threshold = np.median(influence_scores)

    # High influence = likely mislabeled (negatively contributing)
    mislabeled_mask = np.array(influence_scores) > threshold
    return mislabeled_mask, threshold


def compute_score_differences(pairs, labels):
    """Compute sub-objective score differences: r^(chosen) - r^(rejected).

    For each sample, computes the difference between the chosen and rejected
    response's sub-objective scores (correctness, coherence, complexity, verbosity).

    Args:
        pairs: List of pair dicts with 'scores_0' and 'scores_1'.
        labels: Array of preference labels (0 or 1).

    Returns:
        (n_samples, 4) array of score differences.
    """
    diffs = []
    for pair, label in zip(pairs, labels):
        scores_0 = np.array(pair["scores_0"])
        scores_1 = np.array(pair["scores_1"])
        if label == 1:
            # Response 1 is chosen
            diff = scores_1 - scores_0
        else:
            # Response 0 is chosen
            diff = scores_0 - scores_1
        diffs.append(diff)
    return np.array(diffs)


def train_svm_weight_update(score_diffs, mislabeled_mask):
    """Train a linear SVM to learn updated weights from score differences.

    Args:
        score_diffs: (n_samples, 4) array of sub-objective score differences.
        mislabeled_mask: Boolean array where True = likely mislabeled.

    Returns:
        w_svm: (4,) array of SVM-derived weights (Bob's updated strategy).
        svm_model: Trained LinearSVC model.
    """
    # Labels: 0 = likely mislabeled, 1 = correctly labeled
    labels = (~mislabeled_mask).astype(int)

    # Standardize features for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(score_diffs)

    svm = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    svm.fit(X_scaled, labels)

    # Extract weights and de-scale
    w_svm = svm.coef_[0] / scaler.scale_
    # Normalize so weights sum to same magnitude as Alice's
    w_svm = w_svm / np.linalg.norm(w_svm) * np.linalg.norm([1.04, 0.46, 0.47, -0.33])

    return w_svm, svm


def evaluate_weight_update(pairs, w_bob_old, w_bob_new, w_alice):
    """Evaluate the quality of Bob's weight update.

    Args:
        pairs: List of pair dicts with sub-objective scores.
        w_bob_old: Bob's original weight vector.
        w_bob_new: Bob's updated weight vector (from SVM).
        w_alice: Alice's expert weight vector.

    Returns:
        Dict with label_acc_old, label_acc_new, cos_sim_old, cos_sim_new.
    """
    from src.labeling_strategy.helpsteer_data import label_pairs, compute_label_accuracy

    alice_labels = label_pairs(pairs, w_alice)
    bob_old_labels = label_pairs(pairs, w_bob_old)
    bob_new_labels = label_pairs(pairs, w_bob_new)

    label_acc_old = compute_label_accuracy(bob_old_labels, alice_labels)
    label_acc_new = compute_label_accuracy(bob_new_labels, alice_labels)

    cos_sim_old = np.dot(w_bob_old, w_alice) / (np.linalg.norm(w_bob_old) * np.linalg.norm(w_alice))
    cos_sim_new = np.dot(w_bob_new, w_alice) / (np.linalg.norm(w_bob_new) * np.linalg.norm(w_alice))

    return {
        "label_acc_old": label_acc_old,
        "label_acc_new": label_acc_new,
        "label_acc_improvement": label_acc_new - label_acc_old,
        "cos_sim_old": cos_sim_old,
        "cos_sim_new": cos_sim_new,
        "cos_sim_improvement": cos_sim_new - cos_sim_old,
        "w_bob_old": w_bob_old.tolist(),
        "w_bob_new": w_bob_new.tolist(),
        "w_alice": w_alice.tolist(),
    }


def run_full_weight_update(pairs, bob_labels, influence_scores, w_bob, w_alice):
    """Run the complete Algorithm 1 pipeline.

    Args:
        pairs: List of pair dicts with sub-objective scores.
        bob_labels: Bob's preference labels.
        influence_scores: Influence scores from DataInf.
        w_bob: Bob's current weight vector.
        w_alice: Alice's expert weight vector.

    Returns:
        Dict with updated weights and evaluation metrics.
    """
    # Step 1: Partition by influence
    mislabeled_mask, threshold = partition_by_influence(influence_scores)
    n_mislabeled = mislabeled_mask.sum()
    print(f"  Partitioned: {n_mislabeled} likely mislabeled, "
          f"{len(mislabeled_mask) - n_mislabeled} correctly labeled")

    # Step 2: Compute score differences
    score_diffs = compute_score_differences(pairs, bob_labels)

    # Step 3: Train SVM
    w_svm, svm_model = train_svm_weight_update(score_diffs, mislabeled_mask)
    print(f"  SVM weights: {w_svm.tolist()}")

    # Step 4: Evaluate
    results = evaluate_weight_update(pairs, w_bob, w_svm, w_alice)
    print(f"  Label Acc: {results['label_acc_old']:.4f} -> {results['label_acc_new']:.4f} "
          f"(+{results['label_acc_improvement']:.4f})")
    print(f"  Cos Sim:   {results['cos_sim_old']:.4f} -> {results['cos_sim_new']:.4f} "
          f"(+{results['cos_sim_improvement']:.4f})")

    return results
