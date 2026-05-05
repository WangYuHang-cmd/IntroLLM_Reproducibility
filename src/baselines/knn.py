"""
K-Nearest Neighbor distance baseline for bias detection.

Computes the distance from each training sample to its k-th nearest neighbor
in the validation activation space. Higher distance = more likely OOD/biased.
Follows Sun et al. (2022).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_knn_scores(train_activations, eval_activations, k=5):
    """Compute KNN distance of training samples from eval distribution.

    Args:
        train_activations: (n_train, d) array of training activations.
        eval_activations: (n_eval, d) array of evaluation activations.
        k: Number of nearest neighbors.

    Returns:
        (n_train,) array of k-th nearest neighbor distances.
    """
    # Normalize activations
    train_norm = train_activations / (np.linalg.norm(train_activations, axis=1, keepdims=True) + 1e-8)
    eval_norm = eval_activations / (np.linalg.norm(eval_activations, axis=1, keepdims=True) + 1e-8)

    # Fit KNN on eval activations
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(eval_norm)

    # Query distances for each training sample
    distances, _ = nn.kneighbors(train_norm)

    # Use k-th nearest neighbor distance as the score
    return distances[:, -1]
