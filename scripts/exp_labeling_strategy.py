"""
Experiment D: Labeling Strategy Oversight (HelpSteer2).

Demonstrates that influence functions can improve Bob's suboptimal labeling
strategy by identifying likely mislabeled samples, then using an SVM to
learn better sub-objective weights.

Pipeline:
1. Load trained reward model (logs/Qwen2.5-1.5B_helpsteer_bob1)
2. Cache gradients: D_B_1 (train) and D_A (val)
3. Compute influence scores (D_B_1 samples vs D_A distribution)
4. Run SVM weight update (Algorithm 1 from the paper)
5. Report: label accuracy and cosine similarity improvement vs Alice's weights

Prerequisites:
  - Reward model trained: python src/reward_modeling/train.py \\
        --config configs/reward_model_helpsteer_1.5B.yaml
  - Gradient caching:
      python -m src.influence.cache_gradients \\
          --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \\
          --data_path dataset/helpsteer2_b1/train \\
          --save_name rapid_grad_train.pt --K 65536
      python -m src.influence.cache_gradients \\
          --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \\
          --data_path dataset/helpsteer2/D_A \\
          --save_name rapid_grad_val.pt --K 65536

Usage: python scripts/exp_labeling_strategy.py
"""

import json
import sys
from pathlib import Path

import datasets
import numpy as np
import torch

sys.path.insert(0, ".")
from src.influence.datainf import rapid_datainf
from src.labeling_strategy.svm_update import run_full_weight_update

MODEL_DIR = "logs/Qwen2.5-1.5B_helpsteer_bob1"
DATA_DIR  = "dataset/helpsteer2"
K = 65536

# Bob's weight vector #1 and Alice's expert weights (from helpsteer_data.py)
W_ALICE = np.array([1.04, 0.46, 0.47, -0.33])
W_BOB_1 = np.array([1.1, 1.0, 3.1, 3.0])


def build_pairs_from_dataset(ds) -> tuple[list[dict], np.ndarray]:
    """Reconstruct pair dicts and Bob's labels from tokenized dataset.

    In the tokenized D_B_1, chosen_ids was set based on Bob's W_B_1 preference.
    We reconstruct bob_labels by re-applying W_B_1 to the stored sub-objective scores.
    """
    pairs = []
    labels = []
    for ex in ds:
        s0 = np.array(ex["scores_0"], dtype=float)
        s1 = np.array(ex["scores_1"], dtype=float)
        pairs.append({"scores_0": s0.tolist(), "scores_1": s1.tolist()})
        # label=1 if response_1 preferred under W_BOB_1
        label = int(np.dot(W_BOB_1, s1) > np.dot(W_BOB_1, s0))
        labels.append(label)
    return pairs, np.array(labels)


def main():
    train_grad_path = f"{MODEL_DIR}/rapid_grad_train.pt"
    val_grad_path   = f"{MODEL_DIR}/rapid_grad_val.pt"

    for p in [train_grad_path, val_grad_path]:
        if not Path(p).exists():
            print(f"ERROR: {p} not found.")
            print("Run gradient caching first — see the docstring at the top of this file.")
            sys.exit(1)

    print("=" * 60)
    print("Experiment D: HelpSteer2 Labeling Strategy Oversight")
    print("=" * 60)

    print("\nLoading gradient caches...")
    train_grads = torch.load(train_grad_path, weights_only=False)[K]
    val_grads   = torch.load(val_grad_path,   weights_only=False)[K]
    print(f"  Train grads: {len(train_grads)}, Val grads: {len(val_grads)}")

    # Use all D_A samples as validation set
    val_indices = list(range(len(val_grads)))

    print("\nComputing influence scores (D_B_1 vs D_A)...")
    influence = rapid_datainf(train_grads, val_grads, val_indices)
    influence = np.array(influence)
    print(f"  Influence scores: min={influence.min():.4f}, max={influence.max():.4f}")

    # Load D_B_1 with sub-objective scores for SVM
    print("\nLoading D_B_1 pairs with sub-objective scores...")
    db1 = datasets.load_from_disk(f"{DATA_DIR}/D_B_1")
    pairs, bob_labels = build_pairs_from_dataset(db1)
    print(f"  Pairs: {len(pairs)}, Bob label=1 fraction: {bob_labels.mean():.3f}")

    print("\nRunning SVM weight update (Algorithm 1)...")
    results = run_full_weight_update(pairs, bob_labels, influence, W_BOB_1, W_ALICE)

    # Save results
    out = {
        "model": "Qwen2.5-1.5B",
        "bob_variant": 1,
        "w_bob": W_BOB_1.tolist(),
        "w_alice": W_ALICE.tolist(),
        "w_svm": results.get("w_bob_new", []),
        "label_acc_old": results["label_acc_old"],
        "label_acc_new": results["label_acc_new"],
        "label_acc_improvement": results["label_acc_improvement"],
        "cos_sim_old": results["cos_sim_old"],
        "cos_sim_new": results["cos_sim_new"],
        "cos_sim_improvement": results["cos_sim_improvement"],
    }

    out_path = f"{MODEL_DIR}/labeling_strategy_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Label Acc:   {results['label_acc_old']:.4f} → {results['label_acc_new']:.4f}  "
          f"(Δ = {results['label_acc_improvement']:+.4f})")
    print(f"  Cosine Sim:  {results['cos_sim_old']:.4f} → {results['cos_sim_new']:.4f}  "
          f"(Δ = {results['cos_sim_improvement']:+.4f})")


if __name__ == "__main__":
    main()
