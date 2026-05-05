"""
End-to-end influence score computation.

Loads cached compressed gradients, computes influence scores using DataInf,
and saves the results along with ROC AUC metrics.

Usage:
    python src/influence/compute_influence.py \
        --model_path logs/Qwen2.5-7B_length \
        --dataset_dir dataset/length_dataset \
        --bias_type length \
        --K 65536
"""

import argparse
import json
import os

import numpy as np
import torch

from src.influence.datainf import rapid_datainf, get_roc_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--bias_type", type=str,
                        choices=["length", "sycophancy", "safety", "helpsteer"],
                        required=True)
    parser.add_argument("--K", type=int, default=2**16)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.model_path

    # Load compressed gradients
    print("Loading compressed gradients...")
    train_grads_dict = torch.load(f"{args.model_path}/rapid_grad_train.pt", weights_only=False)
    val_grads_dict = torch.load(f"{args.model_path}/rapid_grad_val.pt", weights_only=False)

    train_grads = train_grads_dict[args.K]
    val_grads = val_grads_dict[args.K]
    print(f"  Train: {len(train_grads)} gradients, dim={train_grads[0].shape[0]}")
    print(f"  Val: {len(val_grads)} gradients")

    # Load flipped indices (ground truth)
    flipped_indices = np.load(f"{args.dataset_dir}/flipped_indices.npy")
    flipped_set = set(flipped_indices.tolist())

    # Load validation subset indices
    all_indices = list(range(len(val_grads)))
    if args.bias_type == "length":
        targeted_indices = np.load(f"{args.dataset_dir}/concise_indices.npy").tolist()
        counter_indices = np.load(f"{args.dataset_dir}/verbose_indices.npy").tolist()
        targeted_name = "Concise"
        counter_name = "Verbose"
    elif args.bias_type == "sycophancy":
        targeted_indices = np.load(f"{args.dataset_dir}/less_syco_indices.npy").tolist()
        counter_indices = np.load(f"{args.dataset_dir}/more_syco_indices.npy").tolist()
        targeted_name = "Less Sycophantic"
        counter_name = "More Sycophantic"
    else:
        # safety / helpsteer: use all val indices, no counter split
        concise_path = f"{args.dataset_dir}/concise_indices.npy"
        if os.path.exists(concise_path):
            targeted_indices = np.load(concise_path).tolist()
        else:
            targeted_indices = all_indices
        counter_indices = None
        targeted_name = "Full_Val"
        counter_name = None

    results = {}

    # Compute influence with targeted validation set
    eval_sets = [(targeted_name, targeted_indices), ("Full", all_indices)]
    if counter_indices is not None:
        eval_sets.insert(1, (counter_name, counter_indices))
    for name, indices in eval_sets:
        print(f"\nComputing influence with {name} validation set ({len(indices)} samples)...")
        influence = rapid_datainf(train_grads, val_grads, indices)

        auc_score, fpr, tpr = get_roc_auc(influence, flipped_indices)
        print(f"  AUC = {auc_score:.4f}")

        results[name] = {
            "auc": auc_score,
            "n_val": len(indices),
        }

        # Save influence scores
        np.save(f"{output_dir}/influence_{name.lower().replace(' ', '_')}.npy",
                np.array(influence))

    # Save summary
    summary_path = f"{output_dir}/influence_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
