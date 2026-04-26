"""
Experiment 2C: TracIn vs DataInf comparison across model sizes.

Tests whether DataInf's inverse Hessian approximation amplifies noise for small models.
TracIn uses simple gradient dot products (no Hessian), which might be more robust.

Runs instantly — reuses already cached gradients.

Usage: python scripts/exp_tracin_vs_datainf.py
"""

import numpy as np
import torch
import time
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.influence.datainf import rapid_datainf, rapid_tracin, get_roc_auc


def run_comparison():
    models = [
        ("Qwen2.5-1.5B", "Qwen2.5-1.5B"),
        ("Qwen3-1.7B", "Qwen3-1.7B"),
        ("Llama3.2-3B", "Llama-3.2-3B"),
    ]

    print("=" * 70)
    print("Experiment 2C: TracIn vs DataInf")
    print("=" * 70)
    print(f"\n{'Model':18s} | {'Method':10s} | {'Concise':10s} | {'Verbose':10s} | {'Full':10s}")
    print("-" * 70)

    total_tasks = sum(1 for m, _ in models if Path(f"logs/{m}_length/rapid_grad_train.pt").exists()) * 2 * 3
    done = 0
    for model_name, label in models:
        for bias_type in ["length"]:
            model_dir = f"logs/{model_name}_{bias_type}"
            data_dir = f"dataset/{bias_type}_dataset"

            if not Path(f"{model_dir}/rapid_grad_train.pt").exists():
                print(f"{label:18s} | {'N/A':10s} | skipped")
                continue

            # Load
            t0 = time.time()
            print(f"  Loading gradients for {label}...", end=" ", flush=True)
            train_grads = torch.load(f"{model_dir}/rapid_grad_train.pt", weights_only=False)[65536]
            val_grads = torch.load(f"{model_dir}/rapid_grad_val.pt", weights_only=False)[65536]
            flipped = np.load(f"{data_dir}/flipped_indices.npy")
            concise_idx = np.load(f"{data_dir}/concise_indices.npy").tolist()
            verbose_idx = np.load(f"{data_dir}/verbose_indices.npy").tolist()
            all_idx = list(range(len(val_grads)))
            print(f"done ({time.time()-t0:.1f}s)")

            for method_name, method_fn in [("DataInf", rapid_datainf), ("TracIn", rapid_tracin)]:
                aucs = {}
                for subset_name, indices in [("Concise", concise_idx),
                                              ("Verbose", verbose_idx),
                                              ("Full", all_idx)]:
                    t1 = time.time()
                    print(f"  [{done+1}/{total_tasks}] {label} {method_name} {subset_name}...", end=" ", flush=True)
                    influence = method_fn(train_grads, val_grads, indices)
                    auc, _, _ = get_roc_auc(influence, flipped)
                    aucs[subset_name] = auc
                    done += 1
                    print(f"AUC={auc:.4f} ({time.time()-t1:.1f}s)")

                print(f"{label:18s} | {method_name:10s} | {aucs['Concise']:.4f}     | {aucs['Verbose']:.4f}     | {aucs['Full']:.4f}")

    # Also do sycophancy
    print()
    print(f"\n{'Model':18s} | {'Method':10s} | {'LessSyco':10s} | {'MoreSyco':10s} | {'Full':10s}")
    print("-" * 70)

    total_tasks2 = sum(1 for m, _ in models if Path(f"logs/{m}_sycophancy/rapid_grad_train.pt").exists()) * 2 * 3
    done2 = 0
    for model_name, label in models:
        for bias_type in ["sycophancy"]:
            model_dir = f"logs/{model_name}_{bias_type}"
            data_dir = f"dataset/{bias_type}_dataset"

            if not Path(f"{model_dir}/rapid_grad_train.pt").exists():
                continue

            t0 = time.time()
            print(f"  Loading gradients for {label} sycophancy...", end=" ", flush=True)
            train_grads = torch.load(f"{model_dir}/rapid_grad_train.pt", weights_only=False)[65536]
            val_grads = torch.load(f"{model_dir}/rapid_grad_val.pt", weights_only=False)[65536]
            flipped = np.load(f"{data_dir}/flipped_indices.npy")
            less_idx = np.load(f"{data_dir}/less_syco_indices.npy").tolist()
            more_idx = np.load(f"{data_dir}/more_syco_indices.npy").tolist()
            all_idx = list(range(len(val_grads)))
            print(f"done ({time.time()-t0:.1f}s)")

            for method_name, method_fn in [("DataInf", rapid_datainf), ("TracIn", rapid_tracin)]:
                aucs = {}
                for subset_name, indices in [("LessSyco", less_idx),
                                              ("MoreSyco", more_idx),
                                              ("Full", all_idx)]:
                    t1 = time.time()
                    print(f"  [{done2+1}/{total_tasks2}] {label} {method_name} {subset_name}...", end=" ", flush=True)
                    influence = method_fn(train_grads, val_grads, indices)
                    auc, _, _ = get_roc_auc(influence, flipped)
                    aucs[subset_name] = auc
                    done2 += 1
                    print(f"AUC={auc:.4f} ({time.time()-t1:.1f}s)")

                print(f"{label:18s} | {method_name:10s} | {aucs['LessSyco']:.4f}     | {aucs['MoreSyco']:.4f}     | {aucs['Full']:.4f}")


if __name__ == "__main__":
    run_comparison()
