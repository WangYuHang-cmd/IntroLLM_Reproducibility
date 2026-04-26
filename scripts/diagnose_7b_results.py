"""
Diagnose why Qwen2.5-7B produces AUC ≈ 0.50.

Run this on Colab after influence computation:
    python scripts/diagnose_7b_results.py --model_dir logs_7B/Qwen2.5-7B_length --bias_type length

Or check both:
    for bias in length sycophancy; do
        python scripts/diagnose_7b_results.py --model_dir logs_7B/Qwen2.5-7B_$bias --bias_type $bias
    done
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--bias_type", type=str, choices=["length", "sycophancy"], required=True)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--K", type=int, default=65536)
    return parser.parse_args()


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check_adapter(model_dir):
    section("CHECK 1: LoRA Adapter Files")
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    adapter_model_path = os.path.join(model_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_config_path):
        print(f"  ❌ MISSING: {adapter_config_path}")
        return None
    if not os.path.exists(adapter_model_path):
        print(f"  ❌ MISSING: {adapter_model_path}")
        # Check checkpoints
        import glob
        ckpts = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*")))
        if ckpts:
            print(f"  Found checkpoints: {ckpts}")
            print(f"  → adapter_model.safetensors should be copied from {ckpts[-1]}")
        return None

    with open(adapter_config_path) as f:
        cfg = json.load(f)
    print(f"  ✅ adapter_config.json found")
    print(f"     base_model: {cfg.get('base_model_name_or_path', '???')}")
    print(f"     peft_type: {cfg.get('peft_type', '???')}")
    print(f"     r: {cfg.get('r', '???')}")
    print(f"     target_modules: {cfg.get('target_modules', '???')}")
    print(f"     modules_to_save: {cfg.get('modules_to_save', '???')}")
    return cfg


def check_gradients(model_dir, K):
    section("CHECK 2: Gradient Statistics")
    train_path = os.path.join(model_dir, "rapid_grad_train.pt")
    val_path = os.path.join(model_dir, "rapid_grad_val.pt")

    if not os.path.exists(train_path):
        print(f"  ❌ MISSING: {train_path}")
        return None, None
    if not os.path.exists(val_path):
        print(f"  ❌ MISSING: {val_path}")
        return None, None

    train_dict = torch.load(train_path, weights_only=False)
    val_dict = torch.load(val_path, weights_only=False)

    print(f"  Train keys: {list(train_dict.keys())}")
    print(f"  Val keys: {list(val_dict.keys())}")

    if K not in train_dict:
        print(f"  ❌ Key {K} not in train_dict! Available: {list(train_dict.keys())}")
        return None, None

    train_grads = train_dict[K]
    val_grads = val_dict[K]

    print(f"\n  Train gradients: n={len(train_grads)}, dim={train_grads[0].shape}")
    print(f"  Val gradients:   n={len(val_grads)}, dim={val_grads[0].shape}")

    # Gradient norm stats
    train_norms = [g.norm().item() for g in train_grads[:200]]  # sample 200
    val_norms = [g.norm().item() for g in val_grads[:200]]

    print(f"\n  Train grad norms (first 200 samples):")
    print(f"    mean={np.mean(train_norms):.4f}, std={np.std(train_norms):.4f}")
    print(f"    min={np.min(train_norms):.4f}, max={np.max(train_norms):.4f}")

    print(f"\n  Val grad norms (first 200 samples):")
    print(f"    mean={np.mean(val_norms):.4f}, std={np.std(val_norms):.4f}")
    print(f"    min={np.min(val_norms):.4f}, max={np.max(val_norms):.4f}")

    # Check if gradients are near-zero or identical
    g0 = train_grads[0]
    g1 = train_grads[1]
    g_cos = torch.dot(g0, g1) / (g0.norm() * g1.norm() + 1e-8)
    print(f"\n  Cosine similarity of first two train grads: {g_cos.item():.4f}")
    print(f"  (Should be < 0.9; if ~1.0 → gradients are nearly identical → influence useless)")

    # Check mean gradient variance across samples
    stacked = torch.stack(train_grads[:100])
    grad_var = stacked.var(dim=0).mean().item()
    grad_mean = stacked.mean(dim=0).abs().mean().item()
    print(f"\n  Across-sample gradient variance (first 100): {grad_var:.6f}")
    print(f"  Across-sample gradient mean magnitude:        {grad_mean:.6f}")
    print(f"  SNR estimate: {grad_mean/(grad_var**0.5 + 1e-8):.4f}")
    print(f"  (For 3B this SNR is ~0.01-0.05; <<0.01 → no discriminative signal)")

    return train_grads, val_grads


def check_influence_results(model_dir):
    section("CHECK 3: Influence Results")
    results_path = os.path.join(model_dir, "influence_results.json")
    if not os.path.exists(results_path):
        print(f"  ❌ MISSING: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2))

    # Check influence score files
    for key in results:
        fname = os.path.join(model_dir, f"influence_{key.lower().replace(' ', '_')}.npy")
        if os.path.exists(fname):
            scores = np.load(fname)
            print(f"\n  {key}: n={len(scores)}, mean={scores.mean():.4f}, "
                  f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
            # Check if scores are nearly constant
            if scores.std() < 1e-6:
                print(f"    ⚠️  NEARLY CONSTANT SCORES — influence computation failed!")
        else:
            print(f"  ❌ MISSING influence_{key.lower().replace(' ', '_')}.npy")


def check_flipped_vs_clean(model_dir, bias_type, dataset_dir, train_grads):
    section("CHECK 4: Flipped vs Clean Gradient Separation")
    if train_grads is None:
        print("  Skipping (no gradients loaded)")
        return

    if dataset_dir is None:
        dataset_dir = f"dataset/{bias_type}_dataset"

    flipped_path = os.path.join(dataset_dir, "flipped_indices.npy")
    if not os.path.exists(flipped_path):
        print(f"  ❌ MISSING: {flipped_path}")
        return

    flipped_indices = set(np.load(flipped_path).tolist())
    n_total = len(train_grads)

    flipped_norms = []
    clean_norms = []
    for i, g in enumerate(train_grads):
        norm = g.norm().item()
        if i in flipped_indices:
            flipped_norms.append(norm)
        else:
            clean_norms.append(norm)

    print(f"  N flipped: {len(flipped_norms)}, N clean: {len(clean_norms)}")
    print(f"  Flipped grad norms: mean={np.mean(flipped_norms):.4f}, std={np.std(flipped_norms):.4f}")
    print(f"  Clean grad norms:   mean={np.mean(clean_norms):.4f}, std={np.std(clean_norms):.4f}")

    diff = abs(np.mean(flipped_norms) - np.mean(clean_norms))
    pooled_std = np.sqrt((np.var(flipped_norms) + np.var(clean_norms)) / 2)
    cohens_d = diff / (pooled_std + 1e-8)
    print(f"  Cohen's d (grad norm): {cohens_d:.4f}")
    print(f"  (For Llama-3.2-3B this is ~0.18; for 1.5B ~0.10; ≈0 → no separation)")

    # Check cosine similarity within flipped vs between flipped/clean
    import random
    random.seed(42)
    flip_list = list(flipped_indices)[:50]
    clean_list = [i for i in range(n_total) if i not in flipped_indices][:50]

    within_flip = []
    for i in range(0, len(flip_list)-1, 2):
        g1 = train_grads[flip_list[i]]
        g2 = train_grads[flip_list[i+1]]
        cos = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)
        within_flip.append(cos.item())

    cross = []
    for i in range(min(len(flip_list), len(clean_list))):
        g1 = train_grads[flip_list[i]]
        g2 = train_grads[clean_list[i]]
        cos = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-8)
        cross.append(cos.item())

    print(f"\n  Within-flip cosine sim:  mean={np.mean(within_flip):.4f}")
    print(f"  Cross flip-clean cos:    mean={np.mean(cross):.4f}")
    sep_gap = np.mean(within_flip) - np.mean(cross)
    print(f"  Separation gap: {sep_gap:.4f}")
    print(f"  (For 3B: +0.023 = useful; for 1.5B: ~0 = random; <0 = worse than random)")


def main():
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"  Diagnosing: {args.model_dir}")
    print(f"  Bias type: {args.bias_type}")
    print(f"{'#'*60}")

    cfg = check_adapter(args.model_dir)
    train_grads, val_grads = check_gradients(args.model_dir, args.K)
    check_influence_results(args.model_dir)
    check_flipped_vs_clean(args.model_dir, args.bias_type, args.dataset_dir, train_grads)

    section("SUMMARY")
    if train_grads is None:
        print("  ❌ Gradient files missing — cannot proceed")
    else:
        g0 = train_grads[0]
        g1 = train_grads[1]
        cos01 = torch.dot(g0, g1) / (g0.norm() * g1.norm() + 1e-8)
        if cos01.item() > 0.99:
            print("  🔴 DIAGNOSIS: Gradients are nearly identical across samples")
            print("     → LoRA adapter likely not loaded / all-zero score weights")
        elif g0.norm().item() < 1e-4:
            print("  🔴 DIAGNOSIS: Gradient norms are near zero")
            print("     → Possible: requires_grad not set, or model frozen")
        else:
            print("  🟡 Gradients have variance — issue might be in signal quality")
            print("     Check separation gap (Check 4) for direction")

    print()


if __name__ == "__main__":
    main()
