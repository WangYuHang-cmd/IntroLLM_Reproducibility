"""
Experiment A1: Non-LLM Bias Detection Baselines.

Runs Mahalanobis distance, KNN, self-confidence, and entropy baselines for
4 models × 2 bias types, then compares AUC against influence functions.

Results saved to logs/{model}_{bias}/baseline_results.json

Usage: python scripts/exp_baselines.py [--models MODEL [MODEL ...]] [--bias length sycophancy]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

sys.path.insert(0, ".")
from src.baselines.knn import compute_knn_scores
from src.baselines.mahalanobis import compute_mahalanobis_scores, get_activations
from src.baselines.quality_scores import compute_quality_scores, get_logits
from src.evaluation.metrics import compute_all_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry ─────────────────────────────────────────────────────────────
ALL_MODELS = {
    "Qwen2.5-1.5B": {
        "log_prefix": "Qwen2.5-1.5B",
        "tok_suffix": "",          # dataset/{bias}_dataset_tokenized
        "load_in_4bit": True,
        "batch_size": 4,
    },
    "Qwen3-1.7B": {
        "log_prefix": "Qwen3-1.7B",
        "tok_suffix": "",
        "load_in_4bit": True,
        "batch_size": 4,
    },
    "Qwen2.5-7B": {
        "log_prefix": "Qwen2.5-7B",
        "tok_suffix": "",
        "load_in_4bit": True,
        "batch_size": 2,
    },
    "Llama3.2-1B": {
        "log_prefix": "Llama3.2-1B",
        "tok_suffix": "_llama3.2_3b",  # dataset/{bias}_dataset_tokenized_llama3.2_3b
        "load_in_4bit": False,
        "batch_size": 4,
    },
    "Llama3.2-3B": {
        "log_prefix": "Llama3.2-3B",
        "tok_suffix": "_llama3.2_3b",
        "load_in_4bit": False,
        "batch_size": 2,
    },
}

BIAS_CONFIGS = {
    "length": {
        "data_dir": "dataset/length_dataset",
        "eval_indices_file": "concise_indices.npy",
        "tok_base": "length_dataset_tokenized",
    },
    "sycophancy": {
        "data_dir": "dataset/sycophancy_dataset",
        "eval_indices_file": "less_syco_indices.npy",
        "tok_base": "sycophancy_dataset_tokenized",
    },
}

# Reference influence AUCs (concise val subset) from completed experiments
INFLUENCE_AUCS = {
    "Qwen2.5-1.5B_length":    0.5419,
    "Qwen3-1.7B_length":      0.4791,
    "Qwen2.5-7B_length":      0.5053,
    "Llama3.2-1B_length":     0.6918,
    "Llama3.2-3B_length":     0.6996,
    "Qwen2.5-1.5B_sycophancy": 0.5200,
    "Qwen3-1.7B_sycophancy":   0.5200,
    "Qwen2.5-7B_sycophancy":   0.5054,
    "Llama3.2-1B_sycophancy":  0.6185,
    "Llama3.2-3B_sycophancy":  0.6500,
}


def load_model(model_dir: str, load_in_4bit: bool):
    """Load base model + LoRA adapter using adapter_config.json."""
    adapter_config_path = f"{model_dir}/adapter_config.json"
    with open(adapter_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"    Loading {base_model_name} + LoRA from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.to(device)
    model.eval()

    if model.config.pad_token_id is None:
        eos = model.config.eos_token_id
        model.config.pad_token_id = eos[0] if isinstance(eos, list) else eos

    return model


def run_one(model_name: str, bias_type: str) -> dict | None:
    cfg = ALL_MODELS[model_name]
    bias_cfg = BIAS_CONFIGS[bias_type]

    model_dir = f"logs/{cfg['log_prefix']}_{bias_type}"
    tok_dir = f"dataset/{bias_cfg['tok_base']}{cfg['tok_suffix']}"
    data_dir = bias_cfg["data_dir"]

    if not Path(f"{model_dir}/adapter_model.safetensors").exists():
        print(f"  Skipping {model_name}/{bias_type}: no checkpoint at {model_dir}")
        return None

    # ── Load datasets ────────────────────────────────────────────────────────
    print(f"  Loading tokenized datasets from {tok_dir}")
    train_dataset = datasets.load_from_disk(f"{tok_dir}/train")
    test_dataset = datasets.load_from_disk(f"{tok_dir}/test")

    flipped = np.load(f"{data_dir}/flipped_indices.npy")
    n_total = len(train_dataset)

    # Eval subset: concise (length) or less-sycophantic (sycophancy) validation samples
    eval_idx = np.load(f"{data_dir}/{bias_cfg['eval_indices_file']}")
    eval_subset = test_dataset.select(eval_idx.tolist())
    print(f"  Train: {n_total}, Eval subset: {len(eval_subset)}, Flipped: {len(flipped)}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = load_model(model_dir, cfg["load_in_4bit"])
    bs = cfg["batch_size"]

    results = {"model": model_name, "bias": bias_type}

    # ── Activations (shared by Mahalanobis + KNN) ───────────────────────────
    print("  Extracting train activations...")
    t0 = time.time()
    train_act = get_activations(train_dataset, model, batch_size=bs)
    print(f"  Extracting eval activations...")
    eval_act = get_activations(eval_subset, model, batch_size=bs)
    print(f"  Activations extracted in {(time.time()-t0)/60:.1f} min — shape: {train_act.shape}")

    # ── PCA reduction (when eval samples < feature_dim, covariance is rank-deficient) ──
    from sklearn.decomposition import PCA
    n_eval, feature_dim = eval_act.shape
    if n_eval < feature_dim:
        n_components = min(n_eval - 1, 512)  # cap at 512 for compute speed
        print(f"  PCA: {feature_dim}→{n_components} dims (n_eval={n_eval} < feature_dim)")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(eval_act)
        train_act_pca = pca.transform(train_act)
        eval_act_pca  = pca.transform(eval_act)
    else:
        train_act_pca, eval_act_pca = train_act, eval_act

    # ── Mahalanobis ──────────────────────────────────────────────────────────
    print("  Computing Mahalanobis scores...")
    try:
        maha_scores = compute_mahalanobis_scores(train_act_pca, eval_act_pca)
        maha_metrics = compute_all_metrics(maha_scores, flipped, n_total)
        results["mahalanobis"] = {
            "auc": float(maha_metrics["auc"]),
            "ap":  float(maha_metrics["ap"]),
            "tnr80": float(maha_metrics["tnr80"]),
        }
        print(f"    Mahalanobis AUC: {maha_metrics['auc']:.4f}")
    except np.linalg.LinAlgError as e:
        print(f"    Mahalanobis failed (singular covariance): {e}")
        results["mahalanobis"] = {"auc": None, "ap": None, "tnr80": None, "error": str(e)}

    # ── KNN ──────────────────────────────────────────────────────────────────
    print("  Computing KNN scores...")
    knn_scores = compute_knn_scores(train_act_pca, eval_act_pca, k=5)
    knn_metrics = compute_all_metrics(knn_scores, flipped, n_total)
    results["knn"] = {
        "auc": float(knn_metrics["auc"]),
        "ap":  float(knn_metrics["ap"]),
        "tnr80": float(knn_metrics["tnr80"]),
    }
    print(f"    KNN AUC: {knn_metrics['auc']:.4f}")

    # Free activations
    del train_act, eval_act

    # ── Self-confidence + Entropy ────────────────────────────────────────────
    print("  Computing reward model logits (self-confidence / entropy)...")
    logit_list = get_logits(model, train_dataset, batch_size=bs)
    quality = compute_quality_scores(logit_list)

    sc_metrics = compute_all_metrics(quality["self_confidence"], flipped, n_total)
    ent_metrics = compute_all_metrics(quality["entropy"], flipped, n_total)

    results["self_confidence"] = {
        "auc": float(sc_metrics["auc"]),
        "ap":  float(sc_metrics["ap"]),
        "tnr80": float(sc_metrics["tnr80"]),
    }
    results["entropy"] = {
        "auc": float(ent_metrics["auc"]),
        "ap":  float(ent_metrics["ap"]),
        "tnr80": float(ent_metrics["tnr80"]),
    }
    print(f"    Self-confidence AUC: {sc_metrics['auc']:.4f}")
    print(f"    Entropy AUC:         {ent_metrics['auc']:.4f}")

    # ── Free GPU memory ──────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = f"{model_dir}/baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS.keys()),
                        choices=list(ALL_MODELS.keys()),
                        help="Which models to run (default: all)")
    parser.add_argument("--bias", nargs="+", default=["length", "sycophancy"],
                        choices=["length", "sycophancy"],
                        help="Which bias types to run (default: both)")
    args = parser.parse_args()

    print("=" * 65)
    print("Experiment A1: Non-LLM Bias Detection Baselines")
    print(f"Models: {args.models}")
    print(f"Biases: {args.bias}")
    print("=" * 65)

    all_results = {}
    for model_name in args.models:
        for bias_type in args.bias:
            key = f"{model_name}_{bias_type}"
            print(f"\n{'─'*65}")
            print(f"  {key}")
            print(f"{'─'*65}")
            t0 = time.time()
            r = run_one(model_name, bias_type)
            if r:
                elapsed = (time.time() - t0) / 60
                all_results[key] = r
                print(f"  Finished in {elapsed:.1f} min")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Baseline AUC vs Influence Function AUC")
    print("=" * 80)
    hdr = f"{'Model+Bias':28s} | {'Influence':9s} | {'Mahal':9s} | {'KNN':9s} | {'SelfConf':9s} | {'Entropy':9s}"
    print(hdr)
    print("-" * 80)

    for key, r in all_results.items():
        inf_auc = INFLUENCE_AUCS.get(key, float("nan"))
        m_auc  = r.get("mahalanobis", {}).get("auc") or float("nan")
        k_auc  = r.get("knn", {}).get("auc", float("nan"))
        sc_auc = r.get("self_confidence", {}).get("auc", float("nan"))
        ent_auc = r.get("entropy", {}).get("auc", float("nan"))
        print(f"{key:28s} | {inf_auc:.4f}    | {m_auc:.4f}    | {k_auc:.4f}    | {sc_auc:.4f}    | {ent_auc:.4f}")

    print()


if __name__ == "__main__":
    main()
