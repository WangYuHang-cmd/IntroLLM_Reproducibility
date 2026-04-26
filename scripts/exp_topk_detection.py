"""
Experiment C: Top-k Suspicious Sample Detection (Precision@k).

Uses pre-computed influence scores to measure how well top-k ranked
training samples correspond to truly flipped/biased examples.

No GPU or model loading required — reads existing influence_concise.npy files.

Results saved to logs/topk_detection.json

Usage: python scripts/exp_topk_detection.py
"""

import json
import sys
from pathlib import Path

import numpy as np

K_VALUES = [50, 100, 200, 500, 1000, 2000]

MODELS = [
    {
        "name": "Qwen2.5-1.5B",
        "log_dir": "logs/Qwen2.5-1.5B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Qwen3-1.7B",
        "log_dir": "logs/Qwen3-1.7B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Llama3.2-1B",
        "log_dir": "logs/Llama3.2-1B_length",
        "data_dir": "dataset/length_dataset",
    },
    {
        "name": "Llama3.2-3B",
        "log_dir": "logs/Llama3.2-3B_length",
        "data_dir": "dataset/length_dataset",
    },
]


def precision_at_k(scores: np.ndarray, flipped_set: set, k: int) -> float:
    """Fraction of top-k samples by score that are truly flipped."""
    top_k = np.argsort(scores)[-k:]
    hits = sum(1 for i in top_k if i in flipped_set)
    return hits / k


def run_model(cfg: dict) -> dict | None:
    inf_path = f"{cfg['log_dir']}/influence_concise.npy"
    if not Path(inf_path).exists():
        print(f"  Skipping {cfg['name']}: {inf_path} not found")
        return None

    influence = np.load(inf_path)
    flipped   = np.load(f"{cfg['data_dir']}/flipped_indices.npy")
    flipped_set = set(flipped.tolist())
    n_total   = len(influence)
    base_rate = len(flipped) / n_total

    print(f"  {cfg['name']}: n={n_total}, flipped={len(flipped)}, base_rate={base_rate:.4f}")

    records = []
    for k in K_VALUES:
        if k > n_total:
            continue
        prec = precision_at_k(influence, flipped_set, k)
        enrichment = prec / base_rate
        records.append({
            "k": k,
            "precision": float(prec),
            "base_rate": float(base_rate),
            "enrichment": float(enrichment),
        })
        print(f"    k={k:5d}  Precision={prec:.4f}  Enrichment={enrichment:.2f}x")

    return {"base_rate": float(base_rate), "n_total": n_total,
            "n_flipped": len(flipped), "precision_at_k": records}


def random_baseline(n_total: int, n_flipped: int, k: int, n_trials: int = 100) -> float:
    """Expected precision@k for a random ranker."""
    return n_flipped / n_total   # exact expectation (no need for simulation)


def main():
    print("=" * 60)
    print("Experiment C: Top-k Suspicious Sample Detection")
    print("=" * 60)

    all_results = {}
    for cfg in MODELS:
        print(f"\n--- {cfg['name']} ---")
        r = run_model(cfg)
        if r:
            all_results[cfg["name"]] = r

    out_path = "logs/topk_detection.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table: enrichment at each k
    print("\n" + "=" * 70)
    print("SUMMARY: Enrichment (Precision@k / Base Rate)")
    print("=" * 70)
    k_vals = [r["k"] for r in next(iter(all_results.values()))["precision_at_k"]]
    header = f"{'Model':18s} | " + " | ".join(f"k={k:5d}" for k in k_vals)
    print(header)
    print("-" * len(header))
    for model_name, res in all_results.items():
        enrichments = [f"{r['enrichment']:6.2f}x" for r in res["precision_at_k"]]
        print(f"{model_name:18s} | " + " | ".join(enrichments))

    print()
    print("Random baseline enrichment = 1.00x for all k.")


if __name__ == "__main__":
    main()
