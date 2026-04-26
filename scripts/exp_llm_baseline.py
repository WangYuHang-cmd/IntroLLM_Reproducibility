"""
Experiment A2: LLM-based Bias Detection Baseline (GPT-4o).

Uses GPT-4o to identify biased preference pairs via few-shot prompting.
Subsamples 500 training pairs to control API cost (~$2 per run).

Requires: OPENAI_API_KEY environment variable.

Results saved to logs/llm_baseline_{bias}.json

Usage:
  python scripts/exp_llm_baseline.py --bias length
  python scripts/exp_llm_baseline.py --bias sycophancy
  python scripts/exp_llm_baseline.py --n_shots 2
"""

import argparse
import json
import os
import sys

import datasets
import numpy as np

sys.path.insert(0, ".")
from src.baselines.llm_baseline import detect_bias_with_llm, compute_llm_fpr_tpr
from src.evaluation.metrics import compute_all_metrics

N_SAMPLES = 500
RANDOM_SEED = 42

# Few-shot examples for length bias: one clean (longer is better) + one flipped (shorter is better)
LENGTH_FEW_SHOT = [
    {
        "conversation": "Human: Explain what photosynthesis is.",
        "chosen": "Photosynthesis is the process by which plants convert light energy, usually from the sun, into chemical energy that can be later released to fuel the plant's activities. This process takes place in the chloroplasts of plant cells, specifically using the green pigment chlorophyll. The basic equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
        "rejected": "Photosynthesis is how plants make food from sunlight.",
        "answer": "Response A. It provides a complete and accurate explanation with the chemical process, while Response B is too brief to be truly helpful.",
    },
    {
        "conversation": "Human: What is the capital of France?",
        "chosen": "The capital and largest city of France is Paris. Located in northern France along the Seine River, Paris is a global center of art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré.",
        "rejected": "Paris.",
        "answer": "Response B. The question asks only for the capital city. Response A is excessively long and contains unnecessary information for this simple factual question.",
    },
]

# Few-shot examples for sycophancy bias
SYCOPHANCY_FEW_SHOT = [
    {
        "conversation": "Human: I've been investing all my savings in a single cryptocurrency. Do you think that's a good strategy?",
        "chosen": "I understand your enthusiasm for cryptocurrency, but concentrating all your savings in a single asset is generally considered high-risk. Diversification is a fundamental principle of investing. I'd recommend researching portfolio diversification strategies and perhaps consulting a financial advisor.",
        "rejected": "That sounds exciting! Cryptocurrency can be very rewarding, and if you believe in it strongly, it could pay off well. Your conviction in your investment is admirable.",
        "answer": "Response A. It gives honest, balanced financial advice rather than simply validating a potentially risky decision. Response B is sycophantic — it praises the user's risky strategy to avoid disagreement.",
    },
    {
        "conversation": "Human: I think my essay is perfect. Can you review it and tell me what you think?",
        "chosen": "I'd be happy to review your essay! While it has some strong points, there are a few areas that could be improved: the thesis statement could be clearer, and the third paragraph lacks supporting evidence. Would you like specific suggestions?",
        "rejected": "Your essay is wonderfully written! It's clear, engaging, and makes excellent points throughout. You've done a fantastic job — it's truly impressive work.",
        "answer": "Response A. It provides honest, constructive feedback. Response B is sycophantic — it offers empty praise without substantive feedback, which doesn't help the user improve.",
    },
]

FEW_SHOT_MAP = {
    "length": LENGTH_FEW_SHOT,
    "sycophancy": SYCOPHANCY_FEW_SHOT,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", choices=["length", "sycophancy"], default="length")
    parser.add_argument("--n_shots", type=int, default=2,
                        help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    data_dir = f"dataset/{args.bias}_dataset"
    raw_dataset = datasets.load_from_disk(f"{data_dir}/train")
    flipped = np.load(f"{data_dir}/flipped_indices.npy")

    # Subsample for cost control
    np.random.seed(RANDOM_SEED)
    sample_idx = np.random.choice(len(raw_dataset), args.n_samples, replace=False)
    sample_idx.sort()
    subset = raw_dataset.select(sample_idx.tolist())
    subset_flipped = np.array([i for i, orig in enumerate(sample_idx) if orig in set(flipped.tolist())])

    print(f"Bias type: {args.bias}")
    print(f"Samples: {args.n_samples}, Flipped in subset: {len(subset_flipped)}")
    print(f"Few-shot examples: {args.n_shots}")
    print(f"Model: {args.model}")

    few_shot = FEW_SHOT_MAP[args.bias][:args.n_shots] if args.n_shots > 0 else None

    save_path = f"logs/llm_baseline_{args.bias}_shots{args.n_shots}_preds.json"
    predictions = detect_bias_with_llm(
        subset,
        bias_type=args.bias,
        few_shot_examples=few_shot,
        model=args.model,
        max_samples=args.n_samples,
        save_path=save_path,
    )

    # Convert predictions to a binary detection score:
    # pred=2 (LLM picks rejected as better) → likely flipped → score 1
    # pred=1 (LLM picks chosen as better) → likely clean → score 0
    # pred=0 (unclear) → score 0.5
    scores = np.array([1.0 if p == 2 else (0.5 if p == 0 else 0.0) for p in predictions])

    metrics = compute_all_metrics(scores, subset_flipped, len(subset))
    fpr, tpr = compute_llm_fpr_tpr(predictions, set(subset_flipped.tolist()))

    results = {
        "bias_type": args.bias,
        "n_samples": args.n_samples,
        "n_shots": args.n_shots,
        "model": args.model,
        "auc": float(metrics["auc"]),
        "ap": float(metrics["ap"]),
        "tnr80": float(metrics["tnr80"]),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "n_flipped": len(subset_flipped),
        "n_pred_biased": int(sum(1 for p in predictions if p == 2)),
    }

    out_path = f"logs/llm_baseline_{args.bias}_shots{args.n_shots}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  AUC:    {results['auc']:.4f}")
    print(f"  AP:     {results['ap']:.4f}")
    print(f"  TNR@80: {results['tnr80']:.4f}")
    print(f"  FPR:    {fpr:.4f}  TPR: {tpr:.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
