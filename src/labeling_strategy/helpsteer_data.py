"""
HelpSteer2 dataset processing for the labeling strategy oversight experiment.

Implements Alice (expert) and Bob (non-expert) labeler simulation:
- Each response has 4 sub-objective scores: correctness, coherence, complexity, verbosity
- Alice uses optimal weights w_A = [1.04, 0.46, 0.47, -0.33]
- Bob uses 5 different suboptimal weight vectors
- Preference label z = I(w^T r^(0) < w^T r^(1))

Usage:
    python src/labeling_strategy/helpsteer_data.py --store_dir dataset/helpsteer2
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# Alice's expert weights (from HelpSteer2 RewardBench optimal)
W_ALICE = np.array([1.04, 0.46, 0.47, -0.33])

# Bob's 5 suboptimal weight vectors (from paper Appendix B.2)
W_BOB_LIST = [
    np.array([1.1, 1.0, 3.1, 3.0]),       # w_B^1
    np.array([2.1, 0.5, 4.9, 5.1]),       # w_B^2
    np.array([0.9, 5.9, 2.1, 3.1]),       # w_B^3
    np.array([0.9, 6.1, 5.8, 4.1]),       # w_B^4
    np.array([0.2, 0.9, 0.9, 1.1]),       # w_B^5
]

# Sub-objective column names in HelpSteer2 (exclude helpfulness)
SCORE_COLUMNS = ["correctness", "coherence", "complexity", "verbosity"]


def compute_preference_label(scores_0, scores_1, weights):
    """Compute binary preference label from sub-objective scores and weights.

    z = 1 if w^T r^(1) > w^T r^(0), else 0
    i.e., z=1 means response 1 is preferred.
    """
    score_0 = np.dot(weights, scores_0)
    score_1 = np.dot(weights, scores_1)
    return int(score_1 > score_0)


def prepare_helpsteer2_pairs(split="train"):
    """Load HelpSteer2 and create response pairs grouped by prompt.

    HelpSteer2 has multiple responses per prompt. We pair consecutive responses
    to create preference pairs.

    Returns:
        List of dicts with keys:
            prompt, response_0, response_1, scores_0, scores_1
    """
    dataset = load_dataset("nvidia/HelpSteer2", split=split)

    # Group by prompt
    prompt_to_responses = {}
    for example in dataset:
        prompt = example["prompt"]
        if prompt not in prompt_to_responses:
            prompt_to_responses[prompt] = []
        scores = np.array([example[col] for col in SCORE_COLUMNS])
        prompt_to_responses[prompt].append({
            "response": example["response"],
            "scores": scores,
        })

    # Create pairs from prompts with 2+ responses
    pairs = []
    for prompt, responses in prompt_to_responses.items():
        if len(responses) < 2:
            continue
        # Pair consecutive responses
        for i in range(0, len(responses) - 1, 2):
            pairs.append({
                "prompt": prompt,
                "response_0": responses[i]["response"],
                "response_1": responses[i + 1]["response"],
                "scores_0": responses[i]["scores"],
                "scores_1": responses[i + 1]["scores"],
            })

    return pairs


def label_pairs(pairs, weights):
    """Apply a labeler's weight vector to generate preference labels.

    Args:
        pairs: List of pair dicts from prepare_helpsteer2_pairs.
        weights: Weight vector (4-dim) for scoring.

    Returns:
        List of preference labels (0 or 1).
    """
    labels = []
    for pair in pairs:
        z = compute_preference_label(pair["scores_0"], pair["scores_1"], weights)
        labels.append(z)
    return np.array(labels)


def compute_label_accuracy(labels_bob, labels_alice):
    """Compute agreement between Bob's and Alice's labels."""
    return np.mean(labels_bob == labels_alice)


def tokenize_pairs(pairs, labels, tokenizer, max_length=1024):
    """Tokenize pairs into reward model training format.

    Args:
        pairs: List of pair dicts.
        labels: Array of preference labels (0 or 1).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        List of tokenized examples with input_ids_chosen, attention_mask_chosen,
        input_ids_rejected, attention_mask_rejected.
    """
    examples = []
    for pair, label in zip(pairs, labels):
        prompt = pair["prompt"]
        if label == 1:
            chosen = prompt + "\n" + pair["response_1"]
            rejected = prompt + "\n" + pair["response_0"]
        else:
            chosen = prompt + "\n" + pair["response_0"]
            rejected = prompt + "\n" + pair["response_1"]

        tok_chosen = tokenizer(chosen, truncation=True, max_length=max_length)
        tok_rejected = tokenizer(rejected, truncation=True, max_length=max_length)

        # Skip if either exceeds max_length
        if len(tok_chosen["input_ids"]) > max_length or len(tok_rejected["input_ids"]) > max_length:
            continue

        # Pad to max_length
        pad_id = tokenizer.pad_token_id
        pad_len_c = max_length - len(tok_chosen["input_ids"])
        pad_len_r = max_length - len(tok_rejected["input_ids"])

        examples.append({
            "input_ids_chosen": tok_chosen["input_ids"] + [pad_id] * pad_len_c,
            "attention_mask_chosen": tok_chosen["attention_mask"] + [0] * pad_len_c,
            "input_ids_rejected": tok_rejected["input_ids"] + [pad_id] * pad_len_r,
            "attention_mask_rejected": tok_rejected["attention_mask"] + [0] * pad_len_r,
            "scores_0": pair["scores_0"].tolist(),
            "scores_1": pair["scores_1"].tolist(),
        })

    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", type=str, default="dataset/helpsteer2")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    import os
    from datasets import Dataset

    args = parse_args()
    os.makedirs(args.store_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare pairs
    print("Preparing HelpSteer2 training pairs...")
    train_pairs = prepare_helpsteer2_pairs("train")
    val_pairs = prepare_helpsteer2_pairs("validation")
    print(f"  Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Generate Alice's labels
    alice_train_labels = label_pairs(train_pairs, W_ALICE)
    alice_val_labels = label_pairs(val_pairs, W_ALICE)

    # Generate and evaluate Bob's labels for each weight vector
    for idx, w_bob in enumerate(W_BOB_LIST):
        bob_labels = label_pairs(train_pairs, w_bob)
        acc = compute_label_accuracy(bob_labels, alice_train_labels)
        print(f"  Bob w_B^{idx+1} = {w_bob.tolist()} -> Label Acc: {acc:.4f}")

    # Tokenize with Alice's labels for D_A (validation set)
    print("Tokenizing Alice's validation set (D_A)...")
    alice_val_examples = tokenize_pairs(val_pairs, alice_val_labels, tokenizer, args.max_length)
    alice_val_ds = Dataset.from_list(alice_val_examples)
    alice_val_ds.save_to_disk(f"{args.store_dir}/D_A")
    print(f"  D_A size: {len(alice_val_ds)}")

    # Tokenize with Bob's labels for D_B (training set) — use first Bob weight as default
    for idx, w_bob in enumerate(W_BOB_LIST):
        bob_labels = label_pairs(train_pairs, w_bob)
        print(f"Tokenizing Bob w_B^{idx+1} training set (D_B)...")
        bob_train_examples = tokenize_pairs(train_pairs, bob_labels, tokenizer, args.max_length)
        bob_train_ds = Dataset.from_list(bob_train_examples)
        bob_train_ds.save_to_disk(f"{args.store_dir}/D_B_{idx+1}")
        print(f"  D_B_{idx+1} size: {len(bob_train_ds)}")

    # Save metadata
    np.save(f"{args.store_dir}/w_alice.npy", W_ALICE)
    for idx, w_bob in enumerate(W_BOB_LIST):
        np.save(f"{args.store_dir}/w_bob_{idx+1}.npy", w_bob)

    print("Done!")
