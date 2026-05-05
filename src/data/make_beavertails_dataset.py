"""
Prepare BeaverTails dataset for safety bias detection experiments.

BeaverTails: PKU-Alignment/BeaverTails on HuggingFace
Format: prompt, response, is_safe (bool), category

Strategy:
- Group by prompt; pair safe response (chosen) vs unsafe response (rejected)
- Flip ~6.5% of labels (same ratio as length/sycophancy datasets)
- Create train/test splits matching the reward model pipeline format

Tokenizer: Qwen/Qwen2.5-1.5B-Instruct (to match the 1.5B reward model)

Usage:
    python src/data/make_beavertails_dataset.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import datasets
from transformers import AutoTokenizer

FLIP_RATIO  = 0.065
MAX_LENGTH  = 1024
TRAIN_FRAC  = 0.80   # use 80/20 split regardless of total pair count
SEED        = 42
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_RAW  = "dataset/beavertails_dataset"
OUTPUT_TOK  = "dataset/beavertails_dataset_tokenized"
OUTPUT_TRL  = "dataset/beavertails_dataset_trl"


def build_pairs(ds) -> list[dict]:
    """Group by prompt and create safe-vs-unsafe preference pairs."""
    prompt_groups = defaultdict(lambda: {"safe": [], "unsafe": []})
    for ex in ds:
        key = "safe" if ex["is_safe"] else "unsafe"
        prompt_groups[ex["prompt"]][key].append(ex["response"])

    pairs = []
    for prompt, group in prompt_groups.items():
        if not group["safe"] or not group["unsafe"]:
            continue
        # One pair per prompt: first safe as chosen, first unsafe as rejected
        pairs.append({
            "chosen":   f"\n\nHuman: {prompt}\n\nAssistant: {group['safe'][0]}",
            "rejected": f"\n\nHuman: {prompt}\n\nAssistant: {group['unsafe'][0]}",
            "prompt":   prompt,
        })
    return pairs


def flip_labels(pairs, flip_ratio, rng):
    """Randomly flip chosen/rejected for a fraction of pairs."""
    n = len(pairs)
    n_flip = int(n * flip_ratio)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    flip_set = set(flip_idx.tolist())
    for i in flip_idx:
        pairs[i]["chosen"], pairs[i]["rejected"] = pairs[i]["rejected"], pairs[i]["chosen"]
    return pairs, np.array(sorted(flip_idx))


def tokenize(pairs, tokenizer) -> list[dict]:
    out = []
    for pair in pairs:
        enc_c = tokenizer(
            pair["chosen"], truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors=None,
        )
        enc_r = tokenizer(
            pair["rejected"], truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors=None,
        )
        out.append({
            "input_ids_chosen":      enc_c["input_ids"],
            "attention_mask_chosen": enc_c["attention_mask"],
            "input_ids_rejected":      enc_r["input_ids"],
            "attention_mask_rejected": enc_r["attention_mask"],
        })
    return out


def main():
    rng = np.random.default_rng(SEED)

    print("Loading BeaverTails from HuggingFace...")
    raw = datasets.load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    print(f"  Raw rows: {len(raw)}")

    print("Building preference pairs...")
    pairs = build_pairs(raw)
    print(f"  Pairs formed: {len(pairs)}")

    # Shuffle and 80/20 split
    rng.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    train_pairs = pairs[:n_train]
    test_pairs  = pairs[n_train:]
    print(f"  Split: {len(train_pairs)} train / {len(test_pairs)} test")

    # Flip labels in training split
    print(f"  Flipping {FLIP_RATIO:.1%} of training labels...")
    train_pairs, flipped_idx = flip_labels(train_pairs, FLIP_RATIO, rng)
    print(f"  Flipped: {len(flipped_idx)} samples")

    # Save raw text dataset
    Path(OUTPUT_RAW).mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list([
        {"chosen": p["chosen"], "rejected": p["rejected"], "flipped": False}
        for p in train_pairs
    ]).save_to_disk(f"{OUTPUT_RAW}/train")
    datasets.Dataset.from_list([
        {"chosen": p["chosen"], "rejected": p["rejected"], "flipped": False}
        for p in test_pairs
    ]).save_to_disk(f"{OUTPUT_RAW}/test")
    np.save(f"{OUTPUT_RAW}/flipped_indices.npy", flipped_idx)
    # All test samples used as "safe reference" (no concise split needed)
    np.save(f"{OUTPUT_RAW}/concise_indices.npy", np.arange(len(test_pairs)))
    print(f"  Raw dataset saved to {OUTPUT_RAW}")

    # Tokenize
    print(f"Loading tokenizer {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing training pairs...")
    train_tok = tokenize(train_pairs, tokenizer)
    print("Tokenizing test pairs...")
    test_tok  = tokenize(test_pairs,  tokenizer)

    Path(OUTPUT_TOK).mkdir(parents=True, exist_ok=True)
    train_tok_ds = datasets.Dataset.from_list(train_tok)
    test_tok_ds  = datasets.Dataset.from_list(test_tok)
    train_tok_ds.save_to_disk(f"{OUTPUT_TOK}/train")
    test_tok_ds .save_to_disk(f"{OUTPUT_TOK}/test")
    print(f"  Tokenized dataset saved to {OUTPUT_TOK}")

    # TRL-compatible format (chosen_ids / rejected_ids column names)
    Path(OUTPUT_TRL).mkdir(parents=True, exist_ok=True)
    col_map = {
        "input_ids_chosen":      "chosen_ids",
        "attention_mask_chosen": "chosen_attention_mask",
        "input_ids_rejected":    "rejected_ids",
        "attention_mask_rejected": "rejected_attention_mask",
    }
    train_tok_ds.rename_columns(col_map).save_to_disk(f"{OUTPUT_TRL}/train")
    test_tok_ds .rename_columns(col_map).save_to_disk(f"{OUTPUT_TRL}/test")
    print(f"  TRL dataset saved to {OUTPUT_TRL}")
    print(f"\nDone. Train: {len(train_tok)}, Test: {len(test_tok)}, Flipped: {len(flipped_idx)}")


if __name__ == "__main__":
    main()
