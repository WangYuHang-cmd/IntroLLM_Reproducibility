"""
Prepare Anthropic-HH length and sycophancy bias datasets.

Saves TWO versions of each dataset:
1. Raw text (chosen/rejected columns) -> for TRL RewardTrainer
2. Pre-tokenized + padded (input_ids_*/attention_mask_*) -> for gradient caching

Usage:
    python src/data/make_hh_dataset.py \
        --tokenizer_name Qwen/Qwen2.5-1.5B-Instruct \
        --max_length 1024 \
        --store_dir dataset
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--store_dir", type=str, default="dataset")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    return parser.parse_args()


def tokenize_example(example, tokenizer, max_length):
    """Tokenize a single example with chosen/rejected text columns."""
    tok_c = tokenizer(example["chosen"], truncation=True, max_length=max_length)
    tok_r = tokenizer(example["rejected"], truncation=True, max_length=max_length)
    return {
        "input_ids_chosen": tok_c["input_ids"],
        "attention_mask_chosen": tok_c["attention_mask"],
        "input_ids_rejected": tok_r["input_ids"],
        "attention_mask_rejected": tok_r["attention_mask"],
    }


def pad_example(example, tokenizer, max_length=1024):
    """Pad input_ids and attention_mask to max_length."""
    pad_id = tokenizer.pad_token_id
    for suffix in ["chosen", "rejected"]:
        ids_key = f"input_ids_{suffix}"
        mask_key = f"attention_mask_{suffix}"
        pad_len = max_length - len(example[ids_key])
        example[ids_key] = example[ids_key] + [pad_id] * pad_len
        example[mask_key] = example[mask_key] + [0] * pad_len
    return example


def prepare_length_dataset(args, tokenizer):
    print(f"Loading Length Bias Dataset -> {args.store_dir}/length_dataset")
    dataset = load_dataset("Taywon/HH_length_biased_15k")

    # Step 1: Tokenize (keep original text columns for TRL)
    dataset = dataset.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
    )

    # Step 2: Filter by max_length
    dataset = dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length
        and len(x["input_ids_rejected"]) <= args.max_length
    )

    train_size = len(dataset["train"])
    test_size = len(dataset["test"])
    flipped = np.array(dataset["train"]["flipped"])
    n_flipped = int(flipped.sum())
    print(f"  Train: {train_size}, Test: {test_size}")
    print(f"  Flipped (corrupted): {n_flipped} ({100*n_flipped/train_size:.2f}%)")

    flipped_indices = np.where(flipped)[0]

    # Concise/Verbose indices (based on token length before padding)
    shorter_indices, longer_indices = [], []
    for i, example in enumerate(dataset["test"]):
        if len(example["input_ids_chosen"]) < len(example["input_ids_rejected"]):
            shorter_indices.append(i)
        else:
            longer_indices.append(i)
    print(f"  Concise (shorter chosen): {len(shorter_indices)}")
    print(f"  Verbose (longer chosen): {len(longer_indices)}")

    # Save version 1: raw text for TRL training (chosen, rejected, flipped)
    raw_dataset = dataset.remove_columns([
        "input_ids_chosen", "attention_mask_chosen",
        "input_ids_rejected", "attention_mask_rejected",
    ])
    raw_dataset.save_to_disk(f"{args.store_dir}/length_dataset")

    # Save version 2: tokenized + padded for gradient caching
    tok_dataset = dataset.map(lambda x: pad_example(x, tokenizer, args.max_length))
    tok_dataset = tok_dataset.remove_columns(["chosen", "rejected", "flipped"])
    tok_dataset["train"].save_to_disk(f"{args.store_dir}/length_dataset_tokenized/train")
    tok_dataset["test"].save_to_disk(f"{args.store_dir}/length_dataset_tokenized/test")

    # Save indices
    np.save(f"{args.store_dir}/length_dataset/flipped_indices.npy", flipped_indices)
    np.save(f"{args.store_dir}/length_dataset/concise_indices.npy", np.array(shorter_indices))
    np.save(f"{args.store_dir}/length_dataset/verbose_indices.npy", np.array(longer_indices))

    return dataset


def prepare_sycophancy_dataset(args, tokenizer):
    print(f"Loading Sycophancy Bias Dataset -> {args.store_dir}/sycophancy_dataset")
    dataset = load_dataset("Taywon/HH_sycophancy_biased_15k")

    # Step 1: Tokenize (keep original text columns)
    dataset = dataset.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
    )

    # Step 2: Filter by max_length
    dataset = dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length
        and len(x["input_ids_rejected"]) <= args.max_length
    )

    train_size = len(dataset["train"])
    test_size = len(dataset["test"])
    flipped = np.array(dataset["train"]["flipped"])
    n_flipped = int(flipped.sum())
    print(f"  Train: {train_size}, Test: {test_size}")
    print(f"  Flipped (corrupted): {n_flipped} ({100*n_flipped/train_size:.2f}%)")

    flipped_indices = np.where(flipped)[0]

    # Less/More Sycophantic validation subsets
    less_syco_indices, more_syco_indices = [], []
    if "answer_gpt" in dataset["test"].column_names:
        for i, example in enumerate(dataset["test"]):
            if example["answer_gpt"] == 1:
                less_syco_indices.append(i)
            elif example["answer_gpt"] == 2:
                more_syco_indices.append(i)
    else:
        print("  Note: answer_gpt not in test split, using length-based proxy")
        for i, example in enumerate(dataset["test"]):
            if len(example["input_ids_chosen"]) <= len(example["input_ids_rejected"]):
                less_syco_indices.append(i)
            else:
                more_syco_indices.append(i)
    print(f"  Less Sycophantic validation: {len(less_syco_indices)}")
    print(f"  More Sycophantic validation: {len(more_syco_indices)}")

    # Save version 1: raw text for TRL training
    raw_cols_to_remove = ["input_ids_chosen", "attention_mask_chosen",
                          "input_ids_rejected", "attention_mask_rejected"]
    raw_dataset = dataset.remove_columns(raw_cols_to_remove)
    raw_dataset.save_to_disk(f"{args.store_dir}/sycophancy_dataset")

    # Save version 2: tokenized + padded for gradient caching
    tok_dataset = dataset.map(lambda x: pad_example(x, tokenizer, args.max_length))
    cols_to_keep = ["input_ids_chosen", "attention_mask_chosen",
                    "input_ids_rejected", "attention_mask_rejected"]
    cols_to_remove = [c for c in tok_dataset["train"].column_names if c not in cols_to_keep]
    tok_dataset = tok_dataset.remove_columns(cols_to_remove)
    tok_dataset["train"].save_to_disk(f"{args.store_dir}/sycophancy_dataset_tokenized/train")
    tok_dataset["test"].save_to_disk(f"{args.store_dir}/sycophancy_dataset_tokenized/test")

    # Save indices
    np.save(f"{args.store_dir}/sycophancy_dataset/flipped_indices.npy", flipped_indices)
    np.save(f"{args.store_dir}/sycophancy_dataset/less_syco_indices.npy", np.array(less_syco_indices))
    np.save(f"{args.store_dir}/sycophancy_dataset/more_syco_indices.npy", np.array(more_syco_indices))

    return dataset


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prepare_length_dataset(args, tokenizer)
    prepare_sycophancy_dataset(args, tokenizer)
    print("Done!")
