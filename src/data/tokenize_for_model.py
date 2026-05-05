"""
Tokenize existing raw-text datasets for a specific model's tokenizer.
Used for gradient caching (which needs pre-tokenized + padded data).

The raw dataset (chosen/rejected text) is shared across all models.
Each model needs its own tokenized version due to different tokenizers.

Usage:
    python src/data/tokenize_for_model.py \
        --model_name Qwen/Qwen3-1.7B \
        --tag qwen3_1.7b
"""

import argparse
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Short tag for output dir, e.g. 'qwen3_1.7b'")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--store_dir", type=str, default="dataset")
    return parser.parse_args()


def tokenize_and_pad(example, tokenizer, max_length):
    tok_c = tokenizer(example["chosen"], truncation=True, max_length=max_length)
    tok_r = tokenizer(example["rejected"], truncation=True, max_length=max_length)
    pad_id = tokenizer.pad_token_id

    pad_c = max_length - len(tok_c["input_ids"])
    pad_r = max_length - len(tok_r["input_ids"])

    return {
        "input_ids_chosen": tok_c["input_ids"] + [pad_id] * pad_c,
        "attention_mask_chosen": tok_c["attention_mask"] + [0] * pad_c,
        "input_ids_rejected": tok_r["input_ids"] + [pad_id] * pad_r,
        "attention_mask_rejected": tok_r["attention_mask"] + [0] * pad_r,
    }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for bias in ["length", "sycophancy"]:
        raw_dir = f"{args.store_dir}/{bias}_dataset"
        out_dir = f"{args.store_dir}/{bias}_dataset_tokenized_{args.tag}"

        print(f"Tokenizing {bias} dataset for {args.model_name} -> {out_dir}")

        for split in ["train", "test"]:
            ds = load_from_disk(f"{raw_dir}/{split}")
            print(f"  {split}: {len(ds)} samples")

            ds_tok = ds.map(
                lambda x: tokenize_and_pad(x, tokenizer, args.max_length),
                remove_columns=[c for c in ds.column_names
                                if c not in []],
            )
            # Filter over-length (should be rare since we truncate)
            # Keep only tokenized columns
            keep_cols = ["input_ids_chosen", "attention_mask_chosen",
                         "input_ids_rejected", "attention_mask_rejected"]
            remove_cols = [c for c in ds_tok.column_names if c not in keep_cols]
            ds_tok = ds_tok.remove_columns(remove_cols)

            ds_tok.save_to_disk(f"{out_dir}/{split}")
            print(f"  Saved {split}: {len(ds_tok)} samples")

        # Copy index files from the default tokenized version
        for idx_file in ["flipped_indices.npy", "concise_indices.npy",
                         "verbose_indices.npy", "less_syco_indices.npy",
                         "more_syco_indices.npy"]:
            src = f"{raw_dir}/{idx_file}"
            try:
                arr = np.load(src)
                np.save(f"{args.store_dir}/{bias}_dataset/{idx_file}", arr)
            except FileNotFoundError:
                pass

    print("Done!")


if __name__ == "__main__":
    main()
