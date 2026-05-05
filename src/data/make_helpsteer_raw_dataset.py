"""
Create raw-text HelpSteer2 training dataset for reward model training.

Downloads HelpSteer2, applies Bob's W_B_1 weights to create preference labels,
and saves as a text dataset with 'chosen'/'rejected' columns compatible with
the TRL RewardTrainer.

Also saves D_A (Alice's expert data) as test split.

Usage:
    python src/data/make_helpsteer_raw_dataset.py
"""

import sys
import numpy as np
import datasets as hf_datasets
from pathlib import Path

sys.path.insert(0, ".")
from src.labeling_strategy.helpsteer_data import (
    prepare_helpsteer2_pairs, label_pairs, compute_preference_label
)

W_ALICE = np.array([1.04, 0.46, 0.47, -0.33])
W_BOB_1 = np.array([1.1, 1.0, 3.1, 3.0])

OUTPUT_DIR = "dataset/helpsteer2_bob1_raw"


def pairs_to_text_dataset(pairs, weights):
    """Convert pairs to chosen/rejected text using the given weights."""
    records = []
    for pair in pairs:
        label = compute_preference_label(pair["scores_0"], pair["scores_1"], weights)
        if label == 1:
            chosen_resp, rejected_resp = pair["response_1"], pair["response_0"]
        else:
            chosen_resp, rejected_resp = pair["response_0"], pair["response_1"]

        records.append({
            "chosen":   f"Human: {pair['prompt']}\n\nAssistant: {chosen_resp}",
            "rejected": f"Human: {pair['prompt']}\n\nAssistant: {rejected_resp}",
            "scores_0": list(pair["scores_0"].astype(float)),
            "scores_1": list(pair["scores_1"].astype(float)),
        })
    return records


def main():
    print("Downloading HelpSteer2 (train split)...")
    train_pairs = prepare_helpsteer2_pairs(split="train")
    print(f"  Train pairs: {len(train_pairs)}")

    print("Downloading HelpSteer2 (validation split for D_A)...")
    val_pairs = prepare_helpsteer2_pairs(split="validation")
    print(f"  Val pairs:   {len(val_pairs)}")

    print("Creating Bob's labeled training dataset (W_B_1)...")
    train_records = pairs_to_text_dataset(train_pairs, W_BOB_1)

    print("Creating Alice's labeled test dataset (W_Alice)...")
    val_records = pairs_to_text_dataset(val_pairs, W_ALICE)

    # Check label agreement
    bob_labels   = label_pairs(train_pairs, W_BOB_1)
    alice_labels = label_pairs(train_pairs, W_ALICE)
    acc = np.mean(bob_labels == alice_labels)
    print(f"  Bob vs Alice label agreement (train): {acc:.4f}")

    # Save
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    hf_datasets.Dataset.from_list(train_records).save_to_disk(f"{OUTPUT_DIR}/train")
    hf_datasets.Dataset.from_list(val_records).save_to_disk(f"{OUTPUT_DIR}/test")

    print(f"\nSaved {len(train_records)} train / {len(val_records)} test pairs to {OUTPUT_DIR}")
    print("Train columns:", hf_datasets.load_from_disk(f"{OUTPUT_DIR}/train").column_names)


if __name__ == "__main__":
    main()
