"""
Experiment 3B: Hidden State Probing.

Tests whether small models REPRESENT bias information internally even though
their gradients can't distinguish biased samples.

Extracts last-hidden-state representations and trains a linear probe to classify
flipped vs clean samples. If the probe works on 1.5B but influence fails,
the problem is in gradient computation, not representation.

Usage: python scripts/exp_hidden_probe.py
"""

import numpy as np
import torch
import datasets
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(examples):
    batch = {}
    for key in ["input_ids_chosen", "attention_mask_chosen",
                "input_ids_rejected", "attention_mask_rejected"]:
        batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples]).to(device)
    return batch


def extract_hidden_states(model, dataset, batch_size=2):
    """Extract last-hidden-state for chosen and rejected, concatenate."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_features = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            # Get last hidden state for chosen
            out_c = model.base_model.model.model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
                output_hidden_states=True,
            )
            # Last token of last layer
            h_c = out_c.hidden_states[-1]
            seq_len_c = batch["attention_mask_chosen"].sum(dim=1) - 1
            feat_c = h_c[torch.arange(h_c.size(0)), seq_len_c]

            # Get last hidden state for rejected
            out_r = model.base_model.model.model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
                output_hidden_states=True,
            )
            h_r = out_r.hidden_states[-1]
            seq_len_r = batch["attention_mask_rejected"].sum(dim=1) - 1
            feat_r = h_r[torch.arange(h_r.size(0)), seq_len_r]

            # Concatenate chosen and rejected representations
            features = torch.cat([feat_c, feat_r], dim=1)
            all_features.append(features.cpu().float().numpy())

            torch.cuda.empty_cache()

    return np.concatenate(all_features, axis=0)


def run_probe(model_name, model_id, bias_type="length"):
    """Run linear probe experiment for one model."""
    model_dir = f"logs/{model_name}_{bias_type}"
    data_dir = f"dataset/{bias_type}_dataset"

    if not Path(f"{model_dir}/adapter_model.safetensors").exists():
        print(f"  Skipping {model_name}: no checkpoint")
        return None

    # Determine tokenized data path
    if "llama" in model_id.lower() or "Llama" in model_name:
        tok_dir = f"dataset/{bias_type}_dataset_tokenized_llama3.2_3b"
    else:
        tok_dir = f"dataset/{bias_type}_dataset_tokenized"

    # Load model in 4-bit
    adapter_config_path = f"{model_dir}/adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading {base_model_name} (4-bit) + LoRA from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=1, quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.to(device)
    model.eval()

    # Load tokenized dataset
    train_dataset = datasets.load_from_disk(f"{tok_dir}/train")
    flipped = np.load(f"{data_dir}/flipped_indices.npy")
    flipped_set = set(flipped.tolist())
    labels = np.array([1 if i in flipped_set else 0 for i in range(len(train_dataset))])

    # Extract features (subsample for speed)
    n_total = len(train_dataset)
    n_sample = min(3000, n_total)
    np.random.seed(42)
    sample_idx = np.random.choice(n_total, n_sample, replace=False)
    sample_dataset = train_dataset.select(sample_idx)
    sample_labels = labels[sample_idx]

    print(f"  Extracting hidden states for {n_sample} samples...")
    features = extract_hidden_states(model, sample_dataset, batch_size=1)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Train linear probe (80/20 split)
    n_train = int(0.8 * n_sample)
    perm = np.random.permutation(n_sample)
    X_train, X_test = features[perm[:n_train]], features[perm[n_train:]]
    y_train, y_test = sample_labels[perm[:n_train]], sample_labels[perm[n_train:]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    probe_auc = roc_auc_score(y_test, y_prob)
    probe_acc = clf.score(X_test_s, y_test)

    return {
        "probe_auc": probe_auc,
        "probe_acc": probe_acc,
        "n_flipped_train": int(y_train.sum()),
        "n_clean_train": int((1 - y_train).sum()),
        "feature_dim": features.shape[1],
    }


def main():
    models = [
        ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
        ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
        ("Llama3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
        ("Llama3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ]

    print("=" * 60)
    print("Experiment 3B: Hidden State Probing")
    print("Can a linear probe on hidden states detect biased samples?")
    print("=" * 60)

    import time
    results = {}
    for i, (model_name, model_id) in enumerate(models):
        print(f"\n--- [{i+1}/{len(models)}] {model_name} (expect ~10-15 min) ---")
        t0 = time.time()
        r = run_probe(model_name, model_id, "length")
        if r:
            print(f"  Finished in {(time.time()-t0)/60:.1f} min")
            results[model_name] = r
            print(f"  Probe AUC:      {r['probe_auc']:.4f}")
            print(f"  Probe Accuracy: {r['probe_acc']:.4f}")
            print(f"  Feature dim:    {r['feature_dim']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Hidden Probe vs Influence Function")
    print("=" * 60)
    print(f"{'Model':18s} | {'Probe AUC':12s} | {'Influence AUC':14s} | {'Gap':10s}")
    print("-" * 60)

    influence_aucs = {
        "Qwen2.5-1.5B": 0.5419,
        "Qwen3-1.7B": 0.4791,
        "Qwen2.5-7B": 0.5053,
        "Llama3.2-1B": 0.6918,
        "Llama3.2-3B": 0.6996,
    }

    for model_name in results:
        probe_auc = results[model_name]["probe_auc"]
        inf_auc = influence_aucs.get(model_name, float("nan"))
        gap = probe_auc - inf_auc
        print(f"{model_name:18s} | {probe_auc:.4f}       | {inf_auc:.4f}         | {gap:+.4f}")

    print()
    print("If Probe AUC >> Influence AUC: the representation contains bias info")
    print("  but the gradient computation fails to capture it.")
    print("If Probe AUC ≈ Influence AUC: the model genuinely can't represent")
    print("  the distinction between biased and clean samples.")


if __name__ == "__main__":
    main()
