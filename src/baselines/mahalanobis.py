"""
Mahalanobis distance baseline for bias detection.

Computes the Mahalanobis distance of each training sample's activations
from the validation set distribution. Higher distance = more likely biased.

Follows the approach in Lee et al. (2018) adapted for reward models:
- Extract last-hidden-state activations for chosen and rejected responses
- Concatenate them to form a feature vector
- Fit Gaussian (mean, covariance) on validation activations
- Score each training sample by Mahalanobis distance
"""

import torch
import datasets
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(examples):
    batch = {}
    for key in ["input_ids_chosen", "attention_mask_chosen",
                "input_ids_rejected", "attention_mask_rejected"]:
        batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples]).to(device)
    return batch


def get_activations(dataset, model, layer_idx=-1, batch_size=4):
    """Extract last-hidden-state activations for chosen and rejected responses.

    Args:
        dataset: Tokenized preference dataset.
        model: Reward model with a transformer backbone.
        layer_idx: Which transformer layer to use (-1 = last).
        batch_size: Batch size for inference.

    Returns:
        np.array of shape (n_samples, 2*hidden_dim) — concatenated
        chosen and rejected activations.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_activations = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting activations"):
            # Get hidden states for chosen
            outputs_chosen = model.base_model.model.model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
                output_hidden_states=True,
            )
            # Last token of last hidden state
            hidden_chosen = outputs_chosen.hidden_states[layer_idx]
            # Use the last non-padding token
            seq_lens_chosen = batch["attention_mask_chosen"].sum(dim=1) - 1
            act_chosen = hidden_chosen[torch.arange(hidden_chosen.size(0)), seq_lens_chosen]

            # Get hidden states for rejected
            outputs_rejected = model.base_model.model.model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
                output_hidden_states=True,
            )
            hidden_rejected = outputs_rejected.hidden_states[layer_idx]
            seq_lens_rejected = batch["attention_mask_rejected"].sum(dim=1) - 1
            act_rejected = hidden_rejected[torch.arange(hidden_rejected.size(0)), seq_lens_rejected]

            # Concatenate chosen and rejected activations
            combined = torch.cat([act_chosen, act_rejected], dim=1)
            all_activations.append(combined.cpu().float().numpy())

    return np.concatenate(all_activations, axis=0)


def compute_mahalanobis_scores(train_activations, eval_activations):
    """Compute Mahalanobis distance of train samples from eval distribution.

    Args:
        train_activations: (n_train, d) array.
        eval_activations: (n_eval, d) array.

    Returns:
        (n_train,) array of Mahalanobis distances.
    """
    # Fit Gaussian on eval activations
    mu = np.mean(eval_activations, axis=0)
    diff = eval_activations - mu
    sigma = np.cov(diff, rowvar=False)

    # Regularize covariance for numerical stability
    sigma += 1e-6 * np.eye(sigma.shape[0])
    sigma_inv = np.linalg.inv(sigma)

    # Compute Mahalanobis distance for each training sample
    train_diff = train_activations - mu
    scores = np.sqrt(np.sum(train_diff @ sigma_inv * train_diff, axis=1))

    return scores


def get_mahalanobis_score(model_path, train_datapath, eval_datapath,
                          load_in_4bit=True):
    """End-to-end Mahalanobis baseline computation."""
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, quantization_config=quantization_config
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)
    model.eval()

    train_dataset = datasets.load_from_disk(train_datapath)
    eval_dataset = datasets.load_from_disk(eval_datapath)

    # Try multiple layers, pick the one with best AUC
    train_act = get_activations(train_dataset, model)
    eval_act = get_activations(eval_dataset, model)

    scores = compute_mahalanobis_scores(train_act, eval_act)
    return scores
