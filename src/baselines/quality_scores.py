"""
Self-confidence and Entropy baselines for bias detection.

These baselines use the reward model's own prediction uncertainty to identify
potentially mislabeled samples:
- Self-confidence: -p_chosen (lower confidence in chosen = more likely biased)
- Entropy: higher entropy in preference probability = more uncertain = more likely biased

Signs are flipped so that higher values = more biased (consistent with influence scores).
"""

import torch
import torch.nn.functional as F
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


def get_logits(model, dataset, batch_size=4):
    """Get reward model logits for all samples.

    Returns:
        List of (reward_chosen, reward_rejected) tuples.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logit_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing logits"):
            rewards_chosen = model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
            )["logits"].squeeze(-1)
            rewards_rejected = model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
            )["logits"].squeeze(-1)

            for rc, rr in zip(rewards_chosen.cpu(), rewards_rejected.cpu()):
                logit_list.append((rc.item(), rr.item()))

    return logit_list


def compute_quality_scores(logit_list):
    """Compute self-confidence and entropy scores.

    Args:
        logit_list: List of (reward_chosen, reward_rejected) tuples.

    Returns:
        dict with 'self_confidence' and 'entropy' arrays (higher = more biased).
    """
    self_confidence_scores = []
    entropy_scores = []

    for r_chosen, r_rejected in logit_list:
        # Bradley-Terry probability
        logits = torch.tensor([r_chosen, r_rejected])
        probs = F.softmax(logits, dim=0)

        p_chosen = probs[0].item()
        p_rejected = probs[1].item()

        # Self-confidence: negate p_chosen so higher = more biased
        self_confidence_scores.append(-p_chosen)

        # Entropy: higher entropy = more uncertain = potentially biased
        entropy = -(p_chosen * np.log(p_chosen + 1e-10) + p_rejected * np.log(p_rejected + 1e-10))
        entropy_scores.append(entropy)

    return {
        "self_confidence": np.array(self_confidence_scores),
        "entropy": np.array(entropy_scores),
    }


def get_quality_scores(model_path, train_datapath, load_in_4bit=True):
    """End-to-end quality score baseline computation."""
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

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    train_dataset = datasets.load_from_disk(train_datapath)
    logit_list = get_logits(model, train_dataset)
    return compute_quality_scores(logit_list)
