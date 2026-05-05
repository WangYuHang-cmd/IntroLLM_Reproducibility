"""
Train a reward model with LoRA on Qwen2.5 using TRL RewardTrainer.

This is a standalone script that does NOT depend on the alignment-handbook package.
It reads a YAML config and uses standard transformers/peft/trl APIs.

Usage:
    python src/reward_modeling/train.py --config configs/reward_model_length.yaml
    python src/reward_modeling/train.py --config configs/reward_model_length_1.5B.yaml
"""

import argparse
import warnings
from typing import Dict

import datasets
import numpy as np
import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # TRL >=1.0 returns shape (N, 1) margin or (N, 2) [reward_chosen, reward_rejected]
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Single column: margin (reward_chosen - reward_rejected), positive = correct
        margins = predictions.flatten()
        accuracy = (margins > 0).astype(float).mean().item()
    else:
        # Two columns: [reward_chosen, reward_rejected]
        accuracy = (predictions[:, 0] > predictions[:, 1]).astype(float).mean().item()
    return {"accuracy": accuracy}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Model setup
    model_name = cfg["model_name_or_path"]
    torch_dtype = getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    attn_impl = cfg.get("attn_implementation", None)

    print(f"Loading model: {model_name}")
    model_kwargs = {"num_labels": 1, "trust_remote_code": True}
    # transformers >=4.57 uses 'dtype' instead of 'torch_dtype'
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=torch_dtype, attn_implementation=attn_impl, **model_kwargs,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch_dtype, attn_implementation=attn_impl, **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    # Handle eos_token_id being a list (e.g. Llama-3.2)
    eos_id = model.config.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if model.config.pad_token_id is None:
        model.config.pad_token_id = eos_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    # LoRA config
    peft_config = None
    if cfg.get("use_peft", False):
        peft_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            target_modules=cfg.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ]),
            modules_to_save=cfg.get("lora_modules_to_save", ["score"]),
            bias="none",
            task_type="SEQ_CLS",
        )

    # Load tokenized datasets from disk
    dataset_dir = cfg.get("dataset_dir", None)
    bias_type = cfg["bias_type"]
    if dataset_dir is None:
        dataset_dir = f"dataset/{bias_type}_dataset"

    print(f"Loading datasets from {dataset_dir}")
    train_dataset = datasets.load_from_disk(f"{dataset_dir}/train")
    eval_dataset = datasets.load_from_disk(f"{dataset_dir}/test")
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training config
    output_dir = cfg.get("output_dir", f"logs/{model_name.split('/')[-1]}_{bias_type}")
    reward_config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 4),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 64),
        learning_rate=cfg.get("learning_rate", 2e-5),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),  # deprecated warning is safe to ignore
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg.get("max_length", 1024),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        seed=cfg.get("seed", 42),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    # Train (TRL >=1.0 uses 'processing_class' instead of 'tokenizer')
    try:
        trainer = RewardTrainer(
            model=model,
            processing_class=tokenizer,
            args=reward_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            compute_metrics=compute_accuracy,
        )
    except TypeError:
        trainer = RewardTrainer(
            model=model,
            tokenizer=tokenizer,
            args=reward_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            compute_metrics=compute_accuracy,
        )

    print(f"Starting training -> {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
