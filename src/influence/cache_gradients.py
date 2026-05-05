"""
Cache per-sample gradients with OPORP compression for influence function computation.

For each sample in a dataset, computes the Bradley-Terry preference loss, backpropagates,
collects LoRA parameter gradients, and compresses them with OPORP.

Usage:
    python src/influence/cache_gradients.py \
        --model_path logs/Qwen2.5-7B_length \
        --tokenizer_path Qwen/Qwen2.5-7B-Instruct \
        --data_path dataset/length_dataset/train \
        --save_name rapid_grad_train.pt

    python src/influence/cache_gradients.py \
        --model_path logs/Qwen2.5-7B_length \
        --tokenizer_path Qwen/Qwen2.5-7B-Instruct \
        --data_path dataset/length_dataset/test \
        --save_name rapid_grad_val.pt
"""

import argparse

import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from src.influence.oporp import RapidGrad, pad_gradient


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(examples):
    """Collate function for DataLoader — stacks tokenized preference pairs."""
    batch = {}
    for key in ["input_ids_chosen", "attention_mask_chosen",
                "input_ids_rejected", "attention_mask_rejected"]:
        batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples]).to(device)
    return batch


def compute_preference_loss(model, inputs):
    """Bradley-Terry preference loss: -log sigmoid(r_chosen - r_rejected).
    Runs chosen and rejected sequentially to save VRAM."""
    model.zero_grad()
    # Sequential forward to reduce peak memory
    rewards_chosen = model(
        input_ids=inputs["input_ids_chosen"],
        attention_mask=inputs["attention_mask_chosen"],
        return_dict=True,
    )["logits"]
    torch.cuda.empty_cache()
    rewards_rejected = model(
        input_ids=inputs["input_ids_rejected"],
        attention_mask=inputs["attention_mask_rejected"],
        return_dict=True,
    )["logits"]
    return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()


def prepare_model(model, device, peft_model_id):
    """Load LoRA adapters and set requires_grad for gradient computation."""
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.to(device)
    for name, param in model.named_parameters():
        if "lora" in name or "modules_to_save.default" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.eval()
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained LoRA checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to tokenized dataset on disk")
    parser.add_argument("--save_name", type=str, default="rapid_grad.pt")

    # Quantization (use --no-load_in_4bit for small models that fit in fp16)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")

    # OPORP
    parser.add_argument("--shuffle_lambda", type=int, default=100)
    parser.add_argument("--K", type=lambda x: list(map(int, x.split(","))),
                        default=[2**16],
                        help="Compression target dimensions (powers of 2)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Memory optimization
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()

    # Load model in 4-bit for memory efficiency during gradient caching
    quantization_config = None
    if args.load_in_4bit:
        compute_dtype = getattr(torch, args.torch_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    # Load base model (not the LoRA checkpoint — the original pretrained model)
    # The LoRA adapter config stores the base model name; PeftModel.from_pretrained
    # will load the adapter on top. We load the base model in 4-bit here.
    import json
    adapter_config_path = f"{args.model_path}/adapter_config.json"
    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"Loading base model: {base_model_name} (4-bit quantized)")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapters from: {args.model_path}")
    model = prepare_model(model, device, args.model_path)

    if model.config.pad_token_id is None:
        eos_id = model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        model.config.pad_token_id = eos_id

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = datasets.load_from_disk(args.data_path)

    # Initialize OPORP compressor
    oporp = RapidGrad(args.shuffle_lambda, args.model_path, device, args.seed)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Cache compressed gradients
    rapid_grad_dict = {key: [] for key in args.K}

    print(f"Computing gradients for {len(dataset)} samples...")
    for batch in tqdm(dataloader, desc="Caching gradients"):
        model.zero_grad()
        loss = compute_preference_loss(model, batch)
        loss.backward()

        # Collect LoRA gradients
        grad_list = []
        for name, param in model.named_parameters():
            if param.grad is not None and (
                "lora_A" in name or "lora_B" in name or "modules_to_save.default" in name
            ):
                grad_list.append(param.grad.reshape(-1))

        grad_vec = torch.cat(grad_list)
        grad_vec = pad_gradient(grad_vec)

        # Compress with OPORP
        compressed_vecs = oporp(grad_vec, args.K)
        if not isinstance(compressed_vecs, list):
            compressed_vecs = [compressed_vecs]
        for i, vec in enumerate(compressed_vecs):
            rapid_grad_dict[args.K[i]].append(vec.cpu())

    # Save
    save_path = f"{args.model_path}/{args.save_name}"
    torch.save(rapid_grad_dict, save_path)
    print(f"Saved compressed gradients to {save_path}")
    for k, v in rapid_grad_dict.items():
        print(f"  K={k}: {len(v)} vectors, shape={v[0].shape}")


if __name__ == "__main__":
    main()
