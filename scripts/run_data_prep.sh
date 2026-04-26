#!/bin/bash
# Phase 1: Data preparation
# Downloads and tokenizes Anthropic-HH biased datasets + HelpSteer2

set -e

TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
STORE_DIR="dataset"

echo "=== Phase 1A/1B: Preparing Anthropic-HH length + sycophancy datasets ==="
python src/data/make_hh_dataset.py \
    --tokenizer_name "$TOKENIZER" \
    --max_length 1024 \
    --store_dir "$STORE_DIR"

echo ""
echo "=== Phase 1C: Preparing HelpSteer2 dataset ==="
python src/labeling_strategy/helpsteer_data.py \
    --tokenizer_name "$TOKENIZER" \
    --max_length 1024 \
    --store_dir "$STORE_DIR/helpsteer2"

echo ""
echo "=== Data preparation complete ==="
echo "Datasets saved to $STORE_DIR/"
