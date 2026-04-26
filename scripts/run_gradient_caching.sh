#!/bin/bash
# Phase 3A: Cache per-sample gradients with OPORP compression

set -e

MODEL_SIZE="${1:-1.5B}"
TOKENIZER="Qwen/Qwen2.5-7B-Instruct"

if [ "$MODEL_SIZE" == "1.5B" ]; then
    TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"
fi

echo "=========================================="
echo "Phase 3A: Gradient Caching ($MODEL_SIZE)"
echo "=========================================="

for BIAS_TYPE in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-${MODEL_SIZE}_${BIAS_TYPE}"
    DATA_DIR="dataset/${BIAS_TYPE}_dataset"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model not found: $MODEL_PATH, skipping"
        continue
    fi

    echo ""
    echo "--- Caching train gradients: $BIAS_TYPE ($MODEL_SIZE) ---"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${DATA_DIR}/train" \
        --save_name "rapid_grad_train.pt" \
        --K 65536

    echo ""
    echo "--- Caching validation gradients: $BIAS_TYPE ($MODEL_SIZE) ---"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${DATA_DIR}/test" \
        --save_name "rapid_grad_val.pt" \
        --K 65536
done

echo ""
echo "=== Gradient caching complete ==="
