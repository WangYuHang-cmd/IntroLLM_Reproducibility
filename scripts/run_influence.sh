#!/bin/bash
# Phase 3B + 4A: Compute influence scores and evaluate

set -e

MODEL_SIZE="${1:-1.5B}"

echo "=========================================="
echo "Phase 3B+4A: Influence Computation ($MODEL_SIZE)"
echo "=========================================="

for BIAS_TYPE in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-${MODEL_SIZE}_${BIAS_TYPE}"
    DATA_DIR="dataset/${BIAS_TYPE}_dataset"

    if [ ! -f "$MODEL_PATH/rapid_grad_train.pt" ]; then
        echo "Gradients not found: $MODEL_PATH, skipping"
        continue
    fi

    echo ""
    echo "--- Computing influence: $BIAS_TYPE ($MODEL_SIZE) ---"
    python -m src.influence.compute_influence \
        --model_path "$MODEL_PATH" \
        --dataset_dir "$DATA_DIR" \
        --bias_type "$BIAS_TYPE" \
        --K 65536
done

echo ""
echo "=== Influence computation complete ==="
