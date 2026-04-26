#!/bin/bash
# =============================================================================
# Phase 3 only: Gradient Caching + Influence Computation (1.5B)
# Prerequisites: Phase 1 (data) and Phase 2 (training) already complete
# Usage: CUDA_LAUNCH_BLOCKING=1 bash scripts/run_1.5B_phase3.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="disabled"

TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================="
echo " Phase 3: Gradient Caching + Influence (1.5B)"
echo " $(date)"
echo "============================================="

for BIAS in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-1.5B_${BIAS}"
    TOK_DIR="dataset/${BIAS}_dataset_tokenized"
    DATA_DIR="dataset/${BIAS}_dataset"

    echo ""
    if [ -f "$MODEL_PATH/rapid_grad_train.pt" ]; then
        echo ">>> SKIP: train gradients ($BIAS) already cached <<<"
    else
        echo ">>> Caching train gradients ($BIAS) <<<"
        python -m src.influence.cache_gradients \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER" \
            --data_path "${TOK_DIR}/train" \
            --save_name "rapid_grad_train.pt" \
            --K 65536
    fi

    echo ""
    if [ -f "$MODEL_PATH/rapid_grad_val.pt" ]; then
        echo ">>> SKIP: validation gradients ($BIAS) already cached <<<"
    else
        echo ">>> Caching validation gradients ($BIAS) <<<"
        python -m src.influence.cache_gradients \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER" \
            --data_path "${TOK_DIR}/test" \
            --save_name "rapid_grad_val.pt" \
            --K 65536
    fi

    echo ""
    if [ -f "$MODEL_PATH/influence_results.json" ]; then
        echo ">>> SKIP: influence scores ($BIAS) already computed <<<"
    else
        echo ">>> Computing influence scores ($BIAS) <<<"
        python -m src.influence.compute_influence \
            --model_path "$MODEL_PATH" \
            --dataset_dir "$DATA_DIR" \
            --bias_type "$BIAS" \
            --K 65536
    fi
done

echo ""
echo "============================================="
echo " Done! $(date)"
echo "============================================="
echo "Results:"
cat logs/Qwen2.5-1.5B_length/influence_results.json 2>/dev/null
echo ""
cat logs/Qwen2.5-1.5B_sycophancy/influence_results.json 2>/dev/null
