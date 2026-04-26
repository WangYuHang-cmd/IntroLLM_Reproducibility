#!/bin/bash
# =============================================================================
# Run baseline models on RTX 5080 (16GB)
# Models: Qwen3-1.7B, Qwen3-4B, Llama-3.2-3B-Instruct
#
# Each model: tokenize data -> train -> gradient caching -> influence
# Skips completed steps automatically.
#
# Usage:
#   conda activate if_rlhf
#   CUDA_LAUNCH_BLOCKING=1 bash scripts/run_baseline_models.sh [model]
#
# Examples:
#   bash scripts/run_baseline_models.sh              # run all 3 models
#   bash scripts/run_baseline_models.sh qwen3_1.7b   # run only Qwen3-1.7B
#   bash scripts/run_baseline_models.sh qwen3_4b     # run only Qwen3-4B
#   bash scripts/run_baseline_models.sh llama3.2_3b   # run only Llama-3.2-3B
# =============================================================================
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="disabled"

# Model definitions: tag|model_name|log_prefix|tok_tag
# tok_tag: which tokenized dataset to use (Qwen3 shares tokenizer with Qwen2.5)
MODELS=(
    "qwen3_1.7b|Qwen/Qwen3-1.7B|Qwen3-1.7B|default"
    "qwen3_4b|Qwen/Qwen3-4B|Qwen3-4B|default"
    "llama3.2_3b|meta-llama/Llama-3.2-3B-Instruct|Llama3.2-3B|llama3.2_3b"
)

# Filter by argument if provided
TARGET="${1:-all}"

run_model() {
    local TAG="$1"
    local MODEL_NAME="$2"
    local LOG_PREFIX="$3"
    local TOK_TAG="$4"

    echo ""
    echo "########################################################"
    echo " Model: $MODEL_NAME (tag=$TAG)"
    echo " $(date)"
    echo "########################################################"

    # --- Step 1: Tokenize data for this model ---
    # "default" means use the existing Qwen tokenized data (Qwen2.5/3 share tokenizer)
    if [ "$TOK_TAG" = "default" ]; then
        TOK_SUFFIX="_tokenized"
        echo ">>> Using default (Qwen) tokenized data <<<"
    else
        TOK_SUFFIX="_tokenized_${TOK_TAG}"
        if [ -d "dataset/length_dataset${TOK_SUFFIX}/train" ]; then
            echo ">>> SKIP: Data already tokenized for $TOK_TAG <<<"
        else
            echo ">>> Tokenizing data for $TOK_TAG <<<"
            python src/data/tokenize_for_model.py \
                --model_name "$MODEL_NAME" \
                --tag "$TOK_TAG"
        fi
    fi

    # --- Step 2+3: Train + gradient cache + influence for each bias type ---
    for BIAS in length sycophancy; do
        MODEL_PATH="logs/${LOG_PREFIX}_${BIAS}"
        TOK_DIR="dataset/${BIAS}_dataset${TOK_SUFFIX}"
        DATA_DIR="dataset/${BIAS}_dataset"
        CONFIG="configs/reward_model_${BIAS}_${TAG}.yaml"

        # Check config exists
        if [ ! -f "$CONFIG" ]; then
            echo "WARNING: Config not found: $CONFIG, skipping"
            continue
        fi

        # Train
        if [ -f "$MODEL_PATH/adapter_model.safetensors" ]; then
            echo ">>> SKIP: $LOG_PREFIX $BIAS model already trained <<<"
        else
            echo ""
            echo ">>> Training $LOG_PREFIX $BIAS <<<"
            python src/reward_modeling/train.py --config "$CONFIG"
        fi

        # Gradient caching - train
        if [ -f "$MODEL_PATH/rapid_grad_train.pt" ]; then
            echo ">>> SKIP: $LOG_PREFIX $BIAS train gradients already cached <<<"
        else
            echo ""
            echo ">>> Caching train gradients: $LOG_PREFIX $BIAS <<<"
            python -m src.influence.cache_gradients \
                --model_path "$MODEL_PATH" \
                --tokenizer_path "$MODEL_NAME" \
                --data_path "${TOK_DIR}/train" \
                --save_name "rapid_grad_train.pt" \
                --K 65536
        fi

        # Gradient caching - validation
        if [ -f "$MODEL_PATH/rapid_grad_val.pt" ]; then
            echo ">>> SKIP: $LOG_PREFIX $BIAS val gradients already cached <<<"
        else
            echo ""
            echo ">>> Caching validation gradients: $LOG_PREFIX $BIAS <<<"
            python -m src.influence.cache_gradients \
                --model_path "$MODEL_PATH" \
                --tokenizer_path "$MODEL_NAME" \
                --data_path "${TOK_DIR}/test" \
                --save_name "rapid_grad_val.pt" \
                --K 65536
        fi

        # Influence computation
        if [ -f "$MODEL_PATH/influence_results.json" ]; then
            echo ">>> SKIP: $LOG_PREFIX $BIAS influence already computed <<<"
        else
            echo ""
            echo ">>> Computing influence: $LOG_PREFIX $BIAS <<<"
            python -m src.influence.compute_influence \
                --model_path "$MODEL_PATH" \
                --dataset_dir "$DATA_DIR" \
                --bias_type "$BIAS" \
                --K 65536
        fi
    done

    echo ""
    echo ">>> $LOG_PREFIX complete! <<<"
    for BIAS in length sycophancy; do
        RESULT="logs/${LOG_PREFIX}_${BIAS}/influence_results.json"
        if [ -f "$RESULT" ]; then
            echo "  $BIAS: $(cat $RESULT)"
        fi
    done
}

echo "============================================="
echo " Baseline Models Pipeline (RTX 5080)"
echo " $(date)"
echo "============================================="

for entry in "${MODELS[@]}"; do
    IFS='|' read -r TAG MODEL_NAME LOG_PREFIX TOK_TAG <<< "$entry"
    if [ "$TARGET" = "all" ] || [ "$TARGET" = "$TAG" ]; then
        run_model "$TAG" "$MODEL_NAME" "$LOG_PREFIX" "$TOK_TAG"
    fi
done

echo ""
echo "============================================="
echo " All Baseline Models Complete! $(date)"
echo "============================================="

# Print summary table
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Model             | Bias        | Targeted AUC | Full AUC"
echo "------------------|-------------|-------------|----------"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r TAG MODEL_NAME LOG_PREFIX <<< "$entry"
    for BIAS in length sycophancy; do
        RESULT="logs/${LOG_PREFIX}_${BIAS}/influence_results.json"
        if [ -f "$RESULT" ]; then
            if [ "$BIAS" = "length" ]; then
                TARGETED=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Concise\"][\"auc\"]:.4f}')" 2>/dev/null)
            else
                TARGETED=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Less Sycophantic\"][\"auc\"]:.4f}')" 2>/dev/null)
            fi
            FULL=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Full\"][\"auc\"]:.4f}')" 2>/dev/null)
            printf "%-18s| %-12s| %-12s| %s\n" "$LOG_PREFIX" "$BIAS" "$TARGETED" "$FULL"
        fi
    done
done

# Also include existing Qwen2.5-1.5B results
for BIAS in length sycophancy; do
    RESULT="logs/Qwen2.5-1.5B_${BIAS}/influence_results.json"
    if [ -f "$RESULT" ]; then
        if [ "$BIAS" = "length" ]; then
            TARGETED=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Concise\"][\"auc\"]:.4f}')" 2>/dev/null)
        else
            TARGETED=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Less Sycophantic\"][\"auc\"]:.4f}')" 2>/dev/null)
        fi
        FULL=$(python -c "import json; d=json.load(open('$RESULT')); print(f'{d[\"Full\"][\"auc\"]:.4f}')" 2>/dev/null)
        printf "%-18s| %-12s| %-12s| %s\n" "Qwen2.5-1.5B" "$BIAS" "$TARGETED" "$FULL"
    fi
done
