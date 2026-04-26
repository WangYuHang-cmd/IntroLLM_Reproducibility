#!/bin/bash
# =============================================================================
# Experiment 1A: LoRA Rank Ablation on Qwen2.5-1.5B
#
# Tests whether increasing LoRA rank improves influence function AUC
# by giving gradients more dimensions to encode bias information.
#
# Runs 5 configurations: r=4, 8, 16(done), 32, 64
# Each: train RM → cache gradients → compute influence → diagnostics
#
# Usage:
#   conda activate if_rlhf
#   CUDA_LAUNCH_BLOCKING=1 bash scripts/exp_lora_rank_ablation.sh 2>&1 | tee logs/lora_ablation.log
#
# To run a single rank:
#   CUDA_LAUNCH_BLOCKING=1 bash scripts/exp_lora_rank_ablation.sh 32
# =============================================================================
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="disabled"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"
BIAS="length"
TOK_DIR="dataset/${BIAS}_dataset_tokenized"
DATA_DIR="dataset/${BIAS}_dataset"
TARGET_RANK="${1:-all}"

RANKS=(4 8 16 32 64)

echo "============================================="
echo " Experiment 1A: LoRA Rank Ablation"
echo " Model: Qwen2.5-1.5B, Bias: length"
echo " Ranks: ${RANKS[*]}"
echo " $(date)"
echo "============================================="

run_rank() {
    local R=$1
    local LOG_PREFIX="Qwen2.5-1.5B_r${R}"
    local MODEL_PATH="logs/${LOG_PREFIX}_${BIAS}"
    local CONFIG="configs/reward_model_${BIAS}_r${R}.yaml"

    echo ""
    echo "########################################################"
    echo " LoRA rank r=$R  ($(date))"
    echo "########################################################"

    # --- Generate config ---
    if [ ! -f "$CONFIG" ]; then
        echo ">>> Creating config for r=$R <<<"
        cat > "$CONFIG" <<EOF
model_name_or_path: ${MODEL}
torch_dtype: bfloat16
attn_implementation: sdpa

use_peft: true
lora_r: ${R}
lora_alpha: $((R * 2))
lora_dropout: 0.1
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]
lora_modules_to_save: [score]

num_train_epochs: 4
per_device_train_batch_size: 1
per_device_eval_batch_size: 8
gradient_accumulation_steps: 64
learning_rate: 2.0e-05
optim: paged_adamw_32bit
lr_scheduler_type: linear
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: false
max_length: 1024

dataset_name: Taywon/HH_length_biased_15k
bias_type: ${BIAS}
output_dir: ${MODEL_PATH}

logging_steps: 10
eval_strategy: epoch
save_strategy: epoch
save_total_limit: 1
seed: 42
report_to: none
EOF
    fi

    # --- Train ---
    if [ -f "$MODEL_PATH/adapter_model.safetensors" ]; then
        echo ">>> SKIP: r=$R model already trained <<<"
    else
        echo ">>> Training r=$R <<<"
        python src/reward_modeling/train.py --config "$CONFIG"
    fi

    # --- Cache train gradients ---
    if [ -f "$MODEL_PATH/rapid_grad_train.pt" ]; then
        echo ">>> SKIP: r=$R train gradients already cached <<<"
    else
        echo ">>> Caching train gradients r=$R <<<"
        python -m src.influence.cache_gradients \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER" \
            --data_path "${TOK_DIR}/train" \
            --save_name "rapid_grad_train.pt" \
            --K 65536
    fi

    # --- Cache val gradients ---
    if [ -f "$MODEL_PATH/rapid_grad_val.pt" ]; then
        echo ">>> SKIP: r=$R val gradients already cached <<<"
    else
        echo ">>> Caching val gradients r=$R <<<"
        python -m src.influence.cache_gradients \
            --model_path "$MODEL_PATH" \
            --tokenizer_path "$TOKENIZER" \
            --data_path "${TOK_DIR}/test" \
            --save_name "rapid_grad_val.pt" \
            --K 65536
    fi

    # --- Compute influence ---
    if [ -f "$MODEL_PATH/influence_results.json" ]; then
        echo ">>> SKIP: r=$R influence already computed <<<"
    else
        echo ">>> Computing influence r=$R <<<"
        python -m src.influence.compute_influence \
            --model_path "$MODEL_PATH" \
            --dataset_dir "$DATA_DIR" \
            --bias_type "$BIAS" \
            --K 65536
    fi

    echo ">>> r=$R complete! <<<"
    if [ -f "$MODEL_PATH/influence_results.json" ]; then
        cat "$MODEL_PATH/influence_results.json"
    fi
}

# --- Main loop ---
for R in "${RANKS[@]}"; do
    if [ "$TARGET_RANK" = "all" ] || [ "$TARGET_RANK" = "$R" ]; then
        # Special case: r=16 already done as Qwen2.5-1.5B_length
        if [ "$R" = "16" ]; then
            EXISTING="logs/Qwen2.5-1.5B_${BIAS}"
            TARGET="logs/Qwen2.5-1.5B_r16_${BIAS}"
            if [ -d "$EXISTING" ] && [ ! -d "$TARGET" ]; then
                echo ""
                echo ">>> Linking existing r=16 results <<<"
                ln -s "$(realpath $EXISTING)" "$TARGET"
            fi
        fi
        run_rank "$R"
    fi
done

# --- Summary ---
echo ""
echo "============================================="
echo " LoRA Rank Ablation Results"
echo " $(date)"
echo "============================================="
echo ""
echo "Rank | LoRA Params | Concise AUC | Verbose AUC | Full AUC"
echo "-----|-------------|-------------|-------------|--------"

for R in "${RANKS[@]}"; do
    MODEL_PATH="logs/Qwen2.5-1.5B_r${R}_${BIAS}"
    RESULT="$MODEL_PATH/influence_results.json"
    if [ -f "$RESULT" ]; then
        CONCISE=$(python -c "import json; print(f'{json.load(open(\"$RESULT\"))[\"Concise\"][\"auc\"]:.4f}')" 2>/dev/null)
        VERBOSE=$(python -c "import json; print(f'{json.load(open(\"$RESULT\"))[\"Verbose\"][\"auc\"]:.4f}')" 2>/dev/null)
        FULL=$(python -c "import json; print(f'{json.load(open(\"$RESULT\"))[\"Full\"][\"auc\"]:.4f}')" 2>/dev/null)

        # Count LoRA params from adapter config
        LORA_PARAMS=$(python -c "
import json
try:
    with open('$MODEL_PATH/adapter_config.json') as f:
        cfg = json.load(f)
    r = cfg.get('r', $R)
    # rough estimate: 7 modules * 2 matrices * hidden * r * num_layers
    print(f'~{7*2*1536*r*28/1e6:.1f}M')
except: print('N/A')
" 2>/dev/null)

        printf "r=%-3s| %-12s| %-12s| %-12s| %s\n" "$R" "$LORA_PARAMS" "$CONCISE" "$VERBOSE" "$FULL"
    fi
done

echo ""
echo "Run diagnostics:"
echo "  python scripts/exp_lora_rank_analysis.py"
