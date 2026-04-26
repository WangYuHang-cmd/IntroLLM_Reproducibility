#!/bin/bash
# HelpSteer2 Labeling Strategy Experiment (Experiment D)
# Trains reward model on Bob's labels, caches gradients vs Alice's data,
# then runs SVM weight update.
#
# Usage: bash scripts/run_helpsteer_labeling.sh

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  HelpSteer2 Labeling Strategy Experiment"
echo "  Bob's D_B_1 → reward model → influence → SVM update"
echo "=========================================="

# ---- Step 1: Train reward model on Bob's D_B_1 (TRL format: chosen_ids/rejected_ids) ----
if [ ! -f "logs/Qwen2.5-1.5B_helpsteer_bob1/adapter_model.safetensors" ]; then
    echo ""
    echo "[1/4] Training reward model on Bob's D_B_1 (dataset/helpsteer2_b1_trl)..."
    python src/reward_modeling/train.py \
        --config configs/reward_model_helpsteer_1.5B.yaml \
        2>&1 | tee logs/helpsteer_bob1_train.log
else
    echo "[1/4] Reward model already trained — skipping"
fi

# ---- Step 2: Cache train gradients (D_B_1, original input_ids_chosen format) ----
if [ ! -f "logs/Qwen2.5-1.5B_helpsteer_bob1/rapid_grad_train.pt" ]; then
    echo ""
    echo "[2/4] Caching gradients — D_B_1 (train)..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \
        --data_path dataset/helpsteer2/D_B_1 \
        --save_name rapid_grad_train.pt --K 65536 \
        2>&1 | tee logs/helpsteer_bob1_cache.log
else
    echo "[2/4] Train gradients already cached — skipping"
fi

# ---- Step 3: Cache val gradients (D_A — Alice's expert data) ----
if [ ! -f "logs/Qwen2.5-1.5B_helpsteer_bob1/rapid_grad_val.pt" ]; then
    echo ""
    echo "[3/4] Caching gradients — D_A (Alice's expert validation)..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \
        --data_path dataset/helpsteer2/D_A \
        --save_name rapid_grad_val.pt --K 65536 \
        2>&1 | tee -a logs/helpsteer_bob1_cache.log
else
    echo "[3/4] Val gradients already cached — skipping"
fi

# ---- Step 4: Run SVM weight update ----
echo ""
echo "[4/4] Running labeling strategy experiment (SVM weight update)..."
python scripts/exp_labeling_strategy.py \
    2>&1 | tee logs/helpsteer_bob1_svm.log

echo ""
echo "=========================================="
echo "  DONE"
echo "=========================================="
if [ -f "logs/Qwen2.5-1.5B_helpsteer_bob1/labeling_strategy_results.json" ]; then
    cat logs/Qwen2.5-1.5B_helpsteer_bob1/labeling_strategy_results.json
fi
