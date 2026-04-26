#!/bin/bash
# Run all remaining GPU experiments in sequence after HelpSteer2 training.
#
# Prerequisites:
#   - HelpSteer2 training already done (logs/Qwen2.5-1.5B_helpsteer_bob1/)
#   - All 4 base models trained (Qwen2.5-1.5B, Qwen3-1.7B, Llama3.2-1B, Llama3.2-3B)
#
# Runs:
#   1. HelpSteer2 gradient caching + SVM weight update
#   2. BeaverTails data prep + RM training + influence
#   3. Non-LLM baselines (Mahalanobis, KNN, self-conf, entropy) for all 4 models × 2 biases
#   4. Final figure generation (fig1-15)
#
# Usage: bash scripts/run_all_remaining.sh

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  All Remaining Experiments"
echo "  $(date)"
echo "=========================================="

# ===== Experiment D: HelpSteer2 Labeling Strategy =====
echo ""
echo "===== Experiment D: HelpSteer2 Labeling Strategy ====="

if [ ! -f "logs/Qwen2.5-1.5B_helpsteer_bob1/rapid_grad_train.pt" ]; then
    echo "[D1] Caching train gradients (D_B_1)..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \
        --data_path dataset/helpsteer2/D_B_1 \
        --save_name rapid_grad_train.pt --K 65536 \
        2>&1 | tee logs/helpsteer_bob1_cache.log
fi

if [ ! -f "logs/Qwen2.5-1.5B_helpsteer_bob1/rapid_grad_val.pt" ]; then
    echo "[D2] Caching val gradients (D_A)..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_helpsteer_bob1 \
        --data_path dataset/helpsteer2/D_A \
        --save_name rapid_grad_val.pt --K 65536 \
        2>&1 | tee -a logs/helpsteer_bob1_cache.log
fi

echo "[D3] Running SVM weight update..."
python scripts/exp_labeling_strategy.py 2>&1 | tee logs/helpsteer_bob1_svm.log

# ===== Experiment E: BeaverTails Safety =====
echo ""
echo "===== Experiment E: BeaverTails Safety Extension ====="

if [ ! -d "dataset/beavertails_dataset/train" ]; then
    echo "[E1] Preparing BeaverTails dataset..."
    python src/data/make_beavertails_dataset.py 2>&1 | tee logs/beavertails_dataprep.log
fi

if [ ! -f "logs/Qwen2.5-1.5B_safety/adapter_model.safetensors" ]; then
    echo "[E2] Training BeaverTails reward model..."
    python src/reward_modeling/train.py \
        --config configs/reward_model_beavertails_1.5B.yaml \
        2>&1 | tee logs/beavertails_train.log
fi

if [ ! -f "logs/Qwen2.5-1.5B_safety/rapid_grad_train.pt" ]; then
    echo "[E3] Caching BeaverTails gradients..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_safety \
        --data_path dataset/beavertails_dataset_tokenized/train \
        --save_name rapid_grad_train.pt --K 65536 \
        2>&1 | tee logs/beavertails_cache.log

    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_safety \
        --data_path dataset/beavertails_dataset_tokenized/test \
        --save_name rapid_grad_val.pt --K 65536 \
        2>&1 | tee -a logs/beavertails_cache.log
fi

if [ ! -f "logs/Qwen2.5-1.5B_safety/influence_results.json" ]; then
    echo "[E4] Computing BeaverTails influence..."
    python -m src.influence.compute_influence \
        --model_path logs/Qwen2.5-1.5B_safety \
        --dataset_dir dataset/beavertails_dataset \
        --bias_type safety --K 65536 \
        2>&1 | tee -a logs/beavertails_cache.log
fi

# ===== Experiment A1: Non-LLM Baselines =====
echo ""
echo "===== Experiment A1: Non-LLM Baselines ====="
echo "Running Mahalanobis, KNN, Self-confidence, Entropy baselines..."
python scripts/exp_baselines.py 2>&1 | tee logs/baselines.log

# ===== Generate all figures =====
echo ""
echo "===== Generating all figures (fig1-15) ====="
python scripts/generate_paper_figures.py 2>&1 | tee logs/figures.log

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "=========================================="
