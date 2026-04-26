#!/bin/bash
# Run after BeaverTails RM training completes:
#   1. Cache gradients (train + val)
#   2. Compute influence scores
#   3. Run non-LLM baselines (4 models × 2 biases)
#   4. Regenerate all figures
#
# Usage: bash scripts/run_post_beavertails.sh

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  Post-BeaverTails Experiments"
echo "  $(date)"
echo "=========================================="

# Wait for RM training to finish
echo "Waiting for BeaverTails RM training to complete..."
while [ ! -f "logs/Qwen2.5-1.5B_safety/adapter_model.safetensors" ]; do
    sleep 60
done
echo "  RM training complete: $(date)"

# ---- Cache gradients ----
if [ ! -f "logs/Qwen2.5-1.5B_safety/rapid_grad_train.pt" ]; then
    echo ""
    echo "[1/4] Caching BeaverTails train gradients..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_safety \
        --data_path dataset/beavertails_dataset_tokenized/train \
        --save_name rapid_grad_train.pt --K 65536 \
        2>&1 | tee logs/beavertails_cache.log
fi

if [ ! -f "logs/Qwen2.5-1.5B_safety/rapid_grad_val.pt" ]; then
    echo ""
    echo "      Caching BeaverTails val gradients..."
    python -m src.influence.cache_gradients \
        --model_path logs/Qwen2.5-1.5B_safety \
        --data_path dataset/beavertails_dataset_tokenized/test \
        --save_name rapid_grad_val.pt --K 65536 \
        2>&1 | tee -a logs/beavertails_cache.log
fi

# ---- Compute influence ----
if [ ! -f "logs/Qwen2.5-1.5B_safety/influence_results.json" ]; then
    echo ""
    echo "[2/4] Computing BeaverTails influence scores..."
    python -m src.influence.compute_influence \
        --model_path logs/Qwen2.5-1.5B_safety \
        --dataset_dir dataset/beavertails_dataset \
        --bias_type safety --K 65536 \
        2>&1 | tee -a logs/beavertails_cache.log
fi

echo ""
echo "[3/4] BeaverTails influence done. GPU free — running baselines..."

# ---- Non-LLM baselines (4 models × 2 biases) ----
python scripts/exp_baselines.py 2>&1 | tee logs/baselines.log

# ---- Regenerate figures ----
echo ""
echo "[4/4] Regenerating all figures..."
python scripts/generate_paper_figures.py 2>&1 | tee logs/figures.log

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "  $(date)"
echo "=========================================="
