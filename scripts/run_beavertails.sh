#!/bin/bash
# BeaverTails safety domain extension (Experiment E)
# Full pipeline: data prep → train reward model → cache gradients → compute influence
# Model: Qwen2.5-1.5B-Instruct (fastest, proof of concept)
#
# Usage: bash scripts/run_beavertails.sh

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  BeaverTails Safety Extension"
echo "  Qwen2.5-1.5B-Instruct"
echo "=========================================="

# ---- Step 1: Dataset preparation ----
if [ ! -d "dataset/beavertails_dataset/train" ]; then
    echo ""
    echo "[1/4] Preparing BeaverTails dataset..."
    python src/data/make_beavertails_dataset.py 2>&1 | tee logs/beavertails_dataprep.log
else
    echo "[1/4] BeaverTails dataset already prepared — skipping"
fi

# ---- Step 2: Train reward model ----
if [ ! -f "logs/Qwen2.5-1.5B_safety/adapter_model.safetensors" ]; then
    echo ""
    echo "[2/4] Training reward model on BeaverTails..."
    python src/reward_modeling/train.py \
        --config configs/reward_model_beavertails_1.5B.yaml \
        2>&1 | tee logs/beavertails_train.log
else
    echo "[2/4] Safety reward model already trained — skipping"
fi

# ---- Step 3: Cache gradients ----
echo ""
echo "[3/4] Caching gradients (train)..."
python -m src.influence.cache_gradients \
    --model_path logs/Qwen2.5-1.5B_safety \
    --data_path dataset/beavertails_dataset_tokenized/train \
    --save_name rapid_grad_train.pt --K 65536 \
    2>&1 | tee logs/beavertails_cache.log

echo ""
echo "      Caching gradients (val)..."
python -m src.influence.cache_gradients \
    --model_path logs/Qwen2.5-1.5B_safety \
    --data_path dataset/beavertails_dataset_tokenized/test \
    --save_name rapid_grad_val.pt --K 65536 \
    2>&1 | tee -a logs/beavertails_cache.log

# ---- Step 4: Compute influence ----
echo ""
echo "[4/4] Computing influence scores..."
python -m src.influence.compute_influence \
    --model_path logs/Qwen2.5-1.5B_safety \
    --dataset_dir dataset/beavertails_dataset \
    --bias_type safety --K 65536 \
    2>&1 | tee -a logs/beavertails_cache.log

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
if [ -f "logs/Qwen2.5-1.5B_safety/influence_results.json" ]; then
    cat logs/Qwen2.5-1.5B_safety/influence_results.json
fi

echo ""
echo "Compare against length bias:"
echo "  Qwen2.5-1.5B length AUC: 0.5419"
