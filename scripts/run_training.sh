#!/bin/bash
# Phase 2: Reward model training
# First run 1.5B for debugging, then 7B for final experiments

set -e

echo "=========================================="
echo "Phase 2: Reward Model Training"
echo "=========================================="

# Parse arguments
MODEL_SIZE="${1:-1.5B}"  # default to 1.5B for debugging

if [ "$MODEL_SIZE" == "1.5B" ]; then
    echo ">>> Running 1.5B models (debug mode) <<<"
    echo ""
    echo "--- Training length bias model (1.5B) ---"
    python src/reward_modeling/train.py --config configs/reward_model_length_1.5B.yaml

    echo ""
    echo "--- Training sycophancy bias model (1.5B) ---"
    python src/reward_modeling/train.py --config configs/reward_model_sycophancy_1.5B.yaml

elif [ "$MODEL_SIZE" == "7B" ]; then
    echo ">>> Running 7B models (final experiments) <<<"
    echo ""
    echo "--- Training length bias model (7B) ---"
    python src/reward_modeling/train.py --config configs/reward_model_length.yaml

    echo ""
    echo "--- Training sycophancy bias model (7B) ---"
    python src/reward_modeling/train.py --config configs/reward_model_sycophancy.yaml

elif [ "$MODEL_SIZE" == "all" ]; then
    echo ">>> Running both 1.5B and 7B models <<<"
    bash scripts/run_training.sh 1.5B
    echo ""
    bash scripts/run_training.sh 7B
else
    echo "Usage: bash scripts/run_training.sh [1.5B|7B|all]"
    exit 1
fi

echo ""
echo "=== Training complete ==="
