#!/bin/bash
# =============================================================================
# Resume 1.5B Pipeline: data prep -> training -> gradient caching -> influence
# Usage: conda activate if_rlhf && bash scripts/run_1.5B_resume.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="disabled"

TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================="
echo " IntroLLM 1.5B Full Pipeline"
echo " $(date)"
echo "============================================="

# ----- Phase 1: Data Preparation -----
echo ""
echo ">>> Phase 1: Data Preparation <<<"
python src/data/make_hh_dataset.py \
    --tokenizer_name "$TOKENIZER" \
    --max_length 1024 \
    --store_dir dataset

# ----- Phase 2: Reward Model Training -----
echo ""
echo ">>> Phase 2: Training Length Bias RM (1.5B) <<<"
python src/reward_modeling/train.py --config configs/reward_model_length_1.5B.yaml

echo ""
echo ">>> Phase 2: Training Sycophancy Bias RM (1.5B) <<<"
python src/reward_modeling/train.py --config configs/reward_model_sycophancy_1.5B.yaml

# ----- Phase 3A: Gradient Caching (uses _tokenized datasets) -----
for BIAS in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-1.5B_${BIAS}"
    TOK_DIR="dataset/${BIAS}_dataset_tokenized"

    echo ""
    echo ">>> Phase 3A: Caching train gradients ($BIAS) <<<"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${TOK_DIR}/train" \
        --save_name "rapid_grad_train.pt" \
        --K 65536

    echo ""
    echo ">>> Phase 3A: Caching validation gradients ($BIAS) <<<"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${TOK_DIR}/test" \
        --save_name "rapid_grad_val.pt" \
        --K 65536
done

# ----- Phase 3B: Influence Computation -----
for BIAS in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-1.5B_${BIAS}"
    DATA_DIR="dataset/${BIAS}_dataset"

    echo ""
    echo ">>> Phase 3B: Computing influence scores ($BIAS) <<<"
    python -m src.influence.compute_influence \
        --model_path "$MODEL_PATH" \
        --dataset_dir "$DATA_DIR" \
        --bias_type "$BIAS" \
        --K 65536
done

echo ""
echo "============================================="
echo " Pipeline Complete! $(date)"
echo "============================================="
echo ""
echo "Check results:"
echo "  cat logs/Qwen2.5-1.5B_length/influence_results.json"
echo "  cat logs/Qwen2.5-1.5B_sycophancy/influence_results.json"
