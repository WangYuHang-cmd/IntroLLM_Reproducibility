#!/bin/bash
# =============================================================================
# Full 1.5B Pipeline for RTX 5080 (16GB VRAM)
# Runs all phases: data prep -> training -> gradient caching -> influence
#
# Usage: bash scripts/run_1.5B_full_pipeline.sh
# =============================================================================
set -e
cd "$(dirname "$0")/.."  # cd to project root

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE="disabled"  # disable wandb for debugging; set to "online" to enable

TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"
STORE_DIR="dataset"

echo "============================================="
echo " IntroLLM 1.5B Pipeline (RTX 5080)"
echo " $(date)"
echo "============================================="
echo ""

# ----- Phase 1: Data Preparation -----
echo ">>> Phase 1: Data Preparation <<<"
echo "Downloading and tokenizing datasets..."
python src/data/make_hh_dataset.py \
    --tokenizer_name "$TOKENIZER" \
    --max_length 1024 \
    --store_dir "$STORE_DIR"

echo ""
echo "Preparing HelpSteer2 dataset..."
python src/labeling_strategy/helpsteer_data.py \
    --tokenizer_name "$TOKENIZER" \
    --max_length 1024 \
    --store_dir "$STORE_DIR/helpsteer2"

echo ""
echo ">>> Phase 1 Complete. Intermediate files: <<<"
echo "  dataset/length_dataset/           (train + test splits)"
echo "  dataset/length_dataset/flipped_indices.npy"
echo "  dataset/length_dataset/concise_indices.npy"
echo "  dataset/length_dataset/verbose_indices.npy"
echo "  dataset/sycophancy_dataset/       (train + test splits)"
echo "  dataset/sycophancy_dataset/flipped_indices.npy"
echo "  dataset/sycophancy_dataset/less_syco_indices.npy"
echo "  dataset/sycophancy_dataset/more_syco_indices.npy"
echo "  dataset/helpsteer2/D_A/           (Alice validation set)"
echo "  dataset/helpsteer2/D_B_1..5/      (Bob training sets)"
echo "  dataset/helpsteer2/w_alice.npy"
echo "  dataset/helpsteer2/w_bob_1..5.npy"
echo ""

# ----- Phase 2: Reward Model Training -----
echo ">>> Phase 2: Training Length Bias RM (1.5B) <<<"
python src/reward_modeling/train.py \
    --config configs/reward_model_length_1.5B.yaml

echo ""
echo ">>> Phase 2: Training Sycophancy Bias RM (1.5B) <<<"
python src/reward_modeling/train.py \
    --config configs/reward_model_sycophancy_1.5B.yaml

echo ""
echo ">>> Phase 2 Complete. Intermediate files: <<<"
echo "  logs/Qwen2.5-1.5B_length/        (LoRA checkpoint + adapter_config.json)"
echo "  logs/Qwen2.5-1.5B_sycophancy/     (LoRA checkpoint + adapter_config.json)"
echo ""

# ----- Phase 3A: Gradient Caching -----
for BIAS in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-1.5B_${BIAS}"
    DATA_DIR="dataset/${BIAS}_dataset"

    echo ">>> Phase 3A: Caching train gradients ($BIAS) <<<"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${DATA_DIR}/train" \
        --save_name "rapid_grad_train.pt" \
        --K 65536

    echo ""
    echo ">>> Phase 3A: Caching validation gradients ($BIAS) <<<"
    python -m src.influence.cache_gradients \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER" \
        --data_path "${DATA_DIR}/test" \
        --save_name "rapid_grad_val.pt" \
        --K 65536
    echo ""
done

echo ">>> Phase 3A Complete. Intermediate files: <<<"
echo "  logs/Qwen2.5-1.5B_length/rapid_grad_train.pt"
echo "  logs/Qwen2.5-1.5B_length/rapid_grad_val.pt"
echo "  logs/Qwen2.5-1.5B_length/RapidGrad_D*_n100_seed42.obj"
echo "  logs/Qwen2.5-1.5B_sycophancy/rapid_grad_train.pt"
echo "  logs/Qwen2.5-1.5B_sycophancy/rapid_grad_val.pt"
echo "  logs/Qwen2.5-1.5B_sycophancy/RapidGrad_D*_n100_seed42.obj"
echo ""

# ----- Phase 3B + 4A: Influence Computation -----
for BIAS in length sycophancy; do
    MODEL_PATH="logs/Qwen2.5-1.5B_${BIAS}"
    DATA_DIR="dataset/${BIAS}_dataset"

    echo ">>> Phase 3B: Computing influence scores ($BIAS) <<<"
    python -m src.influence.compute_influence \
        --model_path "$MODEL_PATH" \
        --dataset_dir "$DATA_DIR" \
        --bias_type "$BIAS" \
        --K 65536
    echo ""
done

echo ">>> Phase 3B Complete. Intermediate files: <<<"
echo "  logs/Qwen2.5-1.5B_length/influence_concise.npy"
echo "  logs/Qwen2.5-1.5B_length/influence_verbose.npy"
echo "  logs/Qwen2.5-1.5B_length/influence_full.npy"
echo "  logs/Qwen2.5-1.5B_length/influence_results.json"
echo "  logs/Qwen2.5-1.5B_sycophancy/influence_less_sycophantic.npy"
echo "  logs/Qwen2.5-1.5B_sycophancy/influence_more_sycophantic.npy"
echo "  logs/Qwen2.5-1.5B_sycophancy/influence_full.npy"
echo "  logs/Qwen2.5-1.5B_sycophancy/influence_results.json"
echo ""

echo "============================================="
echo " Pipeline Complete!"
echo " $(date)"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Check influence_results.json for AUC scores"
echo "  2. Run notebooks for visualization and baselines"
echo "  3. Transfer dataset/ and logs/ to 5090 machine for 7B experiments"
