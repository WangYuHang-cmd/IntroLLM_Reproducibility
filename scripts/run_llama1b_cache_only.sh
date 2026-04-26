#!/bin/bash
# Cache gradients and compute influence for Llama-3.2-1B (skips training)
# Fix: use Llama-tokenized dataset (llama3.2_3b tag) to avoid token ID OOB

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  Llama-3.2-1B — Gradient Caching Only"
echo "  Using Llama-tokenized datasets (3b tag)"
echo "=========================================="

# ---- Length bias ----
echo ""
echo "[1/4] Caching gradients — Length bias (train)"
python -m src.influence.cache_gradients \
    --model_path logs/Llama3.2-1B_length \
    --data_path dataset/length_dataset_tokenized_llama3.2_3b/train \
    --save_name rapid_grad_train.pt --K 65536 --no-load_in_4bit \
    2>&1 | tee logs/llama1b_length_cache.log

echo ""
echo "[2/4] Caching gradients — Length bias (val)"
python -m src.influence.cache_gradients \
    --model_path logs/Llama3.2-1B_length \
    --data_path dataset/length_dataset_tokenized_llama3.2_3b/test \
    --save_name rapid_grad_val.pt --K 65536 --no-load_in_4bit \
    2>&1 | tee -a logs/llama1b_length_cache.log

python -m src.influence.compute_influence \
    --model_path logs/Llama3.2-1B_length \
    --dataset_dir dataset/length_dataset \
    --bias_type length --K 65536 \
    2>&1 | tee -a logs/llama1b_length_cache.log

# ---- Sycophancy bias ----
echo ""
echo "[3/4] Caching gradients — Sycophancy bias (train)"
python -m src.influence.cache_gradients \
    --model_path logs/Llama3.2-1B_sycophancy \
    --data_path dataset/sycophancy_dataset_tokenized_llama3.2_3b/train \
    --save_name rapid_grad_train.pt --K 65536 --no-load_in_4bit \
    2>&1 | tee logs/llama1b_syco_cache.log

echo ""
echo "[4/4] Caching gradients — Sycophancy bias (val)"
python -m src.influence.cache_gradients \
    --model_path logs/Llama3.2-1B_sycophancy \
    --data_path dataset/sycophancy_dataset_tokenized_llama3.2_3b/test \
    --save_name rapid_grad_val.pt --K 65536 --no-load_in_4bit \
    2>&1 | tee -a logs/llama1b_syco_cache.log

python -m src.influence.compute_influence \
    --model_path logs/Llama3.2-1B_sycophancy \
    --dataset_dir dataset/sycophancy_dataset \
    --bias_type sycophancy --K 65536 \
    2>&1 | tee -a logs/llama1b_syco_cache.log

# ---- Results ----
echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
for bias in length sycophancy; do
    dir="logs/Llama3.2-1B_${bias}"
    if [ -f "$dir/influence_results.json" ]; then
        echo "--- Llama-3.2-1B $bias ---"
        cat "$dir/influence_results.json"
    fi
done

echo ""
echo "Done. Compare against:"
echo "  Qwen2.5-1.5B length concise AUC: 0.5419"
echo "  Qwen2.5-7B   length concise AUC: 0.5053"
echo "  Llama-3.2-3B length concise AUC: 0.6996"
