#!/bin/bash
# Final experiment runner: waits for Llama baselines, then runs Qwen2.5-7B + figures
#
# Usage: bash scripts/run_final_experiments.sh

set -e
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate if_rlhf

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  Final Experiments"
echo "  $(date)"
echo "=========================================="

# Wait for Llama3.2-3B baselines to finish
echo ""
echo "Waiting for Llama baselines to complete..."
while [ ! -f "logs/Llama3.2-3B_sycophancy/baseline_results.json" ]; do
    sleep 60
done
echo "  Llama baselines complete: $(date)"

# Run Qwen2.5-7B baselines
echo ""
echo "Running Qwen2.5-7B baselines..."
python scripts/exp_baselines.py --models Qwen2.5-7B \
    2>&1 | tee logs/baselines_qwen7b.log

# Regenerate all figures
echo ""
echo "Regenerating all figures..."
python scripts/generate_paper_figures.py 2>&1 | tee logs/figures_final.log

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "  $(date)"
echo "=========================================="
