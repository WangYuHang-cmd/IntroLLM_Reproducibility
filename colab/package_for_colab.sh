#!/bin/bash
# Package IntroLLM project for upload to Google Drive / Colab
# Excludes large intermediate files (gradients, checkpoints)
# Usage: bash colab/package_for_colab.sh

set -e
cd "$(dirname "$0")/.."

OUTPUT="IntroLLM_for_colab.tar.gz"

echo "Packaging IntroLLM for Colab..."
echo "Output: $OUTPUT"
echo ""

# Create a list of what to include
tar --exclude='IF_RLHF_ref' \
    --exclude='logs/Qwen*/rapid_grad_*.pt' \
    --exclude='logs/Qwen*/RapidGrad_*.obj' \
    --exclude='logs/Qwen*/checkpoint-*' \
    --exclude='logs/Llama*/rapid_grad_*.pt' \
    --exclude='logs/Llama*/RapidGrad_*.obj' \
    --exclude='logs/Llama*/checkpoint-*' \
    --exclude='dataset/*/test/cache-*' \
    --exclude='dataset/*/train/cache-*' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='IntroLLM_for_colab.tar.gz' \
    -czf "$OUTPUT" \
    src/ \
    scripts/ \
    configs/ \
    dataset/ \
    logs/Qwen2.5-1.5B_length/adapter_*.* \
    logs/Qwen2.5-1.5B_length/influence_*.* \
    logs/Qwen2.5-1.5B_length/influence_results.json \
    logs/Qwen3-1.7B_length/adapter_*.* \
    logs/Qwen3-1.7B_length/influence_*.* \
    logs/Qwen3-1.7B_length/influence_results.json \
    logs/Llama3.2-3B_length/adapter_*.* \
    logs/Llama3.2-3B_length/influence_*.* \
    logs/Llama3.2-3B_length/influence_results.json \
    paper.pdf \
    Reproducibility*.pdf \
    requirements.txt \
    COLAB_7B_GUIDE.md \
    colab/

echo "Done!"
echo ""
ls -lh "$OUTPUT"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT to Google Drive (e.g., MyDrive/Colab/IntroLLM/)"
echo "2. Open colab/colab_7B_notebook.ipynb in Colab"
echo "3. Follow the notebook step by step"
