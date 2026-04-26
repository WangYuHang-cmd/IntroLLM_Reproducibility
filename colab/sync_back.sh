#!/bin/bash
# After 7B runs on Colab, this script integrates the results into your local project.
# Prerequisite: Drive folder synced to ~/Desktop/Drive/Colab/IntroLLM/logs_7B
# (Install Google Drive for desktop, or use rclone, or manual download)

set -e
cd "$(dirname "$0")/.."

DRIVE_SRC="${1:-$HOME/Desktop/Drive/Colab/IntroLLM/logs_7B}"

if [ ! -d "$DRIVE_SRC" ]; then
    echo "Drive source not found: $DRIVE_SRC"
    echo "Usage: bash colab/sync_back.sh [drive_logs_7B_path]"
    echo ""
    echo "Alternative: download manually from Drive web interface and place in logs/"
    exit 1
fi

echo "Syncing 7B results from $DRIVE_SRC to logs/..."

for model_dir in "$DRIVE_SRC"/*/; do
    name=$(basename "$model_dir")
    echo "  → logs/$name/"
    mkdir -p "logs/$name"
    cp "$model_dir/adapter_config.json" "logs/$name/" 2>/dev/null || true
    cp "$model_dir/adapter_model.safetensors" "logs/$name/" 2>/dev/null || true
    cp "$model_dir"/influence_*.npy "logs/$name/" 2>/dev/null || true
    cp "$model_dir/influence_results.json" "logs/$name/" 2>/dev/null || true
    # Optionally skip large gradient files to save local disk
    # cp "$model_dir"/rapid_grad_*.pt "logs/$name/" 2>/dev/null || true
done

echo ""
echo "Sync complete. Regenerate figures:"
echo "  python scripts/generate_paper_figures.py"
