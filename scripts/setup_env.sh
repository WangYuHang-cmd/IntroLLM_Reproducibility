#!/bin/bash
# =============================================================================
# Create dedicated conda environment for IntroLLM project
# Usage: bash scripts/setup_env.sh
# =============================================================================
set -e

ENV_NAME="if_rlhf"

echo "============================================="
echo " Creating conda environment: $ENV_NAME"
echo "============================================="

# Create env
conda create -n "$ENV_NAME" python=3.10 -y

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(python --version)"
echo "Location: $(which python)"

# Install PyTorch with CUDA 12.8 (matching your system)
echo ""
echo ">>> Installing PyTorch with CUDA 12.8 <<<"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
echo ""
echo ">>> Installing project dependencies <<<"
pip install transformers>=4.44.0 peft>=0.12.0 trl>=0.10.0
pip install datasets>=2.20.0 accelerate>=0.33.0 bitsandbytes>=0.43.0
pip install scikit-learn>=1.5.0 matplotlib numpy scipy
pip install openai wandb pyyaml tqdm

# Verify
echo ""
echo "============================================="
echo " Verifying installation"
echo "============================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import transformers, peft, trl, datasets
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'trl: {trl.__version__}')
print(f'datasets: {datasets.__version__}')
print('All good!')
"

echo ""
echo "============================================="
echo " Environment ready!"
echo " Activate with: conda activate $ENV_NAME"
echo "============================================="
