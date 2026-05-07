#!/bin/bash
# =============================================================================
#  setup_env.sh
#  Duality AI Offroad Segmentation — Environment Setup (Mac / Linux)
#
#  HOW TO RUN:
#    1. Open a Terminal
#    2. Navigate to this folder:
#         cd ~/Desktop/falcon/ENV_SETUP
#    3. Make it executable and run:
#         chmod +x setup_env.sh
#         ./setup_env.sh
#    4. After it finishes, activate with:
#         conda activate EDU
# =============================================================================

echo "============================================================"
echo " Setting up EDU environment for Duality AI Segmentation"
echo "============================================================"

# Create environment
echo ""
echo "[1/5] Creating conda environment 'EDU' with Python 3.9..."
conda create -n EDU python=3.9 -y

# Activate
echo ""
echo "[2/5] Activating EDU environment..."
source activate EDU || conda activate EDU

# Install PyTorch
echo ""
echo "[3/5] Installing PyTorch..."
pip install torch torchvision

# Install dependencies
echo ""
echo "[4/5] Installing other dependencies..."
pip install pillow matplotlib numpy tqdm

# Verify
echo ""
echo "[5/5] Verifying installation..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision OK')"
python -c "import PIL; print('Pillow OK')"

echo ""
echo "============================================================"
echo " Setup complete! Activate with: conda activate EDU"
echo "============================================================"
