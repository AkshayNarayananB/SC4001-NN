#!/bin/bash
# One-time environment setup on ASPIRE 2A. Run on the login node from the repo root.
set -euo pipefail

SCRATCH=/scratch/users/ntu/$USER
ENV_PREFIX=$HOME/envs/vcd

module load miniforge3/25.3.1
eval "$(conda shell.bash hook)"

# Keep the conda package cache off the 50 GB home quota
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$SCRATCH/hf_cache" "$SCRATCH/logs" "$SCRATCH/data"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
