#!/bin/bash
# Build the journal env on ASPIRE (login node, repo root): same versions as the local box.
set -euo pipefail
SCRATCH=/scratch/users/ntu/akshayna
ENV=$HOME/envs/vcdj
module load miniforge3/25.3.1
eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs PIP_CACHE_DIR=$SCRATCH/pip_cache
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.14
conda activate "$ENV"
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-journal.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
