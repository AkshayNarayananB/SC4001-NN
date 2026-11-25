#!/bin/bash
# ASPIRE login node, one time: build the env, then download all three VLMs into scratch.
#   nohup bash scripts/journal/aspire/bootstrap.sh > /scratch/users/ntu/akshayna/logs/bootstrap.log 2>&1 &
set -euo pipefail
cd ~/SC4001-NN
bash scripts/journal/aspire/setup_env.sh
module load miniforge3/25.3.1
eval "$(conda shell.bash hook)"
conda activate $HOME/envs/vcdj
export HF_HOME=/scratch/users/ntu/akshayna/hf_cache HF_HUB_DISABLE_XET=1
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2.5-VL-7B-Instruct", "Salesforce/instructblip-vicuna-7b"]:
    print(r, snapshot_download(r, max_workers=16, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]), flush=True)
PY
echo BOOTSTRAP DONE
