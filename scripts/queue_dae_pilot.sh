#!/bin/bash
# Early answer for probe x views x DAE: smoke test, then the DAE view on blur_s4 / noise_s4 only.
# Resumable. Afterwards resume the full queue with scripts/queue_probe_views_dae.sh.
cd "$(dirname "$0")/.."
export HF_HOME=/mnt/Data/akshay/hf_cache HF_HUB_OFFLINE=1
PY="nice -n 10 /home/akshay/miniconda3/bin/python"
step() { echo "[$(date '+%F %T')] START $1" >> logs/queue.log; shift; "$@"; echo "[$(date '+%F %T')] END   exit=$?" >> logs/queue.log; }
step pvd_smoke    $PY -m scripts.run_layers --conditions noise_s4 --views dae --limit 30 >> logs/run_pvd_smoke.log 2>&1 || exit 1
step pvd_dae_s4   $PY -m scripts.run_layers --conditions blur_s4 noise_s4 --views dae >> logs/run_pvd_dae.log 2>&1
echo "[$(date '+%F %T')] DAE PILOT DONE" >> logs/queue.log
