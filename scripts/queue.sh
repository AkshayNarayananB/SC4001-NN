#!/bin/bash
# Sequential GPU job queue. Every step is resumable, so re-running this script after an interruption
# continues where it stopped. Run detached:  systemd-run --user --unit=vcd-queue bash scripts/queue.sh
cd "$(dirname "$0")/.."
export HF_HOME=/mnt/Data/akshay/hf_cache HF_HUB_OFFLINE=1
PY="nice -n 10 /home/akshay/miniconda3/bin/python"
step() { echo "[$(date '+%F %T')] START $1" >> logs/queue.log; shift; "$@"; echo "[$(date '+%F %T')] END   exit=$?" >> logs/queue.log; }
# done 2026-11-11: step views_pilot   $PY -m scripts.run_layers --conditions blur_s4 noise_s4 --views blur1 blur2 blur3 hflip >> logs/run_views_pilot.log 2>&1
# done 2026-11-12: step patches       $PY -m scripts.run_patches >> logs/run_patches.log 2>&1
step tokens       $PY -m scripts.run_tokens >> logs/run_tokens.log 2>&1
step contrast      $PY -m scripts.run_contrast >> logs/run_contrast.log 2>&1
echo "[$(date '+%F %T')] QUEUE DONE" >> logs/queue.log
