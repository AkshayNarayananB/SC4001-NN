#!/bin/bash
# GPU queue for probe x views x DAE (docs/probe_views_dae.md). Run it on the machine that wrote the
# existing layers_<cond>/ logs (the probes are fit on them; mixing machines changes fp16 numerics).
# Resumable: finished run directories are skipped, so re-running continues where it stopped.
#   VCD_CHECKPOINT_ROOT=/path/to/checkpoints systemd-run --user --unit=vcd-pvd bash scripts/queue_probe_views_dae.sh
# needs $VCD_CHECKPOINT_ROOT/dae/dae_v1/best.pt (copy from ASPIRE:
#   /scratch/users/ntu/c230202/checkpoints/dae/dae_v1/best.pt)
cd "$(dirname "$0")/.."
mkdir -p logs
PY=${PY:-"nice -n 10 python"}
DAE=${DAE:-dae_v1}
step() { echo "[$(date '+%F %T')] START $1" >> logs/queue.log; shift; "$@"; echo "[$(date '+%F %T')] END   exit=$?" >> logs/queue.log; }
[ -f "${VCD_CHECKPOINT_ROOT:-checkpoints}/dae/$DAE/best.pt" ] || { echo "missing DAE checkpoint $DAE"; exit 1; }
# 1. smoke test: 30 questions per split, one condition (writes under smoke_limit30/, not the real logs)
step pvd_smoke  $PY -m scripts.run_layers --conditions noise_s4 --views dae --dae "$DAE" --limit 30 >> logs/run_pvd_smoke.log 2>&1
# 2. the DAE view for all 9 conditions, then ViewLayer's views for the 7 conditions not yet run
#    (blur_s4 / noise_s4 views from the 2026-11-11 pilot are skipped automatically)
step pvd_dae    $PY -m scripts.run_layers --views dae --dae "$DAE" >> logs/run_pvd_dae.log 2>&1
step pvd_views  $PY -m scripts.run_layers --views blur1 blur2 blur3 hflip >> logs/run_pvd_views.log 2>&1
# 3. offline analysis (CPU, minutes)
step pvd_analyze $PY -m scripts.analyze_probe_views_dae >> logs/analyze_pvd.log 2>&1
echo "[$(date '+%F %T')] PVD QUEUE DONE" >> logs/queue.log
