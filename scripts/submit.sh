#!/bin/bash
# Submit GPU jobs on ASPIRE. Run from the repo root.
#   bash scripts/submit.sh configs/stage0/stage0_vcd_greedy.yaml [more.yaml ...]   # one job per config
#   bash scripts/submit.sh configs/stage1/                                          # one job for a whole dir
#     (the model is loaded once and the configs run in order; good for many short runs)
#   WALLTIME=03:00:00 bash scripts/submit.sh ...   # override the default walltime in run_pope.pbs
set -euo pipefail
LOG_DIR=/scratch/users/ntu/c230202/logs
mkdir -p "$LOG_DIR"
for target in "$@"; do
    name=$(basename "$target" .yaml)
    jobid=$(qsub ${WALLTIME:+-l walltime=$WALLTIME} -N "${name:0:15}" -o "$LOG_DIR/$name.log" -v CONFIG="${target%/}" scripts/pbs/run_pope.pbs)
    echo "$jobid  $name"
done
