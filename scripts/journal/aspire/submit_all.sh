#!/bin/bash
# ASPIRE login node: wait for bootstrap, then submit per-model smoke jobs and the TEST/DEV arrays that depend on them.
#   nohup bash scripts/journal/aspire/submit_all.sh > /scratch/users/ntu/akshayna/logs/submit_all.log 2>&1 &
set -uo pipefail
cd ~/SC4001-NN
L=/scratch/users/ntu/akshayna/logs
until grep -q "BOOTSTRAP DONE" $L/bootstrap.log; do
  if ! pgrep -u $USER -f "bootstrap.sh" >/dev/null; then echo "bootstrap died; not submitting"; exit 1; fi
  sleep 60
done
for MODEL in llava-1.5-7b qwen2.5-vl-7b instructblip-7b; do
  S=$(qsub -v MODEL=$MODEL scripts/journal/aspire/smoke.pbs)
  echo "$MODEL smoke $S"
  # TEST (fresh 2000 images) for every model; DEV (500 POPE images) for the models not run locally
  T=$(qsub -J 0-20 -W depend=afterok:$S -v MODEL=$MODEL,POPE_SET=pope_fresh,EXTRA=--vcd scripts/journal/aspire/readout_array.pbs)
  echo "$MODEL test array $T"
  if [ $MODEL != llava-1.5-7b ]; then
    D=$(qsub -J 0-20 -W depend=afterok:$S -v MODEL=$MODEL,POPE_SET=pope,EXTRA=--vcd scripts/journal/aspire/readout_array.pbs)
    echo "$MODEL dev array $D"
  fi
done
echo SUBMITTED
