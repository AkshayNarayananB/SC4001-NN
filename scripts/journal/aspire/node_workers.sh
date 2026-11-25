#!/bin/bash
# Started once per node by pool_multi.pbs (mpiexec --ppn 1): sets up the env and runs one pool_worker per local GPU.
cd ~/SC4001-NN
module load miniforge3/25.3.1
eval "$(conda shell.bash hook)"
conda activate $HOME/envs/vcdj
export HF_HOME=/scratch/users/ntu/akshayna/hf_cache HF_HUB_OFFLINE=1
export VCD_DATA_ROOT=/scratch/users/ntu/akshayna/data VCD_OUTPUT_ROOT=/scratch/users/ntu/akshayna/results/raw
export VCD_CHECKPOINT_ROOT=/scratch/users/ntu/akshayna/checkpoints
N=$(nvidia-smi -L | wc -l)
H=$(hostname -s)
echo "$(date '+%F %T') node $H: $N GPUs"
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python -m scripts.journal.aspire.pool_worker \
    > /scratch/users/ntu/akshayna/logs/pool_${PBS_JOBID%%.*}_${H}_gpu$i.log 2>&1 &
  sleep 15
done
wait
