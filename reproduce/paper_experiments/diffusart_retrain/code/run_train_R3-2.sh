#!/bin/bash
# R3-2: train Diffusart on deterministic (proposed) hints, paper schedule.
# Schedule = paper Sec VI-A & the comp run: AdamW lr2e-5, eff batch20, 200 epochs,
# cosine + 5000-step warmup, EMA 0.995, 256px. Single GPU (set GPU env, default 3).
set -u
GPU=${GPU:-3}
ROOT=/scratch/USER/diffusart_det_R3-2
REPO=/home/USER/gitlab/labrepo/main/Diffusion_v1_tvcg_R3-2
LOG=$ROOT/logs
mkdir -p "$LOG"
source /home/USER/anaconda3/etc/profile.d/conda.sh
conda activate BBDM || exit 1
cd "$REPO"
ts(){ date '+%F %T'; }

echo "[$(ts)] waiting for STAGING_DONE..."
while [ ! -f $ROOT/STAGING_DONE ]; do sleep 120; done
echo "[$(ts)] staging done. waiting for GPU$GPU free (<2000MiB)..."
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  [ "$used" -lt 2000 ] && break
  sleep 180
done
echo "[$(ts)] GPU$GPU free. start scribble."

export DIFFUSART_DET_DATA_ROOT=$ROOT
for entry in train_det_scribble train_det_dot; do
  echo "[$(ts)] ==== $entry start ===="
  CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=1 ${entry}.py \
      > "$LOG/${entry}.log" 2>&1
  rc=$?
  echo "[$(ts)] ==== $entry exit rc=$rc ===="
  [ $rc -ne 0 ] && { echo "[$(ts)] ABORT chain ($entry failed)"; exit $rc; }
done
echo "[$(ts)] ALL TRAINING DONE"
