#!/bin/bash
# R3-2 dot trainer: grabs the first free GPU among 3,2 (GPU2 frees when scribble ends).
set -u
ROOT=/scratch/madono/diffusart_det_R3-2
REPO=/home/madorin/gitlab/tog2024/main/Diffusion_v1_tvcg_R3-2
source /home/madorin/anaconda3/etc/profile.d/conda.sh
conda activate BBDM || exit 1
cd "$REPO"
ts(){ date '+%F %T'; }
echo "[$(ts)] waiting for a free GPU among 3,2 ..."
while :; do
  for g in 3 2; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$used" -lt 2000 ]; then GPU=$g; break 2; fi
  done
  sleep 180
done
echo "[$(ts)] GPU$GPU free. start dot."
export DIFFUSART_DET_DATA_ROOT=$ROOT
CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --master_port=29611 --nproc_per_node=1 train_det_dot.py \
    > "$ROOT/logs/train_det_dot.log" 2>&1
echo "[$(ts)] dot exit rc=$?"
