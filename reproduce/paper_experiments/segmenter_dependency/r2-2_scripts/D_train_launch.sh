#!/bin/bash
# Launch SLIC + Quickshift BBDM training on reaper6 (parallel, 5 GPUs each, DDP).
# Reads data from local /scratch, writes checkpoints to local /scratch (per req [3]).
# Run from cayenne1.  Usage: D_train_launch.sh [host] [smoke]
#   smoke -> --max_steps 4 on 1 GPU each, just to validate the pipeline.
set -u
H=${1:-reaper6}; MODE=${2:-full}; KEY=~/.ssh/id_ed25519
CODE=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
RES=/scratch/madono/seg_retrain/results
LOG=/scratch/madono/seg_retrain/logs
RSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no"

launch() {
  local seg=$1 gpus=$2 port=$3 extra=$4
  $RSH "$H" "mkdir -p $RES/$seg $LOG; cd $CODE && tmux kill-session -t tr_$seg 2>/dev/null; \
    tmux new-session -d -s tr_$seg \"cd $CODE && $PY main.py -c configs/seg_${seg}_illust.yaml \
      --train --save_top --gpu_ids $gpus --port $port -r $RES/$seg $extra \
      > $LOG/train_${seg}.log 2>&1\"; sleep 1; tmux ls 2>/dev/null | grep tr_$seg && echo '$seg launched'"
}

if [ "$MODE" = "smoke" ]; then
  launch slic       0 12360 "--max_steps 4 --max_epoch 1"
  launch quickshift 5 12361 "--max_steps 4 --max_epoch 1"
else
  launch slic       0,1,2,3,4 12360 ""
  launch quickshift 5,6,7,8,9 12361 ""
fi
