#!/bin/bash
# Launch SLIC on reaper6 and Quickshift on reaper2 (2 servers, simultaneous).
# Data read from each host's local /scratch; checkpoints written to local /scratch.
# Run from cayenne1.   Usage: D_train_2server.sh [smoke|full]
set -u
MODE=${1:-full}; KEY=~/.ssh/id_ed25519
CODE=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python   # canonical training env (environment.yml: name=BBDM)
RES=/scratch/madono/seg_retrain/results
LOG=/scratch/madono/seg_retrain/logs
RSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no"

launch() {  # host seg gpus port extra
  local H=$1 seg=$2 gpus=$3 port=$4 extra=$5
  $RSH "$H" "mkdir -p $RES/$seg $LOG; cd $CODE && tmux kill-session -t tr_$seg 2>/dev/null; \
    tmux new-session -d -s tr_$seg \"cd $CODE && $PY main.py -c configs/seg_${seg}_illust.yaml \
      --train --save_top --gpu_ids $gpus --port $port -r $RES/$seg $extra \
      > $LOG/train_${seg}.log 2>&1\"; sleep 2; \
    (tmux ls 2>/dev/null | grep -q tr_$seg && echo '$H/$seg launched') || echo '$H/$seg FAILED'"
}

if [ "$MODE" = "smoke" ]; then
  launch reaper6 slic       0 12360 "--max_steps 4 --max_epoch 1"
  launch reaper2 quickshift 0 12361 "--max_steps 4 --max_epoch 1"
else
  launch reaper6 slic       0,1,2,3,4,5,6,7,8,9 12360 ""   # reaper6 fully free (matches felz 10-GPU)
  launch reaper2 quickshift 0,1,2,3,5,6          12361 ""   # reaper2 free indices only
fi
