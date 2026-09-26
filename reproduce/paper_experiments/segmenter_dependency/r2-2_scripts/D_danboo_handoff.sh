#!/bin/bash
# When DanbooRegion hints are complete: gather -> NFS, stage to reaper2, KILL QS, launch
# DanbooRegion training on reaper2 (10-GPU, 96ch) in QS's place. Run from cayenne1.
set -u
KEY=~/.ssh/id_ed25519
RSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no"
NFS_RD=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data
LOCAL=/scratch/madono/seg_retrain_R2-2/danbooregion
R2DATA=/scratch/madono/seg_retrain/danbooregion
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
RES=/scratch/madono/seg_retrain/results; LOG=/scratch/madono/seg_retrain/logs

echo "[1] gather DanbooRegion hints cayenne1/3/4 -> NFS ($(date))"
mkdir -p "$NFS_RD/danbooregion"
for h in cayenne1 cayenne3 cayenne4; do
  rsync -a -e "$RSH" "$h:$LOCAL/" "$NFS_RD/danbooregion/" 2>/dev/null
done
n=$(find "$NFS_RD/danbooregion" -name '*_scribble_mask64.png' | wc -l)
echo "    gathered: $n hints"

echo "[2] stage DanbooRegion hints NFS -> reaper2:$R2DATA"
$RSH reaper2 "mkdir -p $R2DATA"
rsync -a -e "$RSH" "$NFS_RD/danbooregion/" "reaper2:$R2DATA/"
sn=$($RSH reaper2 "find $R2DATA -name '*_scribble_mask64.png'|wc -l")
echo "    reaper2 danbooregion hints: $sn  (gt/sketch already present from QS)"

echo "[3] KILL QS on reaper2"
$RSH reaper2 "tmux kill-session -t tr_quickshift 2>/dev/null; sleep 3; pgrep -fc '[s]eg_quickshift_illust'" | tail -1 | sed 's/^/    QS procs left: /'

echo "[4] launch DanbooRegion training on reaper2 (10-GPU, 96ch)"
$RSH reaper2 "mkdir -p $RES/danbooregion_96ch $LOG; cd $CODE && tmux kill-session -t tr_danbooregion 2>/dev/null; \
  tmux new-session -d -s tr_danbooregion \"cd $CODE && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    -c configs/seg_danbooregion_illust.yaml --train --save_top --gpu_ids 0,1,2,3,4,5,6,7,8,9 --port 12361 \
    -r $RES/danbooregion_96ch > $LOG/train_danbooregion_96ch.log 2>&1\"; sleep 3; \
  tmux ls 2>/dev/null | grep tr_danbooregion && echo '    DanbooRegion training LAUNCHED'"
echo "handoff done ($(date))"
