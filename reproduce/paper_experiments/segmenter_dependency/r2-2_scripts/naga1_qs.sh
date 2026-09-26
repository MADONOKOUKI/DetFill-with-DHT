#!/bin/bash
# RUN THIS ON naga1 (the user runs it).  Stages Quickshift training data to naga1's
# local /scratch, then launches QS BBDM training on 2 GPUs.
#   usage:  bash naga1_qs.sh [gpu_ids]      e.g.  bash naga1_qs.sh 0,1
# Assumes naga1 mounts the cluster NFS /home/madorin (code + BBDM env + data sources).
set -u
GPUS=${1:-0,1}
DST=/scratch/madono/seg_retrain
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
FX=/home/madorin/datasets/tog2024/main_exp_felzenszwalb_fixdot/illust
RD=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data

echo "== naga1 QS setup =="
[ -x "$PY" ]        || { echo "ERROR: BBDM env not visible at $PY  -> naga1 does not mount NFS; tell Claude."; exit 1; }
[ -f "$CODE/main.py" ] || { echo "ERROR: code not visible at $CODE"; exit 1; }
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" || exit 1

mkdir -p "$DST/gt" "$DST/sketch" "$DST/quickshift" "$DST/results/quickshift" "$DST/logs"

echo "== [1/4] build id lists from NFS txt =="
cat "$CODE"/configs/illust/train_paper.txt "$CODE"/configs/illust/valid_paper.txt "$CODE"/configs/illust/test_paper.txt \
  | sed 's#.*/felzenszwalb/##' | sed '/^$/d' | sort -u > /tmp/naga_gt.txt          # <dir>/<id>.image.png
awk -F'/' '{dir=$1; id=$2; sub(/\.image\.png$/,"",id);
  print "sketch/XDoG/"dir"/"id".png"; print "sketch/pysimp/"dir"/"id".png"; print "sketch/sketchkeras/"dir"/"id".png"}' \
  /tmp/naga_gt.txt | sort -u > /tmp/naga_sk.txt
echo "  gt=$(wc -l < /tmp/naga_gt.txt)  sketch=$(wc -l < /tmp/naga_sk.txt)"

echo "== [2/4] stage GT (NFS -> local /scratch) =="
rsync -a --files-from=/tmp/naga_gt.txt "$FX/segmentation_regions/felzenszwalb/" "$DST/gt/"

echo "== [3/4] stage sketch + quickshift hints =="
rsync -a --files-from=/tmp/naga_sk.txt "$FX/" "$DST/"
rsync -a "$RD/quickshift/" "$DST/quickshift/"

echo "  verify: gt=$(find $DST/gt -name '*.image.png'|wc -l) sketch=$(find $DST/sketch -name '*.png'|wc -l) qs=$(find $DST/quickshift -name '*_scribble_mask64.png'|wc -l)"

echo "== [4/4] launch QS training (tmux tr_qs, GPUs $GPUS) =="
cd "$CODE" && tmux kill-session -t tr_qs 2>/dev/null
tmux new-session -d -s tr_qs "cd $CODE && $PY main.py -c configs/seg_quickshift_illust.yaml \
  --train --save_top --gpu_ids $GPUS --port 12361 -r $DST/results/quickshift \
  > $DST/logs/train_qs.log 2>&1"
sleep 3
tmux ls 2>/dev/null | grep tr_qs && echo "QS LAUNCHED. monitor: tail -f $DST/logs/train_qs.log"
