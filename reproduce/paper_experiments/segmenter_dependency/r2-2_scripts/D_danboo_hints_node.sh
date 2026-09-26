#!/bin/bash
# DanbooRegion Stage 2 on THIS node: make_scribbling over the LOCAL region maps -> hints.
# Usage: D_danboo_hints_node.sh [nproc]
set -u
NPROC=${1:-32}
REGION=/scratch/madono/seg_retrain_R2-2/danboo_regions
OUT=/scratch/madono/seg_retrain_R2-2/danbooregion
PY=/home/madorin/anaconda3/envs/py37/bin/python
GEN=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/scripts/D_danboo_hints.py
FX=/home/madorin/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1
echo "[$(hostname)] Stage2 $NPROC shards over local region maps ($(date))"
for ((s=0; s<NPROC; s++)); do
  nohup "$PY" "$GEN" --region_root "$REGION" --src_root "$FX" --out_root "$OUT" \
        --shard "$s" --nshards "$NPROC" > "$LOG/hints_shard$(printf '%03d' "$s").log" 2>&1 &
done
wait
echo "[$(hostname)] Stage2 DONE ($(date))"
