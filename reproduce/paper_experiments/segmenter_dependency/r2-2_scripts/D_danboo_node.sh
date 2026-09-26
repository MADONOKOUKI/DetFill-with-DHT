#!/bin/bash
# Launch DanbooRegion segmentation shards on THIS cayenne node (CPU).
# Usage: D_danboo_node.sh <nshards_total> <shard_start> <shard_end> [out_root]
set -u
NTOT=${1:?}; S0=${2:?}; S1=${3:?}
OUT=${4:-/scratch/madono/seg_retrain_R2-2/danboo_regions}
PY=/home/madorin/anaconda3/envs/danbooregion/bin/python
GEN=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/scripts/D_danbooregion_seg.py
FX=/home/madorin/datasets/tog2024/main_exp_felzenszwalb_fixdot/illust
TXT=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain/configs/illust
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 NUMBA_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "[$(hostname)] danbooregion shards [$S0,$S1)/$NTOT -> $OUT ($(date))"
for ((s=S0; s<S1; s++)); do
  nohup "$PY" "$GEN" --split all --shard "$s" --nshards "$NTOT" \
        --src_root "$FX" --txt_dir "$TXT" --out_root "$OUT" \
        > "$LOG/shard$(printf '%03d' "$s").log" 2>&1 &
done
wait
echo "[$(hostname)] shards [$S0,$S1) DONE ($(date))"
