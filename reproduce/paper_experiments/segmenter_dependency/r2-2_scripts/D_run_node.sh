#!/bin/bash
# Launch parallel hint-gen shards for ONE segmenter on THIS node (HOST_C-4).
# Writes to LOCAL /scratch by default (fast); gather to NFS afterwards with D_gather.sh.
#
# Usage:
#   D_run_node.sh <segmenter> <nshards_total> <shard_start> <shard_end> [out_root]
#
# Single-node (HOST_C only), 48-way parallel:
#   D_run_node.sh slic       48 0 48
#   D_run_node.sh quickshift 48 0 48
#
# 4-node split (nshards_total=192, 48 per node), run ONE line per node:
#   HOST_C:  D_run_node.sh quickshift 192   0  48
#   HOST_C:  D_run_node.sh quickshift 192  48  96
#   HOST_C:  D_run_node.sh quickshift 192  96 144
#   HOST_C:  D_run_node.sh quickshift 192 144 192
set -u
SEG=${1:?segmenter}; NTOT=${2:?nshards_total}; S0=${3:?shard_start}; S1=${4:?shard_end}
OUT=${5:-/scratch/USER/seg_retrain_R2-2}
PY=/home/USER/anaconda3/envs/py37/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$OUT/logs"; mkdir -p "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[$(hostname)] $SEG shards [$S0,$S1) of $NTOT -> $OUT ($(date))"
for ((s=S0; s<S1; s++)); do
  nohup "$PY" "$HERE/D_retrain_gen_hints.py" --segmenter "$SEG" --split all \
        --shard "$s" --nshards "$NTOT" --out_root "$OUT" \
        > "$LOG/${SEG}_shard$(printf '%03d' "$s").log" 2>&1 &
done
wait
echo "[$(hostname)] $SEG shards [$S0,$S1) DONE ($(date))"
