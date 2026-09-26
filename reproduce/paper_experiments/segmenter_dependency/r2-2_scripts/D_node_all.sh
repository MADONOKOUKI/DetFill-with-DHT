#!/bin/bash
# Run BOTH segmenters (slic then quickshift) for this node's shard range.
# Usage: D_node_all.sh <shard_start> <shard_end> [nshards_total] [out_root]
set -u
S0=${1:?shard_start}; S1=${2:?shard_end}; NTOT=${3:-192}
OUT=${4:-/scratch/USER/seg_retrain_R2-2}
HERE=$(cd "$(dirname "$0")" && pwd)
echo "==== $(hostname) START $(date) shards [$S0,$S1)/$NTOT ===="
bash "$HERE/D_run_node.sh" slic       "$NTOT" "$S0" "$S1" "$OUT"
bash "$HERE/D_run_node.sh" quickshift "$NTOT" "$S0" "$S1" "$OUT"
echo "==== $(hostname) ALL DONE $(date) ===="
