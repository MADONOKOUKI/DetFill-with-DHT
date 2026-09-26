#!/bin/bash
# Gather per-node LOCAL /scratch outputs into one NFS tree (ids are disjoint per node,
# so the merges never collide). Run from HOST_C after generation completes.
# Usage: D_gather.sh [nfs_dest]
set -u
NFS=${1:-/home/USER/datasets/labrepo/seg_retrain_R2-2}
SRC=/scratch/USER/seg_retrain_R2-2
KEY=~/.ssh/id_ed25519
mkdir -p "$NFS"
for h in HOST_C HOST_C HOST_C HOST_C; do
  for seg in slic quickshift; do
    echo "[$(date +%H:%M:%S)] gather $h:$seg -> $NFS/$seg"
    mkdir -p "$NFS/$seg"
    rsync -a -e "ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no" \
      "$h:$SRC/$seg/" "$NFS/$seg/" 2>/dev/null
  done
done
echo "=== gathered counts (NFS) ==="
for seg in slic quickshift; do
  printf '%s: %s scribble_mask64\n' "$seg" \
    "$(find "$NFS/$seg" -name '*_scribble_mask64.png' 2>/dev/null | wc -l)"
done
