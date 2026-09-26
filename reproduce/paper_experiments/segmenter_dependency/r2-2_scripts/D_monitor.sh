#!/bin/bash
# Progress across all 4 nodes (local /scratch). Expected 26000 ids per segmenter.
OUT=/scratch/USER/seg_retrain_R2-2
KEY=~/.ssh/id_ed25519
ts=0; tq=0; tp=0
for h in HOST_C HOST_C HOST_C HOST_C; do
  r=$(ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=8 "$h" \
      "echo \$(find $OUT/slic -name '*_scribble_mask64.png' 2>/dev/null|wc -l) \$(find $OUT/quickshift -name '*_scribble_mask64.png' 2>/dev/null|wc -l) \$(pgrep -f D_retrain_gen_hints|wc -l)" 2>/dev/null)
  set -- $r; s=${1:-0}; q=${2:-0}; p=${3:-0}
  printf '%-9s slic=%-6s qs=%-6s procs=%s\n' "$h" "$s" "$q" "$p"
  ts=$((ts+s)); tq=$((tq+q)); tp=$((tp+p))
done
echo "TOTAL  slic=$ts/26000  qs=$tq/26000  ($((ts+tq))/52000)  procs=$tp"
