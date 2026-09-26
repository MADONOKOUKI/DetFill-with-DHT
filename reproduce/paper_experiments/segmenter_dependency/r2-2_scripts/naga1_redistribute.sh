#!/bin/bash
# RUN ON naga1. Drop quickshift entirely; redistribute the REMAINING slic+danbooregion eval
# combos (proposed 96ch ckpt) across the given GPU list. Resume-safe: done images are skipped,
# so killing the current single-GPU sessions and relaunching here wastes almost nothing.
#   usage: bash naga1_redistribute.sh "0 1"        # rebalance the current 2 GPUs
#          bash naga1_redistribute.sh "0 1 2 3"    # if GPU2/3 become free
set -u
GPUS="${1:-0 1}"
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
CKPT=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/checkpoints/proposed_96ch_scribble.pth
EVAL=/scratch/madono/seg_eval
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"; mkdir -p "$EVAL/logs"

echo "== stop current eval sessions + any quickshift remnants (resume-safe) =="
for s in eval_slic eval_danbooregion eval_quickshift; do tmux kill-session -t $s 2>/dev/null; done
pkill -9 -f '[s]eg_quickshift_eval' 2>/dev/null
pkill -9 -f '[s]eg_slic_eval' 2>/dev/null
pkill -9 -f '[s]eg_danbooregion_eval' 2>/dev/null
sleep 5

echo "== build remaining combo list (slic + danbooregion only) =="
rem=()
for M in slic danbooregion; do
  for R in $RATIOS; do
    P=$(echo "$R" | sed 's/0$//')   # arg->dir name: 0.00->0.0, 0.10->0.1, 0.50->0.5, 1.00->1.0
    for T in 0 1 2; do
      n=$(ls "$EVAL/results/dataset_name/proposed_on_${M}/sample_to_eval/illust/scribble/$T/$P/200" 2>/dev/null | wc -l)
      [ "$n" -lt 3000 ] && rem+=("$M:$R:$T") && echo "  todo: $M r=$R t=$T ($n/3000)"
    done
  done
done
echo "  remaining: ${#rem[@]} / 48 combos"
[ ${#rem[@]} -eq 0 ] && { echo "nothing left - all done!"; exit 0; }

ga=($GPUS); ng=${#ga[@]}
echo "== launch over $ng GPUs: $GPUS =="
for k in $(seq 0 $((ng-1))); do
  g=${ga[$k]}
  body="cd $CODE"
  i=$k
  while [ $i -lt ${#rem[@]} ]; do
    IFS=: read -r M R T <<< "${rem[$i]}"
    body="$body; echo \"== g$g $M r=$R t=$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$g $PY main.py --config configs/seg_${M}_eval.yaml --result_path $EVAL/results --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio $R --sketch_type $T"
    i=$((i+ng))
  done
  body="$body; echo DONE_g$g"
  tmux kill-session -t rd_g$g 2>/dev/null
  tmux new-session -d -s rd_g$g "$body > $EVAL/logs/redist_g$g.log 2>&1"
done
sleep 2
echo "launched $(tmux ls 2>/dev/null | grep -c rd_g) workers (rd_g*). logs: $EVAL/logs/redist_g*.log"
echo "(quickshift stopped & excluded; leftover files still at results/dataset_name/proposed_on_quickshift)"
