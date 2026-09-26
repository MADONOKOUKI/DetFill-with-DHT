#!/bin/bash
# RUN ON reaper2 or reaper3. Launch a SLICE [FROM..TO] of the DanbooRegion-model 72-combo
# eval matrix across 10 GPUs (resume-safe). Combo order matches D_reaper_eval_matrix.sh:
#   index = hint*24 + ratio*3 + type,  hints = felz(0-23) slic(24-47) danbooregion(48-71)
# Static split: reaper2 takes 0-35, reaper3 takes 36-71 -> zero overlap, no sync needed.
#   usage: D_danboo_part.sh <ckpt_path> <from> <to> ["gpu list"]   (default GPUs: 0..9)
set -u
CKPT=${1:?ckpt path}; FROM=${2:?from idx}; TO=${3:?to idx}; GPUS="${4:-0 1 2 3 4 5 6 7 8 9}"
CODE=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
SR=/scratch/madono/seg_retrain
RESULT=$SR/results_eval/danbooregion_model
LOG=$SR/logs/eval_matrix_danbooregion
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
HINTS="felz slic danbooregion"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULT" "$LOG"; cd "$CODE"
[ -f "$CKPT" ] || { echo "ERR: ckpt not found: $CKPT"; exit 1; }

combos=()
for hint in $HINTS; do for R in $RATIOS; do for T in 0 1 2; do combos+=("$hint:$R:$T"); done; done; done
slice=("${combos[@]:$FROM:$((TO-FROM+1))}")
ga=($GPUS); ng=${#ga[@]}
echo "[$(date)] slice [$FROM..$TO] = ${#slice[@]} combos over $ng GPUs ($GPUS) -> $RESULT"

for k in $(seq 0 $((ng-1))); do
  g=${ga[$k]}
  body="cd $CODE"
  i=$k
  while [ $i -lt ${#slice[@]} ]; do
    IFS=: read -r hint R T <<< "${slice[$i]}"
    body="$body; echo \"== g$g $hint r=$R t=$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$g $PY main.py --config configs/eval_on_${hint}.yaml --result_path $RESULT --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio $R --sketch_type $T"
    i=$((i+ng))
  done
  body="$body; echo DONE_g$g"
  tmux kill-session -t em_danbooregion_g$g 2>/dev/null
  tmux new-session -d -s em_danbooregion_g$g "$body > $LOG/g$g.log 2>&1"
done
sleep 3
echo "[$(date)] launched: $(tmux ls 2>/dev/null | grep -c em_danbooregion_g) workers (em_danbooregion_g*) on $(hostname), slice [$FROM..$TO], GPUs: $GPUS"
