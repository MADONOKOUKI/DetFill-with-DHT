#!/bin/bash
# RUN ON a reaper inside tmux/nohup. WAITS for THIS reaper's training to finish, then
# immediately runs the eval matrix on ALL 10 GPUs (so the freed GPUs are grabbed at once):
#   trained model ($SEG) x {felz, slic, danbooregion} hints x 8 ratio x 3 sketch type
#   = 72 combos, round-robin over GPU 0..9. Resume-safe (skips done images).
#   usage: D_reaper_eval_matrix.sh <slic|danbooregion>
set -u
SEG=${1:?slic|danbooregion}
CODE=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
SR=/scratch/madono/seg_retrain
RESULT="$SR/results_eval/${SEG}_model"
LOG="$SR/logs/eval_matrix_${SEG}"
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
HINTS="felz slic danbooregion"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULT" "$LOG"; cd "$CODE"

echo "[$(date)] waiting for ${SEG} training to COMPLETE (latest_model_200.pth present AND proc exited)..."
while true; do
  have200=$(ls $SR/results/$SEG/*/*/checkpoint/latest_model_200.pth 2>/dev/null | head -1)
  if [ -n "$have200" ] && ! pgrep -f "[s]eg_${SEG}_illust" >/dev/null 2>&1; then break; fi
  sleep 300
done
echo "[$(date)] ${SEG} training complete (model_200 present, proc exited)."

CKPT=$(ls $SR/results/$SEG/*/*/checkpoint/latest_model_200.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -v $SR/results/$SEG/*/*/checkpoint/latest_model_*.pth 2>/dev/null | tail -1)  # highest-numbered
[ -z "$CKPT" ] && CKPT=$(ls $SR/results/$SEG/*/*/checkpoint/last_model.pth 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "[$(date)] ERR: no final ckpt for $SEG under $SR/results/$SEG"; exit 1; }
echo "[$(date)] using ckpt=$CKPT"

# data sanity
for hint in $HINTS; do
  n=$(find "$SR/$hint" -name '*_scribble_mask64.png' 2>/dev/null | wc -l)
  echo "  hint=$hint staged=$n"
done

# build 72-combo list, distribute round-robin over 10 GPUs
combos=()
for hint in $HINTS; do for R in $RATIOS; do for T in 0 1 2; do combos+=("$hint:$R:$T"); done; done; done
echo "[$(date)] ${#combos[@]} combos over 10 GPUs -> $RESULT"
for g in $(seq 0 9); do
  body="cd $CODE"
  i=$g
  while [ $i -lt ${#combos[@]} ]; do
    IFS=: read -r hint R T <<< "${combos[$i]}"
    body="$body; echo \"== g$g $hint r=$R t=$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$g $PY main.py --config configs/eval_on_${hint}.yaml --result_path $RESULT --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio $R --sketch_type $T"
    i=$((i+10))
  done
  body="$body; echo DONE_g$g"
  tmux kill-session -t em_${SEG}_g$g 2>/dev/null
  tmux new-session -d -s em_${SEG}_g$g "$body > $LOG/g$g.log 2>&1"
done
sleep 3
echo "[$(date)] launched: $(tmux ls 2>/dev/null | grep -c em_${SEG}_g) GPU workers (em_${SEG}_g0..9). logs: $LOG/g*.log"
