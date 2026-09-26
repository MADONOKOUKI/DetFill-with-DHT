#!/bin/bash
# RUN ON reaper6 in tmux. After reaper6's SLIC-model eval (em_slic_g*) finishes, sweep up
# whatever is LEFT of the DanbooRegion-model matrix (split statically: reaper2=combos 0-35,
# reaper3=36-71). At fire time: pull both reapers' outputs (--ignore-existing), compute the
# remaining (<3000 imgs) combos, run them over 10 GPUs; a background syncer keeps pulling
# every 30min so resume-skip stays aware of r2/r3 concurrent progress. If nothing remains,
# exits cleanly. All workers resume-safe.
set -u
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
SR=/scratch/madono/seg_retrain
CKPT=$SR/handoff_ckpt/danbooregion_model_200.pth
RESULT=$SR/results_eval/danbooregion_model
LOG=$SR/logs/handoff_danboo
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
HINTS="felz slic danbooregion"
RSHE="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RESULT" "$LOG"; cd "$CODE"
[ -f "$CKPT" ] || { echo "[$(date)] ERR: danboo ckpt missing at $CKPT"; exit 1; }

echo "[$(date)] waiting for reaper6 SLIC eval (em_slic_g*) to finish..."
while tmux ls 2>/dev/null | grep -q em_slic_g; do sleep 300; done
echo "[$(date)] SLIC eval done -> sweeping remaining DanbooRegion-model combos."

pull_all() {
  for h in reaper2 reaper3; do
    rsync -a --ignore-existing -e "$RSHE" "$h:$RESULT/" "$RESULT/" 2>>"$LOG/sync.log" || true
  done
}
pull_all
echo "[$(date)] initial pull from reaper2+reaper3 done."

# background syncer: keep pulling so resume-skip sees r2/r3 progress during the sweep
tmux kill-session -t hd_sync 2>/dev/null
tmux new-session -d -s hd_sync "while true; do for h in reaper2 reaper3; do rsync -a --ignore-existing -e \"$RSHE\" \$h:$RESULT/ $RESULT/ 2>>$LOG/sync.log; done; echo \"[\$(date)] pulled\" >> $LOG/sync.log; sleep 1800; done"

# compute remaining combos (<3000 imgs in the 200 dir)
rem=()
for hint in $HINTS; do
  for R in $RATIOS; do
    P=$(echo "$R" | sed 's/0$//')   # arg->dir: 0.00->0.0, 0.10->0.1, 0.50->0.5, 1.00->1.0
    for T in 0 1 2; do
      n=$(ls "$RESULT/dataset_name/on_${hint}/sample_to_eval/illust/scribble/$T/$P/200" 2>/dev/null | wc -l)
      [ "$n" -lt 3000 ] && rem+=("$hint:$R:$T") && echo "  todo: $hint r=$R t=$T ($n/3000)"
    done
  done
done
echo "[$(date)] remaining: ${#rem[@]} / 72 combos"
[ ${#rem[@]} -eq 0 ] && { echo "[$(date)] nothing left - matrix complete, exiting."; tmux kill-session -t hd_sync 2>/dev/null; exit 0; }

for g in $(seq 0 9); do
  body="cd $CODE"
  i=$g
  while [ $i -lt ${#rem[@]} ]; do
    IFS=: read -r hint R T <<< "${rem[$i]}"
    body="$body; echo \"== g$g $hint r=$R t=$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$g $PY main.py --config configs/eval_on_${hint}.yaml --result_path $RESULT --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio $R --sketch_type $T"
    i=$((i+10))
  done
  body="$body; echo DONE_g$g"
  tmux kill-session -t hd_g$g 2>/dev/null
  tmux new-session -d -s hd_g$g "$body > $LOG/g$g.log 2>&1"
done
sleep 3
echo "[$(date)] reaper6 sweeping: $(tmux ls 2>/dev/null | grep -c hd_g) workers (hd_g0..9) + hd_sync."
