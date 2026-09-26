#!/bin/bash
# RUN ON HOST_A. Parallel DanbooRegion eval across HOST_A's 4 GPUs (SLIC eval is already done).
# Distributes the (ratio x sketch_type) combos round-robin over GPU 0..3. Resume-safe:
# the runner skips any test image whose output PNG already exists, so killing the slow
# single-GPU run and relaunching here wastes nothing.
#   usage:  bash danboo_4gpu.sh "0 1 2"     (sketch types; use "0" for 1 type = ~3x faster)
set -u
TYPES="${1:-0 1 2}"
CODE=/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/USER/anaconda3/envs/BBDM/bin/python
CKPT=/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/checkpoints/proposed_96ch_scribble.pth
EVAL=/scratch/USER/seg_eval
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== stop the slow single-GPU danboo run =="
tmux kill-session -t eval_danbooregion 2>/dev/null
pkill -9 -f '[s]eg_danbooregion_eval' 2>/dev/null; sleep 4

# build combo list
combos=(); for R in $RATIOS; do for T in $TYPES; do combos+=("$R:$T"); done; done
echo "== ${#combos[@]} combos over 4 GPUs (types: $TYPES) =="

for g in 0 1 2 3; do
  body="cd $CODE"
  i=$g
  while [ $i -lt ${#combos[@]} ]; do
    RT=${combos[$i]}; R=${RT%:*}; T=${RT#*:}
    body="$body; echo \"== g$g ratio=$R type=$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$g $PY main.py --config configs/seg_danbooregion_eval.yaml --result_path $EVAL/results --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio $R --sketch_type $T"
    i=$((i+4))
  done
  body="$body; echo DONE_g$g"
  tmux kill-session -t db_g$g 2>/dev/null
  tmux new-session -d -s db_g$g "$body > $EVAL/logs/danboo_g$g.log 2>&1"
done
sleep 2
echo "launched: $(tmux ls 2>/dev/null | grep -c db_g) GPU workers (db_g0..3)"
echo "monitor: tail -f $EVAL/logs/danboo_g0.log   |   nvidia-smi"
