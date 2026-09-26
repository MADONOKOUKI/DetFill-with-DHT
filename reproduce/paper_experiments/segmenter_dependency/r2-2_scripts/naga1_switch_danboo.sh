#!/bin/bash
# RUN ON naga1. Replace the running Quickshift eval with a DanbooRegion eval
# (the SLIC eval on the other GPU keeps running). gt/sketch were already staged by
# naga1_eval.sh; this only stages the DanbooRegion test hints + launches.
#   usage:  bash naga1_switch_danboo.sh [gpu]      (default 1 = the GPU QS was using)
set -u
GPU=${1:-1}
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
CKPT=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/checkpoints/proposed_96ch_scribble.pth
RD=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data
EVAL=/scratch/madono/seg_eval
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== stop QS eval =="
tmux kill-session -t eval_quickshift 2>/dev/null && echo "killed eval_quickshift" || echo "(eval_quickshift not found - ok)"

echo "== stage DanbooRegion TEST hints (NFS -> local /scratch) =="
mkdir -p "$EVAL/danbooregion" "$EVAL/logs"
awk '{s=$0; sub(/.*\/felzenszwalb\//,"",s); if(s!="") print s}' "$CODE/configs/illust/test_paper.txt" | sort -u > /tmp/ev_gt.txt
awk -F/ '{d=$1;i=$2;sub(/\.image\.png$/,"",i);
  print "danbooregion/"d"/"i".image_region64.png";
  print "danbooregion/"d"/"i".image_scribble_mask64.png";
  print "danbooregion/"d"/"i".image_scribble_col64.png"}' /tmp/ev_gt.txt > /tmp/ev_db.txt
rsync -a --files-from=/tmp/ev_db.txt "$RD/" "$EVAL/"
echo "  staged: danbooregion=$(find $EVAL/danbooregion -name '*_scribble_mask64.png'|wc -l)  (gt=$(find $EVAL/gt -name '*.image.png'|wc -l) sketch=$(find $EVAL/sketch -name '*.png'|wc -l))"

echo "== launch DanbooRegion eval on GPU$GPU =="
body="cd $CODE; for R in $RATIOS; do for T in 0 1 2; do echo \"=== danbooregion ratio=\$R type=\$T \$(date +%H:%M) ===\"; CUDA_VISIBLE_DEVICES=$GPU $PY main.py --config configs/seg_danbooregion_eval.yaml --result_path $EVAL/results --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio \$R --sketch_type \$T; done; done; echo 'ALL DONE danbooregion'"
tmux kill-session -t eval_danbooregion 2>/dev/null
tmux new-session -d -s eval_danbooregion "$body > $EVAL/logs/eval_danbooregion.log 2>&1"
sleep 2
tmux ls 2>/dev/null | grep eval_danbooregion && echo "danbooregion eval -> GPU$GPU launched (log: $EVAL/logs/eval_danbooregion.log)"
echo "SLIC eval (other GPU) is untouched. monitor: tail -f $EVAL/logs/eval_danbooregion.log"
