#!/bin/bash
# RUN ON naga1. Launch the DanbooRegion eval on one GPU (SLIC eval keeps running on its
# own GPU). 8 hint ratios x 3 sketch types, batch 24, 200 steps, full 3000 test set.
# Resume-safe (the runner skips any test image whose output PNG already exists).
#   usage:  bash naga1_eval_danboo.sh [gpu]      (default 1)
set -u
GPU=${1:-1}
CODE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
CKPT=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/checkpoints/proposed_96ch_scribble.pth
FX=/home/madorin/datasets/tog2024/main_exp_felzenszwalb_fixdot/illust
RD=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data
EVAL=/scratch/madono/seg_eval
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"
mkdir -p "$EVAL/logs"
sed -i '/^  test:/{n; s/batch_size: [0-9]*/batch_size: 24/}' configs/seg_danbooregion_eval.yaml

echo "== ensure DanbooRegion test data staged =="
have=$(find "$EVAL/danbooregion" -name '*_scribble_mask64.png' 2>/dev/null | wc -l)
if [ "$have" -lt 2990 ]; then
  awk '{s=$0; sub(/.*\/felzenszwalb\//,"",s); if(s!="") print s}' configs/illust/test_paper.txt | sort -u > /tmp/ev_gt.txt
  awk -F/ '{d=$1;i=$2;sub(/\.image\.png$/,"",i);
    print "danbooregion/"d"/"i".image_region64.png"; print "danbooregion/"d"/"i".image_scribble_mask64.png"; print "danbooregion/"d"/"i".image_scribble_col64.png"}' /tmp/ev_gt.txt > /tmp/ev_db.txt
  rsync -a --files-from=/tmp/ev_db.txt "$RD/" "$EVAL/"
fi
echo "  danbooregion test hints=$(find $EVAL/danbooregion -name '*_scribble_mask64.png'|wc -l) (gt=$(find $EVAL/gt -name '*.image.png'|wc -l))"

echo "== launch DanbooRegion eval on GPU$GPU =="
body="cd $CODE; for R in $RATIOS; do for T in 0 1 2; do echo \"== danbooregion ratio=\$R type=\$T \$(date +%H:%M) ==\"; CUDA_VISIBLE_DEVICES=$GPU $PY main.py --config configs/seg_danbooregion_eval.yaml --result_path $EVAL/results --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio \$R --sketch_type \$T; done; done; echo ALL_DONE_danbooregion"
tmux kill-session -t eval_danbooregion 2>/dev/null
tmux new-session -d -s eval_danbooregion "$body > $EVAL/logs/eval_danboo2.log 2>&1"
sleep 2
tmux ls 2>/dev/null | grep eval_danbooregion && echo "DanbooRegion eval -> GPU$GPU launched (log: $EVAL/logs/eval_danboo2.log)"
