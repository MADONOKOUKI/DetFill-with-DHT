#!/bin/bash
# RUN ON naga1.  Proposed-checkpoint inference on SLIC & Quickshift hints, area-sorted,
# at the paper's 8 hint ratios x 3 sketch types. SLIC on GPU $G0, Quickshift on GPU $G1
# (parallel). Reads code/env/ckpt from NFS; stages TEST-split data to local /scratch;
# writes outputs to local /scratch (sync to revision_materials later).
#
#   usage:  bash naga1_eval.sh [gpu_slic] [gpu_qs]      e.g.  bash naga1_eval.sh 0 1
set -u
G0=${1:-0}; G1=${2:-1}
CODE=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/BBDM_seg_retrain
PY=/home/madorin/anaconda3/envs/BBDM/bin/python
CKPT=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/checkpoints/proposed_96ch_scribble.pth  # 96ch proposed (paper)
FX=/home/madorin/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust
RD=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data
EVAL=/scratch/madono/seg_eval
RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== sanity (NFS visible?) =="
[ -x "$PY" ]      || { echo "ERR: BBDM env not visible"; exit 1; }
[ -f "$CKPT" ]    || { echo "ERR: proposed ckpt not visible"; exit 1; }
[ -f "$CODE/main.py" ] || { echo "ERR: code not visible"; exit 1; }
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())" || exit 1

echo "== stage TEST-split data -> $EVAL (local /scratch) =="
mkdir -p "$EVAL"/{gt,sketch,slic,quickshift,results,logs}
awk '{s=$0; sub(/.*\/felzenszwalb\//,"",s); if(s!="") print s}' "$CODE/configs/illust/test_paper.txt" | sort -u > /tmp/eval_gt.txt
echo "  test ids: $(wc -l < /tmp/eval_gt.txt)"
rsync -a --files-from=/tmp/eval_gt.txt "$FX/segmentation_regions/felzenszwalb/" "$EVAL/gt/"
awk -F/ '{d=$1;i=$2;sub(/\.image\.png$/,"",i);
  print "sketch/XDoG/"d"/"i".png"; print "sketch/pysimp/"d"/"i".png"; print "sketch/sketchkeras/"d"/"i".png"}' \
  /tmp/eval_gt.txt > /tmp/eval_sk.txt
rsync -a --files-from=/tmp/eval_sk.txt "$FX/" "$EVAL/"
for seg in slic quickshift; do
  awk -F/ -v s=$seg '{d=$1;i=$2;sub(/\.image\.png$/,"",i);
    print s"/"d"/"i".image_region64.png"; print s"/"d"/"i".image_scribble_mask64.png"; print s"/"d"/"i".image_scribble_col64.png"}' \
    /tmp/eval_gt.txt > /tmp/eval_$seg.txt
  rsync -a --files-from=/tmp/eval_$seg.txt "$RD/" "$EVAL/"
done
echo "  staged: gt=$(find $EVAL/gt -name '*.image.png'|wc -l) sketch=$(find $EVAL/sketch -name '*.png'|wc -l) slic=$(find $EVAL/slic -name '*_scribble_mask64.png'|wc -l) qs=$(find $EVAL/quickshift -name '*_scribble_mask64.png'|wc -l)"

run_seg() {  # seg gpu
  local seg=$1 gpu=$2
  local body="cd $CODE; for R in $RATIOS; do for T in 0 1 2; do echo \"=== $seg ratio=\$R type=\$T \$(date +%H:%M) ===\"; CUDA_VISIBLE_DEVICES=$gpu $PY main.py --config configs/seg_${seg}_eval.yaml --result_path $EVAL/results --resume_model $CKPT --sample_to_eval --save_top --gpu_ids 0 --sample_ratio \$R --sketch_type \$T; done; done; echo \"ALL DONE $seg\""
  tmux kill-session -t eval_$seg 2>/dev/null
  tmux new-session -d -s eval_$seg "$body > $EVAL/logs/eval_$seg.log 2>&1"
  sleep 2; tmux ls 2>/dev/null | grep eval_$seg && echo "$seg -> GPU$gpu launched (log: $EVAL/logs/eval_$seg.log)"
}
echo "== launch inference (SLIC on GPU$G0, Quickshift on GPU$G1) =="
run_seg slic       "$G0"
run_seg quickshift "$G1"
echo "done. monitor: tail -f $EVAL/logs/eval_slic.log   /   nvidia-smi"
echo "outputs: $EVAL/results/dataset_name/proposed_on_{slic,quickshift}/sample_to_eval/illust/scribble/<type>/<ratio>/200/"
