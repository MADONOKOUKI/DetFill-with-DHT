#!/usr/bin/env bash
# ColorizeDiffusion v2 (ft) seeded-random-hint runs: scribble / sketchkeras
# (config hint_ratios_scribble_sketch1, sketch_id=1) / 32 runs.
# Uses the DETERMINISTIC pipeline (inference_official_fixed.py, sorted order) with
# sample_ratio=1.0 config; hints come pre-baked from the dataroot wrapper.
#   GPU=1 bash run_seedexp_coldiff.sh
set -uo pipefail
GPU="${GPU:-1}"
REPO=/home/madorin/gitlab/tog2024/main/colorizeDiffusion_v2
CKPT=/scratch/madono/colorizeDiffusion_v2_checkpoints/illust_v2_scr/final/model.safetensors
CFG=configs/inference/hint_ratios_scribble_sketch1/v2_inference_100.yaml   # sample_ratio:1.0, sketch_id:1
source /home/madorin/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate hf || { echo "[ERR] conda hf failed"; exit 1; }
ts(){ date '+%F %T'; }
RATIOS=(0 1 3 5 10 25 50 100)
cd "$REPO"
[ -f "$CKPT" ] || { echo "[ERR] ckpt missing"; exit 1; }
for r in "${RATIOS[@]}"; do
  if [ "$r" = 0 ] || [ "$r" = 100 ]; then SEEDS=(1); else SEEDS=(1 2 3 4 5); fi
  for s in "${SEEDS[@]}"; do
    tag="seed${s}_alpha_${r}"
    wrap="/scratch/madono/seedexp/coldiff_roots/$tag"
    out="/scratch/madono/seedexp/coldiff/$tag"
    n=$(ls "$out/test/samples_cfg_scale_5.00" 2>/dev/null | wc -l)
    if [ "$n" -ge 3000 ]; then echo "[$(ts)] skip $tag (done n=$n)"; continue; fi
    echo "[$(ts)] ### ColDiff $tag (have $n)"
    env CUDA_VISIBLE_DEVICES=$GPU python inference_official_fixed.py \
        --name "$out" --dataroot "$wrap/" \
        --batch_size 16 --num_threads 8 --eval_load_size 256 \
        -cfg "$CFG" -pt "$CKPT" -gs 5 || echo "[$(ts)] [FAIL] $tag"
  done
done
echo "[$(ts)] COLDIFF SEEDEXP ALL DONE"
