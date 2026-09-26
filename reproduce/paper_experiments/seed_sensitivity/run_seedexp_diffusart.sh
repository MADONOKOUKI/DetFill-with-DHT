#!/usr/bin/env bash
# Diffusart seeded-random-hint runs: scribble / sketchkeras (rnd=2 in
# test_seedexp.py) / 32 runs. test_seedexp.py = test_labrepo.py with rnd fixed to
# 2 and SEEDEXP_SAVE_ROOT-driven output. Hints/region are swapped per run by
# re-pointing the global felzenszwalb symlink (runs are strictly sequential).
#   GPU=0 bash run_seedexp_diffusart.sh
set -uo pipefail
GPU="${GPU:-0}"
REPO=/home/USER/gitlab/labrepo/main/Diffusion_v1_comp
LINK=/scratch/USER/main_exp/illust/hint_from_regions/felzenszwalb
SAVE_ROOT=/scratch/USER/seedexp/diffusart
source /home/USER/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate BBDM || { echo "[ERR] conda BBDM failed"; exit 1; }
ts(){ date '+%F %T'; }
RATIOS=(0 1 3 5 10 25 50 100)
cd "$REPO"
for r in "${RATIOS[@]}"; do
  if [ "$r" = 0 ] || [ "$r" = 100 ]; then SEEDS=(1); else SEEDS=(1 2 3 4 5); fi
  for s in "${SEEDS[@]}"; do
    tag="seed${s}_alpha_${r}"
    out="$SAVE_ROOT/$tag/100"     # dir_path = save_path + int(ratio*100)
    n=$(ls "$out" 2>/dev/null | grep -cE "^[0-9]+\.png$")
    if [ "$n" -ge 2999 ]; then echo "[$(ts)] skip $tag (done n=$n)"; continue; fi
    ln -sfn "/scratch/USER/seedexp/diffusart_hints/$tag" "$LINK"
    echo "[$(ts)] ### Diffusart $tag (have $n)"
    env CUDA_VISIBLE_DEVICES=$GPU SEEDEXP_SAVE_ROOT="$SAVE_ROOT" \
      python test_seedexp.py \
        --hint_name scribble \
        --checkpoint_path checkpoint/baseline_scribble/checkpoint_198000.pth \
        --ratio 1.0 --rnd 2 --domain illust --save_name "$tag" \
      || echo "[$(ts)] [FAIL] $tag"
  done
done
echo "[$(ts)] DIFFUSART SEEDEXP ALL DONE"
