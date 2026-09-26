#!/bin/bash
# ============================================================================
# R2-1 — B_eval_dense_curve runner (HOST_B default)
# ============================================================================
# Sequential per-sketch run, NFS-friendly (workers=2, batch_size=8).
# Resumable: kills/OOMs OK — restart and it skips already-done (sketch, ratio,
# image_name) tuples in per_image.csv.
#
# Usage:
#   bash B_run.sh                    # full sweep, all 3 sketches × 101 ratios
#   GPU=1 bash B_run.sh              # use GPU 1
#   DRY=1 bash B_run.sh              # tiny smoke test (1 ratio, 16 imgs)
#   SKETCHES="0" bash B_run.sh       # only sketch 0
#   RATIOS="0.0 0.5 1.0" bash B_run.sh  # only listed ratios
# ============================================================================

set -e
cd "$(dirname "$0")"

PY=${PY:-/home/USER/anaconda3/envs/BBDM/bin/python3}
GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-2}

OUT_DIR="${OUT_DIR:-$(pwd)/../output/B_dense_curve}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-1_alpha_grid_sensitivity/inference_results/scribble}"

# Smoke / debug mode
EXTRA=()
if [ "$DRY" = "1" ]; then
    OUT_DIR="$(pwd)/../output/B_dense_curve_dry"
    RATIOS="${RATIOS:-0.5}"
    EXTRA+=(--limit 16 --no_resume)
    echo "### DRY mode → out=$OUT_DIR  ratios=$RATIOS  limit=16"
fi

SKETCH_ARGS=()
if [ -n "$SKETCHES" ]; then
    # space-separated list to args
    for s in $SKETCHES; do SKETCH_ARGS+=("$s"); done
else
    SKETCH_ARGS=(0 1 2)
fi

RATIO_ARGS=()
if [ -n "$RATIOS" ]; then
    for r in $RATIOS; do RATIO_ARGS+=("$r"); done
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo " [$(hostname -s)] R2-1 dense-curve eval"
echo "=========================================="
echo "  python      : $PY"
echo "  gpu         : $GPU"
echo "  batch_size  : $BATCH_SIZE"
echo "  workers     : $WORKERS"
echo "  sketches    : ${SKETCH_ARGS[*]}"
echo "  ratios      : ${RATIO_ARGS[*]:-<all>}"
echo "  results     : $RESULTS_ROOT"
echo "  out_dir     : $OUT_DIR"
echo "  log         : $LOG"
echo

# build cmd
CMD=("$PY" "$(pwd)/B_eval_dense_curve.py"
     --results_root "$RESULTS_ROOT"
     --sketches "${SKETCH_ARGS[@]}"
     --metrics psnr ssim lpips dreamsim
     --gpu "$GPU"
     --batch_size "$BATCH_SIZE"
     --workers "$WORKERS"
     --out_dir "$OUT_DIR")

if [ ${#RATIO_ARGS[@]} -gt 0 ]; then
    CMD+=(--ratios "${RATIO_ARGS[@]}")
fi
CMD+=("${EXTRA[@]}")

echo "+ ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "$LOG"
