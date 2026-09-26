#!/usr/bin/env bash
# =============================================================================
# run_hauc_pipeline.sh — DetFill inference over the hint-ratio grid + 7-metric Hint-AUC
#
# Generic, runnable version of the launchers used for the paper (Table II, Table III and the
# supplementary alpha-grid / seed / segmenter studies all follow this same loop):
#   for every (hint ratio, sketch type):  detfill/main.py --sample_to_eval          (colorize the test split)
#   then:                                 reproduce/scripts/eval_per_ratio.py       (per-ratio metrics + Hint-AUC)
#
# Required:
#   DATA_ROOT   flat evaluation layout (detfill/README.md, layout A):
#                 <DATA_ROOT>/segmentations/originals/<id>.image.png
#                 <DATA_ROOT>/sketch/{XDoG,pysimp,sketchkeras}/<id>.png
#                 <DATA_ROOT>/hint_from_regions_64_rev/<id>.image_{scribble,dot}_{col,mask}64.png
#                 <DATA_ROOT>/hint_from_regions_256/<id>.image_region64.png
#               The stored test-split hint maps of the paper are release asset test_split_hint_maps_64.tar.gz.
# Optional (defaults reproduce the Table II protocol):
#   HINT=scribble|dot            hint type / checkpoint / config                       (default scribble)
#   CKPT=<path>                  checkpoint (default: detfill/results/.../latest_model_200.pth = checkpoints/README.md)
#   GPU=<id>                     CUDA device index, -1 = CPU                          (default 0)
#   RATIOS="0.00 0.01 ..."       hint ratios (default: the paper grid)
#   TYPES="0 1 2"                sketch types (default all three; see SKETCH TYPES below)
#   HINT_ORDER=area|label        region order: area = largest first (Table II), label = fixed random order (Table III)
#   RESULT_PATH=<dir>            where main.py writes samples (default reproduce/output/<tag>/results)
#   OUT_DIR=<dir>                metrics output (default reproduce/output/<tag>/metrics)
#   METRICS="mse psnr ..."       subset of mse psnr ssim lpips openclip dino dreamsim (default all)
#   LIMIT=<n>                    evaluate only the first n images per cell (smoke test)
#   SKIP_INFER=1                 only evaluate existing samples
#   TEST_BATCH=<n>               inference batch size (default: the config's 5 for scribble / 8 for dot). The test loader
#                                drops the last incomplete batch, so the number of images must be divisible by it
#                                (3,000 is); set TEST_BATCH=1 for arbitrary subsets.
#
# Examples
#   DATA_ROOT=/data/danbooru_test GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh                 # Table II scribble row
#   DATA_ROOT=/data/danbooru_test HINT=dot GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh        # Table II dot row
#   DATA_ROOT=/data/danbooru_test HINT_ORDER=label bash reproduce/scripts/run_hauc_pipeline.sh      # Table III scribble row
#   DATA_ROOT=... RATIOS="$(seq -f %.2f 0 0.02 1)" bash reproduce/scripts/run_hauc_pipeline.sh      # dense alpha sweep (supp.)
#
# SKETCH TYPES: index -> line-art source follows the dataset loader (detfill/datasets/custom.py):
# 0 = sketch simplification (pysimp), 1 = XDoG, 2 = SketchKeras.
# Run time: one (ratio, type) cell = 3,000 images x 200 sampling steps; ~35 min on one RTX A6000 with the
# 96-channel scribble model, so the full 8 x 3 grid is ~14 GPU-hours on that GPU (dot model: about half;
# an RTX 2080 Ti is roughly twice as slow).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the flat evaluation layout (see header)}"
HINT="${HINT:-scribble}"
GPU="${GPU:-0}"
RATIOS="${RATIOS:-0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00}"
TYPES="${TYPES:-0 1 2}"
HINT_ORDER="${HINT_ORDER:-area}"
METRICS="${METRICS:-mse psnr ssim lpips openclip dino dreamsim}"
TAG="${TAG:-${HINT}_${HINT_ORDER}}"
RESULT_PATH="${RESULT_PATH:-$ROOT/reproduce/output/$TAG/results}"
OUT_DIR="${OUT_DIR:-$ROOT/reproduce/output/$TAG/metrics}"
CKPT="${CKPT:-$ROOT/detfill/results/dataset_name/BrownianBridge_${HINT}_illust/checkpoint/latest_model_200.pth}"
PY="${PY:-python}"

case "$HINT" in scribble|dot) ;; *) echo "HINT must be scribble or dot" >&2; exit 1;; esac
case "$HINT_ORDER" in area|label) ;; *) echo "HINT_ORDER must be area or label" >&2; exit 1;; esac
for d in segmentations/originals sketch hint_from_regions_64_rev hint_from_regions_256; do
  [ -d "$DATA_ROOT/$d" ] || { echo "[error] missing $DATA_ROOT/$d (layout: detfill/README.md)" >&2; exit 1; }
done
if [ "${SKIP_INFER:-0}" != 1 ]; then
  [ -f "$CKPT" ] || { echo "[error] checkpoint not found: $CKPT  (see checkpoints/README.md)" >&2; exit 1; }
fi
mkdir -p "$RESULT_PATH" "$OUT_DIR"

# ---- config: released config with the data root and the region order filled in ---------------------
CFG_SRC="$ROOT/detfill/configs/${HINT}_illust.yaml"
CFG="$OUT_DIR/${HINT}_illust_${HINT_ORDER}.yaml"
sed -E "s#^(\s*dataset_path:).*#\1 '$DATA_ROOT'#; s#^(\s*scratch_root:).*#\1 '$DATA_ROOT'#" "$CFG_SRC" > "$CFG"
if grep -qE "^\s*#?\s*hint_order:" "$CFG"; then
  sed -i -E "s|^(\s*)#?\s*hint_order:.*|\1hint_order: '$HINT_ORDER'|" "$CFG"
else
  sed -i -E "s#^(\s*)hint_type:(.*)#\1hint_type:\2\n\1hint_order: '$HINT_ORDER'#" "$CFG"
fi
echo "[config] $CFG  (dataset root $DATA_ROOT, hint_order $HINT_ORDER)"

# ---- inference: one process per (ratio, sketch type), sequential -----------------------------------
if [ "${SKIP_INFER:-0}" != 1 ]; then
  cd "$ROOT/detfill"
  [ -n "${TEST_BATCH:-}" ] && export BATCH_SIZE_OVERRIDE="$TEST_BATCH"
  for R in $RATIOS; do for T in $TYPES; do
    n_have=$(ls "$RESULT_PATH/dataset_name/BrownianBridge_${HINT}_illust/sample_to_eval/illust/$HINT/$T/$(python -c "print(float('$R'))")/200" 2>/dev/null | wc -l || true)
    echo "[infer] $(date '+%F %T') hint=$HINT ratio=$R sketch_type=$T (have $n_have)"
    $PY main.py --config "$CFG" --resume_model "$CKPT" --sample_to_eval --save_top \
        --gpu_ids "$GPU" --sample_ratio "$R" --sketch_type "$T" --result_path "$RESULT_PATH"
  done; done
  cd "$ROOT"
fi

# ---- evaluation ---------------------------------------------------------------------------------------
SAMPLES="$RESULT_PATH/dataset_name/BrownianBridge_${HINT}_illust/sample_to_eval/illust/$HINT"
[ -d "$SAMPLES" ] || { echo "[error] no samples under $SAMPLES" >&2; exit 1; }
EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
$PY "$HERE/eval_per_ratio.py" --results_root "$SAMPLES" --gt_dir "$DATA_ROOT/segmentations/originals" \
    --out_dir "$OUT_DIR" --sketches $TYPES --metrics $METRICS --gpu "$GPU" "${EXTRA[@]}"
echo "[done] per-ratio metrics: $OUT_DIR/per_ratio_summary.csv ; Hint-AUC: $OUT_DIR/hauc.json"
echo "       compare with the paper: python reproduce/scripts/A1_tables_from_released_metrics.py (released numbers)"
