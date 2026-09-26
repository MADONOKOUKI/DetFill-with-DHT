#!/bin/bash
# Single-ratio inference for BBDM_revision_revise_DATESTAMP.
# Runs the 3 sketch types (0/1/2 = pysimp/XDoG/sketchkeras) for one ratio.
#
# Required env:
#   RATIO       : sample_ratio (e.g., 0.02)
# Optional env:
#   GPU_ID      : physical GPU index (default 0)
#   HINT_TYPE   : dot | scribble (default dot)
#   MODEL       : checkpoint path (default: results/.../latest_model_200.pth)
#   CONFIG      : config yaml path (default: configs/<hint>_proposed_illust_200epoch_mr.yaml)
#   TYPES       : sketch types to run (default "0 1 2")
#   RESULT_PATH : where to write outputs (default /scratch/madono/major_revision/results)
set -e

RATIO="${RATIO:?RATIO env var required (e.g., 0.02)}"
GPU_ID="${GPU_ID:-0}"
HINT_TYPE="${HINT_TYPE:-dot}"
TYPES="${TYPES:-0 1 2}"
RESULT_PATH="${RESULT_PATH:-/scratch/madono/major_revision/results}"

# Use BBDM conda env (compatible transformers / dreamsim / torch+cuda).
export PATH=/home/madorin/anaconda3/envs/BBDM/bin:$PATH

# Reduce fragmentation-related OOM for 96-channel model on 11 GB 2080Ti.
: ${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF

cd "$(dirname "$0")"

# Default checkpoint per hint type
if [ -z "$MODEL" ]; then
    case "$HINT_TYPE" in
        dot)
            MODEL="results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth"
            ;;
        scribble)
            MODEL="results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth"
            ;;
        *)
            echo "[error] unknown HINT_TYPE='$HINT_TYPE' (expected dot|scribble)" >&2
            exit 1
            ;;
    esac
fi

# Default config per hint type
if [ -z "$CONFIG" ]; then
    CONFIG="configs/${HINT_TYPE}_proposed_illust_200epoch_mr.yaml"
fi

if [ ! -f "$MODEL" ]; then
    echo "[error] checkpoint not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "[error] config not found: $CONFIG" >&2
    exit 1
fi

echo "### host=$(hostname -s) date=$(date -Iseconds) ratio=$RATIO hint=$HINT_TYPE gpu=$GPU_ID"
echo "### CONFIG=$CONFIG"
echo "### MODEL=$MODEL"
echo "### RESULT_PATH=$RESULT_PATH"
echo "### BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE:-<config default>}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Skip-if-complete: if a ratio×sketch_type already has EXPECTED_COUNT (=3000) outputs,
# don't relaunch python (BBDMRunner.py also skips per-image, but starting python only to
# skip 3000 files wastes ~2 min of model load + dreamsim init per ratio×type).
# Set EXPECTED_COUNT=0 to disable the gate (always run; per-image skip still applies).
EXPECTED_COUNT="${EXPECTED_COUNT:-3000}"
MODEL_NAME="BrownianBridge_${HINT_TYPE}_illust"
RESULT_BASE="${RESULT_PATH}/dataset_name/${MODEL_NAME}/sample_to_eval/illust/${HINT_TYPE}"

for TYPE in $TYPES; do
    OUT_DIR="${RESULT_BASE}/${TYPE}/${RATIO}/200"
    if [ "$EXPECTED_COUNT" -gt 0 ] && [ -d "$OUT_DIR" ]; then
        N=$(ls "$OUT_DIR" 2>/dev/null | wc -l)
        if [ "$N" -ge "$EXPECTED_COUNT" ]; then
            echo "--- skip: ratio=$RATIO sketch_type=$TYPE already complete (n=$N >= $EXPECTED_COUNT) ---"
            continue
        else
            echo "--- resume: ratio=$RATIO sketch_type=$TYPE (have $N/$EXPECTED_COUNT, BBDMRunner will skip existing) ---"
        fi
    else
        echo "--- ratio=$RATIO sketch_type=$TYPE (fresh) ---"
    fi
    python3 main.py \
        --config "$CONFIG" \
        --result_path "$RESULT_PATH" \
        --resume_model "$MODEL" \
        --sample_to_eval --save_top \
        --gpu_ids 0 \
        --sample_ratio "$RATIO" \
        --sketch_type "$TYPE"
done

echo "### ratio=$RATIO hint=$HINT_TYPE DONE"
