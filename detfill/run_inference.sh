#!/bin/bash
# Inference over the hint-ratio grid (the Table II protocol): DetFill colorizes the test split at every hint ratio
# and line-art source. Run from detfill/ with the checkpoints at results/dataset_name/BrownianBridge_<hint>_illust/checkpoint/.
#
# Usage:
#   GPU=<id> [RATIOS="..."] [TYPES="..."] bash run_inference.sh [dot|scribble|all]
#
# Examples:
#   GPU=0 bash run_inference.sh dot                          # dot model, all ratios and line-art sources (0,1,2)
#   GPU=1 bash run_inference.sh scribble                     # scribble model, all ratios
#   GPU=0 RATIOS="0.00 0.01" bash run_inference.sh dot       # selected ratios only
#   GPU=2 TYPES="0 1" bash run_inference.sh all              # line-art sources 0 and 1 only
#   GPU=3 RATIOS="0.10" TYPES="2" bash run_inference.sh dot  # one ratio x one source
# Line-art sources: 0 = sketch simplification, 1 = XDoG, 2 = SketchKeras.

set -e

GPU=${GPU:-0}
MODE=${1:-all}
RATIOS="${RATIOS:-0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00}"
TYPES="${TYPES:-0 1 2}"

DOT_CONFIG="configs/dot_illust.yaml"
DOT_CKPT="results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth"

SCR_CONFIG="configs/scribble_illust.yaml"
SCR_CKPT="results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth"

run_dot() {
    echo "=== DOT inference (ratios: $RATIOS, types: $TYPES) ==="
    for RATIO in $RATIOS; do
        for TYPE in $TYPES; do
            echo "--- dot ratio=$RATIO sketch_type=$TYPE ---"
            python3 main.py \
                --config "$DOT_CONFIG" \
                --resume_model "$DOT_CKPT" \
                --sample_to_eval --save_top \
                --gpu_ids "$GPU" \
                --sample_ratio "$RATIO" \
                --sketch_type "$TYPE"
        done
    done
}

run_scribble() {
    echo "=== SCRIBBLE inference (ratios: $RATIOS, types: $TYPES) ==="
    for RATIO in $RATIOS; do
        for TYPE in $TYPES; do
            echo "--- scribble ratio=$RATIO sketch_type=$TYPE ---"
            python3 main.py \
                --config "$SCR_CONFIG" \
                --resume_model "$SCR_CKPT" \
                --sample_to_eval --save_top \
                --gpu_ids "$GPU" \
                --sample_ratio "$RATIO" \
                --sketch_type "$TYPE"
        done
    done
}

case "$MODE" in
    dot)      run_dot ;;
    scribble) run_scribble ;;
    all)      run_dot; run_scribble ;;
    *)        echo "Usage: GPU=<id> [RATIOS='...'] [TYPES='...'] bash $0 [dot|scribble|all]"; exit 1 ;;
esac

echo "Done."
