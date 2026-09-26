#!/usr/bin/env bash
# =============================================================================
# B9_train_detfill.sh — train DetFill from scratch (the commands behind the released checkpoints)
#
# The released 200-epoch checkpoints were trained with these two commands (the paper's train.sh, with
# the released config names). Data: Danbooru2021 illustrations in the split-based layout
# (detfill/README.md, layout B; ids in detfill/configs/illust/{train,valid,test}.txt), sketches from the
# three extractors and the deterministic 64x64 hint maps (hint_generation/ or the hintauc library).
#
#   HINT=scribble|dot   (default scribble; scribble = 96 base channels, dot = 64 base channels)
#   GPU_IDS=0,1,...     (paper: 10 GPUs, batch 1 per GPU x accumulate_grad_batches 2 = effective batch 20)
#   DATA_ROOT=<split-based dataset root>
#
# Training time for reference: ~4-5 days on 10 RTX 2080 Ti / A6000-class GPUs for 200 epochs.
# Reproduced numbers from a retrained model will differ slightly from the released checkpoint
# (non-deterministic multi-GPU training); the released checkpoints are the reference.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
HINT="${HINT:-scribble}"; GPU_IDS="${GPU_IDS:-0}"; PORT="${PORT:-12356}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (split-based layout, detfill/README.md)}"
CFG_SRC="$ROOT/detfill/configs/${HINT}_illust.yaml"
CFG="$ROOT/reproduce/output/train_${HINT}/${HINT}_illust_train.yaml"
mkdir -p "$(dirname "$CFG")"
sed -E "s#^(\s*dataset_path:).*#\1 '$DATA_ROOT'#; s#^(\s*scratch_root:).*#\1 '$DATA_ROOT'#" "$CFG_SRC" > "$CFG"
cd "$ROOT/detfill"
# verbatim launch (train.sh of the paper repository, config names updated):
python3 main.py --config "$CFG" --train --sample_at_start --save_top --gpu_ids "$GPU_IDS" --port "$PORT" \
    --result_path "$ROOT/reproduce/output/train_${HINT}/results"
# checkpoints: <result_path>/dataset_name/BrownianBridge_<hint>_illust/checkpoint/latest_model_<epoch>.pth
