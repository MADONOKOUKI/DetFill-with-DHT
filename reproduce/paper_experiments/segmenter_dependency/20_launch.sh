#!/bin/bash
# Launcher: runs 10_run_ratio.sh across a GPU slot pool for a list of ratios.
#
# Required env:
#   RATIOS    : space-separated ratios (e.g., "0.00 0.02 0.04 ... 0.50")
# Optional env:
#   HINT_TYPE : dot | scribble (default dot)
#   MODEL     : checkpoint path override (default: per-hint default in 10_run_ratio.sh)
#   GPU_IDS   : space-separated physical GPU indices (default: auto-detect free GPUs, fallback "0")
#   TYPES     : sketch types passed through to 10_run_ratio.sh (default "0 1 2")
#   FREE_THRESHOLD_MB : memory.used <= this is "free" (default 1000)
#   NUM_GPUS_MAX      : cap number of auto-detected GPUs (default unlimited)
#
# Examples:
#   MODEL=/scratch/madono/model_cache/latest_model_200.pth \
#   RATIOS="0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 \
#           0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50" \
#   bash 20_launch.sh
#
#   HINT_TYPE=scribble GPU_IDS="0 1 2 3" RATIOS="0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00" \
#   bash 20_launch.sh

set -e
cd "$(dirname "$0")"

RATIOS="${RATIOS:?RATIOS env var required (e.g., '0.00 0.02 0.04 ...')}"
HINT_TYPE="${HINT_TYPE:-dot}"
MODEL="${MODEL:-}"
CONFIG="${CONFIG:-}"
TYPES="${TYPES:-0 1 2}"
POLL_INTERVAL=2

# ── logs symlink to /scratch (keep NFS write pressure low) ────────────────
SCRATCH_LOGS=/scratch/madono/revision_logs
mkdir -p "$SCRATCH_LOGS"
if [ -L logs ]; then
    rm -f logs
elif [ -d logs ]; then
    cp -an logs/. "$SCRATCH_LOGS/" 2>/dev/null || true
    rm -rf logs
fi
ln -s "$SCRATCH_LOGS" logs

# ── GPU slot pool (auto-detect if GPU_IDS unset) ─────────────────────────
detect_free_gpus() {
    local threshold_mb="${1:-${FREE_THRESHOLD_MB:-1000}}"
    local max_count="${2:-0}"
    local out
    out=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -v t="$threshold_mb" -F, '{
              gsub(/[ \t]+/,"",$1);
              gsub(/[ \t]+/,"",$2);
              if ($2 + 0 <= t + 0) print $1
          }')
    if [ "$max_count" -gt 0 ] 2>/dev/null; then
        out=$(echo "$out" | head -n "$max_count")
    fi
    echo "$out" | tr '\n' ' ' | sed 's/ *$//'
}

if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(detect_free_gpus "" "${NUM_GPUS_MAX:-0}")
    if [ -z "$GPU_IDS" ]; then
        echo "[warn] no free GPUs detected (memory.used <= ${FREE_THRESHOLD_MB:-1000} MB). Falling back to GPU 0." >&2
        GPU_IDS="0"
    else
        echo "[info] auto-detected free GPUs: $GPU_IDS"
    fi
fi

IFS=' ' read -ra GPU_IDS_ARR <<< "$GPU_IDS"
NUM_GPUS=${#GPU_IDS_ARR[@]}

IFS=' ' read -ra RATIOS_ARR <<< "$RATIOS"
NUM_RATIOS=${#RATIOS_ARR[@]}

echo "=========================================="
echo " BBDM_revision_revise_DATESTAMP launch"
echo "=========================================="
echo "  host       : $(hostname -s)"
echo "  num_ratios : $NUM_RATIOS"
echo "  ratios     : ${RATIOS_ARR[0]} .. ${RATIOS_ARR[-1]}"
echo "  hint_type  : $HINT_TYPE"
echo "  model      : ${MODEL:-<default from 10_run_ratio.sh>}"
echo "  config     : ${CONFIG:-<default from 10_run_ratio.sh>}"
echo "  gpu_ids    : ${GPU_IDS_ARR[*]}  ($NUM_GPUS slots)"
echo "  types      : $TYPES"
echo

# ── slot pool ────────────────────────────────────────────────────────────
declare -a SLOT_PID
for g in $(seq 0 $((NUM_GPUS - 1))); do SLOT_PID[$g]=""; done

get_free_slot() {
    while true; do
        for g in $(seq 0 $((NUM_GPUS - 1))); do
            local pid="${SLOT_PID[$g]}"
            if [ -z "$pid" ]; then echo "$g"; return 0; fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                SLOT_PID[$g]=""
                echo "$g"; return 0
            fi
        done
        sleep "$POLL_INTERVAL"
    done
}

LAUNCHED=0
for RATIO in "${RATIOS_ARR[@]}"; do
    SLOT=$(get_free_slot)
    PHYS_GPU="${GPU_IDS_ARR[$SLOT]}"
    LOG="logs/${HINT_TYPE}_${RATIO}.log"
    RATIO="$RATIO" GPU_ID="$PHYS_GPU" HINT_TYPE="$HINT_TYPE" MODEL="$MODEL" CONFIG="$CONFIG" TYPES="$TYPES" \
        BATCH_SIZE_OVERRIDE="$BATCH_SIZE_OVERRIDE" EXPECTED_COUNT="$EXPECTED_COUNT" \
        nohup bash 10_run_ratio.sh > "$LOG" 2>&1 &
    PID=$!
    SLOT_PID[$SLOT]=$PID
    LAUNCHED=$((LAUNCHED + 1))
    echo "[launch] ($LAUNCHED/$NUM_RATIOS) ratio=$RATIO hint=$HINT_TYPE slot=$SLOT gpu=$PHYS_GPU pid=$PID log=$LOG"
done

echo
echo "[launch] all $LAUNCHED dispatched — waiting for remaining tasks"
wait
echo "[launch] done: all $LAUNCHED ratios finished"
