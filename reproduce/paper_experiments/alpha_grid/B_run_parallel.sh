#!/bin/bash
# ============================================================================
# R2-1 — Plan C: 3-GPU parallel dense-curve eval on naga1
# ============================================================================
# - sketch 0 → GPU 2 (A6000, batch=32)
# - sketch 1 → GPU 0 (shared, batch=16, lower VRAM)
# - sketch 2 → GPU 3 (A6000, batch=32)
#
# All 3 processes APPEND to a single per_image.csv. The header is pre-created
# by this launcher so we avoid race conditions on header writes. Each process
# runs with --no_summary; the launcher's "watcher" subprocess builds
# per_ratio_summary.csv once all 3 are finished.
#
# Usage:
#   bash B_run_parallel.sh             # launch + spawn watcher, exit
#   FOREGROUND=1 bash B_run_parallel.sh  # block until everything done
# ============================================================================

set -e
cd "$(dirname "$0")"

PY=${PY:-/home/madorin/anaconda3/envs/BBDM/bin/python3}
OUT_DIR="${OUT_DIR:-$(pwd)/../output/B_dense_curve}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-1_alpha_grid_sensitivity/inference_results/scribble}"

mkdir -p "$OUT_DIR"

PER_IMG="$OUT_DIR/per_image.csv"
HEADER='sketch,sketch_name,ratio,image_name,psnr,ssim,lpips,dreamsim'

# pre-create per_image.csv with header (race-free for parallel writers).
# Python's csv.writer writes CRLF per RFC 4180, so we match that for byte-for-byte
# compatibility with rows the python processes append.
if [ ! -f "$PER_IMG" ]; then
    printf '%s\r\n' "$HEADER" > "$PER_IMG"
    echo "[init] created $PER_IMG with header"
else
    # tolerate either LF or CRLF in existing header by stripping \r before compare
    existing_header=$(head -1 "$PER_IMG" | tr -d '\r')
    if [ "$existing_header" != "$HEADER" ]; then
        echo "[error] $PER_IMG exists but header mismatch:" >&2
        echo "  expected: $HEADER" >&2
        echo "  got     : $existing_header" >&2
        exit 1
    fi
    rows=$(($(wc -l < "$PER_IMG") - 1))
    echo "[init] $PER_IMG already has $rows data rows — resume will skip them"
fi

CMD_BASE=("$PY" "$(pwd)/B_eval_dense_curve.py"
    --results_root "$RESULTS_ROOT"
    --metrics psnr ssim lpips dreamsim
    --workers 4
    --out_dir "$OUT_DIR"
    --no_summary)

LOG0="$OUT_DIR/log_sketch0_gpu2.log"
LOG1="$OUT_DIR/log_sketch1_gpu0.log"
LOG2="$OUT_DIR/log_sketch2_gpu3.log"

# Launch all 3 in background using setsid + nohup + </dev/null so they fully
# detach from this shell's session/process-group (otherwise some sandboxed
# launchers (some tool launchers kill descendants when the launcher
# script exits).
setsid nohup "${CMD_BASE[@]}" --sketches 0 --gpu 2 --batch_size 32 \
    > "$LOG0" 2>&1 < /dev/null &
PID0=$!

setsid nohup "${CMD_BASE[@]}" --sketches 1 --gpu 0 --batch_size 16 \
    > "$LOG1" 2>&1 < /dev/null &
PID1=$!

setsid nohup "${CMD_BASE[@]}" --sketches 2 --gpu 3 --batch_size 32 \
    > "$LOG2" 2>&1 < /dev/null &
PID2=$!

echo "=========================================="
echo " [naga1] R2-1 dense-curve PARALLEL launched"
echo "=========================================="
printf "  sketch 0 (GPU 2, batch 32): pid=%-6s log=%s\n" "$PID0" "$LOG0"
printf "  sketch 1 (GPU 0, batch 16): pid=%-6s log=%s\n" "$PID1" "$LOG1"
printf "  sketch 2 (GPU 3, batch 32): pid=%-6s log=%s\n" "$PID2" "$LOG2"
echo
echo "Monitor:"
echo "  tail -f $OUT_DIR/log_sketch*.log"
echo "  ps -p $PID0 $PID1 $PID2 -o pid,etime,cmd 2>/dev/null"
echo

# Watcher: wait for 3 processes, then build summary once
WATCHER_LOG="$OUT_DIR/watcher.log"
SUMMARY_CMD=("$PY" "$(pwd)/B_build_summary.py"
    --per_image "$PER_IMG"
    --per_ratio "$OUT_DIR/per_ratio_summary.csv"
    --metrics psnr ssim lpips dreamsim)

if [ "$FOREGROUND" = "1" ]; then
    echo "[FOREGROUND] waiting for all 3 evals to finish..."
    wait $PID0 $PID1 $PID2 || true
    echo "[FOREGROUND] all done — building summary"
    "${SUMMARY_CMD[@]}" 2>&1 | tee -a "$WATCHER_LOG"
else
    # spawn detached watcher: poll for the 3 PIDs to die, then build summary.
    # We must POLL (not `wait`) because $PID* are not children of the watcher
    # subshell; setsid above also reparents them to init. setsid here too so
    # the watcher itself survives launcher exit.
    setsid nohup bash -c "
        while kill -0 $PID0 2>/dev/null || kill -0 $PID1 2>/dev/null || kill -0 $PID2 2>/dev/null; do
            sleep 30
        done
        echo \"\$(date): all 3 evals finished — building summary\" >> '$WATCHER_LOG'
        '${SUMMARY_CMD[0]}' '${SUMMARY_CMD[1]}' --per_image '$PER_IMG' --per_ratio '$OUT_DIR/per_ratio_summary.csv' --metrics psnr ssim lpips dreamsim >> '$WATCHER_LOG' 2>&1
        echo \"\$(date): summary built\" >> '$WATCHER_LOG'
    " > /dev/null 2>&1 < /dev/null &
    WATCHER=$!
    echo "[bg watcher] pid=$WATCHER  log=$WATCHER_LOG"
    echo
    echo "Once all 3 sketches finish, watcher writes:"
    echo "  $OUT_DIR/per_ratio_summary.csv  (the headline 303-row CSV)"
    echo
    echo "Launcher returns immediately. tmux/nohup not required."
fi
