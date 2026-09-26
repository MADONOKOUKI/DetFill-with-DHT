#!/bin/bash
# Generic Table III random (id-order) inference launcher with GPU slot queue.
# Env: CODE_DIR CONFIG MODEL RESULT_PATH LOGDIR GPUS JOBS_FILE SHARD_N(optional)
# JOBS_FILE lines: "<ratio> <sketch_type> [<shard_off>]"
set -e
: "${CODE_DIR:?}"; : "${CONFIG:?}"; : "${MODEL:?}"; : "${RESULT_PATH:?}"; : "${LOGDIR:?}"; : "${GPUS:?}"; : "${JOBS_FILE:?}"
export PATH=/home/USER/anaconda3/envs/BBDM/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGDIR"
cd "$CODE_DIR"

IFS=' ' read -ra GPU_ARR <<< "$GPUS"
NUM=${#GPU_ARR[@]}
declare -a SLOT_PID
for i in $(seq 0 $((NUM-1))); do SLOT_PID[$i]=""; done

get_slot() {
  while true; do
    for i in $(seq 0 $((NUM-1))); do
      local pid="${SLOT_PID[$i]}"
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        [ -n "$pid" ] && wait "$pid" 2>/dev/null
        echo "$i"; return 0
      fi
    done
    sleep 20
  done
}

TOTAL=$(grep -c . "$JOBS_FILE")
N=0
while read -r RATIO SK SHOFF; do
  [ -z "$RATIO" ] && continue
  SLOT=$(get_slot)
  GPU="${GPU_ARR[$SLOT]}"
  if [ -n "$SHOFF" ] && [ -n "$SHARD_N" ]; then
    LOG="$LOGDIR/r${RATIO}_sk${SK}_s${SHOFF}.log"
    SHARD_ENV="SHARD_N=$SHARD_N SHARD_OFF=$SHOFF"
  else
    LOG="$LOGDIR/r${RATIO}_sk${SK}.log"
    SHARD_ENV=""
  fi
  CUDA_VISIBLE_DEVICES="$GPU" FIG9_RANDOM=1 env $SHARD_ENV nohup python3 main.py \
    --config "$CONFIG" --result_path "$RESULT_PATH" --resume_model "$MODEL" \
    --sample_to_eval --save_top --gpu_ids 0 \
    --sample_ratio "$RATIO" --sketch_type "$SK" > "$LOG" 2>&1 &
  SLOT_PID[$SLOT]=$!
  N=$((N+1))
  echo "[$(date -Iseconds)] ($N/$TOTAL) ratio=$RATIO sk=$SK shard=$SHOFF gpu=$GPU pid=${SLOT_PID[$SLOT]}"
done < "$JOBS_FILE"
echo "all $N queued/launched — waiting"
wait
echo "TABLEIII_HOST_DONE"
