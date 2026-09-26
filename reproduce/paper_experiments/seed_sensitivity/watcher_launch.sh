#!/usr/bin/env bash
# Launch the seed experiment as HOST_A GPUs free up:
#   GPU0 (seg_slic pid $PID0 ends)        -> DetFill 32 runs, then Diffusart 32 runs
#   GPU1 (seg_danbooregion pid $PID1 ends) -> ColDiff 32 runs
# Polls every 10 min; exits after both chains are launched.
set -u
PID0="${PID0:-3583367}"
PID1="${PID1:-3710453}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ts(){ date '+%F %T'; }
gpu_mem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1; }
launched0=0; launched1=0
echo "[$(ts)] watcher start (PID0=$PID0 PID1=$PID1)"
while [ "$launched0" = 0 ] || [ "$launched1" = 0 ]; do
  if [ "$launched0" = 0 ] && ! kill -0 "$PID0" 2>/dev/null && [ "$(gpu_mem 0)" -lt 5000 ]; then
    echo "[$(ts)] GPU0 free -> launching DetFill + Diffusart chain (+migrate)"
    nohup bash -c "GPU=0 bash '$HERE/run_seedexp_detfill.sh' && GPU=0 bash '$HERE/run_seedexp_diffusart.sh'; bash '$HERE/migrate_seedexp.sh' detfill diffusart hints" \
        > "$HERE/chain_gpu0.log" 2>&1 &
    echo "[$(ts)] chain_gpu0 pid=$!"
    launched0=1
  fi
  if [ "$launched1" = 0 ] && ! kill -0 "$PID1" 2>/dev/null && [ "$(gpu_mem 1)" -lt 5000 ]; then
    echo "[$(ts)] GPU1 free -> launching ColDiff (+migrate)"
    nohup bash -c "GPU=1 bash '$HERE/run_seedexp_coldiff.sh'; bash '$HERE/migrate_seedexp.sh' coldiff" > "$HERE/chain_gpu1.log" 2>&1 &
    echo "[$(ts)] chain_gpu1 pid=$!"
    launched1=1
  fi
  [ "$launched0" = 1 ] && [ "$launched1" = 1 ] && break
  sleep 600
done
echo "[$(ts)] watcher done: both chains launched"
