#!/bin/bash
# 96ch Felz chain for HOST_B GPU 4. 12 cells, sequential.
# Resume-safe: skips n>=299. Cleans up multiprocessing zombies on exit/interrupt.
set -uo pipefail
export PATH=/home/USER/anaconda3/envs/BBDM/bin:$PATH
cd /scratch/USER/seedexp_n300_felz96/code/BBDM
GPU=4
WORK=/scratch/USER/seedexp_n300_felz96

ts() { date '+%F %T'; }

# Cleanup hook: kill any orphan multiprocessing workers if we get interrupted
cleanup() {
  echo "[$(ts)] cleanup: killing orphan multiprocessing workers"
  pkill -P $$ 2>/dev/null
  sleep 1
  pkill -9 -P $$ 2>/dev/null
}
trap cleanup EXIT INT TERM

CELLS=(
  "seed10_alpha_5"
  "seed1_alpha_100"
  "seed2_alpha_1"
  "seed2_alpha_50"
  "seed3_alpha_5"
  "seed4_alpha_3"
  "seed5_alpha_25"
  "seed6_alpha_10"
  "seed7_alpha_1"
  "seed7_alpha_50"
  "seed8_alpha_5"
  "seed9_alpha_3"
)

for tag in "${CELLS[@]}"; do
  OUT=$WORK/detfill/$tag
  CFG=$WORK/configs/$tag.yaml
  LOG=$WORK/logs/cell_${tag}.log
  N=$(ls "$OUT/tog2025_revise/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble/2/1.0/200" 2>/dev/null | wc -l)
  if [ "$N" -ge 299 ]; then
    echo "[$(ts)] gpu=$GPU skip $tag (n=$N)"
    continue
  fi
  echo "[$(ts)] gpu=$GPU ### $tag (have $N/300)"
  MODEL=$WORK/model.pth CUDA_DEVICE_ORDER=PCI_BUS_ID RATIO=1.0 HINT_TYPE=scribble TYPES=2 GPU_ID=$GPU \
    RESULT_PATH="$OUT" CONFIG="$CFG" EXPECTED_COUNT=299 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash 10_run_ratio.sh 2>&1 | tee -a "$LOG"
done
echo "[$(ts)] gpu=$GPU CHAIN DONE"
