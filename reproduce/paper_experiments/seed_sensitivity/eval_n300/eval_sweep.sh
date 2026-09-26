#!/bin/bash
set -uo pipefail
export PATH=/home/USER/anaconda3/envs/BBDM/bin:$PATH
WORK=/scratch/USER/seedexp_n300_felz96
mkdir -p $WORK/metrics
log=$WORK/metrics/sweep.log
EVAL_SCRIPT=$WORK/code/eval_scripts/B_eval_dense_curve.py
DSN=tog2025_revise

ts() { date '+%F %T'; }

loop=0
while true; do
  loop=$((loop+1))
  remaining=0
  progress=0
  while IFS= read -r cell; do
    [ -z "$cell" ] && continue
    csv=$WORK/metrics/$cell/per_image.csv
    if [ -f "$csv" ]; then
      rows=$(wc -l < "$csv")
      [ "$rows" -ge 2 ] && continue
    fi
    out_dir=$WORK/detfill/$cell/$DSN/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble/2/1.0/200
    results_root=$WORK/detfill/$cell/$DSN/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble
    n=$(ls "$out_dir" 2>/dev/null | wc -l)
    if [ "$n" -lt 299 ]; then remaining=$((remaining+1)); continue; fi

    echo "[$(ts)] EVAL $cell (n=$n)" | tee -a $log
    mkdir -p $WORK/metrics/$cell
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
      python3 $EVAL_SCRIPT \
        --results_root "$results_root" \
        --sketches 2 \
        --ratios 1.0 \
        --metrics mse psnr ssim lpips openclip dino dreamsim \
        --out_dir "$WORK/metrics/$cell" \
        --gpu 0 --batch_size 8 \
        >> $log 2>&1
    rc=$?
    csv_rows=$(wc -l < "$csv" 2>/dev/null || echo 0)
    if [ "$rc" -eq 0 ] && [ "$csv_rows" -ge 2 ]; then
      echo "[$(ts)] DONE $cell rows=$csv_rows" | tee -a $log
      progress=$((progress+1))
    else
      echo "[$(ts)] FAIL $cell rc=$rc rows=$csv_rows" | tee -a $log
      rm -f "$csv"
    fi
  done < $WORK/cells_all.txt

  done_count=$(ls $WORK/metrics/*/per_image.csv 2>/dev/null | wc -l)
  echo "[$(ts)] loop=$loop done=$done_count progress=$progress remaining=$remaining" | tee -a $log
  if [ "$remaining" -eq 0 ] && [ "$progress" -eq 0 ] && [ "$done_count" -ge 62 ]; then
    echo "[$(ts)] === ALL DONE ===" | tee -a $log
    break
  fi
  sleep 600
done
