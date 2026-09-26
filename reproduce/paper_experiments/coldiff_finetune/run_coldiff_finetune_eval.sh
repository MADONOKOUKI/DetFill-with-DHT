#!/usr/bin/env bash
# =============================================================================
# ColorizeDiffusion v1/v2 : fine-tune (from official released weights) + inference
# for the Hint-AUC protocol (Table II = deterministic, Table III = random sampling).
#
# Order (single GPU, sequential, HOST_A):
#   (1) v2-dot  (2) v2-scr  (3) v1-dot  (4) v1-scr
# For each model:  TRAIN -> INFER(deterministic, 8 ratios x N sketch) ->
#                  INFER(random sampling = Table III, 8 ratios x N sketch)
#
# Rationale (from the v1/v2 papers): v1 was fine-tuned from Waifu Diffusion (5-7 ep),
# v2 from Stable Diffusion (3 stages), both at lr 1e-5 over MILLIONS of images.
# Here we ADAPT the released checkpoints to the hint setting on ~20k images at a
# low lr (config base_learning_rate = 1e-6), so few epochs are needed.
#
# ---- IMPORTANT, READ BEFORE RUNNING (set these consciously): -----------------
#  * EPOCHS: pretrained-start adaptation converges fast. The old exec.sh used 200,
#    which is almost certainly overkill. Start small (e.g. 20-40) and inspect the
#    per-epoch checkpoints. Inference below uses the "final" checkpoint, written
#    only AFTER training completes EPOCHS.
#  * EFFECTIVE LR = base_learning_rate(config) x BATCH x ACCUM x num_gpus
#    With base=1e-6, BATCH=5, ACCUM=1, 1 GPU  ->  effective lr = 5e-6.
#    If you truly want 1e-6 effective, set the config base to 2e-7 (or BATCH=1).
#  * v2.yaml (scribble) has `base_learning_rate` COMMENTED OUT -> set it to 1.0e-6
#    so v2-scr matches v2-dot. (v2_dot.yaml / mult.yaml / mult_dot.yaml are 1e-6.)
#  * DATA: defaults point at the COMPLETE copy on NFS. HOST_A can read NFS directly.
#    For faster local I/O, rsync to HOST_A:/scratch first and switch *_ROOT below.
#  * accel_config.yaml is MULTI_GPU(4). We bypass it and force a single process.
#  * sample_ratio in v2_dot.yaml is 0.2 (fixed). For paper-style random-ratio
#    training set it to null in the training config (this script does NOT edit it).
#
# Preview without running:   DRY_RUN=1 bash run_coldiff_finetune_eval.sh
# Quick smoke test:          SKETCH_SOURCES="s0"  RATIOS="0 10 100"  EPOCHS=2 ...
# =============================================================================
set -euo pipefail

#### ========================= USER CONFIG ========================= ####
GPU="${GPU:-0}"                     # HOST_A free GPU index
PORT="${PORT:-29555}"              # accelerate main_process_port (avoid clashes)
EPOCHS="${EPOCHS:-30}"            # adaptation epochs (TUNE; see notes above)
BATCH="${BATCH:-5}"               # train micro-batch per GPU
ACCUM="${ACCUM:-1}"               # gradient accumulation
NT_TRAIN="${NT_TRAIN:-4}"         # train dataloader workers
NT_INFER="${NT_INFER:-10}"        # inference dataloader workers
GS="${GS:-5}"                     # guidance scale (matches existing exec scripts)
DO_RANDOM="${DO_RANDOM:-1}"       # 1 = also run Table III (random sampling) inference
DRY_RUN="${DRY_RUN:-0}"           # 1 = print commands, do not execute
LR="${LR:-1e-6}"                  # learning rate passed via --learning_rate (config base_learning_rate is ignored unless --dynamic_lr)

# 3 sketch sources are required to reproduce the paper's mean +/- SD.
# Override for a quick test, e.g.  SKETCH_SOURCES="s0"
SKETCH_SOURCES=(${SKETCH_SOURCES:-s0 s1 s2})
# main.pdf hint-ratio grid (percent): {0,0.01,0.03,0.05,0.10,0.25,0.50,1.00}
RATIOS=(${RATIOS:-0 1 3 5 10 25 50 100})

MAIN=/home/USER/gitlab/labrepo/main
# COMPLETE data lives on NFS (accessible from every host incl. HOST_A):
TRAIN_ROOT="${TRAIN_ROOT:-/home/USER/datasets/revision/colorizeDiffusion/illust/images_train}"
TEST_ROOT="${TEST_ROOT:-/home/USER/datasets/revision/colorizeDiffusion/illust/images_test}"

# checkpoint save roots (train.py writes <save>/<name>/final/model.safetensors)
CKPT_V2="${CKPT_V2:-/scratch/USER/colorizeDiffusion_v2_checkpoints}"
CKPT_V1="${CKPT_V1:-/scratch/USER/colorizeDiffusion_v1/checkpoints}"
# inference output roots (per existing convention)
OUT_V2="${OUT_V2:-/scratch/USER/colorizeDiffusion_v2/checkpoints}"
OUT_V1="${OUT_V1:-/scratch/USER/colorizeDiffusion_v1/checkpoints}"

LOGDIR="${LOGDIR:-$MAIN/coldiff_runs_logs}"
#### =============================================================== ####

# ---- activate the ColDiff conda env (base env has incompatible huggingface-hub 1.7.1) ----
# environment.yml of colorizeDiffusion_v2 is `name: hf` (hf-hub 0.29.3, transformers 4.49.0).
source /home/USER/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-hf}" || { echo "[ERROR] conda activate ${CONDA_ENV:-hf} failed"; exit 1; }

mkdir -p "$LOGDIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say_env() { echo "[env] python=$(command -v python)  hf-hub=$(python -c 'import huggingface_hub as h;print(h.__version__)' 2>/dev/null||echo '??')"; }
say() { echo "[$(ts)] $*"; }

# map (hint_type, sketch source) -> inference config subdirectory
sketch_cfg_dir() {  # $1=dot|scr  $2=s0|s1|s2
  local base; [ "$1" = dot ] && base=hint_ratios_dot || base=hint_ratios_scribble
  case "$2" in
    s0) echo "$base" ;;
    s1) echo "${base}_sketch1" ;;
    s2) echo "${base}_sketch2" ;;
    *)  echo "BAD_SKETCH_$2" ;;
  esac
}

# ---------------------------------------------------------------- pre-flight
preflight() {
  say "===== PRE-FLIGHT ====="
  local ok=1
  # data
  for d in color sketch_p sketch_s sketch_x hint region; do
    local n; n=$(ls "$TRAIN_ROOT/$d" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "  [MISSING] TRAIN_ROOT/$d is empty ($TRAIN_ROOT/$d)"; ok=0; }
  done
  for d in color sketch_p sketch_s sketch_x hint region; do
    local n; n=$(ls "$TEST_ROOT/$d" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "  [MISSING] TEST_ROOT/$d is empty ($TEST_ROOT/$d)"; ok=0; }
  done
  # weights + train cfgs + python entry points
  local checks=(
    "$MAIN/colorizeDiffusion_v2/weights/v2-full.safetensors"
    "$MAIN/colorizeDiffusion/weights/mult-eps-newft.safetensors"
    "$MAIN/colorizeDiffusion_v2/configs/training/sd2.1/v2_dot.yaml"
    "$MAIN/colorizeDiffusion_v2/configs/training/sd2.1/v2.yaml"
    "$MAIN/colorizeDiffusion/configs/training/sd2.1/mult_dot.yaml"
    "$MAIN/colorizeDiffusion/configs/training/sd2.1/mult.yaml"
    "$MAIN/colorizeDiffusion_v2/inference_official_fixed.py"
    "$MAIN/colorizeDiffusion_v2/inference_official_fixed_random.py"
    "$MAIN/colorizeDiffusion/inference_official_fixed.py"
    "$MAIN/colorizeDiffusion/inference_official_fixed_random.py"
  )
  for f in "${checks[@]}"; do [ -e "$f" ] || { echo "  [MISSING] $f"; ok=0; }; done
  # lr sanity
  echo "  --- base_learning_rate per training config ---"
  for f in colorizeDiffusion_v2/configs/training/sd2.1/v2_dot.yaml \
           colorizeDiffusion_v2/configs/training/sd2.1/v2.yaml \
           colorizeDiffusion/configs/training/sd2.1/mult_dot.yaml \
           colorizeDiffusion/configs/training/sd2.1/mult.yaml; do
    echo "    $f : $(grep -m1 'base_learning_rate' "$MAIN/$f" 2>/dev/null | tr -s ' ' || echo '(none)')"
  done
  echo "  NOTE: training uses --learning_rate=$LR (flat; config base_learning_rate applies only with --dynamic_lr, which we do NOT pass)"
  echo "  GPU=$GPU EPOCHS=$EPOCHS BATCH=$BATCH ACCUM=$ACCUM DO_RANDOM=$DO_RANDOM"
  echo "  SKETCH_SOURCES=(${SKETCH_SOURCES[*]})  RATIOS=(${RATIOS[*]})"
  echo "  TRAIN_ROOT=$TRAIN_ROOT"
  echo "  TEST_ROOT=$TEST_ROOT"
  say_env
  [ "$ok" = 1 ] || { echo "  PRE-FLIGHT FAILED — fix the [MISSING] items above."; exit 1; }
  say "pre-flight OK"
}

# ---------------------------------------------------------------- train
train_one() {  # $1 repo  $2 name  $3 train_cfg  $4 pretrained(rel to repo)  $5 save_root
  local repo=$1 name=$2 cfg=$3 pt=$4 save=$5
  local final="$save/$name/final/model.safetensors"
  if [ -f "$final" ]; then say "[skip train] $name (final exists: $final)"; return 0; fi
  say "[train] $name  cfg=$cfg  pt=$pt"
  cd "$MAIN/$repo"
  # plain python, NOT `accelerate launch`: the user's ~/.cache accelerate default
  # config is MULTI_GPU / gpu_ids 0..4 and hijacks GPU 0 (ignoring CUDA_VISIBLE_DEVICES),
  # which caused the OOM on the busy GPU 0. Plain python honours CUDA_VISIBLE_DEVICES.
  local cmd=(env CUDA_VISIBLE_DEVICES=$GPU python train.py
        --name "$name" --dataroot "$TRAIN_ROOT/"
        --batch_size "$BATCH" --num_threads "$NT_TRAIN" --accumulate_batches "$ACCUM"
        --learning_rate "$LR"
        --save_path "$save" --epoch "$EPOCHS"
        -cfg "$cfg" -pt "$pt")
  echo "    + ${cmd[*]}"
  [ "$DRY_RUN" = 1 ] && return 0
  "${cmd[@]}" 2>&1 | tee "$LOGDIR/train_${name}.log"
}

# ---------------------------------------------------------------- inference
infer_block() {  # $1 repo $2 infer_py $3 ckpt $4 cfg_prefix $5 hint_type $6 tag(det|rand) $7 out_root $8 model_name
  local repo=$1 py=$2 ckpt=$3 prefix=$4 ht=$5 tag=$6 outroot=$7 mname=$8
  cd "$MAIN/$repo"
  local sk r cdir cfg name
  for sk in "${SKETCH_SOURCES[@]}"; do
    cdir=$(sketch_cfg_dir "$ht" "$sk")
    for r in "${RATIOS[@]}"; do
      cfg="configs/inference/${cdir}/${prefix}_${r}.yaml"
      name="${outroot}/${mname}/${tag}_${sk}_hint_ratio_${r}"
      if [ ! -f "$cfg" ]; then echo "    [MISSING CFG] $repo/$cfg — skip"; continue; fi
      if [ -d "$name" ] && [ -n "$(ls -A "$name" 2>/dev/null)" ]; then
        echo "    [skip infer] $name (non-empty)"; continue; fi
      say "[infer:$tag] $mname $sk ratio=$r"
      local cmd=(env CUDA_VISIBLE_DEVICES=$GPU python "$py"
            --name "$name" --dataroot "$TEST_ROOT/"
            --batch_size "$BATCH" --num_threads "$NT_INFER"
            -cfg "$cfg" -pt "$ckpt" -gs "$GS")
      echo "    + ${cmd[*]}"
      [ "$DRY_RUN" = 1 ] && continue
      "${cmd[@]}" 2>&1 | tee -a "$LOGDIR/infer_${mname}_${tag}_${sk}.log"
    done
  done
}

# ---------------------------------------------------------------- one full model
run_model() {  # repo name train_cfg pretrained save_root hint_type infer_prefix out_root
  local repo=$1 name=$2 tcfg=$3 pt=$4 save=$5 ht=$6 prefix=$7 outroot=$8
  if [ -n "${ONLY:-}" ] && [ "$name" != "${ONLY}" ]; then say "[skip model] $name (ONLY=${ONLY})"; return 0; fi
  say "############### MODEL: $name ($ht) [$repo] ###############"
  train_one "$repo" "$name" "$tcfg" "$pt" "$save"
  local ckpt="$save/$name/final/model.safetensors"
  if [ "$DRY_RUN" != 1 ] && [ ! -f "$ckpt" ]; then
    echo "  [ERROR] checkpoint not produced: $ckpt"; exit 1; fi
  # Table II : deterministic region-based hints
  infer_block "$repo" inference_official_fixed.py        "$ckpt" "$prefix" "$ht" det  "$outroot" "$name"
  # Table III: random sampling
  if [ "$DO_RANDOM" = 1 ]; then
    infer_block "$repo" inference_official_fixed_random.py "$ckpt" "$prefix" "$ht" rand "$outroot" "$name"
  fi
  say "############### DONE: $name ###############"
}

# ================================ RUN ================================
preflight

# (1) ColorizeDiffusion v2 — dot
run_model colorizeDiffusion_v2 illust_v2_dot \
  configs/training/sd2.1/v2_dot.yaml  weights/v2-full.safetensors \
  "$CKPT_V2" dot v2_inference "$OUT_V2"

# (2) ColorizeDiffusion v2 — scribble
run_model colorizeDiffusion_v2 illust_v2_scr \
  configs/training/sd2.1/v2.yaml      weights/v2-full.safetensors \
  "$CKPT_V2" scr v2_inference "$OUT_V2"

# (3) ColorizeDiffusion v1 — dot
run_model colorizeDiffusion    illust_v1_dot \
  configs/training/sd2.1/mult_dot.yaml weights/mult-eps-newft.safetensors \
  "$CKPT_V1" dot mult_inference "$OUT_V1"

# (4) ColorizeDiffusion v1 — scribble
run_model colorizeDiffusion    illust_v1_scr \
  configs/training/sd2.1/mult.yaml     weights/mult-eps-newft.safetensors \
  "$CKPT_V1" scr mult_inference "$OUT_V1"

say "ALL DONE. logs in $LOGDIR"

# =============================================================================
# STAGE 3 (OPTIONAL) — Hint-AUC from the inference outputs.
# The inference above writes per-(model,ratio,sketch) output folders. Converting
# those to the per-ratio metric CSVs and then to a Hint-AUC summary is a separate
# step that uses Evaluation_paper/calc_hint_auc_manual.py, e.g.:
#
#   cd $MAIN/Evaluation_paper
#   python calc_hint_auc_manual.py \
#     --name illust_v2_dot_s0_det \
#     --alpha_csv 0.00=<...ratio_0.csv> --alpha_csv 0.01=<...ratio_1.csv> \
#     --alpha_csv 0.03=<...ratio_3.csv> --alpha_csv 0.05=<...ratio_5.csv> \
#     --alpha_csv 0.10=<...ratio_10.csv> --alpha_csv 0.25=<...ratio_25.csv> \
#     --alpha_csv 0.50=<...ratio_50.csv> --alpha_csv 1.00=<...ratio_100.csv> \
#     --out_csv .../summary.csv --check_hint_ratio --hint_ratio_scale percent
#
# then average the 3 sketch sources (mean +/- sample SD, n=3, divisor n-1) with
# Evaluation_paper/avg_hint_auc_summary.py. See calc_hint_auc_manual_v2.sh for the
# exact per-ratio CSV path convention. (Left out of this script because it depends
# on the metric->CSV step, which is run separately.)
# =============================================================================
