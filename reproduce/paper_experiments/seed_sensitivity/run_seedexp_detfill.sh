#!/usr/bin/env bash
# DetFill (proposed) seeded-random-hint runs: scribble / sketchkeras(TYPES=2) /
# 5 seeds x 8 alphas (alpha 0,100 are seed-invariant -> seed1 only) = 32 runs.
# Hints are pre-baked per (seed,alpha); inference runs at sample_ratio=1.0 so the
# loader's region selection is the identity.
#   GPU=0 bash run_seedexp_detfill.sh
set -uo pipefail
GPU="${GPU:-0}"
REPO=/home/madorin/gitlab/labrepo/main/BBDM_revision_revise_DATESTAMP
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG_DIR="$HERE/detfill_cfgs"; mkdir -p "$CFG_DIR"
BASE_CFG="$REPO/configs/scribble_proposed_illust_200epoch_mr.yaml"
[ -f "$BASE_CFG" ] || { echo "[ERR] base config missing: $BASE_CFG"; exit 1; }
ts(){ date '+%F %T'; }
RATIOS=(0 1 3 5 10 25 50 100)
cd "$REPO"
for r in "${RATIOS[@]}"; do
  if [ "$r" = 0 ] || [ "$r" = 100 ]; then SEEDS=(1); else SEEDS=(1 2 3 4 5); fi
  for s in "${SEEDS[@]}"; do
    tag="seed${s}_alpha_${r}"
    wrap="/scratch/madono/seedexp/detfill_roots/$tag"
    cfg="$CFG_DIR/$tag.yaml"
    # scratch_root / dataset_path をwrapperへ差し替えたconfigを生成
    sed -E "s#(scratch_root:).*#\1 '$wrap'#; s#(dataset_path:).*#\1 '$wrap'#" "$BASE_CFG" > "$cfg"
    out="/scratch/madono/seedexp/detfill/$tag"
    n=$(ls "$out/dataset_name/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble/2/1.0/200" 2>/dev/null | wc -l)
    if [ "$n" -ge 2999 ]; then echo "[$(ts)] skip $tag (done n=$n)"; continue; fi
    echo "[$(ts)] ### DetFill $tag (have $n)"
    RATIO=1.0 HINT_TYPE=scribble TYPES=2 GPU_ID=$GPU \
      RESULT_PATH="$out" CONFIG="$cfg" EXPECTED_COUNT=2999 \
      bash 10_run_ratio.sh || echo "[$(ts)] [FAIL] $tag"
  done
done
echo "[$(ts)] DETFILL SEEDEXP ALL DONE"
