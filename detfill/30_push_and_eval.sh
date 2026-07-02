#!/bin/bash
# ============================================================================
# 30_push_and_eval.sh — push completed ratios from /scratch to NFS and run eval
# ============================================================================
# Usage:
#   RATIOS="0.03"                         bash 30_push_and_eval.sh    # eval only the listed ratios + already-on-NFS
#   RATIOS="0.03" DELETE_SRC=1            bash 30_push_and_eval.sh    # also remove /scratch source after rsync
#   RATIOS="0.03" SKIP_EVAL=1             bash 30_push_and_eval.sh    # rsync + symlinks only, no eval
#   GPU=0                                  bash 30_push_and_eval.sh    # which GPU for eval (default 0)
#
# What it does (for each listed ratio):
#   1. rsync /scratch/.../<type>/<ratio>/200/ → <NFS_DEST>/<type>/<ratio>/200/  (3 types)
#   2. verify file counts match (must be 3000)
#   3. create ground_truth symlink → ../../_ground_truth  (3 types)
#   4. optionally rm -rf /scratch source
#   5. run B_run.sh (eval over all auto-discovered ratios on NFS)
#   6. run B_auc_grid_sensitivity.py (Hint-AUC under alternative grids)
# ============================================================================

set -e
cd "$(dirname "$0")"

RATIOS="${RATIOS:?RATIOS env var required (space-separated, e.g., '0.03')}"
GPU="${GPU:-0}"
DELETE_SRC="${DELETE_SRC:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

SRC_BASE=/scratch/madono/tvcg26_major_revision/results/dataset_name/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble
DEST_BASE=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-1_alpha_grid_sensitivity/inference_results/scribble
EVAL_DIR=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-1_alpha_grid_sensitivity/scripts
OUT_DIR=/home/madorin/gitlab/tog2024/main/tvcg2026_materials/rebuttal/R2/R2-1_alpha_grid_sensitivity/output/B_dense_curve

PY=/home/madorin/anaconda3/envs/BBDM/bin/python3

echo "=== config ==="
echo "  ratios       : $RATIOS"
echo "  gpu          : $GPU"
echo "  delete_src   : $DELETE_SRC"
echo "  skip_eval    : $SKIP_EVAL"
echo "  src_base     : $SRC_BASE"
echo "  dest_base    : $DEST_BASE"
echo "  out_dir      : $OUT_DIR"
echo

# ── 1. preflight: every (type, ratio) source must have 3000 files in 200/ ────
echo "=== preflight: source completeness ==="
abort=0
for r in $RATIOS; do
    for t in 0 1 2; do
        src="$SRC_BASE/$t/$r/200"
        if [ ! -d "$src" ]; then
            echo "MISSING: $src"; abort=1; continue
        fi
        n=$(ls "$src" 2>/dev/null | wc -l)
        if [ "$n" -lt 3000 ]; then
            echo "INCOMPLETE: type=$t ratio=$r → $n/3000 in /scratch (not yet done)"
            abort=1
        else
            echo "OK: type=$t ratio=$r → $n in /scratch"
        fi
    done
done
[ "$abort" = "1" ] && { echo "[abort] some sources incomplete; rerun when inference finishes"; exit 1; }

# ── 2. rsync + verify + symlink ──────────────────────────────────────────────
echo
echo "=== rsync + symlink ==="
for r in $RATIOS; do
    for t in 0 1 2; do
        src="$SRC_BASE/$t/$r/200/"
        dst="$DEST_BASE/$t/$r/200/"
        mkdir -p "$dst"
        rsync -a "$src" "$dst"
        n_src=$(ls "$SRC_BASE/$t/$r/200" | wc -l)
        n_dst=$(ls "$DEST_BASE/$t/$r/200" | wc -l)
        [ "$n_src" = "$n_dst" ] || { echo "MISMATCH: $t/$r src=$n_src dst=$n_dst"; exit 1; }
        ln -snf ../../_ground_truth "$DEST_BASE/$t/$r/ground_truth"
        echo "type=$t ratio=$r : pushed ($n_dst) + gt symlinked"
    done
done

# ── 3. delete source if requested ───────────────────────────────────────────
if [ "$DELETE_SRC" = "1" ]; then
    echo
    echo "=== deleting /scratch source ==="
    for r in $RATIOS; do
        for t in 0 1 2; do
            rm -rf "$SRC_BASE/$t/$r"
            echo "rm: $SRC_BASE/$t/$r"
        done
    done
fi

# ── 4. eval ─────────────────────────────────────────────────────────────────
if [ "$SKIP_EVAL" = "1" ]; then
    echo "[skip] eval (SKIP_EVAL=1)"
    exit 0
fi

echo
echo "=== B_eval_dense_curve (auto-discover all NFS ratios) ==="
cd "$EVAL_DIR"
GPU="$GPU" RESULTS_ROOT="$DEST_BASE" OUT_DIR="$OUT_DIR" bash B_run.sh

# ── 5. AUC under alternative grids ──────────────────────────────────────────
echo
echo "=== B_auc_grid_sensitivity ==="
$PY B_auc_grid_sensitivity.py \
    --summary "$OUT_DIR/per_ratio_summary.csv" \
    --out_dir "$OUT_DIR"

echo
echo "=== done ==="
echo "  per_image.csv          : $OUT_DIR/per_image.csv"
echo "  per_ratio_summary.csv  : $OUT_DIR/per_ratio_summary.csv"
echo "  auc_grid_sensitivity__*: $OUT_DIR/auc_grid_sensitivity__*.csv|tex|md"
