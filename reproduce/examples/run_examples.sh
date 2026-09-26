#!/usr/bin/env bash
# =============================================================================
# run_examples.sh — reproduce every experiment of the paper on a few example illustrations, with no arguments.
#
#   bash reproduce/examples/run_examples.sh              # "quick" set: ~4 images, 4 hint ratios (about 45 min on one GPU)
#   EXAMPLES_MODE=smoke bash reproduce/examples/run_examples.sh   # 1 image, 2 ratios, scribble model only (~5 min on a GPU)
#   EXAMPLES_MODE=full  bash reproduce/examples/run_examples.sh   # everything in make_examples.py (several GPU-hours)
#
# What it does: (1) creates/reuses the conda environment of replicability/run.sh (`detfill-grsi`) and installs
# the perceptual-metric packages, (2) downloads the released checkpoints and the example bundle from the GitHub
# releases (SHA-256 verified), (3) runs reproduce/examples/make_examples.py, (4) prints where the grids and
# metric files are and how they compare with reproduce/examples/expected/.
# Optional: GRSI_GPU=<id> (default 0; -1 = CPU, slow), GRSI_ENV=<existing conda env name>, GRSI_CKPT_DIR=<dir>.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${EXAMPLES_MODE:-quick}"; GPU="${GRSI_GPU:-0}"; ENV_NAME="${GRSI_ENV:-detfill-grsi}"
CKPT_DIR="${GRSI_CKPT_DIR:-$HERE/checkpoints}"; DATA_DIR="$HERE/data"; OUT="$HERE/output"
BASE=https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/download

echo "== [1/4] environment ($ENV_NAME)"
command -v conda >/dev/null || { echo "conda not found: install Miniconda first (https://docs.conda.io)"; exit 1; }
set +u; eval "$(conda shell.bash hook)"; set -u
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -n "$ENV_NAME" -f "$ROOT/replicability/environment.yml"
fi
set +u; conda activate "$ENV_NAME"; set -u
python -m pip install -q --no-deps -e "$ROOT"
python -m pip install -q -r "$ROOT/reproduce/requirements-metrics.txt"
python - <<'PY'
import torch, hintauc, lpips, open_clip, dreamsim, transformers, torchmetrics
print("   torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| torchmetrics", torchmetrics.__version__)
PY

echo "== [2/4] checkpoints and example data"
mkdir -p "$CKPT_DIR" "$DATA_DIR"
fetch() { # tag file sha256
  local f="$CKPT_DIR/$2"; [ "$2" = "examples_data.tar.gz" ] && f="$DATA_DIR/$2"
  if [ -f "$f" ] && echo "$3  $f" | sha256sum -c --status - 2>/dev/null; then echo "   ok      $2"; return; fi
  echo "   fetch   $2"; curl -fL --retry 3 -o "$f" "$BASE/$1/$2"
  echo "$3  $f" | sha256sum -c --status - || { echo "SHA-256 mismatch for $2"; exit 1; }
}
fetch v1.0        detfill_scribble_illust_200ep.pth               fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4
fetch v1.0        detfill_dot_illust_200ep.pth                    fd872563e17cb1ac09957ed5ef993fac4dbeb2710f43a7ee29203444928d6fd4
fetch v1.1        detfill_scribble_illust_danbooregion_200ep.pth  1770cd61bb496060b4ff9bd8c2ec2e72b549fa99187a0429912100628cad9a78
fetch v1.1        detfill_scribble_illust_slic_200ep.pth          eb25667c1955584f46f34abf98db10eef6d702e4c1078fe3cff388f646cf4217
fetch v1.3        examples_data.tar.gz                            3d7f699da9f8de5a5def35a997e7bb42461abd56babd3ce6c30adfbee9f71c87
if [ "$MODE" = full ]; then
  while read -r tag file sha; do fetch "$tag" "$file" "$sha"; done <<'LIST'
legacy-2024 detfill_scribble_illust_32ch_200ep.pth 4eef804755b9adc4fc0b5234501666138a4a6ecddfbdb28245fd7fc32443ecda
legacy-2024 detfill_scribble_illust_64ch_200ep.pth 8eeee341331ed949cc216049c9fecebc70900d54fb709eee3b97a2b1fc34845b
legacy-2024 legacy2024_detfill_scribble_illust_200ep.pth 7d74ebe7d6c8862286fe9944b7cf9d2448a6e473880c383b13af8a36e6ce0002
legacy-2024 legacy2024_detfill_dot_illust_200ep.pth b4779946f24b5f52cdf640418c73bef67925c845e50c8f36610aeac7d50f212b
LIST
fi
[ -d "$DATA_DIR/examples_data" ] || tar xzf "$DATA_DIR/examples_data.tar.gz" -C "$DATA_DIR"

echo "== [3/4] experiments (mode=$MODE, GPU=$GPU)"
python "$HERE/make_examples.py" --data "$DATA_DIR/examples_data" --ckpt_dir "$CKPT_DIR" --gpu "$GPU" --mode "$MODE" --out "$OUT"

echo "== [4/4] done"
echo "   grids   : $OUT/grids/*.png        (labelled image grids, one per experiment)"
echo "   metrics : $OUT/metrics/*.json     (per-image and mean metrics; Hint-AUC where the full ratio grid was run)"
echo "   compare : reproduce/examples/expected/  holds the authors' grids/metrics and, for the Table II protocol,"
echo "             the paper's archived outputs of the same images (psnr_vs_paper_outputs_dB in E1 metrics)."
python "$HERE/compare_with_expected.py" --out "$OUT" || true
