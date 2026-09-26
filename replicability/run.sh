#!/usr/bin/env bash
# Graphics Replicability Stamp script — reproduces Fig. 9 of
#   Madono, Mingcheng, Simo-Serra, "Hint-AUC: Deterministic Region-based Hint Generation for
#   Line Art Colorization Evaluation", IEEE TVCG (2026).
# Usage:  bash replicability/run.sh          (no arguments; see replicability/README.md)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
ENV_NAME=${GRSI_ENV:-detfill-grsi}

# 1) Python environment (conda; replicability/environment.yml = the paper's runtime pins + hint-generation deps)
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do [ -x "$c/bin/conda" ] && export PATH="$c/bin:$PATH" && break; done
fi
if ! command -v conda >/dev/null 2>&1; then
  # vanilla machine: install Miniconda into $HOME/miniconda3 (no root needed). Set GRSI_NO_AUTO_CONDA=1 to disable.
  [ "${GRSI_NO_AUTO_CONDA:-0}" = 1 ] && { echo "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) and re-run."; exit 1; }
  echo "[0/4] conda not found: installing Miniconda into $HOME/miniconda3 ..."
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  MC=Miniconda3-latest-Linux-x86_64.sh ;;
    Linux-aarch64) MC=Miniconda3-latest-Linux-aarch64.sh ;;
    Darwin-arm64)  MC=Miniconda3-latest-MacOSX-arm64.sh ;;
    Darwin-x86_64) MC=Miniconda3-latest-MacOSX-x86_64.sh ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m); install Miniconda manually and re-run"; exit 1 ;;
  esac
  curl -L --fail --retry 3 -o "/tmp/$MC" "https://repo.anaconda.com/miniconda/$MC"
  bash "/tmp/$MC" -b -p "$HOME/miniconda3" && rm -f "/tmp/$MC"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
command -v conda >/dev/null 2>&1 || { echo "conda is still not available; aborting."; exit 1; }
set +u; eval "$(conda shell.bash hook)"; set -u          # conda activation scripts reference unset variables
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/4] creating conda env '$ENV_NAME' from replicability/environment.yml (one-time, 10-20 minutes) ..."
  conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
else
  echo "[1/4] using existing conda env '$ENV_NAME'"
fi
set +u; conda activate "$ENV_NAME"; set -u
python -m pip install -q --no-deps -e "$ROOT"     # the hintauc package (its dependencies are in the env)

# 2) checkpoints (1.5 GB, verified by SHA-256)
echo "[2/4] checkpoints"; bash "$HERE/fetch_checkpoints.sh"

# 3) reproduce Fig. 9
echo "[3/4] hint generation + DetFill inference + figure assembly"
python "$HERE/make_fig9.py"

# 4) where to look
echo "[4/4] output: $HERE/output/fig9.png   (reference: $HERE/expected/fig9_paper.png)"
