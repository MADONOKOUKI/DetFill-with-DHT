#!/usr/bin/env bash
# Sourced by replicability/run.sh and reproduce/examples/run_examples.sh (expects $ROOT and $ENV_NAME).
# Finds conda (or installs Miniconda into $HOME/miniconda3 on a machine without it), creates the environment from
# replicability/environment.yml when missing, activates it and checks that the imports work. Linux only.
command -v curl >/dev/null 2>&1 || { echo "curl is required (Ubuntu: sudo apt-get install -y curl)"; exit 1; }
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do [ -x "$c/bin/conda" ] && export PATH="$c/bin:$PATH" && break; done
fi
if ! command -v conda >/dev/null 2>&1; then
  [ "${GRSI_NO_AUTO_CONDA:-0}" = 1 ] && { echo "conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) and re-run."; exit 1; }
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  MC=Miniconda3-latest-Linux-x86_64.sh ;;
    Linux-aarch64) MC=Miniconda3-latest-Linux-aarch64.sh ;;
    *) echo "automatic Miniconda install is provided for Linux only; install conda manually and re-run"; exit 1 ;;
  esac
  echo "[0/4] conda not found: installing Miniconda into $HOME/miniconda3 (no root needed; GRSI_NO_AUTO_CONDA=1 disables this) ..."
  curl -L --fail --retry 3 -o "/tmp/$MC" "https://repo.anaconda.com/miniconda/$MC"
  bash "/tmp/$MC" -b -p "$HOME/miniconda3" && rm -f "/tmp/$MC"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
set +u; eval "$(conda shell.bash hook)"; set -u          # conda activation scripts reference unset variables
KNOWN_ENVS="$(conda env list | awk '{print $1}')"
if ! grep -qx "$ENV_NAME" <<<"$KNOWN_ENVS"; then
  echo "[1/4] creating conda env '$ENV_NAME' from replicability/environment.yml (one-time, 10-20 minutes, ~3 GB) ..."
  conda env create -n "$ENV_NAME" -f "$ROOT/replicability/environment.yml"
else
  echo "[1/4] using existing conda env '$ENV_NAME'"
fi
set +u; conda activate "$ENV_NAME"; set -u
if ! python - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "torchvision", "skimage", "fil_finder", "astropy", "yaml", "PIL", "numpy") if importlib.util.find_spec(m) is None]
if missing:
    print("missing packages in the conda env:", ", ".join(missing))
    sys.exit(1)
PY
then
  echo "the conda env '$ENV_NAME' is incomplete (an earlier creation was probably interrupted)."
  echo "Remove it and run again:  conda env remove -n $ENV_NAME"; exit 1
fi
if ! python -c "import cv2" 2>/dev/null; then
  echo "OpenCV cannot be imported. On a minimal Ubuntu image install the system libraries it needs:"
  echo "  sudo apt-get install -y libgl1 libglib2.0-0"; exit 1
fi
python -m pip install -q --no-deps -e "$ROOT"     # the hintauc package itself (its dependencies are in the env)
