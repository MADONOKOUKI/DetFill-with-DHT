#!/usr/bin/env bash
# Graphics Replicability Stamp script — reproduces Fig. 9 of
#   Madono, Mingcheng, Simo-Serra, "Hint-AUC: Deterministic Region-based Hint Generation for
#   Line Art Colorization Evaluation", IEEE TVCG (2026).
# Usage:  bash replicability/run.sh          (no arguments; see replicability/README.md)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)
ENV_NAME=${GRSI_ENV:-detfill-grsi}

# 1) Python environment (conda; replicability/environment.yml = the paper's runtime pins + hint-generation deps)
source "$HERE/conda_env.sh"        # finds or installs conda, creates and activates $ENV_NAME, checks the imports

# 2) checkpoints (1.5 GB, verified by SHA-256)
echo "[2/4] checkpoints"; bash "$HERE/fetch_checkpoints.sh"

# 3) reproduce Fig. 9
echo "[3/4] hint generation + DetFill inference + figure assembly"
python "$HERE/make_fig9.py"

# 4) where to look
echo "[4/4] output: $HERE/output/fig9.png   (reference: $HERE/expected/fig9_paper.png)"
