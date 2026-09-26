#!/usr/bin/env bash
# Download the two released DetFill checkpoints (GitHub release v1.0) into replicability/checkpoints/ and verify SHA-256.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); DST=${GRSI_CKPT_DIR:-$HERE/checkpoints}; mkdir -p "$DST"
BASE=https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/download/v1.0
declare -A SHA=( [detfill_scribble_illust_200ep.pth]=fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4
                 [detfill_dot_illust_200ep.pth]=fd872563e17cb1ac09957ed5ef993fac4dbeb2710f43a7ee29203444928d6fd4 )
for f in "${!SHA[@]}"; do
  if [ -f "$DST/$f" ] && echo "${SHA[$f]}  $DST/$f" | sha256sum -c --quiet - 2>/dev/null; then echo "ok (cached): $f"; continue; fi
  echo "downloading $f ..."; curl -L --fail --retry 3 -o "$DST/$f" "$BASE/$f"
  echo "${SHA[$f]}  $DST/$f" | sha256sum -c --quiet - && echo "ok (verified): $f"
done
