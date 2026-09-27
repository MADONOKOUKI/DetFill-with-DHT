#!/usr/bin/env bash
# Download the two released DetFill checkpoints (GitHub release v1.0) into replicability/checkpoints/ and verify SHA-256.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); DST=${GRSI_CKPT_DIR:-$HERE/checkpoints}; mkdir -p "$DST"
REPO=MADONOKOUKI/DetFill-with-DHT; TAG=v1.0; BASE=https://github.com/$REPO/releases/download/$TAG
declare -A SHA=( [detfill_scribble_illust_200ep.pth]=fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4
                 [detfill_dot_illust_200ep.pth]=b4779946f24b5f52cdf640418c73bef67925c845e50c8f36610aeac7d50f212b )
fetch() {  # $1 = asset name; anonymous HTTPS first, then the GitHub CLI (works for a logged-in user even before the
           # repository is public), otherwise explain what to do.
  if curl -L --fail --retry 3 -o "$DST/$1" "$BASE/$1"; then return 0; fi
  rm -f "$DST/$1"
  if command -v gh >/dev/null 2>&1; then
    echo "  anonymous download failed; retrying with the GitHub CLI ..."
    gh release download "$TAG" -R "$REPO" -p "$1" -D "$DST" --clobber && return 0
  fi
  { echo "[error] could not download $1 from $BASE"
    echo "        (a 404 means the asset is not reachable anonymously, e.g. the repository is not public yet)."
    echo "        Download it with a logged-in GitHub CLI:  gh release download $TAG -R $REPO -p $1 -D $DST"
    echo "        or copy the file into $DST and re-run."; } >&2
  return 1
}
for f in "${!SHA[@]}"; do
  if [ -f "$DST/$f" ] && echo "${SHA[$f]}  $DST/$f" | sha256sum -c --quiet - 2>/dev/null; then echo "ok (cached): $f"; continue; fi
  echo "downloading $f ..."; fetch "$f"
  echo "${SHA[$f]}  $DST/$f" | sha256sum -c --quiet - && echo "ok (verified): $f"
done
