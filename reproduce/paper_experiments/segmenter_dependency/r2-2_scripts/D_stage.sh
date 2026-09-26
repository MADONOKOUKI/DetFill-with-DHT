#!/bin/bash
# Stage GT source images + code + txt to LOCAL /scratch on ALL nodes so generation is
# NFS-free (spares the fileserver). NFS is read exactly ONCE (here -> HOST_C); the
# other nodes are filled HOST_C -> cayenneN over the LAN, never touching NFS.
# Run from HOST_C.
set -u
BASE=/scratch/USER/seg_retrain_R2-2
FIX=/home/USER/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust
TXTD=/home/USER/gitlab/labrepo/main/BBDM_revision_revise_DATESTAMP/configs/illust
REPO=/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/scripts
KEY=~/.ssh/id_ed25519

echo "[1] build list + copy txt"
mkdir -p "$BASE/src" "$BASE/configs/illust" "$BASE/scripts"
cat "$TXTD"/train_paper.txt "$TXTD"/valid_paper.txt "$TXTD"/test_paper.txt \
  | sed 's#.*/illust/##' | sed '/^$/d' | sort -u > "$BASE/configs/gtlist.txt"
cp "$TXTD"/train_paper.txt "$TXTD"/valid_paper.txt "$TXTD"/test_paper.txt "$BASE/configs/illust/"
echo "    $(wc -l < "$BASE/configs/gtlist.txt") ids"

echo "[2] stage GT from NFS -> HOST_C local (one NFS read pass)"
rsync -a --files-from="$BASE/configs/gtlist.txt" "$FIX/" "$BASE/src/"

echo "[3] deploy code -> HOST_C local"
cp "$REPO"/*.py "$REPO"/*.sh "$BASE/scripts/"

echo "[4] distribute src+code+configs HOST_C -> 2/3/4 over LAN"
for h in HOST_C HOST_C HOST_C; do
  echo "    -> $h"
  ssh -i $KEY -o BatchMode=yes "$h" "mkdir -p $BASE/src $BASE/scripts $BASE/configs/illust"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/src/"     "$h:$BASE/src/"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/scripts/" "$h:$BASE/scripts/"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/configs/" "$h:$BASE/configs/"
done

echo "[5] verify staged counts per node"
for h in HOST_C HOST_C HOST_C HOST_C; do
  n=$(ssh -i $KEY -o BatchMode=yes "$h" "find $BASE/src -name '*.image.png' 2>/dev/null | wc -l")
  echo "    $h src=$n"
done
echo "staging done"
