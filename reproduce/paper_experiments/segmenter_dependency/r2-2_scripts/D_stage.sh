#!/bin/bash
# Stage GT source images + code + txt to LOCAL /scratch on ALL nodes so generation is
# NFS-free (spares the fileserver). NFS is read exactly ONCE (here -> cayenne1); the
# other nodes are filled cayenne1 -> cayenneN over the LAN, never touching NFS.
# Run from cayenne1.
set -u
BASE=/scratch/madono/seg_retrain_R2-2
FIX=/home/madorin/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust
TXTD=/home/madorin/gitlab/labrepo/main/BBDM_revision_revise_DATESTAMP/configs/illust
REPO=/home/madorin/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/scripts
KEY=~/.ssh/id_ed25519

echo "[1] build list + copy txt"
mkdir -p "$BASE/src" "$BASE/configs/illust" "$BASE/scripts"
cat "$TXTD"/train_paper.txt "$TXTD"/valid_paper.txt "$TXTD"/test_paper.txt \
  | sed 's#.*/illust/##' | sed '/^$/d' | sort -u > "$BASE/configs/gtlist.txt"
cp "$TXTD"/train_paper.txt "$TXTD"/valid_paper.txt "$TXTD"/test_paper.txt "$BASE/configs/illust/"
echo "    $(wc -l < "$BASE/configs/gtlist.txt") ids"

echo "[2] stage GT from NFS -> cayenne1 local (one NFS read pass)"
rsync -a --files-from="$BASE/configs/gtlist.txt" "$FIX/" "$BASE/src/"

echo "[3] deploy code -> cayenne1 local"
cp "$REPO"/*.py "$REPO"/*.sh "$BASE/scripts/"

echo "[4] distribute src+code+configs cayenne1 -> 2/3/4 over LAN"
for h in cayenne2 cayenne3 cayenne4; do
  echo "    -> $h"
  ssh -i $KEY -o BatchMode=yes "$h" "mkdir -p $BASE/src $BASE/scripts $BASE/configs/illust"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/src/"     "$h:$BASE/src/"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/scripts/" "$h:$BASE/scripts/"
  rsync -a -e "ssh -i $KEY -o BatchMode=yes" "$BASE/configs/" "$h:$BASE/configs/"
done

echo "[5] verify staged counts per node"
for h in cayenne1 cayenne2 cayenne3 cayenne4; do
  n=$(ssh -i $KEY -o BatchMode=yes "$h" "find $BASE/src -name '*.image.png' 2>/dev/null | wc -l")
  echo "    $h src=$n"
done
echo "staging done"
