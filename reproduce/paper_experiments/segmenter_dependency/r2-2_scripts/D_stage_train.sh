#!/bin/bash
# Stage training data to HOST_B:/scratch (NFS-free training). Run from HOST_C.
# Layout produced (matches edited BBDM_seg_retrain/datasets/custom.py):
#   /scratch/USER/seg_retrain/gt/<dir>/<id>.image.png
#   /scratch/USER/seg_retrain/sketch/{XDoG,pysimp,sketchkeras}/<dir>/<id>.png
#   /scratch/USER/seg_retrain/{slic,quickshift}/<dir>/<id>.image_{region,scribble_mask,scribble_col}64.png
set -u
H=${1:-HOST_B}; KEY=~/.ssh/id_ed25519
DST=/scratch/USER/seg_retrain
CAY=/scratch/USER/seg_retrain_R2-2
NFS_RD=/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data
FX=/home/USER/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust
RSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no"

$RSH "$H" "mkdir -p $DST/gt $DST/sketch $DST/slic $DST/quickshift"

echo "[1] GT  HOST_C-local -> $H (LAN)"
rsync -a -e "$RSH" "$CAY/src/segmentation_regions/felzenszwalb/" "$H:$DST/gt/"

echo "[2] hints (full 25998 from NFS gathered retrain_data) -> $H"
# NOTE: HOST_C-local only holds its own shard (~6527); the FULL set is the gathered NFS copy.
rsync -a -e "$RSH" "$NFS_RD/slic/"       "$H:$DST/slic/"
rsync -a -e "$RSH" "$NFS_RD/quickshift/" "$H:$DST/quickshift/"

echo "[3] sketch (3 styles, 26k ids)  fixdot-NFS -> $H"
# gtlist has 'segmentation_regions/felzenszwalb/<dir>/<id>.image.png'; derive sketch paths
awk -F'/' '{dir=$(NF-1); id=$NF; sub(/\.image\.png$/,"",id);
            print "sketch/XDoG/"dir"/"id".png";
            print "sketch/pysimp/"dir"/"id".png";
            print "sketch/sketchkeras/"dir"/"id".png"}' \
    "$CAY/configs/gtlist.txt" | sort -u > /tmp/sketch_list_$H.txt
echo "    sketch files to stage: $(wc -l < /tmp/sketch_list_$H.txt)"
rsync -a --files-from=/tmp/sketch_list_$H.txt -e "$RSH" "$FX/" "$H:$DST/"

echo "[verify] $H counts"
$RSH "$H" "echo gt=\$(find $DST/gt -name '*.image.png'|wc -l) \
slic=\$(find $DST/slic -name '*_scribble_mask64.png'|wc -l) \
qs=\$(find $DST/quickshift -name '*_scribble_mask64.png'|wc -l) \
sketch=\$(find $DST/sketch -name '*.png'|wc -l)"
echo "stage done"
