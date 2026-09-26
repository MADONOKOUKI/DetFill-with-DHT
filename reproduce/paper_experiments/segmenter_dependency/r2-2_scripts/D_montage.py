#!/home/USER/anaconda3/envs/py37/bin/python
"""
Visual comparison montage: GT | felzenszwalb | SLIC | Quickshift, for N sample IDs.
Lets you eyeball whether SLIC/QS coarseness now matches felz.
Builds two PNGs: region maps and scribble-colour hints.

Usage:
  py37 D_montage.py [--n 8] [--kind region|scribble|both] [--out <dir>] [--data <retrain_data>]
"""
import cv2, numpy as np, glob, os, random, argparse

FELZ = "/home/USER/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust/hint_from_regions/felzenszwalb"
GT   = "/home/USER/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust/segmentation_regions/felzenszwalb"
CELL = 200   # upscaled cell size for viewing

def load(p, gt=False):
    if not os.path.exists(p):
        return np.full((CELL, CELL, 3), 64, np.uint8)
    a = cv2.imread(p)
    return cv2.resize(a, (CELL, CELL), interpolation=cv2.INTER_AREA if gt else cv2.INTER_NEAREST)

def ncol(p):
    if not os.path.exists(p): return 0
    a = cv2.imread(p); return int(np.unique(a.reshape(-1, 3), axis=0).shape[0])

def label(img, txt):
    cv2.rectangle(img, (0, 0), (CELL, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--kind", default="both", choices=["region", "scribble", "both"])
    ap.add_argument("--data", default="/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/retrain_data")
    ap.add_argument("--out", default="/home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R2/R2-2_segmentation_dependency/output")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cand = glob.glob(os.path.join(args.data, "slic", "*", "*_region64.png"))
    random.seed(args.seed); random.shuffle(cand)
    ids = []
    for sp in cand:
        dn = os.path.basename(os.path.dirname(sp)); fn = os.path.basename(sp)  # <id>.image_region64.png
        base = fn.replace("_region64.png", "")  # <id>.image
        if os.path.exists(os.path.join(args.data, "quickshift", dn, fn)) and \
           os.path.exists(os.path.join(FELZ, dn, base + "_region64.png")):
            ids.append((dn, base))
        if len(ids) >= args.n: break

    kinds = ["region", "scribble"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        suff = "_region64.png" if kind == "region" else "_scribble_col64.png"
        rows = []
        for dn, base in ids:
            gt   = label(load(os.path.join(GT, dn, base + ".png"), gt=True), f"GT {base.split('.')[0]}")
            fe_p = os.path.join(FELZ, dn, base + suff)
            sl_p = os.path.join(args.data, "slic", dn, base + suff)
            qs_p = os.path.join(args.data, "quickshift", dn, base + suff)
            fe = label(load(fe_p), f"felz ({ncol(os.path.join(FELZ,dn,base+'_region64.png'))})")
            sl = label(load(sl_p), f"SLIC ({ncol(os.path.join(args.data,'slic',dn,base+'_region64.png'))})")
            qs = label(load(qs_p), f"QS ({ncol(os.path.join(args.data,'quickshift',dn,base+'_region64.png'))})")
            rows.append(np.hstack([gt, fe, sl, qs]))
        grid = np.vstack(rows)
        outp = os.path.join(args.out, f"montage_{kind}_felz_slic_qs.png")
        cv2.imwrite(outp, grid)
        print(f"wrote {outp}  ({grid.shape[1]}x{grid.shape[0]}, {len(ids)} ids)  [num = region64 colour count]")

if __name__ == "__main__":
    main()
