#!/usr/bin/env python3
"""
R2-2 / R1-3 segmentation-retrain data generation.

For every image id listed in configs/illust/{train,valid,test}_paper.txt, read the
felzenszwalb GT source image from main_exp_felzenszwalb_fixdot, re-segment it with a
DIFFERENT segmenter (SLIC or Quickshift) using the SAME parameters as the original
felzenszwalb pipeline, and regenerate the deterministic region+scribble hint with the
SAME make_scribbling logic. Outputs use the felz-identical naming so the BBDM dataloader
can read them unchanged.

Segmentation params  : ported verbatim from canonical/all_segmentations.py
Hint (make_scribbling): ported verbatim from
    canonical/hint_dot_generation_20240114_illust_64.py (same logic)

GT/sketch are segmenter-independent and reused from fixdot (NOT regenerated here).
Only region64 / scribble_mask64 / scribble_col64 are produced (training-needed, 64px).

Deps: cv2, scikit-image 0.19, astropy, fil_finder (see requirements.txt).

Usage:
  python D_retrain_gen_hints.py --segmenter slic --split all \
       --src_root /path/to/gt_images --txt_dir /path/to/split_lists --out_root /path/to/output
Resume-safe: skips an id whose _scribble_mask64.png already exists.
"""
import os, sys, argparse, copy, time, traceback
import numpy as np
import cv2
cv2.setNumThreads(1)  # avoid thread oversubscription when many shards share a node
from skimage.morphology import skeletonize
from skimage.segmentation import slic, quickshift
from skimage.util import img_as_float
from fil_finder import FilFinder2D
import astropy.units as u

SIZE = 64  # hint resolution (training reads *_mask64 / *_col64 / *_region64)

# Source layout: <src_root>/<SEG_SUBDIR>/<dir>/<id>.image.png
SEG_SUBDIR = "segmentation_regions/felzenszwalb"   # GT .image.png lives here, per-dir


# ------------------------- segmentation (verbatim params) -------------------------
def run_segmenter(img_bgr, seg, p):
    """SLIC/Quickshift with configurable granularity (tuned to match felz ~290 regions)."""
    if seg == "slic":
        return slic(img_bgr, n_segments=p.slic_n, compactness=p.slic_compactness,
                    sigma=1, start_label=1)
    if seg == "quickshift":
        # skimage 0.19 quickshift needs float input
        return quickshift(img_as_float(img_bgr), kernel_size=p.qs_kernel,
                          max_dist=p.qs_maxdist, ratio=p.qs_ratio)
    raise ValueError(seg)


def colorize_regions(segments):
    """One unique random RGB per label (the color is only a region id; boundaries are
    what matter after NEAREST resize). Matches all_segmentations.py's color_sets idea."""
    h, w = segments.shape
    region = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng()
    used = set()
    for lab in np.unique(segments):
        while True:
            c = (int(rng.integers(1, 255)), int(rng.integers(1, 255)), int(rng.integers(1, 255)))
            if c not in used:
                used.add(c)
                break
        region[segments == lab] = c
    return region


# ------------------- hint generation (verbatim make_scribbling) -------------------
def make_scribbling(img_bgr, region_bgr):
    """Port of hint_dot_generation_20240114_illust_abl_64.py::make_scribbling.
    Returns (region64_bgr, scribble_mask64, scribble_col64) as the felz pipeline did."""
    img = cv2.resize(img_bgr, (SIZE, SIZE))
    region = cv2.resize(region_bgr, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    cand_vals = np.unique(region.reshape((-1, 3)), axis=0)

    scribble_img = np.zeros(region.shape)
    scribbles_single = np.zeros((SIZE, SIZE))
    rng = np.random.default_rng()

    # split disconnected components that share a colour into fresh unique colours
    for i in range(len(cand_vals)):
        mask = np.all(region == cand_vals[i], axis=2).astype(np.uint8) * 255
        _, labeled = cv2.connectedComponents(mask, connectivity=4)
        lbs = np.unique(labeled)
        if lbs.shape[0] > 2:
            for j in range(2, lbs.shape[0]):
                while True:
                    val = np.array([(255 * rng.random()) // 1,
                                    (255 * rng.random()) // 1,
                                    (255 * rng.random()) // 1]).astype(np.uint8)
                    if not any(np.all(v == val) for v in cand_vals):
                        region[labeled == j, :] = val
                        break
    cand_vals = np.unique(region.reshape((-1, 3)), axis=0)

    for i in range(len(cand_vals)):
        skeleton_tmp = np.zeros((SIZE, SIZE))
        idx = np.argwhere(np.all(region == cand_vals[i], axis=2))
        skeleton_tmp[idx[:, 0], idx[:, 1]] = 1
        skeleton_s = skeletonize(skeleton_tmp)

        mval = copy.deepcopy(img[idx[:, 0], idx[:, 1]].mean(axis=0).astype(np.uint8))
        scribble_img[idx[:, 0], idx[:, 1]] = mval

        kernel = np.ones((3, 3), np.uint8)
        skeleton_s = cv2.dilate(skeleton_s.astype(np.uint8), kernel, iterations=1) * 255
        fil = FilFinder2D(skeleton_s, distance=250 * u.pc, mask=skeleton_s)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        try:
            fil.analyze_skeletons(branch_thresh=3 * u.pix, skel_thresh=3 * u.pix,
                                  prune_criteria='length')
        except Exception:
            continue
        scribbles_single += (fil.skeleton_longpath * skeleton_tmp)

    region_out = region
    mask_out = (scribbles_single.astype(np.uint8) * 255)
    col_out = (scribble_img * scribbles_single[:, :, np.newaxis])
    return region_out, mask_out, col_out


# --------------------------------- driver ----------------------------------------
def load_ids(split, txt_dir):
    splits = ["train", "valid", "test"] if split == "all" else [split]
    items = []
    for sp in splits:
        with open(os.path.join(txt_dir, f"{sp}_paper.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fn = line.split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
                dn = line.split('/')[-2]
                items.append((dn, fn))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", required=True, choices=["slic", "quickshift"])
    ap.add_argument("--split", default="all", choices=["all", "train", "valid", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--src_root", required=True, help="GT image root (<src_root>/segmentation_regions/felzenszwalb/<dir>/<id>.image.png)")
    ap.add_argument("--txt_dir", required=True, help="dir with {train,valid,test}_paper.txt (e.g. detfill/configs/illust)")
    ap.add_argument("--limit", type=int, default=0, help="dry-run: process at most N ids")
    # segmentation granularity — tuned on sample so SLIC/QS ~match felz ~290 regions
    #   SLIC n=350 -> ~289 ; Quickshift k5,md12 -> ~305  (felz median ~280)
    ap.add_argument("--slic_n", type=int, default=350)
    ap.add_argument("--slic_compactness", type=float, default=10)
    ap.add_argument("--qs_kernel", type=int, default=5)
    ap.add_argument("--qs_maxdist", type=int, default=12)
    ap.add_argument("--qs_ratio", type=float, default=0.5)
    args = ap.parse_args()

    items = load_ids(args.split, args.txt_dir)
    mine = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    print(f"[{args.segmenter} shard {args.shard}/{args.nshards}] {len(mine)}/{len(items)} ids",
          flush=True)

    done = skip = err = 0
    t0 = time.time()
    for k, (dname, fname) in enumerate(mine):
        out_dir = os.path.join(args.out_root, args.segmenter, dname)
        prefix = os.path.join(out_dir, f"{fname}.image")
        mask_path = prefix + f"_scribble_mask{SIZE}.png"
        if os.path.isfile(mask_path):
            skip += 1
            continue
        gt_path = os.path.join(args.src_root, SEG_SUBDIR, dname, f"{fname}.image.png")
        img = cv2.imread(gt_path)
        if img is None:
            print(f"  MISS gt {gt_path}", flush=True)
            err += 1
            continue
        try:
            segments = run_segmenter(img, args.segmenter, args)
            region = colorize_regions(segments)
            region64, mask64, col64 = make_scribbling(img, region)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(prefix + f"_region{SIZE}.png", region64)
            cv2.imwrite(prefix + f"_scribble_mask{SIZE}.png", mask64.astype(np.uint8))
            cv2.imwrite(prefix + f"_scribble_col{SIZE}.png", col64)
            done += 1
        except Exception:
            err += 1
            print(f"  ERR {dname}/{fname}\n{traceback.format_exc()}", flush=True)
            continue
        if args.limit and done >= args.limit:
            break
        if k % 200 == 0 and k:
            rate = (done + skip) / (time.time() - t0 + 1e-9)
            print(f"  ..{k}/{len(mine)} done={done} skip={skip} err={err} "
                  f"{rate:.1f} img/s", flush=True)

    print(f"[DONE {args.segmenter} shard {args.shard}/{args.nshards}] "
          f"done={done} skip={skip} err={err} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
