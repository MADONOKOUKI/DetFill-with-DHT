#!/usr/bin/env python3
"""
R2-2 / R1-3 segmentation-retrain data generation.

For every image id listed in configs/illust/{train,valid,test}.txt, read the
felzenszwalb GT source image from main_exp_felzenszwalb_fixdot, re-segment it with a
DIFFERENT segmenter (SLIC or Quickshift) using the SAME parameters as the original
felzenszwalb pipeline, and regenerate the deterministic region+scribble hint with the
SAME make_scribbling logic. Outputs use the felz-identical naming so the BBDM dataloader
can read them unchanged.

Segmentation params  : ported verbatim from canonical/all_segmentations.py
Hint (make_scribbling): hintauc.generate_hints (the library's port of canonical/hint_dot_generation.py);
    region ids are assigned deterministically, so the label order used by `hint_order: label` and by the
    region tie-break is reproducible (an earlier version of this script drew random colours)

GT/sketch are segmenter-independent and reused from fixdot (NOT regenerated here).
Only region64 / scribble_mask64 / scribble_col64 are produced (training-needed, 64px).

Deps: cv2, scikit-image 0.19, astropy, fil_finder (see requirements.txt).

Usage:
  python generate_hints.py --segmenter slic --split all \
       --src_root /path/to/gt_images --txt_dir /path/to/split_lists --out_root /path/to/output
Resume-safe: skips an id whose three output files all exist; writes are checked (a failed write is an error).
"""
import os
import sys, argparse, time, traceback
import numpy as np
import cv2
cv2.setNumThreads(1)  # avoid thread oversubscription when many shards share a node
from skimage.segmentation import slic, quickshift
from skimage.util import img_as_float
try:
    import hintauc
except ImportError:  # run from a checkout without installing the package
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import hintauc
from hintauc.hints import _id_to_color

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
    """One unique colour per label, assigned deterministically (the library's collision-free base-255 encoding of
    the label index). The colour is only a region id, but it also fixes the label order that `hint_order: label`
    and the region tie-break use, so it must not be random."""
    h, w = segments.shape
    region = np.zeros((h, w, 3), np.uint8)
    for i, lab in enumerate(np.unique(segments)):
        region[segments == lab] = _id_to_color(i)
    return region


# ------------------- hint generation (the library's port of make_scribbling) -------------------
def make_scribbling(img_bgr, region_bgr, path_method="filfinder"):
    """(region64_bgr, scribble_mask64, scribble_col64) for one image and its region map, computed by
    ``hintauc.generate_hints`` (the maintained port of canonical/hint_dot_generation.py): nearest-neighbour
    resize of the region map to 64 px, deterministic re-colouring of disconnected components, skeleton, longest
    path (FilFinder or geodesic), region-mean colours. Region maps whose colours would collide under the loader's
    base-255 ids (DanbooRegion maps) are re-encoded by the library first."""
    res = hintauc.generate_hints(img_bgr, size=SIZE, region_map=region_bgr, path_method=path_method)
    return res.region, res.scribble_mask, res.scribble_color


OUTPUT_SUFFIXES = (f"_region{SIZE}.png", f"_scribble_mask{SIZE}.png", f"_scribble_col{SIZE}.png")


def outputs_complete(prefix):
    """Resume only skips an id whose three output files all exist (a partial write is redone)."""
    return all(os.path.isfile(prefix + suf) for suf in OUTPUT_SUFFIXES)


def write_outputs(prefix, region64, mask64, col64):
    """Write the three files through hintauc.write_image (creates the directory, raises on failure)."""
    hintauc.write_image(prefix + OUTPUT_SUFFIXES[0], region64)
    hintauc.write_image(prefix + OUTPUT_SUFFIXES[1], np.asarray(mask64).astype(np.uint8))
    hintauc.write_image(prefix + OUTPUT_SUFFIXES[2], np.asarray(col64).astype(np.uint8))


# --------------------------------- driver ----------------------------------------
def load_ids(split, txt_dir):
    splits = ["train", "valid", "test"] if split == "all" else [split]
    items = []
    for sp in splits:
        with open(os.path.join(txt_dir, f"{sp}.txt")) as f:
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
    ap.add_argument("--txt_dir", required=True, help="dir with {train,valid,test}.txt (e.g. detfill/configs/illust)")
    ap.add_argument("--limit", type=int, default=0, help="dry-run: process at most N ids")
    ap.add_argument("--path_method", default="filfinder", choices=["filfinder", "geodesic"],
                    help="longest-path extraction: 'filfinder' (paper) or 'geodesic' (dependency-free); "
                         "geodesic outputs go to <out_root>/<segmenter>_geodesic/")
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
        seg_dir = args.segmenter if args.path_method == "filfinder" else f"{args.segmenter}_{args.path_method}"
        out_dir = os.path.join(args.out_root, seg_dir, dname)
        prefix = os.path.join(out_dir, f"{fname}.image")
        if outputs_complete(prefix):
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
            region64, mask64, col64 = make_scribbling(img, region, args.path_method)
            write_outputs(prefix, region64, mask64, col64)
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
    if err:
        sys.exit(2)


if __name__ == "__main__":
    main()
