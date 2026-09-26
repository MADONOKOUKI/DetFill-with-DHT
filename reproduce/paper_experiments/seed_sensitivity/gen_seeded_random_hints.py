#!/usr/bin/env python3
# Seeded RANDOM region-hint generator for the ranking-stability experiment (supp).
#
# Bakes a seeded uniform-random region selection (NO size sorting) into 64x64
# scribble hint/mask files, so that all three methods (DetFill / ColorizeDiffusion
# v2 / Diffusart) can consume IDENTICAL random hints through their existing
# deterministic inference paths run at sample_ratio = 1.0 (loader-side selection
# becomes the identity).
#
# Exactness: region maps are 64x64 (verified) and every loader upscales them with
# INTER_NEAREST, so masking at 64px here is pixel-identical to the loaders'
# mask-after-resize behaviour. Region ids replicate the loaders' formula on
# cv2 (BGR) arrays: id = c0*255*255 + c1*255 + c2.
#
# Per (seed, image): ONE permutation of region ids; alpha takes its prefix, so
# hint sets are nested across alpha within a seed (mirrors the deterministic
# sorted-prefix structure and yields a proper Hint-AUC curve per seed).
import os
import sys
import argparse
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

REGION_DIR = "/scratch/USER/major_revision/hint_from_regions_256"   # 64x64 region id maps
HINT_DIR = "/scratch/USER/major_revision/hint_from_regions_64_rev/0016"
OUT_ROOT = "/scratch/USER/seedhints_scr"
HINT_TYPE = "scribble"
ALPHAS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]
RATIO_TAG = {0.0: "0", 0.01: "1", 0.03: "3", 0.05: "5", 0.10: "10",
             0.25: "25", 0.50: "50", 1.00: "100"}


def list_ids():
    ids = sorted(f[:-len(".image_region64.png")] for f in os.listdir(REGION_DIR)
                 if f.endswith(".image_region64.png"))
    return ids


def process_one(job):
    seed, idx, img_id = job
    region = cv2.imread(os.path.join(REGION_DIR, f"{img_id}.image_region64.png"))
    if region is None:
        return f"MISS region {img_id}"
    region = cv2.resize(region, (64, 64), interpolation=cv2.INTER_NEAREST)
    r64 = region.astype(np.uint64)
    id_maps = r64[:, :, 0] * 255 * 255 + r64[:, :, 1] * 255 + r64[:, :, 2]
    cand_vals = np.unique(id_maps.reshape(-1))

    col = Image.open(os.path.join(HINT_DIR, f"{img_id}.image_{HINT_TYPE}_col64.png")).convert("RGB")
    msk = Image.open(os.path.join(HINT_DIR, f"{img_id}.image_{HINT_TYPE}_mask64.png"))
    msk_mode = msk.mode
    col_a = np.array(col)
    msk_a = np.array(msk.convert("L"))

    rng = np.random.default_rng([20260611, seed, idx])
    perm = rng.permutation(cand_vals)

    for a in ALPHAS:
        n = int(len(cand_vals) * a)
        keep = perm[:n]
        area = np.isin(id_maps, keep).astype(np.uint8)
        col_out = col_a * area[:, :, None]
        msk_out = msk_a * area
        od = os.path.join(OUT_ROOT, f"seed{seed}", f"alpha_{RATIO_TAG[a]}")
        os.makedirs(od, exist_ok=True)
        Image.fromarray(col_out).save(os.path.join(od, f"{img_id}.image_{HINT_TYPE}_col64.png"))
        Image.fromarray(msk_out).convert(msk_mode).save(
            os.path.join(od, f"{img_id}.image_{HINT_TYPE}_mask64.png"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = list_ids()
    if args.limit:
        ids = ids[: args.limit]
    jobs = [(s, i, img_id) for s in args.seeds for i, img_id in enumerate(ids)]
    print(f"images={len(ids)} seeds={args.seeds} -> jobs={len(jobs)}")
    errs = 0
    with Pool(args.workers) as p:
        for k, r in enumerate(p.imap_unordered(process_one, jobs, chunksize=64)):
            if r:
                errs += 1
                print(" !", r)
            if (k + 1) % 3000 == 0:
                print(f"  {k+1}/{len(jobs)}")
    print(f"DONE errs={errs}")


if __name__ == "__main__":
    main()
