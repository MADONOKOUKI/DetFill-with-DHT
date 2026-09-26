#!/usr/bin/env python3
# Build all symlink wrapper trees for the seeded ranking-stability experiment.
# Methods consume the SAME seeded hints (/scratch/USER/seedhints_scr) through
# their existing deterministic paths at sample_ratio=1.0.
#
# Layout produced (under /scratch/USER/seedexp/):
#   detfill_roots/seed{S}_alpha_{R}/   scratch_root wrapper for BBDM_revision_revise
#   coldiff_roots/seed{S}_alpha_{R}/   dataroot wrapper for colorizeDiffusion_v2
#   diffusart_hints/seed{S}_alpha_{R}/0016/   merged dir (seeded col/mask + region64 links)
# plus static /scratch/USER/main_exp/illust/{segmentation_regions,sketch}/... for Diffusart.
import os

SEEDHINTS = "/scratch/USER/seedhints_scr"
MR = "/scratch/USER/major_revision"
CD_TEST = "/scratch/USER/revision/colorizeDiffusion/illust/images_test"
ROOT = "/scratch/USER/seedexp"
MAIN_EXP = "/scratch/USER/main_exp/illust"
SEEDS = [1, 2, 3, 4, 5]
RATIOS = ["0", "1", "3", "5", "10", "25", "50", "100"]
# alpha 0/100 are seed-invariant (empty / full hints) -> generate seed1 only
def runs():
    for r in RATIOS:
        for s in (SEEDS if r not in ("0", "100") else [1]):
            yield s, r

def ln(src, dst):
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.exists(dst):
        raise RuntimeError(f"refusing to clobber non-symlink: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)

n = 0
for s, r in runs():
    sh = f"{SEEDHINTS}/seed{s}/alpha_{r}"
    assert os.path.isdir(sh), sh
    # --- DetFill scratch_root wrapper ---
    d = f"{ROOT}/detfill_roots/seed{s}_alpha_{r}"
    ln(f"{MR}/segmentations", f"{d}/segmentations")
    ln(f"{MR}/hint_from_regions_256", f"{d}/hint_from_regions_256")
    ln(f"{MR}/sketch", f"{d}/sketch")
    ln(sh, f"{d}/hint_from_regions_64_rev/0016")
    # --- ColDiff dataroot wrapper ---
    c = f"{ROOT}/coldiff_roots/seed{s}_alpha_{r}"
    for comp in ["color", "sketch_p", "sketch_s", "sketch_x", "region"]:
        ln(f"{CD_TEST}/{comp}", f"{c}/{comp}")
    ln(sh, f"{c}/hint")
    n += 1
print(f"detfill/coldiff wrappers: {n} runs")

# --- Diffusart merged hint dirs: seeded col/mask + region64 (per-file links) ---
region_dir = f"{MR}/hint_from_regions_256"
regions = [f for f in os.listdir(region_dir) if f.endswith(".image_region64.png")]
made = 0
for s, r in runs():
    sh = f"{SEEDHINTS}/seed{s}/alpha_{r}"
    dd = f"{ROOT}/diffusart_hints/seed{s}_alpha_{r}/0016"
    os.makedirs(dd, exist_ok=True)
    for f in os.listdir(sh):
        p = os.path.join(dd, f)
        if not os.path.islink(p):
            os.symlink(os.path.join(sh, f), p)
            made += 1
    for f in regions:
        p = os.path.join(dd, f)
        if not os.path.islink(p):
            os.symlink(os.path.join(region_dir, f), p)
            made += 1
print(f"diffusart merged links: {made}")

# --- Diffusart static main_exp parts (illust) ---
ln(f"{MR}/segmentations/originals", f"{MAIN_EXP}/segmentation_regions/felzenszwalb/0016")
for cand, src in [("pysketchsimplify", "pysimp"), ("XDoG", "XDoG"), ("sketchkeras", "sketchkeras")]:
    ln(f"{MR}/sketch/{src}", f"{MAIN_EXP}/sketch/{cand}/0016")
# hint_from_regions/felzenszwalb is re-pointed per run by the runner (ln -sfn)
os.makedirs(f"{MAIN_EXP}/hint_from_regions", exist_ok=True)
print("main_exp static parts done")
print("ALL WRAPPERS BUILT")
