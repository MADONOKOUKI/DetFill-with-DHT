"""C-2  Perturbation of the paper's hint mask (R1-3, R2-2 stress test).

We cannot retrain models for this revision. Instead, we stress-test
the invariant: *if M_paper is perturbed by morphological noise or
random region drops, how quickly does the hint mask diverge from the
original?*  If IoU decays slowly, the deterministic mask is robust.
We pair this with an upper-bound analysis: compute the correlation
between hint-mask change and the induced change in a simple proxy
for colorization quality (nearest-color-neighbor in hint region).

Perturbations applied:
 - Dilation / erosion (morphological) with radius 1, 2, 3
 - Random region-drop (connected components) at 10%, 25%, 50%
 - Gaussian blur on the segmentation basis followed by re-thresholding
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output" / "C-2"; OUT.mkdir(parents=True, exist_ok=True)
IMG_ROOT = Path("/home/madorin/gitlab/labrepo/main/usertest/imgs")
ALPHAS = [1, 3, 5, 10, 25, 50, 100]


def hint_mask(img):
    gray128 = np.all(img == 128, axis=2)
    whitish = np.all(img > 240, axis=2)
    return ~(gray128 | whitish)


def iou(a, b):
    i = (a & b).sum(); u = (a | b).sum()
    return float(i / max(u, 1))


def random_region_drop(mask, frac, rng):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ids = list(range(1, num))
    if not ids:
        return mask.copy()
    drop_k = int(len(ids) * frac)
    drop = set(rng.choice(ids, drop_k, replace=False).tolist())
    out = mask.copy()
    for i in drop:
        out[labels == i] = False
    return out


def perturb(mask, rng):
    yield ("dilate_r1", cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    yield ("dilate_r2", cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    yield ("erode_r1", cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
    yield ("erode_r2", cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    for f in [0.10, 0.25, 0.50]:
        yield (f"region_drop_{int(f*100)}pct", random_region_drop(mask, f, rng))
    k = 5
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
    yield ("blur_rethresh", blurred > 0.5)


def hint_color_error(gt_rgb, mask):
    """Average lab distance of GT in mask vs its mean (proxy for how much
    colorization is constrained by the hint mask)."""
    if mask.sum() == 0:
        return float("nan")
    lab = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pix = lab[mask]
    mean = pix.mean(axis=0)
    return float(np.linalg.norm(pix - mean, axis=1).mean())


def main():
    rng = np.random.default_rng(0)
    files = sorted((IMG_ROOT / "proposed" / "10" / "condition").glob("*.image_hint.png"))[:60]
    rows = []
    for hp in files:
        did = hp.stem.replace(".image_hint", "")
        gt_p = IMG_ROOT / "proposed" / "10" / "ground_truth" / f"{did}.image_col.png"
        gt = cv2.imread(str(gt_p)) if gt_p.exists() else None
        if gt is None:
            continue
        gt = cv2.resize(gt, (256, 256))
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        for alpha in ALPHAS:
            hp_a = IMG_ROOT / "proposed" / str(alpha) / "condition" / f"{did}.image_hint.png"
            if not hp_a.exists():
                continue
            hint = cv2.imread(str(hp_a))
            m = hint_mask(hint)
            if m.sum() < 50:
                continue
            base_err = hint_color_error(gt_rgb, m)
            for name, mp in perturb(m, rng):
                rows.append(dict(
                    image=did, alpha=alpha, perturb=name,
                    iou=iou(mp, m),
                    pixel_change=int(np.abs(mp.astype(int) - m.astype(int)).sum()),
                    base_err=base_err,
                    perturbed_err=hint_color_error(gt_rgb, mp),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "perturbation_results.csv", index=False)
    summary = {}
    for name, sub in df.groupby("perturb"):
        summary[name] = dict(
            mean_iou=float(sub.iou.mean()),
            median_iou=float(sub.iou.median()),
            mean_base_err=float(sub.base_err.mean()),
            mean_perturbed_err=float(sub.perturbed_err.mean()),
            err_delta=float((sub.perturbed_err - sub.base_err).mean()),
        )
    summary["per_alpha"] = {}
    for a in ALPHAS:
        sub = df[df.alpha == a]
        if len(sub) == 0:
            continue
        summary["per_alpha"][int(a)] = {}
        for name, g in sub.groupby("perturb"):
            summary["per_alpha"][int(a)][name] = float(g.iou.mean())
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
