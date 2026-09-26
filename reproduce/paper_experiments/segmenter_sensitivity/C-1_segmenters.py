"""C-1  Segmenter-ablation for hint generation (R1-3, R2-2).

We cannot re-run the full colorization model, but we CAN check whether
the target regions the paper's deterministic hint picks up would change
under a different segmenter. For each ground-truth image we:

  (i)  Extract the paper's hint mask M_paper from
       proposed/{alpha}/condition/{id}.image_hint.png by treating
       any non-gray non-white pixel as a hint.
  (ii) Run alternative segmenters on the GT image at the same 256x256
       resolution:
         - SLIC (skimage) at two compactness settings
         - Felzenszwalb at two scale settings
         - Watershed (OpenCV)
  (iii) For each segmenter, rank segments by area descending, take the
        top-K largest segments whose union covers roughly the same
        number of pixels as M_paper, then report:
          - Jaccard IoU between that union and M_paper (per image, per alpha)

High IoU across segmenters means hint target regions are robust to
segmenter choice. Low IoU justifies R2-2's over-segmentation concern.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import slic, felzenszwalb

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output" / "C-1"; OUT.mkdir(parents=True, exist_ok=True)
IMG_ROOT = Path("/home/madorin/gitlab/labrepo/main/usertest/imgs")
ALPHAS = [1, 3, 5, 10, 25, 50, 100]


def hint_mask(img: np.ndarray) -> np.ndarray:
    gray128 = np.all(img == 128, axis=2)
    whitish = np.all(img > 240, axis=2)
    return ~(gray128 | whitish)


def rank_segments_cover(labels, target_pixels):
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    mask = np.zeros(labels.shape, dtype=bool)
    total = 0
    for idx in order:
        sid = ids[idx]; c = counts[idx]
        mask |= (labels == sid)
        total += c
        if total >= target_pixels:
            break
    return mask


def iou(a, b):
    i = (a & b).sum(); u = (a | b).sum()
    return float(i / max(u, 1))


def segmenters(img_rgb):
    out = {}
    out["slic100"] = slic(img_rgb, n_segments=100, compactness=10.0, start_label=0)
    out["slic300"] = slic(img_rgb, n_segments=300, compactness=10.0, start_label=0)
    out["felzen_scale200"] = felzenszwalb(img_rgb, scale=200, sigma=0.8, min_size=50)
    out["felzen_scale400"] = felzenszwalb(img_rgb, scale=400, sigma=0.8, min_size=50)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thr, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    markers = np.uint8(markers)
    _, markers = cv2.connectedComponents(markers)
    markers += 1
    markers[thr == 0] = 0
    ws = cv2.watershed(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), markers.copy())
    out["watershed"] = ws
    return out


def main():
    cond_dir_10 = IMG_ROOT / "proposed" / "10" / "condition"
    files = sorted(cond_dir_10.glob("*.image_hint.png"))[:60]
    print(f"analyzing {len(files)} images")

    rows = []
    for hp in files:
        did = hp.stem.replace(".image_hint", "")
        gt_p = IMG_ROOT / "proposed" / "10" / "ground_truth" / f"{did}.image_col.png"
        if not gt_p.exists():
            continue
        gt = cv2.imread(str(gt_p))
        if gt is None:
            continue
        gt = cv2.resize(gt, (256, 256))
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        segs = segmenters(gt_rgb)
        for alpha in ALPHAS:
            hp_a = IMG_ROOT / "proposed" / str(alpha) / "condition" / f"{did}.image_hint.png"
            if not hp_a.exists():
                continue
            hint = cv2.imread(str(hp_a))
            if hint is None:
                continue
            m_paper = hint_mask(hint)
            target = int(m_paper.sum())
            if target < 50:
                continue
            row = dict(image=did, alpha=alpha, paper_px=target)
            for nm, lbl in segs.items():
                lbl256 = cv2.resize(lbl.astype(np.int32), (256, 256),
                                     interpolation=cv2.INTER_NEAREST)
                mask = rank_segments_cover(lbl256, target)
                row[f"iou_{nm}"] = iou(mask, m_paper)
                # containment: what fraction of the paper's hint pixels
                # are inside the top-K largest regions of this segmenter?
                row[f"contain_{nm}"] = float((mask & m_paper).sum() / max(target, 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "segmenter_iou.csv", index=False)
    print(df.describe().to_string())

    summary = {"overall_mean_iou": {}, "overall_mean_containment": {}}
    for c in df.columns:
        if c.startswith("iou_"):
            summary["overall_mean_iou"][c] = float(df[c].mean())
        if c.startswith("contain_"):
            summary["overall_mean_containment"][c] = float(df[c].mean())
    summary["per_alpha_mean_iou"] = {}
    summary["per_alpha_mean_containment"] = {}
    for a in ALPHAS:
        sub = df[df.alpha == a]
        if len(sub) == 0:
            continue
        summary["per_alpha_mean_iou"][int(a)] = {c: float(sub[c].mean())
                                                    for c in sub.columns
                                                    if c.startswith("iou_")}
        summary["per_alpha_mean_containment"][int(a)] = {c: float(sub[c].mean())
                                                    for c in sub.columns
                                                    if c.startswith("contain_")}
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
