"""A self-contained example that needs nothing but ``pip install hintauc``.

``python -m hintauc.cli demo`` (or ``hintauc demo``) draws a small synthetic illustration, extracts a line art
from it, generates the deterministic hints, runs the oracle reference colorizer at every hint ratio of the paper
grid and prints the metric curve and the Hint-AUC. It is the same code path as the evaluation of a real model
(:func:`hintauc.evaluate_colorizer`), so it doubles as an installation check: the numbers are deterministic
(``path_method="geodesic"``) and are listed in ``docs/evaluate_your_model.md``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence, Union

import cv2
import numpy as np

from .auc import DEFAULT_ALPHAS
from .evaluate import evaluate_colorizer, hint_fill_colorizer, plot_curves
from .hints import generate_hints, write_image


def synthetic_illustration(seed: int = 0, size: int = 128) -> np.ndarray:
    """HxWx3 uint8 BGR image: light background and three flat colour rectangles (clear Felzenszwalb regions)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 235, np.uint8)
    s = size / 128.0
    boxes = [((10, 10), (60, 70)), ((70, 20), (120, 60)), ((30, 80), (100, 120))]
    for (y0, x0), (y1, x1) in boxes:
        img[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)] = rng.integers(20, 220, size=3, dtype=np.uint8)
    return img


def line_art_of(image_bgr: np.ndarray) -> np.ndarray:
    """A simple line art (white background, black edges) of a colour image, as the model's sketch input."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return (255 - edges).astype(np.uint8)


def run_demo(out: Optional[Union[str, "os.PathLike[str]"]] = None, path_method: str = "geodesic",
             metrics: Sequence[str] = ("mse", "psnr", "ssim"), alphas: Sequence[float] = DEFAULT_ALPHAS,
             hint_type: str = "scribble", size: int = 64, image_size: int = 128, seed: int = 0,
             verbose: bool = True) -> Dict[str, object]:
    """Run the demo; with ``out`` the images (illustration, line art, hints, predictions per ratio), ``result.json``
    and ``curve.png`` (if matplotlib is installed) are written there. Returns the result of
    :func:`hintauc.evaluate_colorizer`."""
    img = synthetic_illustration(seed, image_size)
    sketch = line_art_of(img)
    save_dir = os.path.join(os.fspath(out), "pred") if out is not None else None
    res = evaluate_colorizer(hint_fill_colorizer, [(sketch, img)], alphas=alphas, hint_type=hint_type,
                             metrics=metrics, size=size, path_method=path_method, save_dir=save_dir, oracle=True)
    res["demo"] = {"image": "synthetic illustration", "image_size": image_size, "seed": seed,
                   "colorizer": "hint_fill_colorizer (oracle reference, not a model)"}
    if out is not None:
        out = os.fspath(out)
        write_image(os.path.join(out, "gt", "000000.png"), img)
        write_image(os.path.join(out, "line_art.png"), sketch)
        hints = generate_hints(img, size=size, path_method=path_method)
        for a in (0.10, 1.00):
            color, mask = hints.at_ratio(a, hint_type=hint_type, resize_to=image_size)
            vis = np.full_like(color, 255)
            vis[mask > 0] = color[mask > 0]
            write_image(os.path.join(out, f"hints_{int(round(a * 100))}.png"), vis)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        plot_curves(res["per_alpha"], os.path.join(out, "curve.png"), title="hintauc demo (oracle reference colorizer)")
    if verbose:
        ms = list(metrics)
        print("hint ratio  " + "  ".join(f"{m:>8s}" for m in ms))
        for a in res["alphas"]:
            print(f"   {a:5.2f}    " + "  ".join(f"{res['per_alpha'][a][m]:8.4f}" for m in ms))
        print("Hint-AUC    " + "  ".join(f"{res['hint_auc'][m]:8.4f}" for m in ms))
        b = res["protocol"]["backend"]
        print(f"backend: resize={b['resize']}, ssim={b['ssim']}" + (f"; files in {out}" if out else ""))
    return res
