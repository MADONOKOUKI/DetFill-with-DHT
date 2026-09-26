"""Evaluate your own colorization model with the Hint-AUC protocol.

Two entry points:

* :func:`evaluate_colorizer` — hand in a Python callable that colorizes one image from its line art and the
  hints of one hint ratio. The function generates the deterministic hints, calls the model at every ratio of
  the grid, scores the outputs against the ground truth and integrates the Hint-AUC.
* :func:`hintauc.evaluate_hint_curve` and the command ``hintauc curve`` — hand in directories of images that
  the model produced offline, one directory per hint ratio.

Conventions. Images are ``HxWx3 uint8``. The arrays given to the callable are **BGR** (OpenCV order, like
everything in :mod:`hintauc.hints`) and the callable must return ``HxWx3 uint8`` BGR of the ground-truth size.
Scores are computed by :class:`hintauc.Evaluator` (images resized to 256 x 256, values in [0, 1]); the paper's
seven metrics need ``pip install "hintauc[perceptual]"``.
"""
from __future__ import annotations

import hashlib
import os
import platform
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .auc import DEFAULT_ALPHAS, hint_auc_table
from .hints import HintResult, _load_bgr, generate_hints, region_ids
from .metrics import DEFAULT_METRICS, Evaluator

Sample = Dict[str, object]
Colorizer = Callable[[Sample], np.ndarray]
PathOrArray = Union[str, "os.PathLike[str]", np.ndarray]

PROTOCOL_VERSION = "hint-auc/v1"   # the paper's protocol: size-ordered hints, the 8-point grid, trapezoid, 256 x 256 metrics


def file_sha256(path: Union[str, "os.PathLike[str]"]) -> str:
    """SHA-256 of a file (for recording which checkpoint produced a result)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(module: str) -> Optional[str]:
    try:
        return __import__(module).__version__
    except Exception:  # pragma: no cover - optional dependency
        return None


def protocol_record(evaluator: Optional[Evaluator] = None, alphas: Sequence[float] = DEFAULT_ALPHAS,
                    hint_type: Optional[str] = None, hint_size: Optional[int] = None,
                    path_method: Optional[str] = None, dot_method: Optional[str] = None,
                    **extra) -> Dict[str, object]:
    """Everything a reader needs to interpret a score: protocol, grid, metric settings, library versions."""
    from . import __version__
    alphas = [float(a) for a in alphas]
    rec: Dict[str, object] = {
        "protocol": PROTOCOL_VERSION,
        "paper_grid": alphas == [float(a) for a in DEFAULT_ALPHAS],
        "alphas": alphas,
        "hint_selection": "size-ordered: the largest int(n_regions * alpha) regions keep their hints",
        "aggregation": "trapezoidal integral of the per-alpha mean over alpha in [0, 1]",
        "metric_resize": getattr(evaluator, "resize", 256),
        "metric_input": "images resized to metric_resize x metric_resize, values in [0, 1], RGB",
        "metrics": list(getattr(evaluator, "metrics", ())),
        "hint_type": hint_type, "hint_map_size": hint_size,
        "path_method": path_method, "dot_method": dot_method,
        "versions": {"hintauc": __version__, "python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "scikit-image": _version("skimage"),
                     "torch": _version("torch"), "torchmetrics": _version("torchmetrics")},
    }
    rec.update(extra)
    return rec


def _load_gray(image: PathOrArray, shape_hw: Tuple[int, int]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        g = image if image.ndim == 2 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        g = cv2.imread(os.fspath(image), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(f"could not read line art: {image}")
    if g.shape[:2] != shape_hw:
        g = cv2.resize(g, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)
    return g.astype(np.uint8)


def hint_inputs(hints: HintResult, alpha: float, hint_type: str, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Hint colour and mask at ``alpha`` upsampled (nearest neighbour) to ``height x width``."""
    color, mask = hints.at_ratio(alpha, hint_type=hint_type)
    color = cv2.resize(color, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return color, mask


def _name_of(src: PathOrArray, idx: int) -> str:
    if isinstance(src, np.ndarray):
        return f"{idx:06d}.png"
    base = os.path.basename(os.fspath(src))
    return base if base.lower().endswith(".png") else os.path.splitext(base)[0] + ".png"


def evaluate_colorizer(colorize: Colorizer, samples: Iterable[Tuple[PathOrArray, PathOrArray]],
                       alphas: Sequence[float] = DEFAULT_ALPHAS, hint_type: str = "scribble",
                       metrics: Sequence[str] = DEFAULT_METRICS, evaluator: Optional[Evaluator] = None,
                       size: int = 64, path_method: str = "filfinder", dot_method: str = "medoid",
                       save_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
                       verbose: bool = False) -> Dict[str, object]:
    """Hint-AUC of a colorization model given as a Python callable.

    ``samples`` yields ``(line_art, ground_truth)`` pairs (paths or arrays). For each pair the deterministic
    hints are generated from the ground truth, and for every ``alpha`` in ``alphas`` the callable receives::

        {"index": i, "alpha": alpha, "hint_type": hint_type,
         "line_art": HxW uint8, "hint_color": HxWx3 uint8 BGR, "hint_mask": HxW {0,255},
         "hints": HintResult, "ground_truth": HxWx3 uint8 BGR}

    and must return the colorization as ``HxWx3 uint8`` BGR. With ``save_dir`` the outputs are written to
    ``save_dir/<alpha:.2f>/<name>.png`` (the layout of ``hintauc curve``). Returns per-alpha mean metrics,
    the Hint-AUC of every metric and a :func:`protocol_record`.
    """
    ev = evaluator or Evaluator(metrics=metrics)
    alphas = [float(a) for a in alphas]
    if any(a < 0 or a > 1 for a in alphas) or alphas[0] != 0.0 or alphas[-1] != 1.0:
        raise ValueError("alphas must be sorted, start at 0.0 and end at 1.0 (paper grid: hintauc.DEFAULT_ALPHAS)")
    sums: Dict[float, Dict[str, float]] = {a: {} for a in alphas}
    n = 0
    for idx, (sketch_src, gt_src) in enumerate(samples):
        gt = _load_bgr(gt_src)
        h, w = gt.shape[:2]
        sketch = _load_gray(sketch_src, (h, w))
        hints = generate_hints(gt, size=size, path_method=path_method, dot_method=dot_method)
        name = _name_of(gt_src, idx)
        for a in alphas:
            color, mask = hint_inputs(hints, a, hint_type, h, w)
            pred = colorize({"index": idx, "alpha": a, "hint_type": hint_type, "line_art": sketch,
                             "hint_color": color, "hint_mask": mask, "hints": hints, "ground_truth": gt})
            pred = np.asarray(pred)
            if pred.dtype != np.uint8 or pred.shape != (h, w, 3):
                raise ValueError(f"the colorizer must return an HxWx3 uint8 BGR image of shape {(h, w, 3)}, "
                                 f"got {pred.dtype} {pred.shape}")
            if save_dir is not None:
                d = os.path.join(os.fspath(save_dir), f"{a:.2f}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, name), pred)
            scores = ev(pred[:, :, ::-1], gt[:, :, ::-1])       # the evaluator takes RGB arrays
            for k, v in scores.items():
                sums[a][k] = sums[a].get(k, 0.0) + float(v)
        n += 1
        if verbose:
            print(f"[{idx}] {name}: {hints.n_regions()} regions", flush=True)
    if n == 0:
        raise ValueError("no samples")
    per_alpha = {a: {k: v / n for k, v in sums[a].items()} for a in alphas}
    return {"hint_type": hint_type, "alphas": alphas, "n_images": n, "per_alpha": per_alpha,
            "hint_auc": hint_auc_table(per_alpha),
            "protocol": protocol_record(ev, alphas, hint_type, size, path_method, dot_method)}


def hint_fill_colorizer(sample: Sample) -> np.ndarray:
    """Reference baseline for demos and tests: every hinted region is filled with its hint colour (the region
    mean), unhinted regions stay light gray, and the line art is multiplied on top."""
    hints: HintResult = sample["hints"]  # type: ignore[assignment]
    gt = sample["ground_truth"]
    h, w = gt.shape[:2]
    ids = region_ids(cv2.resize(hints.region, (w, h), interpolation=cv2.INTER_NEAREST))
    flat = cv2.resize(hints.flatten, (w, h), interpolation=cv2.INTER_NEAREST)
    hinted = np.unique(ids[sample["hint_mask"] > 0])
    out = np.full((h, w, 3), 235, np.uint8)
    sel = np.isin(ids, hinted)
    out[sel] = flat[sel]
    line = (sample["line_art"].astype(np.float32) / 255.0)[:, :, None]
    return np.clip(out.astype(np.float32) * line, 0, 255).astype(np.uint8)


def plot_curves(per_alpha: Mapping[float, Mapping[str, float]], path: Union[str, "os.PathLike[str]"],
                title: Optional[str] = None, metrics: Optional[Sequence[str]] = None) -> bool:
    """Metric-vs-hint-ratio curves as a PNG (needs matplotlib; returns False if it is not installed)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return False
    alphas = sorted(float(a) for a in per_alpha)
    ms = list(metrics) if metrics else sorted({m for a in alphas for m in per_alpha[a]})
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.0), squeeze=False)
    for ax, m in zip(axes[0], ms):
        ax.plot([100 * a for a in alphas], [per_alpha[a][m] for a in alphas], marker="o")
        ax.set_xlabel("hint ratio (%)"); ax.set_title(m.upper()); ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.fspath(path), dpi=150)
    plt.close(fig)
    return True
