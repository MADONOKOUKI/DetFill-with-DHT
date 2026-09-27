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
import json
import os
import platform
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .auc import DEFAULT_ALPHAS, alpha_dir_name, check_alphas, hint_auc_table
from .hints import HintResult, _load_bgr, generate_hints, region_ids, write_image
from .metrics import DEFAULT_METRICS, Evaluator

Sample = Dict[str, object]
Colorizer = Callable[[Sample], np.ndarray]
PathOrArray = Union[str, "os.PathLike[str]", np.ndarray]

PROTOCOL_VERSION = "hint-auc/v1"   # the paper's protocol: size-ordered hints, the 8-point grid, trapezoid, 256 x 256 metrics

#: keys of the sample dictionary a colorizer receives (the model's observations)
OBSERVED_KEYS = ("index", "name", "alpha", "hint_type", "line_art", "hint_color", "hint_mask")
#: keys added with ``oracle=True`` (ground-truth information; for reference baselines only)
ORACLE_KEYS = ("hints", "ground_truth", "n_regions")


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
    """Everything a reader needs to interpret a score: protocol, grid, metric settings, the evaluation backend
    (resize and SSIM implementation) and library versions."""
    from . import __version__
    ev = evaluator or Evaluator(metrics=("mse",))
    alphas = [float(a) for a in alphas]
    rec: Dict[str, object] = {
        "protocol": PROTOCOL_VERSION,
        "paper_grid": alphas == [float(a) for a in DEFAULT_ALPHAS],
        "alphas": alphas,
        "hint_selection": "size-ordered: the largest int(n_regions * alpha) regions keep their hints",
        "aggregation": "trapezoidal integral of the per-alpha mean over alpha in [0, 1]",
        "metric_resize": ev.resize,
        "metric_input": "images resized to metric_resize x metric_resize, values in [0, 1], RGB",
        "metrics": list(ev.metrics),
        "backend": ev.backend(),
        "hint_type": hint_type, "hint_map_size": hint_size,
        "path_method": path_method, "dot_method": dot_method,
        "versions": {"hintauc": __version__, "python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "scikit-image": _version("skimage"), "pillow": _version("PIL"),
                     "torch": _version("torch"), "torchvision": _version("torchvision"),
                     "torchmetrics": _version("torchmetrics")},
    }
    rec.update(extra)
    return rec


def _load_gray(image: PathOrArray, shape_hw: Tuple[int, int]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        g = _load_bgr(image)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    else:
        g = cv2.imread(os.fspath(image), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(f"could not read line art: {image}")
    if g.shape[:2] != shape_hw:
        g = cv2.resize(g, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)
    return g.astype(np.uint8)


def hint_inputs(hints: HintResult, alpha: float, hint_type: str, height: int, width: int,
                tie_break: str = "stable") -> Tuple[np.ndarray, np.ndarray]:
    """Hint colour and mask at ``alpha`` upsampled (nearest neighbour) to ``height x width``.

    ``tie_break`` orders regions of equal area: ``"stable"`` (default here; ascending label, identical on every
    machine) or ``"default"`` (NumPy's default argsort as in the DetFill loader, whose order among equal areas
    depends on the NumPy build and on the CPU's SIMD sort path)."""
    color, mask = hints.at_ratio(alpha, hint_type=hint_type, tie_break=tie_break)
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
                       oracle: bool = False, tie_break: str = "stable", verbose: bool = False) -> Dict[str, object]:
    """Hint-AUC of a colorization model given as a Python callable.

    ``samples`` yields ``(line_art, ground_truth)`` pairs (paths or arrays). For each pair the deterministic
    hints are generated from the ground truth, and for every ``alpha`` in ``alphas`` the callable receives the
    model's observations::

        {"index": i, "name": "<file>.png", "alpha": alpha, "hint_type": hint_type,
         "line_art": HxW uint8, "hint_color": HxWx3 uint8 BGR, "hint_mask": HxW {0,255}}

    and must return the colorization as ``HxWx3 uint8`` BGR. The ground truth and the full hint structure are
    *not* passed by default, so an adapter cannot use them by accident; ``oracle=True`` adds ``"ground_truth"``,
    ``"hints"`` (the :class:`~hintauc.hints.HintResult`) and ``"n_regions"`` for reference baselines such as
    :func:`hint_fill_colorizer`, whose scores must not be compared with those of real models.

    ``alphas`` must be strictly increasing, start at 0 and end at 1 (:func:`hintauc.check_alphas`). Regions of equal
    area are ordered by ``tie_break``: ``"stable"`` (default; ascending label, the same on every machine) or
    ``"default"`` (NumPy's default argsort, the DetFill loader's rule, whose order among equal areas depends on the
    NumPy build and on the CPU's SIMD sort path, so results can differ between machines). With
    ``save_dir`` the outputs are written to ``save_dir/<ratio>/<name>.png`` (``<ratio>`` from
    :func:`hintauc.alpha_dir_name`, the layout of ``hintauc curve``) together with ``save_dir/manifest.json``
    (exact ratios, directory names, image names, protocol); a write failure raises ``OSError``. Sample names
    must be unique (files with the same basename would overwrite each other). Returns the per-alpha mean
    metrics, the Hint-AUC of every metric, the evaluated names and a :func:`protocol_record`.
    """
    ev = evaluator or Evaluator(metrics=metrics)
    alphas = check_alphas(alphas, full_range=True)
    dir_names = {a: alpha_dir_name(a) for a in alphas}
    sums: Dict[float, Dict[str, float]] = {a: {} for a in alphas}
    names: List[str] = []
    for idx, (sketch_src, gt_src) in enumerate(samples):
        gt = _load_bgr(gt_src)
        h, w = gt.shape[:2]
        sketch = _load_gray(sketch_src, (h, w))
        hints = generate_hints(gt, size=size, path_method=path_method, dot_method=dot_method)
        name = _name_of(gt_src, idx)
        if name in names:
            raise ValueError(f"duplicate sample name {name!r} (sample {idx}): ground-truth files must have unique "
                             "basenames, or pass arrays (named by index)")
        names.append(name)
        for a in alphas:
            color, mask = hint_inputs(hints, a, hint_type, h, w, tie_break=tie_break)
            sample: Sample = {"index": idx, "name": name, "alpha": a, "hint_type": hint_type, "line_art": sketch,
                              "hint_color": color, "hint_mask": mask}
            if oracle:
                sample.update({"hints": hints, "ground_truth": gt, "n_regions": hints.n_regions()})
            pred = np.asarray(colorize(sample))
            if pred.dtype != np.uint8 or pred.shape != (h, w, 3):
                raise ValueError(f"the colorizer must return an HxWx3 uint8 BGR image of shape {(h, w, 3)}, "
                                 f"got {pred.dtype} {pred.shape}")
            if save_dir is not None:
                write_image(os.path.join(os.fspath(save_dir), dir_names[a], name), pred)
            scores = ev(pred[:, :, ::-1], gt[:, :, ::-1])       # the evaluator takes RGB arrays
            for k, v in scores.items():
                sums[a][k] = sums[a].get(k, 0.0) + float(v)
        if verbose:
            print(f"[{idx}] {name}: {hints.n_regions()} regions", flush=True)
    n = len(names)
    if n == 0:
        raise ValueError("no samples")
    per_alpha = {a: {k: v / n for k, v in sums[a].items()} for a in alphas}
    result = {"hint_type": hint_type, "alphas": alphas, "n_images": n, "names": names, "per_alpha": per_alpha,
              "hint_auc": hint_auc_table(per_alpha),
              "protocol": protocol_record(ev, alphas, hint_type, size, path_method, dot_method, oracle=oracle,
                                          tie_break=tie_break)}
    if save_dir is not None:
        manifest = {"protocol": PROTOCOL_VERSION, "hint_type": hint_type, "alphas": alphas,
                    "dirs": {dir_names[a]: a for a in alphas}, "names": names, "n_images": n,
                    "note": "one directory per hint ratio; `hintauc curve <this dir> <gt dir>` re-scores the images"}
        with open(os.path.join(os.fspath(save_dir), "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    return result


def hint_fill_colorizer(sample: Sample) -> np.ndarray:
    """Oracle reference baseline for demos and tests (not a model): every hinted region is filled with its hint
    colour (the region mean), unhinted regions stay light gray, and the line art is multiplied on top. It reads
    the ground-truth regions, so it needs ``evaluate_colorizer(..., oracle=True)``."""
    if "hints" not in sample:
        raise KeyError("hint_fill_colorizer is an oracle baseline that uses the ground-truth regions; call "
                       "evaluate_colorizer(hint_fill_colorizer, samples, oracle=True)")
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
