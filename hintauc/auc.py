"""Hint-AUC: integrate per-ratio metric scores over the hint-ratio grid.

Port of ``evaluation/calc_hint_auc.py``: plain trapezoidal integration
of metric(alpha) over alpha in [0, 1] (the alpha range has length 1, so the
integral equals the range-normalized value).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

#: the paper's front-loaded hint-ratio grid
DEFAULT_ALPHAS = (0.00, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00)


def alpha_dir_name(alpha: float) -> str:
    """Directory name of a hint ratio: the shortest decimal exact to 1e-6 with at least two decimals
    (``0.00, 0.01, ..., 0.10, ..., 1.00`` for the paper grid; ``0.001`` for finer ratios). Read back by
    ``hintauc curve`` with ``float()``."""
    s = f"{float(alpha):.6f}".rstrip("0")
    if s.endswith("."):
        s += "00"
    elif len(s.split(".")[1]) < 2:
        s += "0"
    return s


def check_alphas(alphas: Sequence[float], full_range: bool = True) -> List[float]:
    """Validate a hint-ratio grid and return it as a list of floats.

    Requires at least two finite values in [0, 1], strictly increasing (no duplicates: a repeated ratio would be
    integrated twice) and, with ``full_range`` (the Hint-AUC protocol), a grid that starts at 0 and ends at 1.
    Two ratios that round to the same :func:`alpha_dir_name` are rejected as well.
    """
    try:
        vals = [float(a) for a in alphas]
    except (TypeError, ValueError) as e:
        raise ValueError(f"alphas must be numbers, got {alphas!r}") from e
    if len(vals) < 2:
        raise ValueError(f"need at least two hint ratios (got {vals}); the paper grid is hintauc.DEFAULT_ALPHAS")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError(f"alphas must be finite, got {vals}")
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"alphas must lie in [0, 1], got {vals}")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"alphas must be strictly increasing without duplicates, got {vals}")
    if full_range and (vals[0] != 0.0 or vals[-1] != 1.0):
        raise ValueError(f"the Hint-AUC grid must start at 0.0 and end at 1.0 (got {vals[0]} .. {vals[-1]}); "
                         "the paper grid is hintauc.DEFAULT_ALPHAS")
    names = [alpha_dir_name(v) for v in vals]
    if len(set(names)) != len(names):
        raise ValueError(f"two hint ratios are closer than 1e-6 and would share a directory name: {vals}")
    return vals


def trapz(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoidal rule (port of calc_hint_auc.trapz); ``xs`` must be strictly increasing."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need >= 2 (x, y) points with matching lengths")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"x values must be strictly increasing, got {list(xs)}")
    area = 0.0
    for i in range(1, len(xs)):
        area += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return area


def hint_auc(scores_by_alpha: Mapping[float, float]) -> float:
    """Hint-AUC of one metric from ``{alpha: score}``.

    >>> hint_auc({0.0: 0.44, 0.01: 0.33, ..., 1.0: 0.10})
    """
    xs = sorted(scores_by_alpha)
    ys = [scores_by_alpha[a] for a in xs]
    return trapz(xs, ys)


def hint_auc_table(
    metrics_by_alpha: Mapping[float, Mapping[str, float]],
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Hint-AUC per metric from ``{alpha: {metric: score}}``."""
    alphas = sorted(metrics_by_alpha)
    if metrics is None:
        metrics = sorted({m for a in alphas for m in metrics_by_alpha[a]})
    out = {}
    for m in metrics:
        out[m] = hint_auc({a: metrics_by_alpha[a][m] for a in alphas})
    return out


def curve_from_manifest(pred_root: str, gt_dir: str, manifest: Mapping[str, object],
                        allow_missing: bool = False) -> Dict[float, List[tuple]]:
    """(prediction, ground truth, name) triples per ratio, taken from the manifest of :func:`evaluate_colorizer`.

    The manifest names the ratio directories and, for every prediction, the ground-truth file it was made for
    (``sources``), so JPEG ground truths saved as PNG predictions pair correctly and a stale directory of another run
    cannot slip in: ratio directories under ``pred_root`` that the manifest does not list raise ``ValueError``.
    """
    import warnings
    from .evaluate import _ratio_subdirs
    dirs = {str(k): float(v) for k, v in manifest["dirs"].items()}
    on_disk = _ratio_subdirs(pred_root)
    stale = sorted(set(on_disk) - set(dirs))
    if stale:
        raise ValueError(f"{pred_root} holds ratio directories that its manifest does not list ({', '.join(stale)}): "
                         "predictions of another run. Remove them, re-run with overwrite=True, or pass "
                         "--ignore-manifest to score every directory on disk")
    missing_dirs = sorted(set(dirs) - set(on_disk))
    if missing_dirs:
        raise ValueError(f"{pred_root}: ratio directories listed in the manifest are missing: {', '.join(missing_dirs)}")
    names = list(manifest["names"])
    sources = manifest.get("sources") or {n: n for n in names}
    gts = {}
    for n in names:
        src = sources.get(n, n)
        candidates = ([src] if src != "<array>" else []) + [n, os.path.splitext(n)[0] + ".png"]
        found = next((c for c in candidates if os.path.isfile(os.path.join(gt_dir, c))), None)
        if found is None:
            raise ValueError(f"ground truth of {n} not found in {gt_dir} (looked for {', '.join(candidates)})")
        gts[n] = os.path.join(gt_dir, found)
    pairs: Dict[float, List[tuple]] = {}
    dropped = set()
    for d, a in dirs.items():
        missing = [n for n in names if not os.path.isfile(os.path.join(on_disk[d], n))]
        if missing:
            msg = f"{on_disk[d]}: {len(missing)} prediction(s) of the manifest are missing ({', '.join(missing[:3])})"
            if not allow_missing:
                raise ValueError(msg)
            warnings.warn(msg + "; evaluating the common subset", stacklevel=2)
            dropped.update(missing)
        pairs[a] = [(os.path.join(on_disk[d], n), gts[n], n) for n in names]
    if dropped:
        pairs = {a: [t for t in v if t[2] not in dropped] for a, v in pairs.items()}
    return pairs


def evaluate_hint_curve(
    preds_by_alpha: Mapping[float, str],
    gt_dir: str,
    evaluator=None,
    metrics: Sequence[str] = ("mse", "psnr", "ssim"),
    pairing: str = "sorted",
    limit: int = 0,
    allow_missing: bool = False,
    full_range: bool = False,
    manifest: Optional[Mapping[str, object]] = None,
    pred_root: Optional[str] = None,
) -> Dict[str, object]:
    """End-to-end Hint-AUC over per-ratio prediction directories.

    ``preds_by_alpha`` maps each hint ratio alpha to a directory of the colorizations produced with hints at that
    ratio (paper grid: :data:`DEFAULT_ALPHAS`). Before anything is scored, the file sets of all ratio directories
    are checked against the ground truth **and against each other**: every ratio must contain the same images
    (``pairing='name'``) or the same number of images (``pairing='sorted'``); a missing prediction at one ratio is
    an error, because it would silently change the evaluated set of that ratio only. With ``allow_missing=True``
    the images common to all ratios are evaluated and a warning lists the rest. ``full_range=True`` additionally
    requires the grid to start at 0 and end at 1 (the Hint-AUC protocol).

    With ``manifest`` (the ``manifest.json`` of :func:`evaluate_colorizer`, read with :func:`hintauc.read_manifest`)
    and ``pred_root`` (its directory), the ratio directories, the prediction names and their ground-truth files come
    from the manifest instead of the directory listing (:func:`curve_from_manifest`), and the result carries the
    run's attributes under ``"run"``.

    Returns ``{"alphas", "n_images", "names", "per_alpha", "hint_auc"}``: the evaluated names, the per-alpha mean
    metrics and the Hint-AUC of every metric.
    """
    import warnings
    from .metrics import Evaluator, evaluate_pairs, list_pairs

    ev = evaluator or Evaluator(metrics=metrics)
    if manifest is not None:
        if pred_root is None:
            raise ValueError("pred_root (the directory that holds the manifest) is required with manifest")
        pairs = curve_from_manifest(os.fspath(pred_root), gt_dir, manifest, allow_missing=allow_missing)
        alphas = check_alphas(sorted(pairs), full_range=full_range)
        if limit:
            pairs = {a: pairs[a][:limit] for a in alphas}
        names = [n for _, _, n in pairs[alphas[0]]]
        per_alpha = {a: evaluate_pairs(pairs[a], ev) for a in alphas}
        run = {k: manifest.get(k) for k in ("hint_type", "oracle", "tie_break", "path_method", "dot_method",
                                            "hint_map_size", "hintauc")}
        return {"alphas": alphas, "n_images": len(names), "names": names, "per_alpha": per_alpha,
                "hint_auc": hint_auc_table(per_alpha), "run": run}

    alphas = check_alphas(sorted(float(a) for a in preds_by_alpha), full_range=full_range)
    dirs = {float(a): d for a, d in preds_by_alpha.items()}
    pairs = {a: list_pairs(dirs[a], gt_dir, pairing=pairing, allow_missing=allow_missing) for a in alphas}
    if pairing == "name":
        common = set.intersection(*(set(n for _, _, n in pairs[a]) for a in alphas))
        for a in alphas:
            names_a = set(n for _, _, n in pairs[a])
            if names_a != common:
                msg = (f"ratio {alpha_dir_name(a)} ({dirs[a]}) does not hold the same images as the other ratios: "
                       f"{len(names_a - common)} extra / {len(common - names_a)} missing (e.g. "
                       f"{', '.join(sorted(names_a ^ common)[:3])})")
                if not allow_missing:
                    raise ValueError(msg + "; every hint ratio must be evaluated on the same images "
                                     "(pass allow_missing=True / --allow-missing to use the common subset)")
                warnings.warn(msg + "; using the common subset", stacklevel=2)
        pairs = {a: [t for t in pairs[a] if t[2] in common] for a in alphas}
    else:
        counts = {a: len(pairs[a]) for a in alphas}
        if len(set(counts.values())) != 1:
            msg = f"the ratio directories hold different numbers of images: { {alpha_dir_name(a): n for a, n in counts.items()} }"
            if not allow_missing:
                raise ValueError(msg + " (pass allow_missing=True / --allow-missing to evaluate the first min(n) of each)")
            warnings.warn(msg + "; evaluating the first min(n) files of each", stacklevel=2)
            n = min(counts.values())
            pairs = {a: pairs[a][:n] for a in alphas}
    if limit:
        pairs = {a: pairs[a][:limit] for a in alphas}
    names = [n for _, _, n in pairs[alphas[0]]]
    per_alpha: Dict[float, Dict[str, float]] = {a: evaluate_pairs(pairs[a], ev) for a in alphas}
    return {"alphas": alphas, "n_images": len(names), "names": names,
            "per_alpha": per_alpha, "hint_auc": hint_auc_table(per_alpha)}
