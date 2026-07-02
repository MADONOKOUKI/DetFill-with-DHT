"""Hint-AUC: integrate per-ratio metric scores over the hint-ratio grid.

Port of ``evaluation/calc_hint_auc_manual.py``: plain trapezoidal integration
of metric(alpha) over alpha in [0, 1] (the alpha range has length 1, so the
integral equals the range-normalized value).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

#: the paper's front-loaded hint-ratio grid
DEFAULT_ALPHAS = (0.00, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00)


def trapz(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Trapezoidal rule (port of calc_hint_auc_manual.trapz)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need >= 2 (x, y) points with matching lengths")
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


def evaluate_hint_curve(
    preds_by_alpha: Mapping[float, str],
    gt_dir: str,
    evaluator=None,
    metrics: Sequence[str] = ("mse", "psnr", "ssim"),
    pairing: str = "sorted",
    limit: int = 0,
) -> Dict[str, object]:
    """End-to-end Hint-AUC over per-ratio prediction directories.

    ``preds_by_alpha`` maps each hint ratio alpha to a directory of the
    colorizations produced with hints at that ratio (paper grid:
    :data:`DEFAULT_ALPHAS`).  Returns per-alpha mean metrics and the
    Hint-AUC of every metric.
    """
    from .metrics import Evaluator, evaluate_dirs

    ev = evaluator or Evaluator(metrics=metrics)
    per_alpha: Dict[float, Dict[str, float]] = {}
    for alpha in sorted(preds_by_alpha):
        per_alpha[alpha] = evaluate_dirs(
            preds_by_alpha[alpha], gt_dir, evaluator=ev,
            pairing=pairing, limit=limit)
    return {"per_alpha": per_alpha, "hint_auc": hint_auc_table(per_alpha)}
