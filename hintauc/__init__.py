"""hintauc — Deterministic region-based hint generation (DHT) and Hint-AUC
evaluation for line-art colorization.

Library form of the method proposed in
"Hint-AUC: Deterministic Region-based Hint Generation for Line Art
Colorization Evaluation" (Madono, Mingcheng, Simo-Serra).

Quick start
-----------
>>> import hintauc
>>> hints = hintauc.generate_hints("illustration.png")       # 1. hints
>>> color, mask = hints.at_ratio(0.10, hint_type="scribble") # top-10% regions
>>> hints.save("out/illustration")                           # canonical files

>>> ev = hintauc.Evaluator(metrics=("mse", "psnr", "ssim", "lpips"))
>>> scores = ev("colorized.png", "ground_truth.png")         # 2. evaluation
>>> auc = hintauc.hint_auc({0.0: 0.44, 0.01: 0.33, 0.03: 0.27,
...                         0.05: 0.24, 0.10: 0.20, 0.25: 0.15,
...                         0.50: 0.12, 1.00: 0.10})
"""

from .longest_path import geodesic_longest_path
from .hints import (
    DEFAULT_HINT_SIZE,
    FELZENSZWALB_PARAMS,
    HintResult,
    generate_hints,
    region_ids,
    segment_regions,
)
from .metrics import (
    DEFAULT_METRICS,
    LOWER_IS_BETTER,
    Evaluator,
    evaluate_dirs,
    evaluate_pair,
)
from .auc import (
    DEFAULT_ALPHAS,
    evaluate_hint_curve,
    hint_auc,
    hint_auc_table,
    trapz,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_HINT_SIZE",
    "DEFAULT_METRICS",
    "FELZENSZWALB_PARAMS",
    "LOWER_IS_BETTER",
    "Evaluator",
    "HintResult",
    "evaluate_dirs",
    "evaluate_hint_curve",
    "evaluate_pair",
    "generate_hints",
    "hint_auc",
    "hint_auc_table",
    "region_ids",
    "segment_regions",
    "trapz",
    "__version__",
]
