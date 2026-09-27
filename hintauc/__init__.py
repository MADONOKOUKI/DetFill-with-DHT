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
    reencode_region_map,
    region_ids,
    segment_regions,
    write_image,
)
from .metrics import (
    ALL_METRICS,
    DEFAULT_METRICS,
    EXTRA_METRICS,
    LOWER_IS_BETTER,
    SET_METRICS,
    Evaluator,
    evaluate_dirs,
    evaluate_pair,
    evaluate_pairs,
    evaluate_set,
    list_pairs,
)
from .evaluate import (
    PROTOCOL_VERSION,
    evaluate_colorizer,
    file_sha256,
    hint_fill_colorizer,
    hint_inputs,
    plot_curves,
    protocol_record,
)
from .auc import (
    DEFAULT_ALPHAS,
    alpha_dir_name,
    check_alphas,
    evaluate_hint_curve,
    hint_auc,
    hint_auc_table,
    trapz,
)
from .demo import run_demo, synthetic_illustration

__version__ = "0.3.1"

__all__ = [
    "ALL_METRICS",
    "EXTRA_METRICS",
    "SET_METRICS",
    "evaluate_set",
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
    "geodesic_longest_path",
    "PROTOCOL_VERSION",
    "evaluate_colorizer",
    "file_sha256",
    "hint_fill_colorizer",
    "hint_inputs",
    "plot_curves",
    "protocol_record",
    "hint_auc",
    "hint_auc_table",
    "region_ids",
    "segment_regions",
    "trapz",
    "alpha_dir_name",
    "check_alphas",
    "evaluate_pairs",
    "list_pairs",
    "reencode_region_map",
    "run_demo",
    "synthetic_illustration",
    "write_image",
    "__version__",
]
