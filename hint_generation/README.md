# Deterministic hint generation (DHT)

Pipeline per image: Felzenszwalb segmentation (scale=100, sigma=0.5, min_size=100)
-> per-region skeletonization (Zhang-Suen via skimage) -> 3x3 dilation (1 iteration)
-> FilFinder2D longest path (branch/skel threshold 3 px, prune by length)
-> scribble mask = longest path within the region; region color = per-region mean color;
dot hint = single pixel at the mean coordinate of the longest path.
Outputs per image: `*_region64.png`, `*_scribble_mask64.png`, `*_scribble_col64.png`,
`*_flatten_img64.png`, `*_dot_mask64.png`, `*_dot_col64.png` (and 256px variants).

Files:
- `canonical/hint_dot_generation_20240114_illust_64.py`
  -- the original generation script used to build the paper dataset (hardcoded cluster paths;
  kept verbatim for provenance). Variants for ImageNet / 256-px hints / the superpixel
  ablation differed only in path constants and `hint_img_size` and were removed in the
  2026-07 cleanup (available in the git history).
- `canonical/all_segmentations.py` -- segmentation stage (Felzenszwalb / SLIC / Quickshift / Watershed).
- `D_retrain_gen_hints.py` -- cleaned, argparse-based port of the same `make_scribbling` logic
  with SLIC/Quickshift segmenters (used for the supplementary segmenter-robustness study).
  Recommended starting point.
- `D_danboo_hints.py` -- DanbooRegion-segmenter variant (imports `make_scribbling` from the above).

Environment: python 3.7-3.9, `pip install -r requirements.txt`.

At evaluation time, hints for a hint ratio alpha are the top-alpha regions ordered by pixel
area (descending); see `detfill/datasets/custom.py` (`sample_ratio`).
