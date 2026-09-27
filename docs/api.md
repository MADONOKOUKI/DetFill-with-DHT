# API reference: the `hintauc` library

Every public function and command, with its role, arguments, and an input/output example. All examples were run on
the illustration shipped in `replicability/data/4731016.image.png` (800 × 1200 px, 1,701 regions) with `hintauc` 0.3.3,
Python 3.9, NumPy 1.26, OpenCV 4.11, scikit-image 0.24, PyTorch 2.5.1, torchvision 0.20, torchmetrics 1.4.0; the
numbers are the actual outputs of that run (FilFinder examples can differ by a few pixels between runs, see
`generate_hints`).

```bash
pip install hintauc                 # hints, pixel metrics, Hint-AUC
pip install "hintauc[paper]"        # + torchvision / torchmetrics 1.4.0: the paper's metric backend
pip install "hintauc[perceptual]"   # + LPIPS / OpenCLIP / DINOv2 / DreamSim / DISTS / FID / KID
```

**Conventions.** Images are `HxWx3 uint8`; floating-point arrays are accepted only with values in [0, 1] (scaled by
255 and rounded), anything else raises `ValueError` instead of being cast silently. Everything produced by the hint
generator (`HintResult` arrays, the callback inputs of `evaluate_colorizer`) is **BGR** (OpenCV order, `cv2.imread`).
`Evaluator` accepts file paths or arrays that it takes as **RGB**, so pass `arr[:, :, ::-1]` when the array comes
from OpenCV. Files are decoded by Pillow and converted to RGB, whatever their mode (palette, grayscale, LA, RGBA,
CMYK), on both backends; 16-bit and floating-point files are rejected. The pixel metrics (MSE, PSNR, SSIM, MAE,
MS-SSIM, ΔE) and LPIPS / DISTS use the image resized to 256 × 256 with values in [0, 1] (antialiased bilinear resize
of float32 channels: torchvision when installed, otherwise Pillow; the two agree to about 1e-6); OpenCLIP, DINOv2 and
DreamSim apply their own preprocessing to the decoded image, as in the paper's evaluator.

Contents: [constants](#constants) · [hint generation](#hint-generation) · [evaluation](#evaluation) ·
[Hint-AUC](#hint-auc) · [your own model](#evaluating-your-own-model) · [command line](#command-line) ·
[what was verified](#what-was-verified)

## Constants

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_ALPHAS` | `(0.0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0)` | the paper's hint-ratio grid |
| `DEFAULT_HINT_SIZE` | `64` | resolution of the hint maps |
| `FELZENSZWALB_PARAMS` | `{'scale': 100, 'sigma': 0.5, 'min_size': 100}` | segmentation parameters of the paper |
| `DEFAULT_METRICS` | `('mse', 'psnr', 'ssim', 'lpips', 'openclip', 'dino', 'dreamsim')` | the seven metrics of the paper |
| `EXTRA_METRICS` | `('mae', 'ms_ssim', 'deltae', 'lpips_vgg', 'dists')` | added after the paper |
| `ALL_METRICS` | `DEFAULT_METRICS + EXTRA_METRICS` | everything `Evaluator` accepts |
| `SET_METRICS` | `('fid', 'kid')` | directory-level metrics of `evaluate_set` |
| `LOWER_IS_BETTER` | `{'mse': True, 'psnr': False, ...}` | direction of every metric |
| `PROTOCOL_VERSION` | `'hint-auc/v1'` | protocol tag written into every result |
| `evaluate.OBSERVED_KEYS`, `evaluate.ORACLE_KEYS` | `('index', 'name', 'alpha', 'hint_type', 'line_art', 'hint_color', 'hint_mask')`, `('hints', 'ground_truth', 'n_regions')` | what a colorizer receives; the second set only with `oracle=True` |

## Hint generation

### `generate_hints(image, size=64, segmenter="felzenszwalb", region_map=None, verbose=False, path_method="filfinder", dot_method="medoid", **seg_kwargs) -> HintResult`

Turns a colour image into the deterministic hints of the paper: Felzenszwalb regions, one scribble (longest path of
the region skeleton) and one dot per region, coloured with the region mean, at `size` × `size`.

| Argument | Meaning |
|---|---|
| `image` | path or `HxWx3 uint8` BGR array (float arrays in [0, 1] accepted); segmentation runs at the input resolution and the region map is downsampled to `size` — the paper's stored maps come from the original-resolution files, so feed those to compare |
| `size` | hint-map resolution (paper: 64) |
| `segmenter` | `felzenszwalb` (paper), `slic`, `quickshift`; `**seg_kwargs` override the segmenter parameters |
| `region_map` | a precomputed region-colour map (e.g. from DanbooRegion), any colours; then `segmenter` is ignored. Colours that would collide under the loader's base-255 ids (a channel value of 255) are re-encoded first (`reencode_region_map`) |
| `path_method` | `filfinder` (paper; FilFinder's medial axis breaks ties with an unseeded generator, so two runs differ in a few pixels: six runs on `4942016` gave six scribble masks of 1,960–1,963 pixels, same 938 regions and dots) or `geodesic` (dependency-free, bit-reproducible; slightly different scribbles, never mix the two in one evaluation) |
| `dot_method` | `medoid` (rule of the released maps: the in-region path pixel with the smallest total L1 distance to the others), `mean` (truncated mean of the path, may leave the region), `nearest_mean` |

```python
>>> import hintauc
>>> h = hintauc.generate_hints("replicability/data/4731016.image.png", size=64)
>>> h.n_regions(), h.failed_regions, int((h.scribble_mask > 0).sum()), int((h.dot_mask > 0).sum())
(1701, 0, 2926, 1698)
>>> h.region.shape, h.scribble_mask.shape, h.dot_color.shape, h.region.dtype
((64, 64, 3), (64, 64), (64, 64, 3), dtype('uint8'))
```
About 70 s with FilFinder on this 800 × 1200 image (2 CPU threads), 4 s with `path_method="geodesic"`.

### `HintResult`

The container returned by `generate_hints`. Fields (all `uint8`, BGR): `region` (S×S×3 region-id colours),
`scribble_mask` (S×S, {0, 255}), `scribble_color` (S×S×3, region-mean colour on scribble pixels), `flatten` (S×S×3,
region-mean colour everywhere), `dot_mask` (S×S, one pixel per region), `dot_color`; plus `size`, `failed_regions`,
`path_method`, `dot_method`.

- **`at_ratio(alpha, hint_type="scribble", resize_to=None, tie_break="default") -> (color, mask)`** — the hints of
  the largest `int(n_regions * alpha)` regions (the paper's size-ordered selection). `tie_break="stable"` orders
  equal-area regions by label instead of NumPy's default argsort order. `resize_to=N` upsamples both arrays to N × N
  with nearest neighbour.
  ```python
  >>> [(a, int((h.at_ratio(a)[1] > 0).sum())) for a in (0.0, 0.1, 1.0)]
  [(0.0, 0), (0.1, 878), (1.0, 2926)]
  >>> color, mask = h.at_ratio(0.1, hint_type="dot", resize_to=256); color.shape, int((mask > 0).sum())
  ((256, 256, 3), 2688)          # 168 dots x 4x4 blocks
  ```
- **`n_regions() -> int`** — `1701` here.
- **`save(stem) -> dict`** — writes the six canonical files read by the DetFill loader; the parent directory is
  created when missing and a file that cannot be written raises `OSError` (the returned paths always exist):
  ```python
  >>> h.save("out/4731016.image")                       # "out/" did not exist
  {'region': 'out/4731016.image_region64.png', 'scribble_mask': 'out/4731016.image_scribble_mask64.png',
   'scribble_col': 'out/4731016.image_scribble_col64.png', 'flatten_img': 'out/4731016.image_flatten_img64.png',
   'dot_mask': 'out/4731016.image_dot_mask64.png', 'dot_col': 'out/4731016.image_dot_col64.png'}
  ```
- **`write_image(path, array) -> path`** — the `cv2.imwrite` used by `save` and by `evaluate_colorizer`: creates the
  directory and raises `OSError` instead of returning `False`.

### `segment_regions(image, segmenter="felzenszwalb", **seg_kwargs) -> ndarray`

Segmentation only: an `HxWx3 uint8` region-colour map at the input resolution (every region a unique colour).
```python
>>> reg = hintauc.segment_regions("replicability/data/4731016.image.png"); reg.shape
(1200, 800, 3)
>>> len(np.unique(hintauc.region_ids(reg)))
1527                                          # regions before the 64-px map splits disconnected parts
>>> len(np.unique(hintauc.region_ids(hintauc.segment_regions(small, "slic", n_segments=100))))
58
```

### `region_ids(region, check=True) -> ndarray`

Decodes a region-colour map into an `int64` id map with the loader's base-255 rule (`c0·255² + c1·255 + c2`). The
rule is only injective for channel values below 255 — the maps written by the library and the stored paper maps —
so with `check=True` a map whose colours collide raises `ValueError`.
```python
>>> ids = hintauc.region_ids(h.region); ids.shape, ids.dtype, len(np.unique(ids))
((64, 64), dtype('int64'), 1701)
>>> bad = np.zeros((8, 8, 3), np.uint8); bad[:, :4] = (0, 255, 0); bad[:, 4:] = (1, 0, 0)   # both decode to 65025
>>> hintauc.region_ids(bad)
ValueError: region map: two different colours decode to the same region id under the base-255 encoding ...
```

### `reencode_region_map(region) -> ndarray`

Re-encodes an arbitrary region-colour map with the library's collision-free colours (one id per distinct colour,
numbered in sorted colour order). `generate_hints(region_map=...)` applies it when needed.
```python
>>> len(np.unique(hintauc.region_ids(hintauc.reencode_region_map(bad))))
2

### `geodesic_longest_path(skeleton) -> PathResult`

Longest geodesic path on a binary skeleton (the `geodesic` backend): the exact geodesic diameter of the 8-connected
skeleton graph, for tree-shaped components (start points = endpoints) and for components with cycles (every pixel is
a start point). `PathResult` has `mask` (bool array), `length` (1 per orthogonal step, √2 per diagonal), `start`,
`end`. `hintauc.longest_path.longest_path_pixels(skeleton)` returns the mask as `{0, 1} uint8`.
```python
>>> skel = np.zeros((9, 9), np.uint8); skel[4, 1:8] = 1; skel[1:8, 4] = 1     # a cross
>>> r = hintauc.geodesic_longest_path(skel); r.length, int(r.mask.sum()), r.start, r.end
(6.0, 7, (1, 4), (4, 1))
```

## Evaluation

### `Evaluator(metrics=DEFAULT_METRICS, device=None, resize=256)`

Loads the metric backbones once and scores prediction/ground-truth pairs: `ev(pred, gt) -> {metric: float}`.
`pred`/`gt` are paths or RGB arrays (uint8, or float in [0, 1]). `device` is `"cuda"`/`"cpu"` (default: CUDA when
available). `ev.backend()` reports the implementation in use — `{'resize': 'torchvision' | 'pillow', 'ssim':
'torchmetrics' | 'scikit-image', ...versions}` — and is written into every protocol record; `ev.resize_backend()` and
`ev.ssim_backend()` return the two names.

| Metric | Backend | Notes |
|---|---|---|
| `mse`, `psnr`, `mae` | NumPy | on [0, 1] images, 256 × 256; the resize is torchvision's antialiased bilinear (paper) or Pillow's float32 bilinear (agree to ~1e-6) |
| `ssim`, `ms_ssim` | torchmetrics (1.4.0 for the published SSIM values) | scikit-image fallback with the same Gaussian window (a few 1e-4 away), with a warning |
| `deltae` | CIEDE2000 (scikit-image) | mean colour difference |
| `lpips`, `lpips_vgg` | `lpips` (AlexNet / VGG) | inputs in [−1, 1] |
| `openclip` | OpenCLIP ViT-B/32 (LAION-2B) | `(cosine + 1) / 2` |
| `dino` | DINOv2-base (`transformers`) | `(cosine + 1) / 2` |
| `dreamsim` | DreamSim | distance; weights cached under `~/.cache/hintauc` |
| `dists` | `dists-pytorch` | added after the paper |

```python
>>> ev = hintauc.Evaluator(metrics=("mse", "psnr", "ssim"))
>>> ev(gt, gt)                                            # identical images
{'mse': 0.0, 'psnr': 100.0, 'ssim': 1.0}
>>> ev("shifted_by_16px.png", gt)
{'mse': 0.0547, 'psnr': 12.6217, 'ssim': 0.1856}
>>> ev(shifted_bgr[:, :, ::-1], gt_bgr[:, :, ::-1])        # arrays (RGB) give the same numbers as files
{'mse': 0.0547, 'psnr': 12.6217, 'ssim': 0.1856}
>>> ev(np.full((16, 16, 3), 0.5, np.float32), np.zeros((16, 16, 3), np.uint8))["mse"]   # float in [0, 1] = mid gray
0.2520                                                     # (128/255)^2; before 0.3.1 the array was cast to 0
>>> ev(np.full((16, 16, 3), 128.0, np.float32), np.zeros((16, 16, 3), np.uint8))
ValueError: image: floating-point images must be in [0, 1] (got min 128, max 128); pass uint8 (0..255) or scale ...
>>> hintauc.Evaluator(metrics=hintauc.ALL_METRICS, device="cpu")("4942016.image.png", gt)
{'mse': 0.2023, 'psnr': 6.9396, 'ssim': 0.0784, 'mae': 0.3651, 'ms_ssim': 0.0, 'deltae': 33.2338,
 'lpips': 0.6179, 'lpips_vgg': 0.6712, 'dists': 0.325, 'openclip': ..., 'dino': ..., 'dreamsim': ...}
```

### `evaluate_pair(pred, gt, metrics=("mse", "psnr", "ssim"), device=None, resize=256) -> dict`

One-shot version of `Evaluator` for a single pair.
```python
>>> hintauc.evaluate_pair("4942016.image.png", gt, metrics=("mse", "mae", "deltae"))
{'mse': 0.2023, 'mae': 0.3651, 'deltae': 33.2338}
```

### `evaluate_dirs(pred_dir, gt_dir, evaluator=None, metrics=DEFAULT_METRICS, pairing="sorted", limit=0, allow_missing=False) -> dict`

Mean metrics over two directories. `pairing="sorted"` pairs files in sorted order and refuses different file counts;
`pairing="name"` pairs files with identical names and refuses a ground truth without a prediction or a prediction
without a ground truth (both would silently change the evaluated set). `allow_missing=True` evaluates the common
files and warns about the rest. `limit=N` uses the first N pairs.
```python
>>> hintauc.evaluate_dirs("pred/", "gt/", metrics=("mse", "psnr"), pairing="name")
{'mse': 0.0437, 'psnr': 13.7315}
>>> hintauc.evaluate_dirs("pred_2_files/", "gt_3_files/", metrics=("mse",))
ValueError: pred/gt counts differ (2 in pred_2_files/ vs 3 in gt_3_files/); use pairing='name' for files that share names, or align the directories
>>> hintauc.evaluate_dirs("pred_2_files/", "gt_3_files/", metrics=("mse",), pairing="name")
ValueError: pred_2_files/: 1 ground-truth image(s) have no prediction (2.png) and 0 prediction(s) have no ground truth (); pass allow_missing=True (CLI: --allow-missing) to evaluate the common files only
```

### `list_pairs(pred_dir, gt_dir, pairing="sorted", allow_missing=False, limit=0) -> [(pred, gt, name), ...]`, `evaluate_pairs(pairs, evaluator) -> dict`

The pairing step and the scoring step of `evaluate_dirs` as separate functions (`evaluate_hint_curve` uses them to
check all ratio directories before scoring anything).

### `evaluate_set(pred_dir, gt_dir, metrics=SET_METRICS, device=None, resize=256, kid_subset_size=None) -> dict`

FID / KID between two image sets (Inception-v3 features; needs `torchmetrics` and `torch-fidelity`). FID needs a few
hundred images per side to be meaningful; KID is preferable for small sets.
```python
>>> hintauc.evaluate_set("pred/", "gt/", device="cpu", kid_subset_size=2)      # 4 images per side
{'fid': 21.223, 'kid': 0.018, 'kid_std': 0.007, 'n_pred': 4, 'n_gt': 4}
```

## Hint-AUC

### `check_alphas(alphas, full_range=True) -> list`, `alpha_dir_name(alpha) -> str`

Validation of a hint-ratio grid (at least two finite values in [0, 1], strictly increasing, no duplicates, and with
`full_range` from 0 to 1) and the directory name of a ratio (shortest decimal exact to 1e-6, at least two decimals).
```python
>>> hintauc.check_alphas((0, .5, 1)), hintauc.check_alphas((0.2, 0.7), full_range=False)
([0.0, 0.5, 1.0], [0.2, 0.7])
>>> hintauc.check_alphas((0, .5, .5, 1))
ValueError: alphas must be strictly increasing without duplicates, got [0.0, 0.5, 0.5, 1.0]
>>> [hintauc.alpha_dir_name(a) for a in (0, 0.01, 0.1, 0.001, 1)]
['0.00', '0.01', '0.10', '0.001', '1.00']
```

### `trapz(xs, ys) -> float`, `hint_auc(scores_by_alpha) -> float`, `hint_auc_table(metrics_by_alpha, metrics=None) -> dict`

The trapezoidal integral over the hint ratio; the paper reports it over `DEFAULT_ALPHAS`. `xs` must be strictly
increasing (a repeated ratio would be integrated twice).
```python
>>> hintauc.trapz([0, 0.5, 1], [0, 1, 2])
1.0
>>> hintauc.hint_auc({0.0: 10.5, 0.01: 13.9, 0.03: 16.0, 0.05: 17.1, 0.10: 18.6, 0.25: 20.3, 0.50: 21.4, 1.00: 22.3})
20.7
>>> hintauc.hint_auc_table({0.0: {"psnr": 10.0, "lpips": 0.4}, 1.0: {"psnr": 20.0, "lpips": 0.2}})
{'lpips': 0.3, 'psnr': 15.0}
```

### `evaluate_hint_curve(preds_by_alpha, gt_dir, evaluator=None, metrics=("mse", "psnr", "ssim"), pairing="sorted", limit=0, allow_missing=False, full_range=False, manifest=None, pred_root=None) -> dict`

Per-ratio means and Hint-AUC from one prediction directory per ratio. Before scoring, every ratio directory is
paired with the ground truth and the ratios are compared with each other: with `pairing="name"` all ratios must hold
the same image names, with `"sorted"` the same number of files; otherwise `ValueError` names the ratio and the files
(`allow_missing=True`: the common subset, with a warning). `full_range=True` also requires a grid from 0 to 1. With
`manifest=hintauc.read_manifest(pred_root)` the directories, names and ground-truth files come from the manifest that
`evaluate_colorizer` wrote (`curve_from_manifest`): a ratio directory on disk that the manifest does not list is an
error (predictions of another run), and the result carries the run's attributes (`oracle`, `tie_break`, ...) under
`"run"`.
```python
>>> hintauc.evaluate_hint_curve({0.0: "pred/0.00", 0.1: "pred/0.10", 1.0: "pred/1.00"}, "gt/", metrics=("mse", "psnr"), pairing="name")
{'alphas': [0.0, 0.1, 1.0], 'n_images': 1, 'names': ['4731016.image.png'],
 'per_alpha': {0.0: {'mse': 0.2127, 'psnr': 6.7228}, 0.1: {'mse': 0.0890, 'psnr': 10.5046}, 1.0: {'mse': 0.0263, 'psnr': 15.7989}},
 'hint_auc': {'mse': 0.0670, 'psnr': 12.6979}}
>>> hintauc.evaluate_hint_curve({0.0: "p/0.00", 0.5: "p/0.50", 1.0: "p/1.00"}, "gt2/", metrics=("mse",), pairing="name")  # b.png missing at 0.50
ValueError: p/0.50: 1 ground-truth image(s) have no prediction (b.png) and 0 prediction(s) have no ground truth (); pass allow_missing=True (CLI: --allow-missing) to evaluate the common files only
```

## Evaluating your own model

### `evaluate_colorizer(colorize, samples, alphas=DEFAULT_ALPHAS, hint_type="scribble", metrics=DEFAULT_METRICS, evaluator=None, size=64, path_method="filfinder", dot_method="medoid", save_dir=None, oracle=False, tie_break="stable", verbose=False) -> dict`

Runs a Python callable at every hint ratio and integrates the Hint-AUC. `samples` yields `(line_art, ground_truth)`
paths or arrays. For each ratio the callable receives one dict and returns `HxWx3 uint8` BGR of the ground-truth size:

| key | value |
|---|---|
| `line_art` | `HxW uint8` |
| `hint_color`, `hint_mask` | `HxWx3 uint8` BGR and `HxW {0, 255}`, upsampled to the image size |
| `alpha`, `hint_type`, `index`, `name` | the ratio, `"scribble"` / `"dot"`, sample index, output file name |
| `hints`, `ground_truth`, `n_regions` | **only with `oracle=True`**: the `HintResult`, the `HxWx3` BGR ground truth, the region count |

A model adapter therefore cannot read the ground truth by accident; `oracle=True` is for reference baselines such as
`hint_fill_colorizer`. `alphas` goes through `check_alphas` (strictly increasing from 0 to 1). Regions of equal area
are ordered by `tie_break`: `"stable"` (ascending label, identical on every machine) or `"default"` (the DetFill
loader's NumPy argsort order, which differs between NumPy builds and between CPUs with and without AVX-512). `save_dir` writes the
outputs as `save_dir/<alpha_dir_name(alpha)>/<name>.png` (the layout of `hintauc curve`; `0.00 … 1.00` for the paper
grid, `0.001` for finer ratios) plus `save_dir/manifest.json`: the exact ratios and directory names, the prediction
names with the ground-truth file each belongs to, `oracle`, `tie_break`, `path_method`, `dot_method`, `hint_map_size`.
A `save_dir` that already holds ratio directories or a manifest of another run raises `ValueError` (`overwrite=True`
removes them first); a write that fails raises `OSError`, and two samples with the same file name raise
`ValueError`. The result holds `alphas`, `n_images`, `names`, `per_alpha`, `hint_auc`, a `protocol` record and, with
`save_dir`, the `manifest` path.
```python
>>> res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, [("sketch/4731016.png", "4731016.image.png")],
...                                  alphas=(0.0, 0.1, 1.0), metrics=("mse", "psnr"), save_dir="pred", oracle=True)
>>> res["per_alpha"], res["hint_auc"]
({0.0: {'mse': 0.2127, 'psnr': 6.7228}, 0.1: {'mse': 0.0890, 'psnr': 10.5046}, 1.0: {'mse': 0.0263, 'psnr': 15.7989}},
 {'mse': 0.0670, 'psnr': 12.6979})
>>> res["names"], sorted(os.listdir("pred"))
(['4731016.image.png'], ['0.00', '0.10', '1.00', 'manifest.json'])
>>> res["protocol"]["backend"]
{'resize': 'torchvision', 'resize_filter': 'bilinear, antialiased, float32 in [0, 1]', 'ssim': 'torchmetrics',
 'pillow': '10.4.0', 'torchvision': '0.20.1+cu124', 'torchmetrics': '1.4.0.post0', 'scikit-image': '0.24.0'}
>>> hintauc.evaluate_colorizer(model, samples, alphas=(0, .5, .5, 1))
ValueError: alphas must be strictly increasing without duplicates, got [0.0, 0.5, 0.5, 1.0]
```
A callable that returns the wrong shape or dtype raises `ValueError`.

### `hint_fill_colorizer(sample) -> ndarray`

Oracle reference baseline used by the demo, the quick start and the tests: hinted *regions* are filled with their
hint colour (it reads the ground-truth region map, hence `oracle=True`; without it a `KeyError` explains why), other
regions stay light gray, the line art is multiplied on top. Its PSNR rises from 6.7 dB (no hints) to 15.8 dB (all
hints) on the example above. Its scores are not comparable with a real model's.

### `hint_inputs(hints, alpha, hint_type, height, width, tie_break="stable") -> (color, mask)`

The hint colour and mask at `alpha` (equal-area regions in stable order by default), upsampled with nearest neighbour
to an arbitrary `height × width`.
```python
>>> c, m = hintauc.hint_inputs(h, 0.1, "scribble", 512, 512); c.shape, int((m > 0).sum())
((512, 512, 3), 56128)
```

### `protocol_record(evaluator=None, alphas=DEFAULT_ALPHAS, hint_type=None, hint_size=None, path_method=None, dot_method=None, **extra) -> dict`

The settings and library versions that make a score interpretable: `protocol`, `paper_grid` (True when `alphas` is the
paper grid), `alphas`, `hint_selection`, `aggregation`, `metric_resize`, `metric_input`, `metrics`, `backend` (the
resize and SSIM implementation in use, with versions), `hint_type`, `hint_map_size`, `path_method`, `dot_method`,
`versions`, plus any `extra` keys (`evaluate_colorizer` adds `oracle` and `tie_break`). Written by `evaluate_colorizer`, `hintauc curve` and
`reproduce/scripts/eval_per_ratio.py`.

### `run_demo(out=None, path_method="geodesic", metrics=("mse", "psnr", "ssim"), ...) -> dict`, `synthetic_illustration(seed=0, size=128) -> ndarray`

The self-contained example behind `hintauc demo` (`hintauc/demo.py`): a synthetic illustration (three flat colour
rectangles), its Canny line art, deterministic hints, the oracle reference colorizer at every ratio of the paper grid,
Hint-AUC. With `out` it writes the images, `pred/<ratio>/`, `result.json` and `curve.png`. Its numbers are recorded
in `examples/expected_numbers.json` ([Evaluate your own model](evaluate_your_model.md#which-numbers-to-expect)).

### `plot_curves(per_alpha, path, title=None, metrics=None) -> bool`

Metric-versus-ratio curves as a PNG (one panel per metric). Returns `False` when matplotlib is not installed.

### `file_sha256(path) -> str`

SHA-256 of a file, for recording which checkpoint produced a result (`eval_per_ratio.py --ckpt`).

## Command line

`hintauc --help` lists four commands. Every command exits with status 1 and a one-line `error: ...` message when an
input check fails (missing files, mismatched directories, invalid grids, unwritable outputs).

### `hintauc generate IMAGE [-o STEM] [--size 64] [--segmenter ...] [--ratio A] [--hint_type scribble|dot] [--path_method filfinder|geodesic] [--tie_break default|stable] [--dot_method medoid|mean|nearest_mean] [-v]`

Writes the six canonical files next to `STEM` (missing directories are created; nothing is reported as written unless
it is on disk); with `--ratio` also the masked hints at that ratio (`*_r10.png` for 0.1).
```
$ hintauc generate 4731016.image.png -o out/4731016.image --ratio 0.1
{"n_regions": 1701, "failed_regions": 0, "path_method": "filfinder", "dot_method": "medoid",
 "outputs": {"region": "out/4731016.image_region64.png", "scribble_mask": "...", ...}}
$ ls out/
4731016.image_dot_col64.png  4731016.image_dot_mask64.png  4731016.image_flatten_img64.png  4731016.image_region64.png
4731016.image_scribble_col64.png  4731016.image_scribble_col64_r10.png  4731016.image_scribble_mask64.png  4731016.image_scribble_mask64_r10.png
```

### `hintauc eval PRED GT [--metrics ...] [--set_metrics [fid kid]] [--pairing sorted|name] [--allow-missing] [--resize 256] [--device DEV] [--limit N]`

Scores one image pair or two directories (unmatched files are an error unless `--allow-missing`); prints JSON.
```
$ hintauc eval pred/ gt/ --metrics mse psnr --pairing name
{
 "mse": 0.04373995028436184,
 "psnr": 13.731498107027996
}
```

### `hintauc curve PRED_ROOT GT [--metrics ...] [--pairing name|sorted] [--allow-missing] [--ignore-manifest] [--resize 256] [--device DEV] [--limit N] [--json FILE] [--plot FILE.png]`

Hint-AUC from one sub-directory per ratio (`PRED_ROOT/0.00`, `PRED_ROOT/0.01`, …; `1%` … `100%` and bare percentages
above 1 such as `10` are read as well; two directories denoting the same ratio are an error). When `PRED_ROOT`
holds the `manifest.json` of `evaluate_colorizer`, the ratio directories, the prediction names and their ground-truth
files come from it: a ratio directory that the manifest does not list stops the command (predictions of another run;
`--ignore-manifest` scores every directory on disk instead), and the run's attributes (`oracle`, `tie_break`, ...)
are copied into the protocol record. Without a manifest, files are paired by name; every ratio directory must hold
the same images as the ground truth and as the other ratios, otherwise the command stops and names the files
(`--allow-missing` evaluates the common subset). The JSON on stdout carries the number of images; `--json` also
writes their names.
```
$ hintauc curve pred/ gt/ --metrics mse psnr --json result.json --plot curve.png
{
 "alphas": [0.0, 0.1, 1.0],
 "n_images": 1,
 "per_alpha": {"0.0": {"mse": 0.2127, "psnr": 6.7228}, "0.1": {"mse": 0.0890, "psnr": 10.5046}, "1.0": {"mse": 0.0263, "psnr": 15.7989}},
 "hint_auc": {"mse": 0.0670, "psnr": 12.6979},
 "protocol": {"protocol": "hint-auc/v1", "paper_grid": false, "alphas": [0.0, 0.1, 1.0], "backend": {"resize": "torchvision", ...}, ...}
}
$ hintauc curve pred_missing_b_at_050/ gt/ --metrics mse
error: pred_missing_b_at_050/0.50: 1 ground-truth image(s) have no prediction (b.png) and 0 prediction(s) have no ground truth (); pass allow_missing=True (CLI: --allow-missing) to evaluate the common files only
```

### `hintauc demo [--out DIR] [--path_method geodesic|filfinder] [--metrics ...] [--json]`

The self-contained example (no data): prints the per-ratio metric table, the Hint-AUC and the backend in use; `--out`
writes the images, `result.json` and `curve.png`. About 10 s on a laptop CPU.
```
$ hintauc demo
hint ratio       mse      psnr      ssim
    0.00      0.1034    9.8537    0.7038
    ...
    1.00      0.0136   18.6650    0.8248
Hint-AUC      0.0359   15.9028    0.7790
backend: resize=torchvision, ssim=torchmetrics
```

## What was verified

Every public symbol of `hintauc.__all__` was called once on the shipped illustration in the reference environment
(CPU for the metric backbones): `generate_hints` with both path methods, all three dot rules, the three segmenters,
a precomputed region map and array input; `HintResult.at_ratio` (both hint types, `resize_to`, `tie_break`), `save`,
`n_regions`; `segment_regions`, `region_ids`, `reencode_region_map`, `geodesic_longest_path`, `longest_path_pixels`;
`Evaluator` with all twelve metrics for file and array inputs (identical numbers), `evaluate_pair`, `evaluate_dirs`
(both pairings, `limit`, the mismatch errors), `evaluate_set` (FID / KID); `trapz`, `hint_auc`, `hint_auc_table`,
`check_alphas`, `alpha_dir_name`, `evaluate_hint_curve`; `evaluate_colorizer` with `hint_fill_colorizer`,
`hint_inputs`, `plot_curves`, `protocol_record`, `file_sha256`, `run_demo`; and the four commands.

Two different scopes: that manual pass ran the perceptual metrics (LPIPS, OpenCLIP, DINOv2, DreamSim, DISTS, FID,
KID) with their downloaded weights on the reference machine. The continuous tests (`tests/`, CPU, no downloads) cover
the pixel metrics, the hint generation, the Hint-AUC arithmetic, the command line and the input checks (missing
predictions, duplicate ratios, percent directories, unwritable outputs, float and non-RGB inputs, region-id
collisions, cyclic skeletons, stale prediction directories, manifest-driven re-scoring, the evaluation script's image-set
and resume checks, the training loop's budget and error handling) on synthetic images, and check the recorded demo
and quick-start numbers with both backends; the perceptual metrics are exercised there only through the registry
and the `torchmetrics` SSIM/MS-SSIM, not with the downloaded models.
