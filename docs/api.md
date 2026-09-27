# API reference: the `hintauc` library

Every public function and command, with its role, arguments, and an input/output example. All examples were run on
the illustration shipped in `replicability/data/4731016.image.png` (800 × 1200 px, 1,701 regions) with `hintauc` 0.3.0,
Python 3.9, NumPy 1.26, OpenCV 4.11, scikit-image 0.24, PyTorch 2.5.1, torchmetrics 1.4.0; the numbers are the actual
outputs of that run.

```bash
pip install hintauc                 # hints, pixel metrics, Hint-AUC
pip install "hintauc[perceptual]"   # + LPIPS / OpenCLIP / DINOv2 / DreamSim / DISTS / FID / KID
```

**Conventions.** Images are `HxWx3 uint8`. Everything produced by the hint generator (`HintResult` arrays, the
callback inputs of `evaluate_colorizer`) is **BGR** (OpenCV order, `cv2.imread`). `Evaluator` accepts file paths (decoded
as RGB) or arrays that it takes as **RGB**, so pass `arr[:, :, ::-1]` when the array comes from OpenCV. Metrics are
computed on images resized to 256 × 256 with values in [0, 1].

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

## Hint generation

### `generate_hints(image, size=64, segmenter="felzenszwalb", region_map=None, verbose=False, path_method="filfinder", dot_method="medoid", **seg_kwargs) -> HintResult`

Turns a colour image into the deterministic hints of the paper: Felzenszwalb regions, one scribble (longest path of
the region skeleton) and one dot per region, coloured with the region mean, at `size` × `size`.

| Argument | Meaning |
|---|---|
| `image` | path or `HxWx3 uint8` BGR array; segmentation runs at the input resolution |
| `size` | hint-map resolution (paper: 64) |
| `segmenter` | `felzenszwalb` (paper), `slic`, `quickshift`; `**seg_kwargs` override the segmenter parameters |
| `region_map` | a precomputed region-colour map (e.g. from DanbooRegion); then `segmenter` is ignored |
| `path_method` | `filfinder` (paper; FilFinder's medial axis breaks ties randomly, so regeneration is not bit-exact) or `geodesic` (dependency-free, fully deterministic; slightly different scribbles, never mix the two in one evaluation) |
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
- **`save(stem) -> dict`** — writes the six canonical files read by the DetFill loader:
  ```python
  >>> h.save("out/4731016.image")
  {'region': 'out/4731016.image_region64.png', 'scribble_mask': 'out/4731016.image_scribble_mask64.png',
   'scribble_col': 'out/4731016.image_scribble_col64.png', 'flatten_img': 'out/4731016.image_flatten_img64.png',
   'dot_mask': 'out/4731016.image_dot_mask64.png', 'dot_col': 'out/4731016.image_dot_col64.png'}
  ```

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

### `region_ids(region) -> ndarray`

Decodes a region-colour map into an `int64` id map with the loader's base-255 rule.
```python
>>> ids = hintauc.region_ids(h.region); ids.shape, ids.dtype, len(np.unique(ids))
((64, 64), dtype('int64'), 1701)
```

### `geodesic_longest_path(skeleton) -> PathResult`

Longest geodesic path on a binary skeleton (the `geodesic` backend). `PathResult` has `mask` (bool array),
`length` (1 per orthogonal step, √2 per diagonal), `start`, `end`. `hintauc.longest_path.longest_path_pixels(skeleton)`
returns the mask as `{0, 1} uint8`.
```python
>>> skel = np.zeros((9, 9), np.uint8); skel[4, 1:8] = 1; skel[1:8, 4] = 1     # a cross
>>> r = hintauc.geodesic_longest_path(skel); r.length, int(r.mask.sum()), r.start, r.end
(6.0, 7, (1, 4), (4, 1))
```

## Evaluation

### `Evaluator(metrics=DEFAULT_METRICS, device=None, resize=256)`

Loads the metric backbones once and scores prediction/ground-truth pairs: `ev(pred, gt) -> {metric: float}`.
`pred`/`gt` are paths or RGB arrays. `device` is `"cuda"`/`"cpu"` (default: CUDA when available).

| Metric | Backend | Notes |
|---|---|---|
| `mse`, `psnr`, `mae` | NumPy | on [0, 1] images, 256 × 256 |
| `ssim`, `ms_ssim` | torchmetrics (1.4.0 for the published SSIM values) | scikit-image fallback with the same Gaussian window, with a warning |
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

### `evaluate_dirs(pred_dir, gt_dir, evaluator=None, metrics=DEFAULT_METRICS, pairing="sorted", limit=0) -> dict`

Mean metrics over two directories. `pairing="sorted"` pairs files in sorted order and refuses different file counts;
`pairing="name"` pairs files with identical names and warns about predictions without a partner. `limit=N` uses the
first N pairs.
```python
>>> hintauc.evaluate_dirs("pred/", "gt/", metrics=("mse", "psnr"), pairing="name")
{'mse': 0.0437, 'psnr': 13.7315}
>>> hintauc.evaluate_dirs("pred_2_files/", "gt_3_files/", metrics=("mse",))
ValueError: pred/gt counts differ (2 vs 3); use pairing='name' or align the directories
```

### `evaluate_set(pred_dir, gt_dir, metrics=SET_METRICS, device=None, resize=256, kid_subset_size=None) -> dict`

FID / KID between two image sets (Inception-v3 features; needs `torchmetrics` and `torch-fidelity`). FID needs a few
hundred images per side to be meaningful; KID is preferable for small sets.
```python
>>> hintauc.evaluate_set("pred/", "gt/", device="cpu", kid_subset_size=2)      # 4 images per side
{'fid': 21.223, 'kid': 0.018, 'kid_std': 0.007, 'n_pred': 4, 'n_gt': 4}
```

## Hint-AUC

### `trapz(xs, ys) -> float`, `hint_auc(scores_by_alpha) -> float`, `hint_auc_table(metrics_by_alpha, metrics=None) -> dict`

The trapezoidal integral over the hint ratio; the paper reports it over `DEFAULT_ALPHAS`.
```python
>>> hintauc.trapz([0, 0.5, 1], [0, 1, 2])
1.0
>>> hintauc.hint_auc({0.0: 10.5, 0.01: 13.9, 0.03: 16.0, 0.05: 17.1, 0.10: 18.6, 0.25: 20.3, 0.50: 21.4, 1.00: 22.3})
20.7
>>> hintauc.hint_auc_table({0.0: {"psnr": 10.0, "lpips": 0.4}, 1.0: {"psnr": 20.0, "lpips": 0.2}})
{'lpips': 0.3, 'psnr': 15.0}
```

### `evaluate_hint_curve(preds_by_alpha, gt_dir, evaluator=None, metrics=("mse", "psnr", "ssim"), pairing="sorted", limit=0) -> dict`

Per-ratio means and Hint-AUC from one prediction directory per ratio.
```python
>>> hintauc.evaluate_hint_curve({0.0: "pred/0.00", 0.1: "pred/0.10", 1.0: "pred/1.00"}, "gt/", metrics=("mse", "psnr"), pairing="name")
{'per_alpha': {0.0: {'mse': 0.2127, 'psnr': 6.7228}, 0.1: {'mse': 0.0890, 'psnr': 10.5069}, 1.0: {'mse': 0.0263, 'psnr': 15.7989}},
 'hint_auc': {'mse': 0.0670, 'psnr': 12.6991}}
```

## Evaluating your own model

### `evaluate_colorizer(colorize, samples, alphas=DEFAULT_ALPHAS, hint_type="scribble", metrics=DEFAULT_METRICS, evaluator=None, size=64, path_method="filfinder", dot_method="medoid", save_dir=None, verbose=False) -> dict`

Runs a Python callable at every hint ratio and integrates the Hint-AUC. `samples` yields `(line_art, ground_truth)`
paths or arrays. For each ratio the callable receives one dict and returns `HxWx3 uint8` BGR of the ground-truth size:

| key | value |
|---|---|
| `line_art` | `HxW uint8` |
| `hint_color`, `hint_mask` | `HxWx3 uint8` BGR and `HxW {0, 255}`, upsampled to the image size |
| `alpha`, `hint_type`, `index` | the ratio, `"scribble"` / `"dot"`, sample index |
| `hints`, `ground_truth` | the `HintResult`, the `HxWx3` BGR ground truth |

`save_dir` writes the outputs as `save_dir/<alpha:.2f>/<name>.png` (the layout of `hintauc curve`). The result holds
`alphas`, `n_images`, `per_alpha`, `hint_auc` and a `protocol` record.
```python
>>> res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, [("sketch/4731016.png", "4731016.image.png")],
...                                  alphas=(0.0, 0.1, 1.0), metrics=("mse", "psnr"), save_dir="pred")
>>> res["per_alpha"], res["hint_auc"]
({0.0: {'mse': 0.2127, 'psnr': 6.7228}, 0.1: {'mse': 0.0890, 'psnr': 10.5069}, 1.0: {'mse': 0.0263, 'psnr': 15.7989}},
 {'mse': 0.0670, 'psnr': 12.6991})
>>> res["protocol"]["versions"]
{'hintauc': '0.3.0', 'python': '3.9.16', 'numpy': '1.26.4', 'opencv': '4.11.0', 'scikit-image': '0.24.0', 'torch': '2.5.1+cu124', 'torchmetrics': '1.4.0.post0'}
```
A callable that returns the wrong shape or dtype raises `ValueError`; `alphas` must start at 0 and end at 1.

### `hint_fill_colorizer(sample) -> ndarray`

Reference baseline used by the quick start and the tests: hinted regions are filled with their hint colour, other
regions stay light gray, the line art is multiplied on top. Its PSNR rises from 6.7 dB (no hints) to 15.8 dB (all
hints) on the example above.

### `hint_inputs(hints, alpha, hint_type, height, width) -> (color, mask)`

The hint colour and mask at `alpha`, upsampled with nearest neighbour to an arbitrary `height × width`.
```python
>>> c, m = hintauc.hint_inputs(h, 0.1, "scribble", 512, 512); c.shape, int((m > 0).sum())
((512, 512, 3), 56192)
```

### `protocol_record(evaluator=None, alphas=DEFAULT_ALPHAS, hint_type=None, hint_size=None, path_method=None, dot_method=None, **extra) -> dict`

The settings and library versions that make a score interpretable: `protocol`, `paper_grid` (True when `alphas` is the
paper grid), `alphas`, `hint_selection`, `aggregation`, `metric_resize`, `metric_input`, `metrics`, `hint_type`,
`hint_map_size`, `path_method`, `dot_method`, `versions`, plus any `extra` keys. Written by `evaluate_colorizer`,
`hintauc curve` and `reproduce/scripts/eval_per_ratio.py`.

### `plot_curves(per_alpha, path, title=None, metrics=None) -> bool`

Metric-versus-ratio curves as a PNG (one panel per metric). Returns `False` when matplotlib is not installed.

### `file_sha256(path) -> str`

SHA-256 of a file, for recording which checkpoint produced a result (`eval_per_ratio.py --ckpt`).

## Command line

`hintauc --help` lists three commands.

### `hintauc generate IMAGE [-o STEM] [--size 64] [--segmenter ...] [--ratio A] [--hint_type scribble|dot] [--path_method filfinder|geodesic] [--tie_break default|stable] [--dot_method medoid|mean|nearest_mean] [-v]`

Writes the six canonical files next to `STEM`; with `--ratio` also the masked hints at that ratio (`*_r10.png` for 0.1).
```
$ hintauc generate 4731016.image.png -o out/4731016.image --ratio 0.1
{"n_regions": 1701, "failed_regions": 0, "path_method": "filfinder", "dot_method": "medoid",
 "outputs": {"region": "out/4731016.image_region64.png", "scribble_mask": "...", ...}}
$ ls out/
4731016.image_dot_col64.png  4731016.image_dot_mask64.png  4731016.image_flatten_img64.png  4731016.image_region64.png
4731016.image_scribble_col64.png  4731016.image_scribble_col64_r10.png  4731016.image_scribble_mask64.png  4731016.image_scribble_mask64_r10.png
```

### `hintauc eval PRED GT [--metrics ...] [--set_metrics [fid kid]] [--pairing sorted|name] [--resize 256] [--device DEV] [--limit N]`

Scores one image pair or two directories; prints JSON.
```
$ hintauc eval pred/ gt/ --metrics mse psnr --pairing name
{
 "mse": 0.04373995028436184,
 "psnr": 13.731498107027996
}
```

### `hintauc curve PRED_ROOT GT [--metrics ...] [--pairing name|sorted] [--resize 256] [--device DEV] [--limit N] [--json FILE] [--plot FILE.png]`

Hint-AUC from one sub-directory per ratio (`PRED_ROOT/0.00`, `PRED_ROOT/0.01`, …; `10` is read as 10 %). Files are
paired by name by default; a sub-directory with a different number of images or no common names is an error.
```
$ hintauc curve pred/ gt/ --metrics mse psnr --json result.json --plot curve.png
{
 "alphas": [0.0, 0.1, 1.0],
 "per_alpha": {"0.0": {"mse": 0.2127, "psnr": 6.7228}, "0.1": {"mse": 0.0890, "psnr": 10.5069}, "1.0": {"mse": 0.0263, "psnr": 15.7989}},
 "hint_auc": {"mse": 0.0670, "psnr": 12.6991},
 "protocol": {"protocol": "hint-auc/v1", "paper_grid": false, "alphas": [0.0, 0.1, 1.0], ...}
}
```

## What was verified

Every public symbol of `hintauc.__all__` was called once on the shipped illustration in the reference environment
(CPU for the metric backbones): `generate_hints` with both path methods, all three dot rules, the three segmenters,
a precomputed region map and array input; `HintResult.at_ratio` (both hint types, `resize_to`, `tie_break`), `save`,
`n_regions`; `segment_regions`, `region_ids`, `geodesic_longest_path`, `longest_path_pixels`; `Evaluator` with all
twelve metrics for file and array inputs (identical numbers), `evaluate_pair`, `evaluate_dirs` (both pairings, `limit`,
the count-mismatch error), `evaluate_set` (FID / KID); `trapz`, `hint_auc`, `hint_auc_table`, `evaluate_hint_curve`;
`evaluate_colorizer` with `hint_fill_colorizer`, `hint_inputs`, `plot_curves`, `protocol_record`, `file_sha256`; and the
three commands. The continuous tests (`tests/`, 41 tests) cover the same surface on synthetic images without
downloads.
