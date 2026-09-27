# Evaluate your own model

Hint-AUC scores a hint-based colorization model over the whole range of hint ratios: the model is run with the
deterministic hints of the largest 0 %, 1 %, 3 %, 5 %, 10 %, 25 %, 50 % and 100 % of the regions, each output is
compared with the ground truth, and the metric curve is integrated over the ratio. One image with one set of hints
is therefore not enough; you need one output per hint ratio. Two ways to get there:

| | You provide | Command / call | Good for |
|---|---|---|---|
| **A. Images per hint ratio** | a directory per ratio with your model's outputs | `hintauc curve <pred_root> <gt_dir>` | any framework; run your model wherever it lives |
| **B. A Python function** | a callable that colorizes one image from line art + hints | `hintauc.evaluate_colorizer(fn, samples)` | quick experiments; hints and evaluation in one process |

`pip install "hintauc[perceptual]"` gives all seven metrics of the paper (MSE, PSNR, SSIM, LPIPS, OpenCLIP, DINO,
DreamSim); the plain install has the pixel metrics, `pip install "hintauc[paper]"` adds the exact metric backend of the
paper (torchvision resize, torchmetrics 1.4.0 SSIM). Two worked examples: `hintauc demo` needs no data at all (a
synthetic illustration; about 10 s) and `python examples/quickstart_cpu.py` runs both A and B on the two illustrations
shipped with the repository (about a minute with `--fast`). Both print numbers that are recorded in
`examples/expected_numbers.json` and checked in CI ([which numbers to expect](#which-numbers-to-expect)).

## What the protocol fixes

- **Hints.** `hintauc.generate_hints(ground_truth, size=64)` — Felzenszwalb regions, one scribble (longest skeleton
  path) and one dot per region, coloured with the region mean. At ratio α the largest `int(n_regions * α)` regions keep
  their hints (`HintResult.at_ratio`); regions of equal area are ordered by ascending label (`tie_break="stable"`,
  identical on every machine; `"default"` is the DetFill loader's NumPy argsort order, which depends on the NumPy
  build and on the CPU's SIMD sort path). The 64 × 64 maps are upsampled to the image size with nearest-neighbour
  interpolation (4 × 4 blocks at 256 px, as DetFill was trained).
- **Inputs to the model.** The line art (the paper reports the mean over three extractors: sketch simplification,
  XDoG, SketchKeras), the hint colour image and the hint mask. Arrays from `hintauc` are BGR (OpenCV order).
- **Scoring.** Prediction and ground truth are decoded to 8-bit RGB (files of any mode through Pillow — palette,
  grayscale, CMYK, RGBA — on both backends; arrays as uint8 or float in [0, 1]; 16-bit and float files are rejected),
  converted to float in [0, 1] and resized to 256 × 256 with an antialiased bilinear filter; then MSE, PSNR,
  SSIM (torchmetrics 1.4.0), LPIPS-AlexNet on that image, and OpenCLIP ViT-B/32, DINOv2-base, DreamSim with their own
  preprocessing of the decoded image. The resize is torchvision's
  when torch and torchvision are installed (the paper's evaluator) and Pillow's float32 bilinear filter otherwise; the
  two agree to about 1e-6 per pixel, so MSE and PSNR do not depend on which one is present. SSIM falls back to
  scikit-image without torchmetrics (a few 1e-4 away from torchmetrics 1.4.0, with a warning).
- **Aggregation.** Trapezoidal integral over α ∈ [0, 1] of the per-ratio means; the paper reports mean ± SD over the
  three line-art sources. `result["protocol"]` records the grid, the resize, the metric list, the backend actually
  used (`resize`, `ssim` and their versions) and the library versions next to every score, and states whether the grid
  is the paper grid.

## A. Images per hint ratio

1. Get the hints. For the paper's test split download `test_split_hint_maps_64.tar.gz` (release v1.0) and the line
   art (v1.3), or generate them for your own images with `hintauc generate image.png -o out/image` /
   `hintauc.generate_hints(...)`. `HintResult.at_ratio(alpha, hint_type, resize_to=...)` gives the colour and mask
   at each ratio; `detfill/datasets/custom.py` shows the exact loader used for DetFill.
2. Run your model at every ratio and save the outputs as PNGs named like the ground truth, one directory per ratio:
   ```
   pred_root/0.00/<id>.png   pred_root/0.01/<id>.png   ...   pred_root/1.00/<id>.png
   gt_dir/<id>.png
   ```
   Directory names are the ratios as decimals (`hintauc.alpha_dir_name(0.01) == "0.01"`, `0.001` for finer grids);
   `1%` … `100%` and bare percentages above 1 (`10`) are read as well. Two directories that denote the same ratio
   are an error.
3. Score:
   ```bash
   hintauc curve pred_root gt_dir --metrics mse psnr ssim lpips openclip dino dreamsim --json result.json --plot curve.png
   ```
   Before anything is scored the file sets are checked: every ground truth needs a prediction, every prediction a
   ground truth, and **every ratio directory must hold the same images** — a prediction missing at one ratio would
   otherwise change the evaluated set of that ratio alone and make a failed inference look like a better score.
   Such a mismatch stops the command with the names involved; `--allow-missing` evaluates the images common to all
   ratios instead and lists the rest. Files are paired by name (`--pairing sorted` pairs by sorted order and checks
   the counts instead). The JSON holds the number of images and their names (`--json`), the per-ratio means, the
   Hint-AUC per metric and the protocol record; the PNG holds the curves.

For a DetFill-style output tree (`.../sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/200/`) use
`reproduce/scripts/eval_per_ratio.py`, which also writes the `per_ratio_summary.csv` format of the released metric files
and can record the checkpoint hash (`--ckpt`).

## B. A Python function

```python
import hintauc

def my_model(sample):
    # sample["line_art"]   HxW  uint8         sample["hint_color"]  HxWx3 uint8 BGR
    # sample["hint_mask"]  HxW  {0, 255}      sample["alpha"], sample["hint_type"], sample["index"], sample["name"]
    return colorize(sample["line_art"], sample["hint_color"], sample["hint_mask"])   # HxWx3 uint8 BGR

samples = [("line_art/0001.png", "ground_truth/0001.png"), ...]
result = hintauc.evaluate_colorizer(my_model, samples, hint_type="scribble",
                                    metrics=("mse", "psnr", "ssim", "lpips", "dreamsim"),
                                    save_dir="pred_root")          # optional: also write the images of path A
print(result["hint_auc"])            # {'psnr': ..., 'lpips': ..., ...}
hintauc.plot_curves(result["per_alpha"], "curve.png")
```

The callable receives only what a model may observe: the line art, the hint colour and mask, the ratio, the hint type
and the sample's index and name. The ground truth and the full hint structure are held back so that an adapter
cannot use them by accident; `oracle=True` adds `sample["ground_truth"]`, `sample["hints"]` (the `HintResult`) and
`sample["n_regions"]` for reference baselines. `hintauc.hint_fill_colorizer` is such an oracle baseline (hinted
*regions* filled with their hint colour, which needs the ground-truth region map): it shows the expected shape of a
curve, and its scores must not be compared with a real model's. `alphas` must be strictly increasing from 0 to 1
(`hintauc.check_alphas`). `save_dir` gets one directory per ratio plus `manifest.json`, which records the run (exact
ratios, prediction names and their ground-truth files, `oracle`, `tie_break`, path method); `hintauc curve` on that
directory re-scores exactly those files, so a JPEG ground truth saved as a PNG prediction still pairs, and the `oracle`
flag stays visible in the result. A `save_dir` that already holds another run is refused (`overwrite=True` replaces
it), because a stale ratio directory would otherwise be scored together with the new ones; a write failure is an
error. Repeat with `hint_type="dot"` for the dot protocol, and with the three line-art sources to report the paper's
mean ± SD.

Signatures and examples of every function used above: [API reference](api.md).

## Which numbers to expect

`hintauc demo` and `examples/quickstart_cpu.py --fast` are deterministic (synthetic image or the two shipped
illustrations, geodesic paths) and their Hint-AUC values are recorded in `examples/expected_numbers.json`, for both
backends; `python examples/check_expected.py <out>/result.json demo|quickstart_fast` compares a run with them (CI does
this on every push, with and without PyTorch, and before every PyPI release against the built wheel).

| Example | Backend | PSNR Hint-AUC | MSE Hint-AUC | SSIM Hint-AUC |
|---|---|---:|---:|---:|
| `hintauc demo` | torchvision + torchmetrics | 15.9028 | 0.0359 | 0.7790 |
| `hintauc demo` | Pillow + scikit-image | 15.9028 | 0.0359 | 0.7790 |
| `quickstart_cpu.py --fast` | torchvision + torchmetrics | 14.2770 | 0.0429 | 0.5472 |
| `quickstart_cpu.py --fast` | Pillow + scikit-image | 14.2770 | 0.0429 | 0.5472 |

Both examples order equal-area regions with `tie_break="stable"`, so the values are the same on every CPU (NumPy's
default argsort, used by the DetFill loader, orders equal areas differently on machines with and without AVX-512).
The default quick start (FilFinder paths, the paper's generator) gave a PSNR Hint-AUC of 14.18 in two runs, but
FilFinder breaks ties with an unseeded generator, so a few scribble pixels, and with them the last digits, can differ
between runs. Comparisons with the paper's numbers use the stored maps, never regenerated ones.

## Reporting checklist

- hint type (scribble / dot), the ratio grid (paper: 0, 1, 3, 5, 10, 25, 50, 100 %), that the selection is
  size-ordered (the Table III variant uses the fixed label order, `hint_order: label` in the DetFill loader) and the
  tie-break (`result["protocol"]["tie_break"]`);
- the line-art source(s) and the test split (ids in `reproduce/data/splits/test.txt`);
- metric settings (256 × 256, torchmetrics 1.4.0 for SSIM) and the backend — `result["protocol"]["backend"]` and
  `hauc.json` carry them;
- which checkpoint produced the outputs (SHA-256; `eval_per_ratio.py --ckpt` records it) and the sampler seed;
- whether the hints were the stored maps or regenerated ones (and with which `path_method`).

Numbers obtained this way are comparable with Table II of the paper only when the inputs are the stored test-split
maps and line art; the released per-ratio files in `reproduce/expected/` are the reference.
