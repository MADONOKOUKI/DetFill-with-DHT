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
DreamSim); the plain install has the pixel metrics. A worked example that runs on CPU in about a minute:
`python examples/quickstart_cpu.py` (it does both A and B on the two illustrations shipped with the repository).

## What the protocol fixes

- **Hints.** `hintauc.generate_hints(ground_truth, size=64)` — Felzenszwalb regions, one scribble (longest skeleton
  path) and one dot per region, coloured with the region mean. At ratio α the largest `int(n_regions * α)` regions keep
  their hints (`HintResult.at_ratio`). The 64 × 64 maps are upsampled to the image size with nearest-neighbour
  interpolation (4 × 4 blocks at 256 px, as DetFill was trained).
- **Inputs to the model.** The line art (the paper reports the mean over three extractors: sketch simplification,
  XDoG, SketchKeras), the hint colour image and the hint mask. Arrays from `hintauc` are BGR (OpenCV order).
- **Scoring.** Prediction and ground truth are resized to 256 × 256, values in [0, 1]; SSIM with torchmetrics 1.4.0
  (the environment pin), LPIPS-AlexNet, OpenCLIP ViT-B/32, DINOv2-base, DreamSim.
- **Aggregation.** Trapezoidal integral over α ∈ [0, 1] of the per-ratio means; the paper reports mean ± SD over the
  three line-art sources. `result["protocol"]` records the grid, the resize, the metric list and the library versions
  next to every score, and states whether the grid is the paper grid.

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
   (directory names are the ratios as decimals; `10` is also read as 10 %).
3. Score:
   ```bash
   hintauc curve pred_root gt_dir --metrics mse psnr ssim lpips openclip dino dreamsim --json result.json --plot curve.png
   ```
   Files are paired by name (`--pairing sorted` pairs by sorted order instead); a ratio directory with a different
   number of images or without common names is an error, not a silent mismatch. The JSON holds the per-ratio means,
   the Hint-AUC per metric and the protocol record; the PNG holds the curves.

For a DetFill-style output tree (`.../sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/200/`) use
`reproduce/scripts/eval_per_ratio.py`, which also writes the `per_ratio_summary.csv` format of the released metric files
and can record the checkpoint hash (`--ckpt`).

## B. A Python function

```python
import hintauc

def my_model(sample):
    # sample["line_art"]   HxW  uint8         sample["hint_color"]  HxWx3 uint8 BGR
    # sample["hint_mask"]  HxW  {0, 255}      sample["alpha"], sample["hint_type"], sample["hints"] (HintResult)
    return colorize(sample["line_art"], sample["hint_color"], sample["hint_mask"])   # HxWx3 uint8 BGR

samples = [("line_art/0001.png", "ground_truth/0001.png"), ...]
result = hintauc.evaluate_colorizer(my_model, samples, hint_type="scribble",
                                    metrics=("mse", "psnr", "ssim", "lpips", "dreamsim"),
                                    save_dir="pred_root")          # optional: also write the images of path A
print(result["hint_auc"])            # {'psnr': ..., 'lpips': ..., ...}
hintauc.plot_curves(result["per_alpha"], "curve.png")
```

`hintauc.hint_fill_colorizer` is a trivial reference model (hinted regions filled with their hint colour) that shows
the expected shape of a curve. Repeat with `hint_type="dot"` for the dot protocol, and with the three line-art
sources to report the paper's mean ± SD.

## Reporting checklist

- hint type (scribble / dot), the ratio grid (paper: 0, 1, 3, 5, 10, 25, 50, 100 %), and that the selection is
  size-ordered (the Table III variant uses the fixed label order, `hint_order: label` in the DetFill loader);
- the line-art source(s) and the test split (ids in `reproduce/data/splits/test.txt`);
- metric settings (256 × 256, torchmetrics 1.4.0 for SSIM) — `result["protocol"]` and `hauc.json` carry them;
- which checkpoint produced the outputs (SHA-256; `eval_per_ratio.py --ckpt` records it) and the sampler seed.

Numbers obtained this way are comparable with Table II of the paper only when the inputs are the stored test-split
maps and line art; the released per-ratio files in `reproduce/expected/` are the reference.
