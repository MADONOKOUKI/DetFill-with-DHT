# DetFill-with-DHT

Official research code for

> **Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation**
> Koki Madono, Yuan Mingcheng, Edgar Simo-Serra
> (under review, IEEE TVCG)

Hint-based line-art colorization has been evaluated with *randomly* sampled color hints, which
makes scores unstable and comparisons unfair. This repository provides:

- **DHT (Deterministic HinT generation)** — a reproducible, region-based hint generation pipeline:
  the same image always yields the same scribble/dot hints.
- **Hint-AUC** — an evaluation protocol that scores a colorization model across the *whole range*
  of hint ratios (from no hints to fully hinted) and integrates the metric curve into one number.
- **DetFill** — a pixel-space Brownian Bridge diffusion colorization model trained with DHT hints.

![Deterministic hint generation pipeline](assets/readme/dht_pipeline.png)
*DHT pipeline: region segmentation → skeleton → longest path per region → color scribble map.*

---

## Quick start (pip)

```bash
pip install git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git
# with the perceptual metrics (LPIPS / OpenCLIP / DINOv2 / DreamSim):
pip install "hintauc[perceptual] @ git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git"
```

```python
import hintauc

# 1) image -> deterministic hints
hints = hintauc.generate_hints("illustration.png", size=64)
color, mask = hints.at_ratio(0.10, hint_type="scribble")   # top-10% regions by area
hints.save("out/illustration")                             # canonical *_region64 / *_scribble_col64 / ... files

# 2) evaluate a colorization against ground truth (same formulas as the paper)
ev = hintauc.Evaluator(metrics=("mse", "psnr", "ssim", "lpips", "dreamsim"))
scores = ev("colorized.png", "ground_truth.png")

# 3) Hint-AUC over the paper's hint-ratio grid
result = hintauc.evaluate_hint_curve(
    {a: f"preds/ratio_{a}" for a in hintauc.DEFAULT_ALPHAS}, "gt_dir",
    metrics=("mse", "lpips", "dreamsim"))
print(result["hint_auc"])
```

Command line: `hintauc generate image.png --ratio 0.1` /
`hintauc eval pred_dir gt_dir --metrics mse psnr ssim` — see
[examples/basic_usage.py](examples/basic_usage.py).

---

## What the library produces

`hintauc.generate_hints()` output for one illustration (all generated with the code in this repo):

| Input image | Region map | Flatten (region mean) | Scribble hints | Dot hints |
|:---:|:---:|:---:|:---:|:---:|
| ![input](assets/readme/demo_input.png) | ![region](assets/readme/demo_region.png) | ![flatten](assets/readme/demo_flatten.png) | ![scribble](assets/readme/demo_scribble100.png) | ![dot](assets/readme/demo_dot100.png) |

`HintResult.at_ratio(α)` masks the hints to the largest-area top-α regions — deterministically, so
every run and every paper reproduces the identical hint sets:

| α = 1% | α = 10% | α = 50% | α = 100% |
|:---:|:---:|:---:|:---:|
| ![r1](assets/readme/demo_scribble_r1.png) | ![r10](assets/readme/demo_scribble_r10.png) | ![r50](assets/readme/demo_scribble_r50.png) | ![r100](assets/readme/demo_scribble_r100.png) |

## DetFill colorization with DHT hints

DetFill colorizations for one test sketch as the (deterministic scribble) hint ratio grows —
the colorization converges to the ground truth as more regions are hinted:

| Sketch | α = 1% | α = 10% | α = 50% | α = 100% | Ground truth |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![sketch](assets/readme/ex_sketch.png) | ![h1](assets/readme/ex_hint_r1.png) | ![h10](assets/readme/ex_hint_r10.png) | ![h50](assets/readme/ex_hint_r50.png) | ![h100](assets/readme/ex_hint_r100.png) | ![gt](assets/readme/ex_gt.png) |
| *hints* → | ![d1](assets/readme/ex_detfill_r1.png) | ![d10](assets/readme/ex_detfill_r10.png) | ![d50](assets/readme/ex_detfill_r50.png) | ![d100](assets/readme/ex_detfill_r100.png) | |

## Hint-AUC evaluation

![Hint-AUC overview](assets/readme/hintauc_overview.png)

For each hint ratio α ∈ {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00} the model colorizes the
test set with the deterministic hints at that ratio; each per-ratio score curve
(MSE / PSNR / SSIM / LPIPS / OpenCLIP / DINOv2 / DreamSim) is integrated over α with the
trapezoidal rule into a single **Hint-AUC** value. Because the hints are deterministic, the whole
protocol is exactly reproducible.

---

## Repository layout

| Directory | Contents |
|---|---|
| `hintauc/` | **pip-installable library**: hint generation (`hints.py`), evaluation metrics (`metrics.py`), Hint-AUC (`auc.py`), CLI (`cli.py`). |
| `hint_generation/` | The original research scripts behind the library: canonical 2024 generation scripts (`canonical/`), and the cleaned segmenter-robustness port (`D_retrain_gen_hints.py`, `D_danboo_hints.py`). |
| `detfill/` | DetFill model — a pixel-space Brownian Bridge diffusion fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT) adapted to sketch + deterministic-hint conditioning. Training, inference, per-ratio sampling. |
| `evaluation/` | The paper's evaluation pipelines: per-ratio metrics (`eval_single_run.py`, `dense/B_eval_dense_curve.py`) and Hint-AUC aggregation (`calc_hint_auc_manual.py`, `dense/B_auc_grid_sensitivity_7m.py`). |
| `checkpoints/` | Pointers to the released model weights (below). |
| `examples/` | Library usage examples. |

## Checkpoints

The paper checkpoints (DetFill, 96 base channels, 200 epochs, Danbooru2021 illustrations) are
attached to the GitHub Release of this repository:

- `detfill_scribble_illust_200ep.pth` (463 MB) — scribble-hint model (Table II scribble results)
- `detfill_dot_illust_200ep.pth` (463 MB) — dot-hint model

Place them under `detfill/results/dataset_name/BrownianBridge_{scribble,dot}_illust/checkpoint/latest_model_200.pth`
(or pass `--resume_model` explicitly). DetFill is pixel-space: no VQGAN / latent-diffusion weights are required.
Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically by their pip packages.

## Reproducing the paper experiments

```bash
# environment for training / inference
conda env create -f detfill/environment.yml && conda activate BBDM

# 1. generate the deterministic hint dataset (or use the hintauc library)
python hint_generation/D_retrain_gen_hints.py --help

# 2. inference over the full hint-ratio grid (x 3 sketch extractors)
cd detfill
GPU=0 bash run_inference_mr.sh scribble
GPU=0 RATIOS="0.10" TYPES="2" bash run_inference_mr.sh scribble   # single cell

# 3. per-ratio metrics + Hint-AUC
python evaluation/dense/B_eval_dense_curve.py --help
python evaluation/calc_hint_auc_manual.py --help
```

## Notes

- This is research code: some scripts contain absolute dataset paths from our cluster
  (`/scratch/...`, `/home/madorin/...`) that need to be adapted to your environment
  (`dataset_path` / `scratch_root` in `detfill/configs/*.yaml`).
- The `hintauc` library re-implements the canonical scripts faithfully (3x3 dilation,
  region-mean colors, base-255 region-id encoding compatible with the dataloader);
  region-id colors are assigned deterministically instead of the original random sampling.

## License and acknowledgements

MIT License. The `detfill/` directory is derived from
[BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM)
(© 2023 xuekt98, MIT) — see `detfill/LICENSE`.

## Citation

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {under review},
  year    = {2026}
}
```
