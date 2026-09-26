# DetFill-with-DHT

Official research code for

> **Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation**
> Koki Madono, Yuan Mingcheng, Edgar Simo-Serra
> IEEE Transactions on Visualization and Computer Graphics, 2026 (accepted; DOI to follow)

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

## Replicability

`bash replicability/run.sh` reproduces **Fig. 9** of the paper end-to-end with no arguments (conda environment →
released checkpoints → DHT hint generation → DetFill inference → figure). See [replicability/README.md](replicability/README.md)
for requirements and run times. This is the script submitted for the Graphics Replicability Stamp.

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

Everything above (fresh-venv install, generation, evaluation, CLI) plus the full
DetFill inference and evaluation pipeline below is exercised end-to-end on a clean
clone as part of the release checks; `replicability/run.sh` is the one-command version.

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

| Input sketch | Ground truth |
|:---:|:---:|
| ![sketch](assets/readme/ex_sketch.png) | ![gt](assets/readme/ex_gt.png) |

| | α = 1% | α = 10% | α = 50% | α = 100% |
|:--|:---:|:---:|:---:|:---:|
| **Scribble hints** | ![h1](assets/readme/ex_hint_r1.png) | ![h10](assets/readme/ex_hint_r10.png) | ![h50](assets/readme/ex_hint_r50.png) | ![h100](assets/readme/ex_hint_r100.png) |
| **DetFill colorization** | ![d1](assets/readme/ex_detfill_r1.png) | ![d10](assets/readme/ex_detfill_r10.png) | ![d50](assets/readme/ex_detfill_r50.png) | ![d100](assets/readme/ex_detfill_r100.png) |

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
| `hint_generation/` | The original research scripts behind the library: canonical 2024 generation scripts (`canonical/`), and the cleaned segmenter-robustness port (`generate_hints.py`, `danbooregion_hints.py`). |
| `detfill/` | DetFill model — a pixel-space Brownian Bridge diffusion fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT) adapted to sketch + deterministic-hint conditioning. Training, inference, per-ratio sampling. |
| `evaluation/` | The paper's evaluation pipelines: per-ratio metrics (`eval_single_run.py`, `dense/eval_curve.py`) and Hint-AUC aggregation (`calc_hint_auc.py`, `dense/auc_grids.py`). |
| `checkpoints/` | Pointers to the released model weights (below). |
| `examples/` | Library usage examples. |

## Checkpoints

The paper checkpoints (DetFill, 200 epochs, Danbooru2021 illustrations) are attached to the
GitHub Release of this repository:

- `detfill_scribble_illust_200ep.pth` (1.08 GB) — scribble-hint model, 96 base channels (Table II/III scribble results)
- `detfill_dot_illust_200ep.pth` (0.48 GB) — dot-hint model, 64 base channels (Table II/III dot results)

Place them under `detfill/results/dataset_name/BrownianBridge_{scribble,dot}_illust/checkpoint/latest_model_200.pth`
(or pass `--resume_model` explicitly; see `checkpoints/README.md` for SHA-256 checksums and the v1.1
segmenter-retrained models of the supplementary study). DetFill is pixel-space: no VQGAN / latent-diffusion weights are required.
Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically by their pip packages.

## Longest-path extraction: two implementations

The scribble of a region is the longest path of its skeleton. Two implementations are
available and selected with `path_method` (library) / `--path_method` (CLI and batch generator):

| `path_method` | How the path is found | Dependencies | Deterministic? |
|---|---|---|---|
| `filfinder` (default, **used for all paper results**) | 3x3 dilation of the Zhang--Suen skeleton, then FilFinder2D 1.7.2 (medial axis, branch/skeleton threshold 3 px, prune by length, longest path) | `fil_finder`, `astropy` | Deterministic given a skeleton *except* that FilFinder's medial-axis step (`skimage.morphology.medial_axis`) breaks pixel ties with an unseeded random generator, so regeneration is not bit-exact |
| `geodesic` | Longest shortest path (geodesic diameter) of the 8-connected Zhang--Suen skeleton itself, orthogonal step 1 / diagonal step sqrt(2), no corner cutting; every tie is broken in raster order (`hintauc/longest_path.py`, pure NumPy) | none | Fully deterministic; the path is always inside the region |

```python
hints = hintauc.generate_hints("illustration.png", path_method="geodesic")
hints.path_method   # 'geodesic'
```

```bash
hintauc generate image.png --ratio 0.1 --path_method geodesic
python hint_generation/generate_hints.py ... --path_method geodesic   # outputs under <out_root>/<segmenter>_geodesic/
```

The two methods produce slightly different scribbles (the geodesic path skips no corner
pixels and applies no branch pruning). **Do not mix hint maps produced with different
methods within one evaluation**; the paper's numbers correspond to `filfinder`, and the
stored test-split hint maps attached to the v1.0 release were produced with it.
`hint_generation/compare_path_methods.py` compares the two methods on stored region maps
(path overlap, dot displacement, failure counts, speed).

**Dot placement** is selected with `dot_method` / `--dot_method`. `medoid` (default) is the rule behind the
paper's stored hint maps (training data and the released test-split maps; verified on six test images:
every stored dot is the in-region longest-path pixel with the smallest total Manhattan distance to the other
in-region path pixels, first index on ties, so every dot lies on its own scribble inside its region — see
`hint_generation/canonical/hint_dot_generation_fixdot.py`). `mean` truncates the mean row/column of the whole
longest path to integers without projecting onto the region (the January-2024 generation script and the wording
of Sec. IV-A of the paper); it can place a dot outside a non-convex region and is kept only for reference.
`nearest_mean` places the dot on the in-region path pixel nearest to that mean.

**Tie-breaking of equal-area regions** in `HintResult.at_ratio(..., tie_break=...)` / `--tie_break`:
`default` (paper) uses NumPy's default `argsort`, whose order among equal-area regions depends on the
NumPy build; `stable` breaks ties by ascending label value and is independent of the NumPy version and
hardware. Equal-area ties are frequent at 64x64 (about a thousand regions per map), so the two settings
select slightly different small regions at intermediate ratios; the reported results use `default`.

Comparison on the first 20 stored test-split region maps (16,098 regions;
`python hint_generation/compare_path_methods.py --region_dir <region64 dir> --limit 20`):

| | FilFinder (`filfinder`) | Geodesic (`geodesic`) |
|---|---|---|
| Regions without a path | 414 | 0 |
| Time per region | 40.1 ms | 0.07 ms |
| Dot outside its region | 3.4% | 3.3% |

Path agreement on the 15,684 regions where both methods return a path: identical path in
91.8% of regions, mean IoU 0.969 (median 1.000); the geodesic path has
+0.02 pixels on average relative to the FilFinder path; the dot positions coincide in
95.5% of regions and are within one pixel in 99.7%.
The FilFinder failure count of this run (414/16,098) is higher than the share of regions
without a stored scribble in the released test-split maps (0.26%), which illustrates the environment sensitivity of the
FilFinder path (randomized medial-axis tie-breaking, library versions); the geodesic method has no such dependence.

## Reproducing the paper's tables and supplementary results

`reproduce/` is the reproduction package (see [reproduce/README.md](reproduce/README.md)):

```bash
# A. rebuild every Hint-AUC table of the paper from the released per-ratio metrics and check each printed cell (CPU, <1 min)
python reproduce/scripts/A1_tables_from_released_metrics.py
# A. user study: Tables IV/V and the GLMM statistics from the released trial table
python reproduce/scripts/A2_userstudy_glmm.py
# B. re-run DetFill inference over the hint-ratio grid + 7-metric Hint-AUC with the released checkpoints (GPU)
DATA_ROOT=/data/danbooru_test GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh            # Table II scribble row
DATA_ROOT=/data/danbooru_test HINT_ORDER=label bash reproduce/scripts/run_hauc_pipeline.sh  # Table III scribble row
# C. the verbatim experiment/training launchers used for the paper and its revision (example code)
ls reproduce/paper_experiments/
```

Status: A1 rebuilds 322 of 324 printed cells to the printed precision and A2 reproduces all user-study
numbers exactly; the remaining two cells and everything that is *not* covered are listed in
`reproduce/README.md` ("Known deviations"). Training from scratch: `reproduce/scripts/B9_train_detfill.sh`.

## Notes

- All data locations are user-specified: set `data.dataset_config.dataset_path` /
  `scratch_root` in `detfill/configs/*.yaml` to your dataset root (the expected
  directory layout is documented in `detfill/README.md`). Hints default to the paper's
  64×64 resolution.
- The `hintauc` library re-implements the canonical scripts faithfully (3x3 dilation,
  region-mean colors, base-255 region-id encoding compatible with the dataloader);
  region-id colors are assigned deterministically instead of the original random sampling.

## Refactoring note (2026-07)

The initial release (`bfa3a84`) mirrored our internal research tree. It was then
simplified to the minimal set needed to reproduce the paper, **without changing any
behavior of the shipped pipeline**:

- `detfill/`: removed ~40 per-ratio `test_*.sh` launchers (superseded by
  `run_inference.sh`), cluster-specific ops scripts (`10_run_ratio.sh`,
  `20_launch.sh`, `30_push_and_eval.sh`), an unused alternate entry point
  (`main_colorization.py`), upstream BBDM evaluation utilities
  (`preprocess_and_evaluation.py`, `evaluation/` — our Hint-AUC evaluation lives at the
  repository root), the vendored `dreamsim/` repository (its only reference in the model
  code was a dead import, now removed; perceptual metrics are provided by the pip
  packages), the seed-experiment dataset variants, seven unused config variants, and a
  fully commented-out duplicate dataset class in `datasets/custom.py`.
- `evaluation/`: removed `eval_single_run_v2.py` (a 9-line near-duplicate of
  `eval_single_run.py`) and the superseded 4-metric AUC script.
- `hint_generation/canonical/`: kept the canonical generator
  (`hint_dot_generation.py`) and `all_segmentations.py`; the
  ImageNet / 256-px / superpixel-ablation variants differed only in path constants and
  were removed.

A follow-up pass made every data location user-configurable: the config files ship
with `/path/to/dataset` placeholders, the dataset loaders derive all directories from
`dataset_path` / `scratch_root` (raising a clear error when unset), script argument
defaults that pointed at our experiment environment became required arguments, and
committed build artifacts were removed from version control.

A final pass simplified every file name (dates and internal prefixes removed):
`run_inference.sh`, `configs/{dot,scribble}_{illust,real}.yaml` (the duplicate
training configs were merged into these), `configs/<domain>/{train,valid,test}.txt`,
`hint_generation/generate_hints.py` / `danbooregion_hints.py` /
`canonical/hint_dot_generation.py`, and `evaluation/{calc_hint_auc,avg_hint_auc}.py`,
`evaluation/dense/{eval_curve,build_summary,auc_grids}.py`.

Everything removed remains available in the git history.

## License and acknowledgements

MIT License, with the third-party components listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
(notably the ColorizeDiffusion-derived inference wrappers under `reproduce/paper_experiments/coldiff_finetune/`,
which stay under the upstream CC BY-NC-SA 4.0 license). The `detfill/` directory is derived from
[BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM)
(© 2023 xuekt98, MIT) — see `detfill/LICENSE`.

## Citation

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026}
}
```
