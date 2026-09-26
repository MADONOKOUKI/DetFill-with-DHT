# DetFill-with-DHT

Official code, models and reproduction package for

> **Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation**
> Koki Madono, Yuan Mingcheng, Edgar Simo-Serra
> IEEE Transactions on Visualization and Computer Graphics, 2026. DOI: [10.1109/TVCG.2026.3738401](https://doi.org/10.1109/TVCG.2026.3738401)

Hint-based line-art colorization has been evaluated with *randomly* sampled colour hints, which makes scores
unstable and comparisons unfair. This repository provides three things:

- **DHT (Deterministic HinT generation)** — a reproducible, region-based hint generation pipeline: the same
  illustration always yields the same scribble and dot hints.
- **Hint-AUC** — an evaluation protocol that scores a colorization model across the *whole range* of hint
  ratios (from no hints to fully hinted) and integrates the metric curve into one number.
- **DetFill** — a pixel-space Brownian Bridge diffusion colorization model trained with DHT hints, with all
  checkpoints of the paper.

![Deterministic hint generation pipeline](assets/readme/dht_pipeline.png)
*DHT pipeline: region segmentation → skeleton → longest path per region → colour scribble map.*

---

## Reproducing the paper: start here

Everything below runs from this repository plus files attached to its [GitHub releases](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases)
(checkpoints and data; every file has a SHA-256 in `checkpoints/README.md`). No account, no private data, no
cluster is needed. Requirements: Linux, Miniconda/Anaconda, ~10 GB of disk, an NVIDIA GPU (CUDA 12 driver) for the
inference scripts — the CPU fallback works but is slow (times below).

| What you want to check | Command (no arguments) | Time | What you get and how to compare |
|---|---|---|---|
| **Fig. 9 of the paper** (deterministic vs. random hint selection, DetFill outputs) — the Replicability-Stamp script | `bash replicability/run.sh` | GPU ≈ 3 min, CPU ≈ 30 min, plus a one-time environment setup | `replicability/output/fig9.png`; compare with `replicability/expected/fig9_paper.png` (the printed figure) |
| **Every table of the paper rebuilt from the released metric files** (Tables II–V and the supplementary α-grid, segmentation-dependency, seed-sensitivity, Diffusart-retrain tables, GLMM statistics) | `python reproduce/scripts/A1_tables_from_released_metrics.py` and `python reproduce/scripts/A2_userstudy_glmm.py` | < 2 min, CPU only | Markdown tables in `reproduce/output/`; each printed number is checked automatically (322 of 324 cells match; the two exceptions are explained in `reproduce/README.md`). Reference copies: `reproduce/expected/recomputed_tables/` |
| **Every experiment re-run on a few example illustrations with the released checkpoints** (Table II and Table III protocols, segmentation dependency, seed sensitivity, dense ratio curve, channel ablation, hint regeneration, 2024 models) | `bash reproduce/examples/run_examples.sh` (`EXAMPLES_MODE=smoke` ≈ 5 min, default `quick` ≈ 45 min, `full` = several GPU-hours) | see left | Labelled image grids and per-image metrics in `reproduce/examples/output/`; the authors' run of the same script and the paper's archived outputs of the same images are in `reproduce/examples/expected/`, and the script prints a side-by-side comparison |
| **A full table row on the 3,000 test images** | `DATA_ROOT=… bash reproduce/scripts/run_hauc_pipeline.sh` | ≈ 14 GPU-hours per row | Needs the Danbooru2021 originals (not redistributable) plus the released line art and hint maps; details in `reproduce/README.md` part B |
| **Training from scratch** | `bash reproduce/scripts/B9_train_detfill.sh` | days on 10 GPUs | The commands behind the released checkpoints |

The verbatim launcher scripts that produced the paper and its revision (cluster paths included, kept as example
code) are in `reproduce/paper_experiments/`. What is bit-exact and what is not is listed in
`reproduce/README.md` → "Known deviations".

---

## Quick start with the library (pip)

```bash
pip install git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git
# with the perceptual metrics (LPIPS / OpenCLIP / DINOv2 / DreamSim):
pip install "hintauc[perceptual] @ git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git"
```

```python
import hintauc

# 1) image -> deterministic hints
hints = hintauc.generate_hints("illustration.png", size=64)
color, mask = hints.at_ratio(0.10, hint_type="scribble")   # the largest 10 % of the regions
hints.save("out/illustration")                             # canonical *_region64 / *_scribble_col64 / ... files

# 2) evaluate a colorization against its ground truth (same formulas as the paper)
ev = hintauc.Evaluator(metrics=("mse", "psnr", "ssim", "lpips", "dreamsim"))
scores = ev("colorized.png", "ground_truth.png")

# 3) Hint-AUC over the paper's hint-ratio grid
result = hintauc.evaluate_hint_curve(
    {a: f"preds/ratio_{a}" for a in hintauc.DEFAULT_ALPHAS}, "gt_dir",
    metrics=("mse", "lpips", "dreamsim"))
print(result["hint_auc"])
```

Command line: `hintauc generate image.png --ratio 0.1` and `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`
(see `examples/basic_usage.py`).

## What the library produces

`hintauc.generate_hints()` output for one illustration (all generated with the code in this repository):

| Input image | Region map | Flatten (region mean) | Scribble hints | Dot hints |
|:---:|:---:|:---:|:---:|:---:|
| ![input](assets/readme/demo_input.png) | ![region](assets/readme/demo_region.png) | ![flatten](assets/readme/demo_flatten.png) | ![scribble](assets/readme/demo_scribble100.png) | ![dot](assets/readme/demo_dot100.png) |

`HintResult.at_ratio(α)` keeps the hints of the largest-area regions only — deterministically, so every run and
every paper reproduces the identical hint sets:

| α = 1% | α = 10% | α = 50% | α = 100% |
|:---:|:---:|:---:|:---:|
| ![r1](assets/readme/demo_scribble_r1.png) | ![r10](assets/readme/demo_scribble_r10.png) | ![r50](assets/readme/demo_scribble_r50.png) | ![r100](assets/readme/demo_scribble_r100.png) |

## DetFill colorization with DHT hints

DetFill colorizations for one test sketch as the (deterministic scribble) hint ratio grows — the colorization
converges to the ground truth as more regions are hinted:

| Input sketch | Ground truth |
|:---:|:---:|
| ![sketch](assets/readme/ex_sketch.png) | ![gt](assets/readme/ex_gt.png) |

| | α = 1% | α = 10% | α = 50% | α = 100% |
|:--|:---:|:---:|:---:|:---:|
| **Scribble hints** | ![h1](assets/readme/ex_hint_r1.png) | ![h10](assets/readme/ex_hint_r10.png) | ![h50](assets/readme/ex_hint_r50.png) | ![h100](assets/readme/ex_hint_r100.png) |
| **DetFill colorization** | ![d1](assets/readme/ex_detfill_r1.png) | ![d10](assets/readme/ex_detfill_r10.png) | ![d50](assets/readme/ex_detfill_r50.png) | ![d100](assets/readme/ex_detfill_r100.png) |

## Hint-AUC evaluation

![Hint-AUC overview](assets/readme/hintauc_overview.png)

For each hint ratio α ∈ {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00} the model colorizes the test set with the
deterministic hints at that ratio; each per-ratio score curve (MSE / PSNR / SSIM / LPIPS / OpenCLIP / DINOv2 /
DreamSim) is integrated over α with the trapezoidal rule into a single **Hint-AUC** value. Because the hints are
deterministic, the whole protocol is exactly reproducible.

---

## Repository layout

| Directory | Contents |
|---|---|
| `replicability/` | **The one-command reproduction of Fig. 9** (`run.sh`), its conda environment, the two input illustrations, and the figure as printed for comparison. |
| `reproduce/` | **Reproduction package**: `scripts/` (table recomputation, generic inference + Hint-AUC pipeline, training commands, sketch-extractor wrappers), `examples/` (example-based re-run of every experiment), `expected/` (released per-ratio metric files and reference results), `data/` (user-study trial table, data split lists), `paper_experiments/` (verbatim launchers of the paper and the revision). Read `reproduce/README.md` first. |
| `hintauc/` | **pip-installable library**: hint generation (`hints.py`), evaluation metrics (`metrics.py`), Hint-AUC (`auc.py`), command line (`cli.py`). |
| `detfill/` | The DetFill model — a pixel-space Brownian Bridge diffusion fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT) adapted to sketch + deterministic-hint conditioning. Training, inference (`run_inference.sh`), configs, data-layout documentation. |
| `hint_generation/` | The original research scripts behind the library: the canonical 2024 generation scripts (`canonical/`) and the cleaned segmenter-robustness port. |
| `evaluation/` | The paper's evaluation pipelines: per-ratio metrics and Hint-AUC aggregation. |
| `checkpoints/` | **List of every released checkpoint and data file with sizes, SHA-256 and what it reproduces.** |
| `examples/` | Library usage examples. |
| `assets/` | README images and the 250×250 representative image for the Replicability Stamp (`assets/grsi/`). |

## Checkpoints and data (summary)

All files are attached to the GitHub releases; `checkpoints/README.md` lists them with hashes. In short:

- **v1.0** — the paper's models: `detfill_scribble_illust_200ep.pth` (96 base channels) and `detfill_dot_illust_200ep.pth`
  (64 base channels), plus the stored hint maps of the 3,000 test images. Behind Tables II/III and Fig. 9.
- **v1.1** — scribble models trained with DanbooRegion and SLIC hints (supplementary segmentation-dependency study).
- **v1.2** — Diffusart retrained with our hints (supplementary "additional training" table) and the natural-image
  (ImageNet) DetFill models with their test hint maps.
- **v1.3** — the test-split line art (three extractors), the alternative-segmenter hint maps, the 12-image example
  bundle, and the user-study stimuli.
- **legacy-2024** — the models of the 2024 submission and the 32/64-channel models of the channel ablation.

## Longest-path extraction: two implementations

The scribble of a region is the longest path of its skeleton. Two implementations are available and selected with
`path_method` (library) / `--path_method` (command line and batch generator):

| `path_method` | How the path is found | Dependencies | Deterministic? |
|---|---|---|---|
| `filfinder` (default, **used for all paper results**) | 3×3 dilation of the Zhang–Suen skeleton, then FilFinder2D 1.7.2 (medial axis, branch/skeleton threshold 3 px, prune by length, longest path) | `fil_finder`, `astropy` | Up to the unseeded medial-axis tie-breaking inside FilFinder (a few pixels may move between environments) |
| `geodesic` | Longest shortest path (geodesic diameter) of the 8-connected Zhang–Suen skeleton itself, orthogonal step 1 / diagonal step √2, no corner cutting; every tie is broken in raster order | none beyond NumPy | Yes, bit-exact everywhere |

```python
hints = hintauc.generate_hints("illustration.png", path_method="geodesic")
```

The two methods produce slightly different scribbles (the geodesic path skips no corner pixels and applies no
branch pruning). **Do not mix hint maps produced with different methods within one evaluation**; the paper's numbers
correspond to `filfinder`, and the stored test-split hint maps of the v1.0 release were produced with it.

**Dot placement** is selected with `dot_method` / `--dot_method`. `medoid` (default) is the rule behind the paper's
stored hint maps: the dot is the in-region longest-path pixel with the smallest total Manhattan distance to the
other in-region path pixels (first index on ties), so every dot lies on its own scribble inside its region — verified
against the released maps. The paper's text (Sec. IV-A) describes a truncated mean of the path coordinates; that rule
is available as `dot_method="mean"` for reference but it is not what produced the data. See
`hint_generation/README.md`.

**Tie-breaking of equal-area regions** in `HintResult.at_ratio(..., tie_break=...)`: `default` (paper) uses NumPy's
default `argsort`, whose order among equal-area regions depends on the NumPy build; `stable` breaks ties by ascending
label value. The reported results use `default` with the pinned NumPy.

## Notes

- Data locations are user-specified: set `data.dataset_config.dataset_path` / `scratch_root` in `detfill/configs/*.yaml`
  to your dataset root (layout documented in `detfill/README.md`). Hints default to the paper's 64×64 resolution.
- The `hintauc` library re-implements the canonical scripts faithfully (3×3 dilation, region-mean colours,
  base-255 region-id encoding compatible with the data loader).
- The evaluator reproduces the paper's per-image metrics to numerical precision when `torchmetrics==1.4.0` is used
  (pinned in `replicability/environment.yml` and `reproduce/requirements-metrics.txt`); newer torchmetrics versions
  changed the SSIM implementation.

## License and acknowledgements

MIT License, with the third-party components listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
(notably the ColorizeDiffusion-derived inference wrappers under `reproduce/paper_experiments/coldiff_finetune/`,
which stay under the upstream CC BY-NC-SA 4.0 license). The `detfill/` directory is derived from
[BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM)
(© 2023 xuekt98, MIT) — see `detfill/LICENSE`. Released data files are derived from Danbooru2021 illustrations and
are provided for non-commercial research use; the original artworks remain the property of their creators.

Contact: Koki Madono (see the paper for the address).

## Citation

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026},
  doi     = {10.1109/TVCG.2026.3738401}
}
```
