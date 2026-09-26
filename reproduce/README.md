# Reproducing the paper's results

This directory is the reproduction package for

> Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation
> (IEEE TVCG 2026, DOI 10.1109/TVCG.2026.3738401). Main paper: Tables II–V and Fig. 9. Supplement: the
> hint-ratio-grid (α-grid) study, the segmentation-dependency study, the seed-sensitivity study, the additional
> training comparisons, the natural-image tables and the user study.

It has four layers, from cheapest to most expensive. Everything runs from this repository plus the files attached to
its GitHub releases (see `checkpoints/README.md`); nothing else is needed except the Danbooru2021 originals for
layer D.

| Layer | What it does | Needs | Time |
|---|---|---|---|
| **A. Recompute the tables from the released metric files** | Rebuilds every Hint-AUC table of the paper from the per-ratio metric files and the user-study trial table shipped in `expected/` and `data/`, and checks each printed cell | numpy, pandas, statsmodels (CPU) | < 2 min |
| **B. Example-based re-run of every experiment** | Runs the released checkpoints on 12 example illustrations for each experiment of the paper and the supplement, writes labelled image grids and per-image metrics, and compares them with the authors' run and with the paper's archived outputs | one GPU (CPU possible but slow); assets are downloaded automatically | 5 min (smoke) / ≈ 45 min (default) / hours (full) |
| **C. Verbatim experiment launchers** (`paper_experiments/`) | The scripts that were actually run for the paper and its revision, kept as example code (cluster paths hard-coded) | our cluster | days |
| **D. Full-scale re-run of a table row** | Colorizes the 3,000 test images at every hint ratio and recomputes the seven metrics and Hint-AUC | GPU + the Danbooru2021 originals + the released line art, hint maps and checkpoints | ≈ 14 GPU-hours per row |

`../replicability/run.sh` (Fig. 9, one command, ≈ 3 min on a GPU) is the Graphics-Replicability-Stamp entry point and
is independent of this directory.

---

## A. Recompute the tables from the released metric files (no GPU)

```bash
conda env create -f replicability/environment.yml && conda activate detfill-grsi   # or any env with numpy, pandas, statsmodels
python reproduce/scripts/A1_tables_from_released_metrics.py   # Tables II/III (DetFill rows) + supplementary tables
python reproduce/scripts/A2_userstudy_glmm.py                 # Tables IV/V + the GLMM statistics of Sec. VII
```

`A1_tables_from_released_metrics.py` writes Markdown tables to `reproduce/output/tables/` and a cell-by-cell comparison
with the numbers printed in the paper (`check_report.csv`). Reference copies of both scripts' outputs are in
`expected/recomputed_tables/`.

Result of the authors' run (2026-09-26): **A1 rebuilds 322 of the 324 compared cells to the printed precision**;
**A2 reproduces all 30 percentages of Tables IV/V exactly and the three GLMM statistics** (z = 30.18;
χ²(7) = 618.33, p = 2.75e-129; χ²(14) = 324.04, p = 1.12e-60). The two A1 exceptions are a rounding artefact and one
provenance gap, both listed under "Known deviations" below; neither changes a conclusion.

What each table is rebuilt from (the rule is always the paper's: Hint-AUC = trapezoidal integral of the per-ratio
metric over the hint ratio α ∈ {0, 1, 3, 5, 10, 25, 50, 100} %, computed per line-art source and reported as
mean ± sample standard deviation over the three sources — XDoG, sketch simplification, SketchKeras):

| Paper item | Where in A1 | Released input under `reproduce/expected/` |
|---|---|---|
| Table II, DetFill scribble row (7 metrics) | part 1 | `alpha_grid/B_dense_curve/B_dense_curve__per_ratio_summary*.csv` — per-source, per-ratio means of the released 96-channel scribble model; the dense hint-ratio sweep contains the eight paper ratios |
| Supp. table "sensitivity of Hint-AUC to the α grid" (5 grids) | part 2 | the same dense sweep, re-integrated on each grid (linear interpolation where a grid point was not measured) |
| Supp. table "segmentation dependency" (3 training × 3 evaluation segmenters) | part 3 | `segmenter_dependency/hauc_7m/<training segmenter>_model__on_<evaluation segmenter>/per_ratio_summary.csv`, plus the dense sweep for the Felzenszwalb → Felzenszwalb cell |
| Table III, DetFill scribble row (fixed random region order) | part 4 | `table3_scribble/hauc7_scribble.json` |
| Supp. table "additional training comparisons", Diffusart-retrain row (PSNR / SSIM / LPIPS) | part 5 | `additional_training/diffusart_retrain_R3-2/hauc_runs_v3_*/hauc_summary.json` (see the NOTE.md next to it) |
| Supp. table "size-ordered vs. random selection" (10 seeds, 300 images) | part 6 | `seed_sensitivity/metrics_felz96/seed*_alpha_*/per_ratio_summary.csv` (random rows) and `seed_sensitivity/aggregated_sort.csv` (size-ordered row); α = 0 % and 100 % do not depend on the seed and were measured once |
| Tables IV and V (user study) | A2 | `data/userstudy/glmm_trials.csv` — 32 participants × 192 forced-choice trials, anonymised |
| GLMM statistics (Sec. VII) | A2 | same file; `statsmodels` `BinomialBayesMixedGLM` (variational Bayes) with random intercepts for participant and image; model A = intercept only, model B = opponent × ratio (sum coding), diagonal Wald χ² |

`expected/` also contains the analyses written for the review response that are not printed in the paper
(`userstudy_rank_stability/`, `segmenter_sensitivity/`, `alpha_grid/A-5/`), with their scripts under `paper_experiments/`.

## B. Example-based re-run of every experiment (one GPU, no data preparation)

```bash
bash reproduce/examples/run_examples.sh                       # default "quick" set (≈ 45 min on one GPU)
EXAMPLES_MODE=smoke bash reproduce/examples/run_examples.sh   # 1 image, 2 ratios, scribble model only (≈ 5 min)
EXAMPLES_MODE=full  bash reproduce/examples/run_examples.sh   # all experiments, 12 images, 8 ratios (several GPU-hours)
```

The script creates (or reuses) the conda environment of `replicability/run.sh`, downloads the checkpoints and the
12-image example bundle from the releases (SHA-256 verified), runs each experiment with the same `detfill/main.py`
entry point and the same `hintauc` evaluator as the full-scale experiments, and writes:

- `reproduce/examples/output/grids/*.png` — one labelled image grid per experiment (ground truth, line art, hints,
  outputs per hint ratio / per model),
- `reproduce/examples/output/metrics/*.json` — per-image and mean values of the seven metrics, and Hint-AUC where the
  full ratio grid was run,
- a printed comparison with `reproduce/examples/expected/` (the authors' run of the same script) and, for the
  Table II protocol, the PSNR between each new output and the paper's archived output of the same image.

| Experiment | Paper item | What the example shows |
|---|---|---|
| E1 Table II protocol | Table II, Fig. 8 | scribble (96-ch) and dot (64-ch) models with size-ordered hints over the ratio grid |
| E2 Table III protocol | Table III | the scribble model with the fixed random (ascending-label) region order |
| E3 Segmentation dependency | supp. segmentation-dependency table and figures | scribble models trained on Felzenszwalb / DanbooRegion / SLIC hints, each evaluated with each segmenter's hints |
| E4 Seed sensitivity | supp. "size-ordered vs. random" table and figure | size-ordered selection vs. the paper's seeded random selection at the same ratios |
| E5 Dense ratio curve | supp. α-grid study | per-image metric curves on a fine ratio grid and the Hint-AUC on the dense vs. the paper grid |
| E8 Channel ablation | supp. channel-ablation figure | 32 / 64 / 96 base-channel scribble models on the two images of that figure |
| E9 Hint regeneration | Sec. IV, supp. Sec. I | the deterministic hint generator re-run with the `hintauc` library vs. the stored maps |
| E11 Legacy 2024 models | (provenance) | the scribble/dot models of the 2024 submission on the same images |

`reproduce/examples/README.md` documents each experiment, the images used and the authors' numbers.

## C. The verbatim launchers

`paper_experiments/README.md` lists, per experiment, the shell and Python files that were actually run for the paper
and the revision (with our cluster's paths and environment names). They are not runnable as-is elsewhere; the
runnable equivalents are the scripts in `scripts/` and `examples/`.

## D. Full-scale re-run of a table row

### D.1 Data

1. **Test images.** The test split is the 3,000 Danbooru2021 images listed in `data/splits/test.txt` (ids ending in
   `016`). Obtain them from the Danbooru2021 distribution (we do not redistribute the images) and store them as
   `<DATA_ROOT>/segmentations/originals/<id>.image.png` (512 × 512; the preprocessing of `hint_generation/canonical/all_segmentations.py`).
2. **Line art.** Release v1.3 ships the exact line-art files used for every reported number
   (`test_split_sketch_{sketchkeras,pysimp,XDoG}.tar.gz` → `<DATA_ROOT>/sketch/<extractor>/<id>.png`). The wrappers
   that generated them are in `scripts/sketch_tools/` (XDoG draws its σ/k jitter from an unseeded generator, so
   regenerated XDoG line art is not bit-identical).
3. **Deterministic hints.** Release v1.0 ships the stored maps (`test_split_hint_maps_64.tar.gz` →
   `<DATA_ROOT>/hint_from_regions_64_rev/` and `region64/*` → `<DATA_ROOT>/hint_from_regions_256/`). They are the
   exact inputs of every reported number.
4. **Checkpoints.** `checkpoints/README.md`.

### D.2 One table row = one command

```bash
# Table II, DetFill scribble row (default), 7 metrics, Hint-AUC over the paper grid
DATA_ROOT=/data/danbooru_test GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# Table II, DetFill dot row
DATA_ROOT=/data/danbooru_test HINT=dot GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# Table III (fixed random region order = ascending label order of the stored region map)
DATA_ROOT=/data/danbooru_test HINT_ORDER=label GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# supplementary α-grid study: dense sweep, then integrate on the alternative grids
DATA_ROOT=/data/danbooru_test RATIOS="$(seq -f %.2f 0 0.02 1 | tr '\n' ' ') 0.01 0.03 0.05 0.25" TAG=dense \
    bash reproduce/scripts/run_hauc_pipeline.sh
python reproduce/paper_experiments/alpha_grid/B_auc_grid_sensitivity_7m.py \
    --summary reproduce/output/dense/metrics/per_ratio_summary.csv --out_dir reproduce/output/dense/auc
# smoke test of the pipeline (20 images per cell, ~10 min on one GPU)
DATA_ROOT=/data/danbooru_test LIMIT=20 TYPES="2" GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
```

`run_hauc_pipeline.sh` = `detfill/main.py --sample_to_eval` for every (ratio, line-art source) followed by
`scripts/eval_per_ratio.py`, which uses the released evaluator `hintauc.metrics.Evaluator` (the paper's
`evaluation/eval_single_run.py`: 256 × 256, MSE/PSNR/SSIM on [0, 1]; LPIPS-Alex; OpenCLIP ViT-B-32 laion2b;
DINOv2-base; DreamSim) and writes `per_ratio_summary.csv` in the same format as the files in `expected/`, plus
`hauc.json`. The perceptual metrics need `pip install -r reproduce/requirements-metrics.txt`.

Expected agreement (checked on 160 image pairs of the archived paper outputs): with `requirements-metrics.txt` the
evaluator reproduces the archived per-image values to numerical precision for MSE, PSNR, LPIPS, DINO and DreamSim
(max |Δ| ≤ 3e-6), SSIM to 5e-5 with the pinned `torchmetrics==1.4.0` (torchmetrics 1.8 changed the SSIM
implementation: per-image |Δ| up to 0.04, so the pin matters), and OpenCLIP within 8e-4 (the archived run scored
OpenCLIP in half precision). Inference is seeded and the stored hint maps remove the hint-generation variance, but
the diffusion sampler is not bit-exact across GPU generations or on CPU: re-generated outputs of the same image agree with
the paper's archived outputs only approximately (about 25–30 dB PSNR in our checks, see `reproduce/examples/`). We have
not re-run a complete table row on different hardware; expect small differences in the last printed digit.

### D.3 Other studies at full scale

- **Seed sensitivity (supp.)** — `paper_experiments/seed_sensitivity/gen_seeded_random_hints.py` bakes a seeded
  uniformly random region order into 64 × 64 hint maps (10 seeds × 6 intermediate ratios; α = 0 % and 100 % are
  seed-independent); each (seed, α) set is then run through the pipeline above at `RATIOS=1.00` with `DATA_ROOT`
  pointing at a data root whose `hint_from_regions_64_rev/` holds that seed's maps, evaluated on the first 300 test
  ids with SketchKeras line art, and aggregated exactly as in A1 part 6. `examples/` does this for three images.
- **Segmentation dependency (supp.)** — release v1.3 ships the DanbooRegion and SLIC hint maps of the test split,
  release v1.1 the two retrained models; run the pipeline with `CKPT=<v1.1 file>` and `DATA_ROOT` pointing at a copy
  of the data root whose hint maps were replaced by the segmenter's maps. Regeneration of the maps for other images:
  `paper_experiments/segmenter_dependency/r2-2_scripts/` (`D_danbooregion_seg.py`, `D_retrain_gen_hints.py`,
  `D_danboo_hints.py`) and `hint_generation/generate_hints.py --segmenter slic`. Training of the two models:
  `D_train_launch.sh` / `D_stage_train.sh` (200 epochs, same protocol as `scripts/B9_train_detfill.sh`).
- **Training DetFill from scratch** — `scripts/B9_train_detfill.sh` (the paper's `train.sh` with the released config
  names; ~4–5 days on 10 GPUs).
- **Diffusart-retrain (supp.)** — `paper_experiments/diffusart_retrain/code/` (our Diffusart re-implementation:
  training loop, deterministic-hint loader, inference and evaluation; README inside) with the two released EMA
  checkpoints of release v1.2.
- **ColorizeDiffusion fine-tuning (Table II/III "200ep", supp. "7ep")** — `paper_experiments/coldiff_finetune/`
  (launchers for the official ColorizeDiffusion v1/v2 code; the fine-tuned weights were not preserved).

## Known deviations and gaps (honest list)

- **Dot placement rule.** Sec. IV-A describes the dot as the truncated mean of the longest-path pixels; the stored
  hint maps (training data and the released test maps) were produced with the medoid rule (`dot_method="medoid"`,
  the library default; verified on the released maps). All reported numbers use the stored maps, so results are
  unaffected; `dot_method="mean"` reproduces the text's rule.
- **Table II, DetFill scribble SSIM** is printed as 0.724; the value is 0.72348 (printed as 0.7235 in the
  supplementary α-grid table). A rounding artefact in the main-table typesetting.
- **Diffusart-retrain SSIM.** The recovered per-ratio record of that run reproduces PSNR 19.457 and LPIPS 0.177
  exactly but gives SSIM 0.7085 where the table prints 0.717 ± 0.066; the seven-metric evaluation the other cells were
  taken from was not recovered (`expected/additional_training/diffusart_retrain_R3-2/NOTE.md`).
- **SSIM depends on the torchmetrics version.** The published SSIM values were computed with torchmetrics 1.4.0; use
  the pinned version (`requirements-metrics.txt`, `replicability/environment.yml`) — 1.8.x gives per-image SSIM values
  that differ by up to 0.04 (all other metrics are unaffected).
- **Line-art source labels.** The DetFill data loader maps `sketch_type` 0 / 1 / 2 to sketch simplification / XDoG /
  SketchKeras; the archived per-ratio files name index 0 "XDoG" and 1 "pysimp" (a label swap in the old evaluation
  script). Every published value is a mean ± SD over the three sources, so no number depends on this; the per-source
  tables written by A1 use the loader's mapping.
- **Not covered by released per-ratio data:** the Table II/III *dot* rows of DetFill, all baseline rows
  (PaintsTorch, Diffusart, ColorizeDiffusion v1/v2), the natural-image (ImageNet) tables and the legacy per-source
  tables of the supplement (2024 checkpoints). The dot row and the baselines can be regenerated with layer D and the
  respective official code (the ColorizeDiffusion fine-tuned weights were not preserved). The natural-image DetFill
  models and the ImageNet test hint maps are released (v1.2), but the natural-image loader path still uses our absolute
  split lists (`detfill/configs/real/*.txt`) and the archive cannot tie the printed ImageNet values to one checkpoint
  file (`checkpoints/README.md`), so those tables are "same protocol", not bit-exact, reproductions.
- **User study.** The released trial table is anonymised and reproduces Tables IV/V and the GLMM exactly; the stimuli
  of three of the four methods are released (v1.3), the ColorizeDiffusion-v2 stimuli were not preserved. The
  response-letter analyses in `paper_experiments/userstudy_rank_stability/` read the raw per-participant CSVs through
  `common/ab_loader.py`, which are not released.

## Layout

```
reproduce/
  scripts/            A1_tables_from_released_metrics.py  A2_userstudy_glmm.py  eval_per_ratio.py
                      run_hauc_pipeline.sh  B9_train_detfill.sh  sketch_tools/
  examples/           run_examples.sh  make_examples.py  compare_with_expected.py  expected/  README.md
  expected/           released per-ratio metric files and JSON summaries (inputs of A1), recomputed_tables/
  data/               userstudy/glmm_trials.csv, splits/{train,valid,test}.txt
  paper_experiments/  verbatim launchers and analysis scripts of the paper and the revision (README.md there)
  requirements-metrics.txt   perceptual-metric packages (pinned)
  output/             created by the scripts (git-ignored)
```
