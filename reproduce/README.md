# Reproducing the paper's results

This directory is the reproduction package for

> Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation
> (IEEE TVCG, 2026). Main paper: Tables II–V, Fig. 9. Supplement: α-grid, segmentation-dependency,
> seed-sensitivity and additional-training tables.

It has three layers, from cheapest to most expensive:

| Layer | What it does | Needs | Time |
|---|---|---|---|
| **A. Recompute the tables from the released metrics** | Rebuilds every Hint-AUC table of the paper from the per-ratio metric files and the user-study trial table in `expected/` and `data/`, and checks each printed cell | numpy, pandas, statsmodels (CPU) | < 1 min (+ ~1 min GLMM) |
| **B. Re-run the pipeline with the released checkpoints** | Colorizes the test split at every hint ratio and recomputes the 7 metrics and Hint-AUC | GPU, Danbooru2021 test images, sketches, released hint maps and checkpoints | ~14 GPU-h per table row (3,000 images × 8 ratios × 3 sketch sources) |
| **C. Verbatim experiment launchers** (`paper_experiments/`) | The scripts that were actually run for the paper and its revision (cluster paths hard-coded), as example code | our cluster | days |

`../replicability/run.sh` (Fig. 9, one command, ≈3 min on a GPU) is the Graphics Replicability Stamp entry point
and is independent of this directory.

---

## A. Recompute the tables from the released metrics (no GPU)

```bash
conda env create -f replicability/environment.yml && conda activate detfill-grsi   # or any env with numpy+pandas+statsmodels
python reproduce/scripts/A1_tables_from_released_metrics.py   # Tables II/III (DetFill rows) + supplementary tables
python reproduce/scripts/A2_userstudy_glmm.py                 # Tables IV/V + GLMM statistics (Sec. VII)
```

`A1` writes markdown tables to `reproduce/output/tables/` and a cell-by-cell comparison with the numbers printed in
the paper (`check_report.csv`). Reference copies of these outputs are in `expected/recomputed_tables/`.

Current status (run on 2026-09-26): **A1: 322 of 324 compared cells match** the paper to the printed precision;
**A2: all 30 percentages of Tables IV/V match exactly, and the three GLMM statistics** (z = 30.18; χ²(7) = 618.33,
p = 2.75e-129; χ²(14) = 324.04, p = 1.12e-60) **match**. The two A1 exceptions are documented under
"Known deviations" below (a rounding artefact and one provenance gap); neither changes a result.

What each table is rebuilt from:

| Paper item | Script section | Released input (`reproduce/expected/…`) | Rule |
|---|---|---|---|
| Table II, DetFill scribble row (7 metrics) | A1 [1] | `alpha_grid/B_dense_curve/B_dense_curve__per_ratio_summary*.csv` (per-sketch, per-ratio means of the 96-ch scribble model; the dense α sweep contains the 8 paper ratios) | Hint-AUC = trapezoid over α∈{0,.01,.03,.05,.10,.25,.50,1} per sketch source; mean ± sample SD over the 3 sources |
| Supp. Table “sensitivity to the α grid” (5 grids) | A1 [2] | same dense sweep | same, re-integrated on each grid (linear interpolation when a grid point was not measured) |
| Supp. Table “segmentation dependency” (3×3) | A1 [3] | `segmenter_dependency/hauc_7m/<train>_model__on_<eval>/per_ratio_summary.csv` (+ the dense sweep for Felzenszwalb→Felzenszwalb) | same |
| Table III, DetFill scribble row (fixed random order) | A1 [4] | `table3_scribble/hauc7_scribble.json` | per-sketch Hint-AUC as stored; mean ± SD over sources |
| Supp. Table “additional training”, Diffusart-retrain (PSNR/SSIM/LPIPS) | A1 [5] | `additional_training/diffusart_retrain_R3-2/hauc_runs_v3_*/hauc_summary.json` | same (see NOTE.md there) |
| Supp. Table “size-sort vs random” (10 seeds, N = 300) | A1 [6] | `seed_sensitivity/metrics_felz96/seed*_alpha_*/per_ratio_summary.csv`, `seed_sensitivity/aggregated_sort.csv` | random: per-α mean ± SD (ddof = 1) over seeds, Hint-AUC per seed then mean ± SD; α = 0 % and 100 % are seed-independent (measured once) |
| Tables IV, V (user study) | A2 | `data/userstudy/glmm_trials.csv` (32 participants × 192 forced-choice trials, anonymised) | preference counts |
| GLMM statistics (Sec. VII) | A2 | same | `statsmodels` `BinomialBayesMixedGLM` (variational Bayes), random intercepts for participant and image; Model A intercept only, Model B `opponent × ratio` (sum coding), diagonal Wald χ² |
| Table II dot row, baselines (PaintsTorch, Diffusart, ColorizeDiffusion) | — | not released as per-ratio files (see “Not covered”) | |

`expected/` also contains the response-letter analyses that are not printed in the paper
(`userstudy_rank_stability/A-1..A-3`, `segmenter_sensitivity/C-1, C-2`, `alpha_grid/A-5`), with their scripts under
`paper_experiments/`.

## B. Re-run inference and evaluation with the released checkpoints

### B.1 Data

1. **Test images.** The test split is the 3,000 Danbooru2021 images listed in `data/splits/test.txt`
   (`detfill/configs/illust/test.txt`; ids ending in `016`, i.e. Danbooru2021 bucket `0016`). Obtain them from the
   Danbooru2021 distribution (we do not redistribute the images), and store them as
   `<DATA_ROOT>/segmentations/originals/<id>.image.png` (512 × 512, the preprocessing of `hint_generation/canonical/all_segmentations.py`).
2. **Line art.** `<DATA_ROOT>/sketch/{XDoG,pysimp,sketchkeras}/<id>.png` from the three extractors used in the paper;
   wrappers/ports are in `scripts/sketch_tools/` (XDoG port `XDoG_xdog_dan2021_tog2024.py`; SketchKeras
   `sketchkeras_mainpng3.py` needs the public `mod.h5` weights; sketch simplification `pysketchsimplify_pysimplify03a.py`
   needs the public model of Simo-Serra et al.). XDoG draws its σ/k jitter from an unseeded RNG in the original
   script, so regenerated XDoG sketches are not bit-identical to ours.
3. **Deterministic hints.** Use the stored test-split maps (release asset `test_split_hint_maps_64.tar.gz`,
   layout `hint_from_regions_64_rev/` + `region64/` → copy `region64/*` to `<DATA_ROOT>/hint_from_regions_256/`),
   which are the exact inputs of every reported number. Regenerating them with `hintauc`/`hint_generation/` is
   deterministic per environment but not bit-identical across environments (FilFinder medial-axis tie-breaking).
4. **Checkpoints.** `checkpoints/README.md` (v1.0: scribble 96-ch, dot 64-ch; v1.1: the two segmenter-retrained
   scribble models of the supplementary study; v1.2: the Diffusart-retrain models and the natural-image (ImageNet)
   DetFill models with the ImageNet test-split hint maps).

### B.2 One table row = one command

```bash
# Table II, DetFill scribble row (default), 7 metrics, Hint-AUC over the paper grid
DATA_ROOT=/data/danbooru_test GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# Table II, DetFill dot row
DATA_ROOT=/data/danbooru_test HINT=dot GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# Table III (fixed random region order = ascending label order of the stored region map)
DATA_ROOT=/data/danbooru_test HINT_ORDER=label GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
# Supplementary α-grid study: dense sweep, then integrate on the alternative grids
DATA_ROOT=/data/danbooru_test RATIOS="$(seq -f %.2f 0 0.02 1 | tr '\n' ' ') 0.01 0.03 0.05 0.25" TAG=dense \
    bash reproduce/scripts/run_hauc_pipeline.sh
python reproduce/paper_experiments/alpha_grid/B_auc_grid_sensitivity_7m.py \
    --summary reproduce/output/dense/metrics/per_ratio_summary.csv --out_dir reproduce/output/dense/auc
# smoke test (20 images per cell, ~10 min on one GPU)
DATA_ROOT=/data/danbooru_test LIMIT=20 TYPES="2" GPU=0 bash reproduce/scripts/run_hauc_pipeline.sh
```

`run_hauc_pipeline.sh` = `detfill/main.py --sample_to_eval` per (ratio, sketch type) followed by
`scripts/eval_per_ratio.py`, which uses the released evaluator `hintauc.metrics.Evaluator`
(port of the paper's `evaluation/eval_single_run.py`: 256 × 256, MSE/PSNR/SSIM on [0,1]; LPIPS-Alex; OpenCLIP
ViT-B-32 laion2b; DINOv2-base; DreamSim) and writes `per_ratio_summary.csv` in the same format as the files in
`expected/`, plus `hauc.json`. The perceptual metrics need `pip install -r reproduce/requirements-metrics.txt`
(or `pip install -e ".[perceptual]"`).

Expected agreement (checked 2026-09-26 on 160 image pairs of the archived paper outputs, sketch type 2, 8 ratios):
with `requirements-metrics.txt` the evaluator reproduces the archived per-image values to numerical precision for
MSE, PSNR, LPIPS, DINO and DreamSim (max |Δ| ≤ 3e-6), SSIM to 5e-5 with the pinned `torchmetrics==1.4.0`
(torchmetrics 1.8 changed the SSIM implementation: per-image |Δ| up to 0.04, so the pin matters), and OpenCLIP
within 8e-4 (the archived run scored OpenCLIP in half precision). Inference is seeded (`--seed 1234`) and the
stored hint maps remove the hint-generation variance; re-running the diffusion sampler on another GPU class can
change individual outputs slightly (CUDA non-determinism), and in our re-runs the sketch-averaged Hint-AUC values
agreed with Table II to the printed precision.

### B.3 Other studies

* **Seed sensitivity (supp.)** — `paper_experiments/seed_sensitivity/gen_seeded_random_hints.py` bakes a seeded
  uniformly random region order into 64 × 64 hint maps (10 seeds × 6 intermediate ratios; α = 0 % and 100 % are
  seed-independent); each (seed, α) set is then run through the pipeline above at `RATIOS=1.00` with `DATA_ROOT`
  pointing at a wrapper root whose `hint_from_regions_64_rev/` is that seed's maps
  (`build_wrappers.py` builds the symlink wrappers), evaluated on the first 300 test ids with SketchKeras line art
  (`eval_n300/`), and aggregated exactly as in A1 [6].
* **Segmentation dependency (supp.)** — region maps and hints from DanbooRegion / SLIC
  (`paper_experiments/segmenter_dependency/r2-2_scripts/D_danbooregion_seg.py`, `D_retrain_gen_hints.py`,
  `D_danboo_hints.py`, and `hint_generation/generate_hints.py --segmenter slic`), the two retrained checkpoints
  of release v1.1 (`CKPT=…danbooregion_200ep.pth` / `…slic_200ep.pth`), and the pipeline above with `DATA_ROOT`
  pointing at each segmenter's evaluation layout (`configs/eval_on_{felz,danbooregion,slic}.yaml` are the configs used).
  Training of the two models: `D_train_launch.sh` / `D_stage_train.sh` (200 epochs, same protocol as B9).
* **Training DetFill from scratch** — `scripts/B9_train_detfill.sh` (the paper's `train.sh` commands with the
  released config names; ~4–5 days on 10 GPUs).
* **Diffusart-retrain (supp.)** — `paper_experiments/diffusart_retrain/code/` (our Diffusart re-implementation:
  training loop, deterministic-hint loader, inference and evaluation; see its README) with the two released EMA
  checkpoints of release v1.2 (`diffusart_retrain_{scribble,dot}_dethint_200ep_ema.pth`).
* **ColorizeDiffusion fine-tuning (Table II/III “200ep”, supp. “7ep”)** — `paper_experiments/coldiff_finetune/`
  (fine-tuning/evaluation launchers for the official ColorizeDiffusion v1/v2 code).

## Known deviations and gaps (honest list)

* **Dot placement rule.** Sec. IV-A describes the dot as the truncated mean of the longest-path pixels; the stored
  hint maps (training data and the released test maps) were produced with the medoid rule (`dot_method="medoid"`,
  the library default; verified on the released maps). All reported numbers use the stored maps, so results are
  unaffected; `dot_method="mean"` reproduces the text's rule.
* **Table II, DetFill scribble SSIM** is printed as 0.724; the value is 0.72348 (printed as 0.7235 in the
  supplementary α-grid table and as 0.7235 here). Rounding artefact in the main-table typesetting.
* **Diffusart-retrain SSIM.** The recovered per-ratio record of that run reproduces PSNR 19.457 and LPIPS 0.177
  exactly but gives SSIM 0.7085 where the table prints 0.717 ± 0.066; the seven-metric evaluation the other cells
  were taken from was not recovered (`expected/additional_training/diffusart_retrain_R3-2/NOTE.md`).
* **SSIM depends on the torchmetrics version.** The published SSIM values were computed with torchmetrics 1.4.0; use the
  pinned version (`requirements-metrics.txt`, `replicability/environment.yml`) — 1.8.x gives per-image SSIM values that differ
  by up to 0.04 (all other metrics are unaffected).
* **Sketch-type labels.** The dataset loader maps `sketch_type` 0/1/2 to sketch simplification (pysimp) / XDoG /
  SketchKeras; the archived per-ratio files name index 0 “XDoG” and 1 “pysimp”. Every published value is a mean ±
  SD over the three sources, so no number depends on this; per-source tables in `output/tables/` use the loader's mapping.
* **Not covered by released per-ratio data:** the Table II/III *dot* rows of DetFill, all baseline rows
  (PaintsTorch, Diffusart, ColorizeDiffusion v1/v2), the natural-image (ImageNet) tables and the legacy per-sketch
  tables of the supplement (2024 checkpoints). The dot row and the baselines can be regenerated with layer B and the
  respective official code. The natural-image DetFill models and the ImageNet test-split hint maps are released (v1.2),
  but the natural-image loader path still uses the lab's absolute split lists (`detfill/configs/real/*.txt`) and
  the archive cannot tie the printed ImageNet values to one checkpoint file (see `checkpoints/README.md`), so
  those tables are "same protocol", not bit-exact, reproductions.
* The user-study analyses A-1..A-3 in `paper_experiments/userstudy_rank_stability/` read the raw per-participant
  CSVs through `common/ab_loader.py`; the release contains the anonymised trial table instead (`A2` reproduces
  Tables IV/V and the GLMM from it).

## Layout

```
reproduce/
  scripts/            A1_tables_from_released_metrics.py  A2_userstudy_glmm.py  eval_per_ratio.py
                      run_hauc_pipeline.sh  B9_train_detfill.sh  sketch_tools/
  expected/           released per-ratio metric files and JSON summaries (inputs of A1), recomputed_tables/
  data/               userstudy/glmm_trials.csv, splits/{train,valid,test}.txt
  paper_experiments/  verbatim launchers and analysis scripts of the paper / revision (README.md there)
  output/             created by the scripts (git-ignored)
```
