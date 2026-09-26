# The launchers that were actually run

The shell and Python files that produced the paper's and the revision's numbers, copied verbatim from our lab tree
(2024–2026). They are **example code**: cluster paths (`/scratch/madono/...`,
`/home/madorin/gitlab/...`), conda environment names (`BBDM`, `py310`) and GPU ids are hard-coded, several were
written for a specific host queue, and they are not maintained as runnable entry points. The runnable, path-free
equivalents are `../scripts/run_hauc_pipeline.sh`, `../scripts/eval_per_ratio.py`, `../scripts/A1_*` / `A2_*` and
`../examples/`. Directory names with "R1-1", "R2-2" and so on refer to the reviewer comments of the first TVCG revision
(reviewer number – comment number) that the experiment answered.

| Directory | Paper item | Contents |
|---|---|---|
| `tableII_inference/` | Table II (DetFill rows) | `run_inference_mr.sh`: the loop over 8 hint ratios × 3 line-art sources (`main.py --sample_to_eval`) for the dot and scribble models. Released as `detfill/run_inference.sh`. |
| `table3_label_order/` | Table III (DetFill rows) | `launch_tableIII.sh`: the same loop with the fixed ascending-label region order (`hint_order: label`). |
| `alpha_grid/` | Table II scribble row; supplementary hint-ratio-grid (α-grid) table and dense-curve figure | `B_run.sh` / `B_run_parallel.sh` → `B_eval_dense_curve.py` (per-image metrics over the 57-ratio sweep) → `B_build_summary.py` (per-ratio summary) → `B_auc_grid_sensitivity_7m.py` and `B_vary_N_auc.py` (Hint-AUC on alternative grids) → `B_plot_dense_curve.py`. `A-5_alpha_grids.py`: user-study win-rate AUC under alternative grids (review response). Results: `../expected/alpha_grid/`. |
| `seed_sensitivity/` | Supplementary "size-ordered vs. random selection" table and figure | `gen_seeded_random_hints.py` (seeded random region order baked into 64 × 64 maps), `build_wrappers.py` (symlinked data roots per seed and ratio), `run_seedexp_{detfill,diffusart,coldiff}.sh`, `watcher_launch.sh` (queueing), `detfill_cfgs/` (the generated configs), `eval_n300/` (`eval_sweep.sh` + the seven-metric evaluator restricted to the 300-image SketchKeras subset; the size-ordered counterpart used the same scripts on the deterministic hints). Results: `../expected/seed_sensitivity/`. |
| `segmenter_dependency/` | Supplementary segmentation-dependency table and figures | `10_run_ratio.sh` / `20_launch.sh` / `30_push_and_eval.sh` (per-ratio inference across a GPU pool, copy to the file server, evaluation), `configs/` (training configs of the 96-channel model and the `eval_on_{felz,danbooregion,slic}.yaml` evaluation configs), `r2-2_scripts/` (DanbooRegion/SLIC segmentation and hint generation: `D_danbooregion_seg.py`, `D_danboo_hints.py`, `D_retrain_gen_hints.py`; retraining: `D_train_launch.sh`, `D_stage_train.sh`; cross-evaluation matrix: `D_reaper_eval_matrix.sh`; figure montage: `D_build_allratio_figs.py`, `D_montage.py`; `C-1_segmenters.py`, `C-2_perturbation.py` = hint-containment and perturbation analyses of the review response), `R2-2_README_original.md`. Checkpoints: release v1.1; test hint maps: release v1.3. Results: `../expected/segmenter_dependency/`, `../expected/segmenter_sensitivity/`. |
| `segmenter_sensitivity/` | Review response | `C-1_segmenters.py`, `C-2_perturbation.py` (same as above, original location). |
| `userstudy_rank_stability/` | Review response | `A-1_bootstrap.py` (per-participant bootstrap of win rates and rank flips), `A-2_binomial.py`, `A-3_bradley_terry.py`; they read the raw per-participant CSVs through the loader in `common/` (not released; the anonymised trial table is `../data/userstudy/glmm_trials.csv`). Results: `../expected/userstudy_rank_stability/`. |
| `diffusart_retrain/` | Diffusart baseline rows (Tables II/III) and the supplementary "additional training comparisons" (Diffusart-retrain) | `code/`: our complete Diffusart re-implementation (training loop, deterministic-hint loader `data/data_load_det.py`, `train_det_{scribble,dot}.py`, `infer_det_proposed.py`, evaluation; README inside); `R3-2_README_original.md`. Checkpoints: release v1.2. Results: `../expected/additional_training/`. |
| `coldiff_finetune/` | Table II/III "ColDiff (200ep)", supplementary "(7ep)" | `run_coldiff_finetune_eval_naga1.sh`, `coldiff_v{1,2}_inference_official_fixed*.py` (deterministic-hint inference wrappers around the official ColorizeDiffusion code; CC BY-NC-SA 4.0, see `LICENSE` in that directory), `EXPERIMENT_coldiff_finetune.md` (protocol notes). |

Conventions shared by all launchers: hint ratios `0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00`; `sketch_type` 0 / 1 / 2 =
sketch simplification / XDoG / SketchKeras in the dataset loader; outputs land in
`<result_path>/dataset_name/BrownianBridge_<hint>_illust/sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/200/<id>.image.png`;
inference seed 1234; the evaluation resizes to 256 × 256 and the Hint-AUC is the trapezoidal integral over α ∈ [0, 1].
The split lists referenced by the launchers (`configs/illust/{train,valid,test}_paper.txt`) are identical to
`../data/splits/{train,valid,test}.txt` (see `SPLIT_LISTS.md`).
