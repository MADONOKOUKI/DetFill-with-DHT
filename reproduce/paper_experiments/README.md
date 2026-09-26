# paper_experiments — the launchers that were actually run

These are the shell and Python files used to produce the paper's and the revision's numbers, copied verbatim
from our lab tree (2024–2026). They are **example code**: cluster paths (`/scratch/madono/...`,
`/home/madorin/gitlab/...`), conda environment names (`BBDM`, `py310`) and GPU ids are hard-coded, several were
written for a specific host queue, and they are not maintained as runnable entry points. The runnable,
path-free equivalents are `../scripts/run_hauc_pipeline.sh`, `../scripts/eval_per_ratio.py` and
`../scripts/A1_*/A2_*`. Each directory below says what the scripts produced and where the resulting numbers are.

| Directory | Paper item | Contents |
|---|---|---|
| `tableII_inference/` | Table II (DetFill rows) | `run_inference_mr.sh`: the 8-ratio × 3-sketch inference loop (`main.py --sample_to_eval`) for the dot/scribble models. Released as `detfill/run_inference.sh`. |
| `table3_label_order/` | Table III (DetFill rows) | `launch_tableIII.sh`: same loop with the fixed ascending-label region order (`hint_order: label`). |
| `alpha_grid/` | Table II scribble row, supp. α-grid table, Fig. of the dense curve | `B_run.sh` / `B_run_parallel.sh` → `B_eval_dense_curve.py` (per-image 7 metrics over the 57-ratio sweep) → `B_build_summary.py` (per_ratio_summary.csv) → `B_auc_grid_sensitivity_7m.py` / `B_vary_N_auc.py` (Hint-AUC on alternative grids) → `B_plot_dense_curve.py`. `A-5_alpha_grids.py`: user-study win-rate AUC under alternative grids (response letter). Results: `../expected/alpha_grid/`. |
| `seed_sensitivity/` | Supp. “size-sort vs random” table and figure | `gen_seeded_random_hints.py` (seeded random region order baked into 64×64 maps), `build_wrappers.py` (symlink data roots per (seed, α)), `run_seedexp_{detfill,diffusart,coldiff}.sh`, `watcher_launch.sh` (queueing), `detfill_cfgs/` (the generated configs), `eval_n300/` (`eval_sweep.sh` + the 7-metric evaluator restricted to the 300-image SketchKeras subset; the size-sort counterpart used the same scripts on the deterministic hints). Results: `../expected/seed_sensitivity/`. |
| `segmenter_dependency/` | Supp. “segmentation dependency” table and figures | `10_run_ratio.sh` / `20_launch.sh` / `30_push_and_eval.sh` (per-ratio inference across a GPU pool, rsync to NFS, evaluation), `configs/` (training configs of the 96-ch model and the `eval_on_{felz,danbooregion,slic}.yaml` evaluation configs), `r2-2_scripts/` (DanbooRegion/SLIC segmentation and hint generation `D_danbooregion_seg.py`, `D_danboo_hints.py`, `D_retrain_gen_hints.py`; retraining `D_train_launch.sh`, `D_stage_train.sh`; cross-evaluation matrix `D_reaper_eval_matrix.sh`; figure montage `D_build_allratio_figs.py`, `D_montage.py`; `C-1_segmenters.py`, `C-2_perturbation.py` = hint-containment / perturbation analyses of the response letter), `R2-2_README_original.md`. Checkpoints: release v1.1. Results: `../expected/segmenter_dependency/`, `../expected/segmenter_sensitivity/`. |
| `segmenter_sensitivity/` | Response letter (R1-3) | `C-1_segmenters.py`, `C-2_perturbation.py` (same as above, original location). |
| `userstudy_rank_stability/` | Response letter (R1-1/R1-2) | `A-1_bootstrap.py` (per-participant bootstrap of win rates and rank flips), `A-2_binomial.py`, `A-3_bradley_terry.py`; they read the raw per-participant CSVs through the loader in `common/` (not released; the anonymised trial table is `../data/userstudy/glmm_trials.csv`). Results: `../expected/userstudy_rank_stability/`. |
| `diffusart_retrain/` | Supp. “additional training comparisons” (Diffusart-retrain) | `run_train_R3-2.sh`, `training/training_det.py`, `data/data_load_det.py` (Diffusart trained with our deterministic hints, same 200-epoch protocol), `infer_det_proposed.py`, `eval_test.py`, `hint_gen_test.py`, `R3-2_README_original.md`; `test_wacv_randhint.py` (random-hint test-time variant). Results: `../expected/additional_training/`. |
| `coldiff_finetune/` | Table II/III “ColDiff (200ep)”, supp. “(7ep)” | `run_coldiff_finetune_eval_naga1.sh`, `coldiff_v{1,2}_inference_official_fixed*.py` (deterministic-hint inference wrappers around the official ColorizeDiffusion code), `EXPERIMENT_coldiff_finetune.md` (protocol notes). |

Conventions shared by all launchers: hint ratios `0.00 0.01 0.03 0.05 0.10 0.25 0.50 1.00`; `sketch_type` 0/1/2 =
sketch simplification / XDoG / SketchKeras in the dataset loader; outputs
`<result_path>/dataset_name/BrownianBridge_<hint>_illust/sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/200/<id>.image.png`;
inference seed 1234; the evaluation resizes to 256 × 256 and the Hint-AUC is the trapezoidal integral over α ∈ [0, 1].
