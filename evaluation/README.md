# Hint-AUC evaluation

Two pipelines (both compute MSE / PSNR / SSIM / LPIPS / OpenCLIP / DINO / DreamSim per
hint ratio, then integrate over the ratio grid with the trapezoidal rule):

- Original submission pipeline:
  `eval_single_run.py` (per-ratio metrics over one inference output dir)
  -> `calc_hint_auc_manual.py` (Hint-AUC over alpha in {0, .01, .03, .05, .10, .25, .50, 1.0})
  -> `avg_hint_auc_summary.py` (mean +/- SD aggregation across sketch sources).
- Revision (dense-grid) pipeline in `dense/`:
  `B_eval_dense_curve.py` (resumable batch evaluation over `{sketch_type}/{ratio}/200/` outputs)
  -> `B_auc_grid_sensitivity_7m.py` (7-metric Hint-AUC on multiple alpha grids;
  the front-loaded row is the paper value).

Metric backbones (lpips, open_clip, DINOv2 via transformers, dreamsim) are downloaded
automatically on first run.
