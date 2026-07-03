# Hint-AUC evaluation

Scores colorization outputs per hint ratio (MSE / PSNR / SSIM / LPIPS / OpenCLIP /
DINOv2 / DreamSim) and integrates each metric curve over the ratio grid
α ∈ {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00} with the trapezoidal rule
(**Hint-AUC**). The `hintauc` library exposes the same computation as functions
(`hintauc.Evaluator`, `hintauc.evaluate_hint_curve`, `hintauc.hint_auc`).

## Recommended pipeline (`dense/`, used for the paper's revision results)

Works directly on the output layout of `detfill/run_inference.sh`
(`.../sample_to_eval/illust/<hint_type>/<sketch_type>/<ratio>/{200,ground_truth}/`):

```bash
# 1. per-ratio metrics (resumable; writes per_image.csv + per_ratio_summary.csv)
python evaluation/dense/eval_curve.py \
    --results_root detfill/results/dataset_name/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble \
    --sketches 0 1 2 \
    --metrics psnr ssim lpips dreamsim \
    --out_dir eval_out

# 2. aggregate per_image.csv -> per_ratio_summary.csv (if you skipped it above)
python evaluation/dense/build_summary.py --help

# 3. Hint-AUC over the ratio grid (per metric, incl. alternative grids)
python evaluation/dense/auc_grids.py --summary eval_out/per_ratio_summary.csv
```

`per_ratio_summary.csv` columns: `sketch, sketch_name, ratio, n, <metric>_mean, <metric>_std`.

## Original-submission pipeline

- `eval_single_run.py` — metrics for ONE inference run directory
  (`--run_dir .../<run>` with samples under `test/samples_cfg_scale_5.00`, generated
  `A_B.png` files paired with `--gt_root` by dataloader order). Writes a one-row CSV.
- `calc_hint_auc.py` — Hint-AUC from eight per-ratio CSVs
  (`--alpha_csv 0.00=... --alpha_csv 0.01=...`, stdlib-only).
- `avg_hint_auc.py` — mean ± sample SD across summary CSVs
  (e.g., over the three sketch extractors).

## Notes

- Pixel metrics are computed on [0,1] tensors resized to 256×256; LPIPS uses the
  AlexNet backbone in [-1,1]; OpenCLIP (ViT-B-32, laion2b_s34b_b79k) and DINOv2-base
  report `(cosine+1)/2`; DreamSim reports its distance. Formulas match
  `hintauc/metrics.py` one-to-one.
- OpenCLIP / DINOv2 / DreamSim download their weights on first use (network needed);
  LPIPS weights ship with the pip package.
