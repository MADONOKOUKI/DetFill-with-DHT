# Hint-AUC evaluation: original scripts

These are the evaluation scripts used for the paper. They score colorizations per hint ratio and integrate each
metric curve over the ratio grid α ∈ {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00} with the trapezoidal rule
(**Hint-AUC**). The `hintauc` library exposes the same computation (`hintauc.Evaluator`,
`hintauc.evaluate_hint_curve`, `hintauc.hint_auc`), and `../reproduce/scripts/eval_per_ratio.py` is the maintained
command-line version that writes the released `per_ratio_summary.csv` format.

## Metrics

- Pixel metrics on [0, 1] images resized to 256 × 256: MSE, PSNR, SSIM (torchmetrics 1.4.0).
- LPIPS (AlexNet, inputs in [−1, 1]); OpenCLIP (ViT-B-32, laion2b) and DINOv2-base as `(cosine + 1) / 2`; DreamSim distance.
- Weights of the perceptual metrics are downloaded on first use.
- Metrics added after the paper (MAE, MS-SSIM, CIEDE2000, LPIPS-VGG, DISTS, set-level FID / KID) live in the library only.

## Pipeline used for the revision results (`dense/`)

Works on the output layout of `../detfill/run_inference.sh`
(`.../sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/{200,ground_truth}/`):

```bash
python evaluation/dense/eval_curve.py --results_root <.../sample_to_eval/illust/scribble> \
    --sketches 0 1 2 --metrics psnr ssim lpips dreamsim --out_dir eval_out     # per_image.csv + per_ratio_summary.csv
python evaluation/dense/build_summary.py --help                                # per_image.csv -> per_ratio_summary.csv
python evaluation/dense/auc_grids.py --summary eval_out/per_ratio_summary.csv  # Hint-AUC on the paper grid and alternatives
```

`per_ratio_summary.csv` columns: `sketch, sketch_name, ratio, n, <metric>_mean, <metric>_std`.

## Scripts of the original submission

| File | Role |
|---|---|
| `eval_single_run.py` | metrics for one inference run directory (pairs generated `A_B.png` files with the ground truth in loader order); one-row CSV |
| `calc_hint_auc.py` | Hint-AUC from eight per-ratio CSVs (`--alpha_csv 0.00=... --alpha_csv 0.01=...`), standard library only |
| `avg_hint_auc.py` | mean ± sample SD across summary CSVs (for example over the three line-art extractors) |
