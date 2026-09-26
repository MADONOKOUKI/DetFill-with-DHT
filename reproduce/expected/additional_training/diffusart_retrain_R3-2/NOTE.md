# Diffusart-retrain (supplementary "Additional training comparisons")

`hauc_runs_v3_scribble_psnr_ssim_lpips/` is the per-ratio PSNR/SSIM/LPIPS record of the inference run
behind the **Diffusart-retrain, scribble** row of the supplementary table (Diffusart trained with our
deterministic hints under the same 200-epoch protocol as DetFill; 3 sketch sources x 8 hint ratios x
3,000 Danbooru2021 test images).

* Source run: an internal per-cell inference run (100 sampling steps, batch 16, the 64 × 64 hint maps of the test split
  upsampled to 256 × 256 with nearest-neighbour interpolation), checkpoint
  `Diffusion_v1_tvcg_R3-2/checkpoint/tog2024_scribble/final_model_ema.pth` (= release v1.2
  `diffusart_retrain_scribble_dethint_200ep_ema.pth`, SHA-256 `03ed65e3…`)
  (repository `Diffusion_v1_tvcg_R3-2` = Diffusart code trained on deterministic hints; see
  `reproduce/paper_experiments/diffusart_retrain/`).
* `A1_tables_from_released_metrics.py` recomputes the sketch-averaged Hint-AUC from `hauc_summary.json`:
  PSNR 19.457 and LPIPS 0.177 match the printed row exactly; the recomputed SSIM is 0.7085 while the
  printed row says 0.717 +- 0.066. The seven-metric evaluation from which the printed SSIM, MSE, OpenCLIP,
  DINO and DreamSim values (and the SDs) were taken was not recovered from the lab archive when this
  package was assembled, so those five cells are not covered by the released data.
* The dot row of the same table is not covered either.
