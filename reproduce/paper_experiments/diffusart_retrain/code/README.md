# Diffusart re-implementation (baseline of the paper) and its deterministic-hint retraining

This directory is a copy of our lab's PyTorch re-implementation of **Diffusart** (Carrillo et al., CVPRW 2023;
no official training code was available), which produced the paper's Diffusart baseline rows, plus the
deterministic-hint retraining used for the supplementary "additional training comparisons"
(Diffusart-retrain). It is our own code (MIT), kept verbatim as example code: cluster paths and the `BBDM`
conda environment are hard-coded in the launchers, and several files are exploratory variants.

Entry points that matter:

| File | Purpose |
|---|---|
| `train_det_scribble.py`, `train_det_dot.py` | Diffusart-retrain: same architecture and 200-epoch schedule as the baseline (AdamW 2e-5, effective batch 20, cosine + 5,000-step warm-up, EMA 0.995, 256 px), trained on the deterministic region-based hints (`data/data_load_det.py`, the loader of `training/training_det.py`). Run names `baseline_scribble` / `baseline_dot`; the released EMA checkpoints (`diffusart_retrain_{scribble,dot}_dethint_200ep_ema.pth`, GitHub release v1.2) are their final states. |
| `run_train_R3-2.sh` | Launcher (staging wait, GPU wait, both trainings). |
| `infer_det_proposed.py` | Inference with the deterministic, area-sorted hints at a given ratio and sketch type (the Hint-AUC protocol); writes the per-ratio output tree consumed by `reproduce/scripts/eval_per_ratio.py`. |
| `eval_test.py`, `hint_gen_test.py`, `gen_scrib.py` | Evaluation / hint-generation checks used during the revision. |
| `main*.py`, `test_*.py`, `data/test_wacv*.py`, `*_64*.py` | Earlier baseline training/testing variants (random-hint baseline, 64-px hint experiments); not needed for the released checkpoints. |
| `models/`, `training/`, `utils.py`, `configs/` | Network definitions (U-Net with cross-attention conditioning), training loop, utilities and the data-split lists. |

Numbers reproduced with these checkpoints: the Diffusart-retrain scribble row of the supplementary table
(PSNR 19.457, LPIPS 0.177; see `reproduce/expected/additional_training/diffusart_retrain_R3-2/`).
