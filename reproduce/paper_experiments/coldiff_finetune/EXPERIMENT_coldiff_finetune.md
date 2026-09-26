# ColDiff v1/v2 fine-tune + Hint-AUC inference — experiment script

**Script:** [`run_coldiff_finetune_eval.sh`](run_coldiff_finetune_eval.sh) (in this folder)

**Answers:** R1-4 (strong modern baseline) **and** R2-3 (fairness of ColDiff retraining).

## What it does
Single-GPU, sequential on HOST_A. For each of 4 models, in order
**① v2-dot → ② v2-scr → ③ v1-dot → ④ v1-scr**:
1. **Fine-tune** from the *official released* ColorizeDiffusion weights
   (v2 ← `weights/v2-full.safetensors`, v1 ← `weights/mult-eps-newft.safetensors`)
   on ~20k illust images at config `base_learning_rate = 1e-6` (adaptation, not full retrain).
2. **Inference — Table II** (deterministic region-based hints): 8 hint ratios
   {0,1,3,5,10,25,50,100}% × 3 sketch sources (for mean±SD).
3. **Inference — Table III** (random sampling): same grid, `inference_official_fixed_random.py`.

## Code it drives (NOT copied here — lives in the model repos)
- v2: `/home/USER/gitlab/labrepo/main/colorizeDiffusion_v2/` (`train.py`, `inference_official_fixed[_random].py`, `configs/`)
- v1: `/home/USER/gitlab/labrepo/main/colorizeDiffusion/`

## Run (on HOST_A)
```bash
cd /home/USER/gitlab/labrepo/main/revision_materials/rebuttal/R1/R1-4_scope_baselines
DRY_RUN=1 bash run_coldiff_finetune_eval.sh            # preview + data check
GPU=<free> bash run_coldiff_finetune_eval.sh           # real run
```
(absolute paths inside; runnable from anywhere)

## Decide before running (see script header)
- **EPOCHS** (default 30; pretrained-start converges fast — 200 is overkill)
- **effective lr = base × batch × accum × gpu** → base 1e-6 × batch 5 = **5e-6**
- **`configs/training/sd2.1/v2.yaml` has `base_learning_rate` commented out** → set to `1.0e-6`
- **data**: defaults to the complete NFS copy `/home/USER/datasets/revision/colorizeDiffusion/illust/{images_train,images_test}`; rsync to HOST_A `/scratch` for speed
- **`sample_ratio: 0.2`** in `v2_dot.yaml` → set `null` for paper-style random-ratio training

## Outputs
- checkpoints: `/scratch/USER/colorizeDiffusion_v2_checkpoints/…`, `/scratch/USER/colorizeDiffusion_v1/checkpoints/…`
- inference: `/scratch/USER/colorizeDiffusion_v{1,2}/checkpoints/<model>/…`
- logs: `labrepo/main/coldiff_runs_logs/`
- **Hint-AUC (mean±SD) is STAGE 3 — commented template at the end of the script**
  (`Evaluation_paper/calc_hint_auc_manual.py` + `avg_hint_auc_summary.py`); not yet automated.

