# Diffusion_v1_tvcg_R3-2 — Diffusart trained on deterministic hints (revision R3-2)

Purpose: complete the 2x2 (training-hint distribution x evaluation protocol) asked by R3-2.
The paper's Diffusart (Diffusion_v1_comp, ckpt labrepo_*/checkpoint_198000.pth) is trained
with its ORIGINAL random-hint simulator (making_mask_v3). This repo trains the SAME
architecture/schedule on the PROPOSED deterministic hints instead.

## Provenance
- Code base: copy of Diffusion_v1_comp (paper version), code only (no checkpoints/results).
- Det-hint loader: data/data_load_det.py = verbatim port of
  Diffusion_v1/data/data_load.py::MyData_train_scrib (approach='proposed'):
  region map (image_region256) -> regions sorted by size desc -> random-ratio PREFIX
  -> mask applied to deterministic scribble/dot col+mask 256 files -> random dilate 1-4 + blur.
  Only change: hardcoded /scratch/USER/main_exp root -> env DIFFUSART_DET_DATA_ROOT.
- Trainer: training/training_det.py = training_multi.py with:
  (1) t sampled with original.shape[0] (was module-global batch_size),
  (2) warmup scaled by accumulation_steps (no-op at accum=1),
  (3) EMA gated to optimizer steps (no-op at accum=1),
  (4) ckpt names in optimizer-step units,
  (5) final save path fixed (full_checkpoint_dir was undefined upstream).

## Schedule match (paper Sec VI-A / comp run)
- AdamW lr 2e-5, 200 epochs, EFFECTIVE batch 20, cosine + 5000-step warmup, EMA 0.995, 256px.
- comp run realized eff-20 as 10 DDP ranks x batch 2 (SyncBN batch 20, ckpt_198000 = epoch 198).
- here: single GPU batch 20 x accum 1 -> SAME SyncBN batch (20), same optimizer-step count
  (~1000/epoch x 200 = 200k), same lr curve at optimizer steps.
- compare at matched step: our checkpoint_198000.pth (non-EMA) vs comp checkpoint_198000.pth.

## Data
- /scratch/USER/diffusart_det_R3-2/illust/{hint_from_regions,segmentation_regions}/felzenszwalb,
  sketch/{pysimp,XDoG,sketchkeras} — staged from
  /home/USER/datasets/labrepo/main_exp_felzenszwalb_fixdot/illust (same source the felz/
  DetFill pipeline uses; see rebuttal/R2/R2-2_segmentation_dependency/D_RETRAIN_DATA.md).
- Train list: configs/illust/train_paper.txt (19,999 ids) — identical to comp run.

## Run
bash run_train_R3-2.sh            # waits for staging + GPU3 free, scribble then dot
# logs: /scratch/USER/diffusart_det_R3-2/logs/train_det_{scribble,dot}.log

## After training (TODO)
- inference: test_labrepo.py-style det + rand runs (8 ratios x 3 sketches) with both new ckpts
- eval: Hint-AUC via Evaluation_paper/eval_single_run_batched.py (gt_order per mode)
- deliver: Table II/III rows "Diffusart (det-trained)" + R3-2 coverletter fill
