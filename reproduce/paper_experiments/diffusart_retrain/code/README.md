# Diffusart re-implementation (baseline of the paper) and its deterministic-hint retraining

This directory is a copy of our lab's PyTorch re-implementation of **Diffusart** (Carrillo et al., CVPRW 2023;
no official training code was available), which produced the paper's Diffusart baseline rows, plus the
deterministic-hint retraining used for the supplementary "additional training comparisons"
(Diffusart-retrain). It is our own code (MIT), kept verbatim as example code: cluster paths and the `BBDM`
conda environment are hard-coded in the launchers, and several files are exploratory variants.

Inputs. `infer_det_proposed.py` reads the **64 × 64** deterministic hint maps `<id>.image_{scribble,dot}_{col,mask}64.png`
and the region map `<id>.image_region64.png` from `<data_root>/<domain>/hint_from_regions/<segmenter>/<bucket>/` and
upsamples them to 256 × 256 with nearest-neighbour interpolation (the maps of release v1.0; this is how the paper's
Diffusart-retrain record was produced), the line art from `<data_root>/<domain>/sketch/<type>/<bucket>/<id>.png` and the
colour image from `<data_root>/<domain>/segmentation_regions/<segmenter>/<bucket>/<id>.image.png` (`<bucket>` is the
directory in the split list, e.g. `0016`; the lists in `configs/illust/` are relative `<bucket>/<id>.image.png` lines).
Environment: PyTorch 2.5 and `diffusers` 0.32 (`pip install torch torchvision diffusers==0.32.2 scikit-image opencv-python`).
Example (12 images, full hints, SketchKeras line art; the record used `--num_steps 100 --batch_size 16`):

```bash
DIFFUSART_DET_DATA_ROOT=/data/diffusart python infer_det_proposed.py --checkpoint_path diffusart_retrain_scribble_dethint_200ep_ema.pth \
    --hint scribble --ratio 1.0 --sketch_index 2 --data_root /data/diffusart --region_root /data/diffusart/illust/hint_from_regions/felzenszwalb \
    --test_list configs/illust/test_paper.txt --out_dir out/scribble --num_steps 100 --batch_size 16 --limit 12
```

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
