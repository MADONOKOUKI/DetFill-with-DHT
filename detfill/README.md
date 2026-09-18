# DetFill

Pixel-space Brownian Bridge diffusion model for sketch + deterministic-hint line-art
colorization. Fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT, © 2023 xuekt98)
adapted to the Hint-AUC protocol — see the repository root README for the full context.

## 1. Environment

```bash
conda env create -f environment.yml && conda activate BBDM
```

(Python 3.9.16, PyTorch 2.5.1 + CUDA 12.4, NumPy 2.0.2; see `environment.yml` for the import-verified pins used for the paper results.)

## 2. Checkpoints

Download from the [v1.0 release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0)
and place as:

```
results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth   # detfill_scribble_illust_200ep.pth
results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth        # detfill_dot_illust_200ep.pth
```

(see `../checkpoints/README.md` for download commands and SHA-256 checksums; any other
location works with `--resume_model <path>`).

## 3. Data layout (user-specified)

All data locations are set by you in `configs/*.yaml`; hints are the paper's 64×64
setting by default.

**A. Flat evaluation layout** — set `data.dataset_config.scratch_root` (used by the
`*_illust.yaml` configs and `run_inference.sh`):

```
<scratch_root>/
  segmentations/originals/<id>.image.png       # ground-truth color images
  sketch/{XDoG,pysimp,sketchkeras}/<id>.png    # line art from the 3 extractors
  hint_from_regions_64_rev/<id>.image_scribble_{col,mask}64.png
                           <id>.image_dot_{col,mask}64.png
  hint_from_regions_256/<id>.image_region64.png
```

Hint files may also live in a `hint_from_regions_64_rev/0016/` subdirectory (the layout
of the original experiments) — both are detected automatically.

**B. Split-based layout** — set `data.dataset_config.dataset_path` (used when
`scratch_root` is absent, e.g. the `*_real_*.yaml` configs). Train/valid/test ids come
from `configs/<domain>/{train,valid,test}.txt`:

```
<dataset_path>/
  sketch/{XDoG,pysimp,sketchkeras}/...
  hint_from_regions_64_rev/   hint_from_regions_256/
  segmentation_regions/felzenszwalb/
```

Hint files are produced by the `hintauc` library or `../hint_generation/` (both
implement the same deterministic pipeline).

## 4. Inference over the Hint-AUC ratio grid (tested)

```bash
GPU=0 bash run_inference.sh scribble                      # all ratios x 3 sketch types
GPU=0 RATIOS="0.10" TYPES="2" bash run_inference.sh dot   # a single cell
```

Outputs land in:

```
results/dataset_name/BrownianBridge_<hint>_illust/sample_to_eval/illust/<hint>/
    <sketch_type>/<ratio>/200/<id>.image.png        # colorizations
    <sketch_type>/<ratio>/ground_truth/             # matching GT copies
```

which is exactly the layout `../evaluation/dense/eval_curve.py` consumes.
Ratios follow the paper grid {0.00, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00}; sketch
types are 0 = sketch simplification (pysimp), 1 = XDoG, 2 = SketchKeras.

## 5. Training

```bash
bash train.sh   # the two illustration configs; ~200 epochs
```

The natural-image (ImageNet) configs are included commented-out in `train.sh` — enable
them once the corresponding dataset (layout B) is prepared. Adjust `--gpu_ids` to your
hardware; the paper models were trained with batch size 20 and gradient accumulation.

## Notes

- `main.py` flags: `--train`, `--sample_to_eval`, `--resume_model`, `--sample_ratio`,
  `--sketch_type`, `--gpu_ids`, `--port`.
- The dataset loader raises a clear error if `dataset_path` / `scratch_root` is unset
  or still a `/path/to/...` placeholder.
