# DetFill

Pixel-space Brownian Bridge diffusion model for sketch + deterministic-hint
colorization. Fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT, (c) 2023 xuekt98)
adapted to the Hint-AUC protocol — see the repository root README for the full context.

## Environment

```bash
conda env create -f environment.yml && conda activate BBDM
```

## Data layout (user-specified)

All data locations are set by you in `configs/*.yaml`. Hints default to the paper's
64×64 resolution.

**A. Flat evaluation layout** — set `data.dataset_config.scratch_root`:

```
<scratch_root>/
  segmentations/originals/<id>.image.png      # ground-truth color images
  sketch/{XDoG,pysimp,sketchkeras}/<id>.png   # line art (3 extractors)
  hint_from_regions_64_rev/                   # 64px hints (see hint_generation/)
  hint_from_regions_256/<id>.image_region64.png
```

**B. Split-based layout** — set `data.dataset_config.dataset_path` (used when
`scratch_root` is absent, e.g. the `*_real_*.yaml` configs); train/valid/test ids come
from `configs/<domain>/{train,valid,test}_paper.txt`:

```
<dataset_path>/
  sketch/{XDoG,pysimp,sketchkeras}/...
  hint_from_regions_64_rev/
  hint_from_regions_256/
  segmentation_regions/felzenszwalb/
```

The hint files themselves are produced by the `hintauc` library or
`../hint_generation/` scripts.

## Checkpoints

Download from the GitHub Release (v1.0) and place as
`results/dataset_name/BrownianBridge_{scribble,dot}_illust/checkpoint/latest_model_200.pth`.

## Inference over the Hint-AUC ratio grid

```bash
GPU=0 bash run_inference_mr.sh scribble                      # all ratios x 3 sketch types
GPU=0 RATIOS="0.10" TYPES="2" bash run_inference_mr.sh dot   # a single cell
```

## Training

```bash
bash train.sh   # runs the two illustration configs; natural-image configs are
                # included commented-out — enable them once their dataset is prepared
```
