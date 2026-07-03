# DetFill

Pixel-space Brownian Bridge diffusion model for sketch + deterministic-hint
colorization. Fork of [BBDM](https://github.com/xuekt98/BBDM) (MIT, (c) 2023 xuekt98)
adapted to the Hint-AUC protocol — see the repository root README for the full context.

## Environment

```bash
conda env create -f environment.yml && conda activate BBDM
```

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
bash train.sh   # uses configs/{dot,scribble}_proposed_illust_200epoch.yaml
```

Set `dataset_path` / `scratch_root` in `configs/*.yaml` to your dataset location
(layout: `sketch/{XDoG,pysimp,sketchkeras}/`, `hint_from_regions_64_rev/`,
`hint_from_regions_256/`, `segmentations/originals/`). The `*_real_*.yaml` configs
still contain our cluster paths and need the same adaptation.
