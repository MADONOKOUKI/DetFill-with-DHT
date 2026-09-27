# DetFill: the colorization model

DetFill is the pixel-space diffusion colorization model of the paper: a fork of
[BBDM](https://github.com/xuekt98/BBDM) (Brownian Bridge Diffusion Models, MIT) conditioned on line art and
deterministic colour hints. This directory holds its training and inference code and configs.

- **Models:** scribble-hint model (96 base channels) and dot-hint model (64 base channels), 200 epochs on
  Danbooru2021; see the [model zoo](../checkpoints/README.md).
- **Protocols:** hints of the largest regions first (Table II, `hint_order: area`) or in a fixed random order
  (Table III, `hint_order: label`); hint ratio set per run with `--sample_ratio`.
- **Runs on:** one NVIDIA GPU (< 4 GB) or CPU (`--gpu_ids -1`, slow).

## Setup

```bash
conda env create -f ../replicability/environment.yml && conda activate detfill-grsi   # PyTorch 2.5.1, CUDA 12.4 (runs on CPU too)
# or the paper's original training environment: conda env create -f environment.yml && conda activate BBDM
```

Checkpoints: download from the [v1.0 release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0)
and place them as

```
results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth   # <- detfill_scribble_illust_200ep.pth
results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth        # <- detfill_dot_illust_200ep.pth
```

(any other location works with `--resume_model <file>`).

## Data layout

Set the data root in `configs/*.yaml` (`data.dataset_config.scratch_root`). Hints are the paper's 64 × 64 maps.

```
<DATA_ROOT>/
  segmentations/originals/<id>.image.png        ground-truth colour images (512 x 512)
  sketch/{XDoG,pysimp,sketchkeras}/<id>.png     line art from the three extractors
  hint_from_regions_64_rev/<id>.image_{scribble,dot}_{col,mask}64.png
  hint_from_regions_256/<id>.image_region64.png
```

- Images: [Danbooru2021](https://gwern.net/danbooru2021) (split lists in `configs/illust/`); not redistributed. The
  512 × 512 copies are the ground truth; the stored hint maps were generated from the original-resolution files.
- Line art and hint maps of the test split: releases [v1.3](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.3)
  and [v1.0](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0); for new images use the `hintauc`
  library (`hints.save(...)` writes exactly these files).
- Hint files may also sit in a `hint_from_regions_64_rev/0016/` sub-directory (original experiment layout); both are detected.
- **Split-based layout** (training, and the natural-image configs `*_real.yaml`): set `dataset_path` instead of
  `scratch_root` and run `main.py` from `detfill/`; the split lists `configs/{illust,real}/{train,valid,test}.txt`
  hold `<bucket>/<id>.image.png` lines and the files live in
  `<dataset_path>/segmentation_regions/felzenszwalb/<bucket>/<id>.image.png` (colour image),
  `<dataset_path>/sketch/<type>/<bucket>/<id>.png` (`<id>.image.png` for natural images),
  `<dataset_path>/hint_from_regions_64_rev/<bucket>/<id>.image_*64.png` and
  `<dataset_path>/hint_from_regions_256/<bucket>/<id>.image_region64.png`.

## Inference over the hint-ratio grid

```bash
GPU=0 bash run_inference.sh scribble                      # 8 ratios x 3 line-art extractors
GPU=0 RATIOS="0.10" TYPES="2" bash run_inference.sh dot   # one cell
```

- Outputs: `results/dataset_name/BrownianBridge_<hint>_illust/sample_to_eval/illust/<hint>/<sketch_type>/<ratio>/200/<id>.image.png`
  (plus `ground_truth/` copies) — the layout consumed by `../reproduce/scripts/eval_per_ratio.py` and `../evaluation/`.
  Each `<ratio>` directory carries a `run_manifest.json` (checkpoint SHA-256, seed, hint type and order, ratio, sketch
  type, sampling steps, config hash, data root); existing images are reused only when a new run has the same
  conditions, otherwise it stops. A resumed run draws the sampler's noise in a different order than an
  uninterrupted one, so resumed images are not bit-identical to a fresh run. `--sample_ratio` must lie in [0, 1]; the
  config value `eta` has no effect in this sampler (fixed-variance step).
- `sketch_type` 0 / 1 / 2 = sketch simplification / XDoG / SketchKeras; ratios follow the paper grid
  {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00}; the sampler is seeded (`--seed 1234`).
- Full row with metrics in one command: `../reproduce/scripts/run_hauc_pipeline.sh`.

## Training

```bash
bash train.sh          # the two illustration configs, 200 epochs (~4-5 days on 10 GPUs); or ../reproduce/scripts/B9_train_detfill.sh
```

- Effective batch 20 (batch 1 per GPU × 10 GPUs × gradient accumulation 2), Adam 1e-4, EMA 0.995.
- Budget: `training.n_epochs` (200) and `training.n_steps` (400,000 micro-batches *per process*; checked before every
  batch, `--max_steps` overrides it, 0 = no step budget). 400,000 equals 200 epochs of the 20,000-image split on
  10 GPUs; on one GPU the same value stops after 20 epochs, so pass `--max_steps 4000000` for a 200-epoch single-GPU run.
  A training error is re-raised after the emergency checkpoint (`last_model.pth`), so a failed job exits non-zero.
- Training-time hint sampling follows the paper: the number of hinted regions is uniform on {0, …, n−1}, so the fully
  hinted case is never seen in training. `dataset_config.include_full_hint: true` includes it. The released checkpoints
  use the paper setting.

## Files

| Path | Role |
|---|---|
| `main.py` | entry point (`--train`, `--sample_to_eval`, `--resume_model`, `--sample_ratio`, `--sketch_type`, `--gpu_ids`, `--seed`) |
| `run_inference.sh`, `train.sh` | the inference-grid and training launchers |
| `configs/` | `scribble_illust.yaml`, `dot_illust.yaml`, `scribble_real.yaml`, `dot_real.yaml`; split lists in `illust/`, `real/` |
| `datasets/custom.py` | data loader (flat and split-based layouts, region selection, `hint_order`) |
| `model/`, `runners/` | the Brownian Bridge model and training/sampling loops (from BBDM) |
| `environment.yml` | the paper's training environment |
