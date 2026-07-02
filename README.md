# DetFill-with-DHT

Official research code for

> **Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation**
> Koki Madono, Yuan Mingcheng, Edgar Simo-Serra
> (under review, IEEE TVCG)

**DHT** = **D**eterministic **H**in**T** generation: a reproducible, region-based color-hint
generation pipeline (segmentation → skeleton → longest-path scribble / dot → region-medoid color),
combined with the **Hint-AUC** protocol that evaluates hint-based line-art colorization across the
full range of hint ratios instead of a single hint setting.

## Repository layout

| Directory | Contents |
|---|---|
| `hint_generation/` | Deterministic region-based hint generation (Felzenszwalb default; SLIC / Quickshift variants used in the supplementary robustness study). Produces `*_region64.png`, `*_scribble_mask64.png`, `*_scribble_col64.png` (+ dot variants) per image. |
| `detfill/` | DetFill colorization model — a pixel-space Brownian Bridge diffusion model (fork of [BBDM](https://github.com/xuekt98/BBDM), MIT) adapted to sketch + deterministic-hint conditioning. Training, inference, and per-ratio sampling scripts. |
| `evaluation/` | Hint-AUC computation: per-ratio metrics (MSE / PSNR / SSIM / LPIPS / OpenCLIP / DINO / DreamSim) and the trapezoidal Hint-AUC integral over the front-loaded ratio grid α ∈ {0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00}. |
| `checkpoints/` | Pointers to the released model weights (see below). |

## Checkpoints

The paper checkpoints (DetFill, 96 base channels, 200 epochs, Danbooru2021 illustrations) are
attached to the GitHub Release of this repository:

- `detfill_scribble_illust_200ep.pth` (463 MB) — scribble-hint model (used for the Table II scribble results)
- `detfill_dot_illust_200ep.pth` (463 MB) — dot-hint model

Place them under `detfill/results/dataset_name/BrownianBridge_{scribble,dot}_illust/checkpoint/latest_model_200.pth`
(or pass `--resume_model` explicitly). DetFill is pixel-space: no VQGAN / latent-diffusion weights are required.
Metric backbones (LPIPS, OpenCLIP, DINO, DreamSim) are downloaded automatically by their pip packages.

## Setup

```bash
# model (training / inference)
conda env create -f detfill/environment.yml
conda activate BBDM

# hint generation (lightweight; separate env is fine)
pip install opencv-python scikit-image==0.19.* fil_finder==1.7.2 astropy natsort
```

## Usage

### 1. Deterministic hint generation

```bash
python hint_generation/generate_hints.py \
    --src_root <dir with ground-truth images> \
    --out_root <output dir> --segmenter felzenszwalb
```

Per image this produces the 64×64 region map, scribble mask, and scribble color map that the
model and the evaluation protocol consume. Generation is fully deterministic: the same image
always yields the same hints, and hint subsets at ratio α are the top-α regions by pixel area.

### 2. Inference over the hint-ratio grid

```bash
cd detfill
GPU=0 bash run_inference_mr.sh scribble          # all ratios x 3 sketch types
GPU=0 RATIOS="0.10" TYPES="2" bash run_inference_mr.sh scribble
```

### 3. Hint-AUC evaluation

```bash
python evaluation/calc_hint_auc_manual.py --help
```

Computes the per-ratio metrics over the generated outputs and integrates them into Hint-AUC
with the trapezoidal rule.

## Notes

- This is research code: some scripts contain absolute dataset paths from our cluster
  (`/scratch/...`, `/home/madorin/...`) that need to be adapted to your environment
  (`dataset_path` / `scratch_root` in `detfill/configs/*.yaml`).
- A pip-installable package of the deterministic hint generation is under preparation.

## License and acknowledgements

MIT License. The `detfill/` directory is derived from
[BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM)
(© 2023 xuekt98, MIT) — see `detfill/LICENSE`.

## Citation

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {under review},
  year    = {2026}
}
```
