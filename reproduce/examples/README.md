# Reproduce every experiment on example images

`run_examples.sh` re-runs each experiment of the paper and its supplement on a few test illustrations with the
released checkpoints, using the same inference program (`detfill/main.py`) and the same evaluator (the `hintauc`
library) as the full-scale experiments. It needs no data preparation: the checkpoints and the 12-image example bundle
(release v1.3, `examples_data.tar.gz`) are downloaded automatically and checked against their SHA-256.

```bash
bash reproduce/examples/run_examples.sh                       # default: "quick" (4 images, 4 ratios)  ≈ 1 h on one GPU (2080 Ti class)
EXAMPLES_MODE=smoke bash reproduce/examples/run_examples.sh   # 1 image, 2 ratios, scribble model only    ≈ 5 min on a GPU
EXAMPLES_MODE=full  bash reproduce/examples/run_examples.sh   # all experiments, 12 images, 8 ratios      several GPU-hours
```

Options: `GRSI_GPU=<id>` (default 0; `-1` = CPU, roughly 10× slower), `GRSI_ENV=<conda env>` to reuse an environment,
`GRSI_CKPT_DIR=<dir>` if the checkpoints were downloaded by hand.

## What you get

- `output/grids/<experiment>*.png` — one labelled image grid per experiment: ground truth, line art, the hints
  actually given to the model at each ratio, and the model outputs.
- `output/metrics/<experiment>.json` — the seven metrics (MSE, PSNR, SSIM, LPIPS, OpenCLIP, DINOv2, DreamSim) per
  image and averaged, and Hint-AUC when the full ratio grid was run.
- A printed comparison (`compare_with_expected.py`) with `expected/metrics/`, the authors' run of the same script,
  and — for the Table II protocol — the PSNR between every new output and the paper's archived output of the same
  image (`expected/paper_outputs/`).

## The experiments

| Id | Paper item | Images | What is run | How to read the result |
|---|---|---|---|---|
| **E1** Table II protocol | Table II, Fig. 8 | 12 test images (4 in quick mode) | scribble (96-ch) and dot (64-ch) models, size-ordered hints, ratio grid, SketchKeras line art | quality improves monotonically with the hint ratio; per-image metrics and the Hint-AUC of the examples are in the JSON; `psnr_vs_paper_outputs_dB` says how close each re-generated image is to the paper's own output |
| **E2** Table III protocol | Table III | same | scribble model with the fixed random (ascending-label) region order | lower scores than E1 at the same ratio, as in Table III vs. Table II |
| **E3** Segmentation dependency | supp. table and figures "robustness to the region segmenter" | 1019016, 1023016 (the figure's images) | the three scribble models (trained on Felzenszwalb / DanbooRegion / SLIC hints) each evaluated with each segmenter's hints | the evaluation segmenter dominates (SLIC hints hurt every model), as in the supplement |
| **E4** Seed sensitivity | supp. table "size-ordered vs. random selection" | 3 images | size-ordered selection vs. the paper's seeded uniformly random selection (seeds 1–3) at the intermediate ratios | size-ordered wins at every intermediate ratio; the spread over seeds is small |
| **E5** Dense ratio curve | supp. α-grid study | 2 images | 24 ratios from 0 to 1 | the metric curve is smooth; Hint-AUC on the dense grid and on the paper grid are close |
| **E8** Channel ablation | supp. channel-ablation figure | 3421016, 4417016 (the figure's images) | 32 / 64 / 96 base-channel scribble models at α = 10 % | the 32-channel model loses colour fidelity, as in the figure |
| **E9** Hint regeneration | Sec. IV, supp. Sec. I | 12 images | `hintauc.generate_hints` re-run on the illustrations vs. the stored maps | the regenerated maps agree with the stored ones up to FilFinder's environment-dependent tie-breaking (the JSON gives the mask IoU and pixel counts) |
| **E11** Earlier checkpoint | provenance | 4 images | the scribble model of the earlier submission (release v1.4) | the current 96-channel model produces sharper colours; the earlier model is sampled with 1,000 steps as in its archived config |

Not part of the automatic run: the Diffusart-retrain models (v1.2; their inference needs the separate Diffusart code
base and `diffusers`, see `reproduce/paper_experiments/diffusart_retrain/code/README.md`) and the natural-image
models (v1.2; the ImageNet originals cannot be redistributed — point `run_hauc_pipeline.sh` at your own ImageNet copy).

## Authors' reference run

`expected/` holds the grids and metrics of the authors' `full` run (NVIDIA GeForce RTX 2080 Ti 11 GB, 2 × Intel Xeon Gold 6226R,
environment of `replicability/environment.yml`) and the paper's archived outputs of the 12 example images for the
Table II protocol. Because the diffusion sampler is seeded but not bit-exact across GPU generations, your outputs will
differ from both at the pixel level; `compare_with_expected.py` flags metric differences beyond a generous tolerance
(PSNR 0.6 dB, LPIPS 0.02, DreamSim 0.015). A few flagged cells on a different GPU generation or on CPU are expected
and do not indicate a problem as long as the trends described in the table above hold.

## Files

| File | Role |
|---|---|
| `run_examples.sh` | no-argument driver: environment → downloads → `make_examples.py` → comparison |
| `make_examples.py` | the experiments (`--mode smoke|quick|full`, `--only E1 E9 …`, `--ids …`, `--gpu`) |
| `compare_with_expected.py` | compares `output/metrics/` with `expected/metrics/` |
| `test_image_ids.txt` | the 3,000 test ids (the seeded random selection of E4 uses an image's index in this list, as the paper did) |
| `expected/` | authors' grids and metrics; `paper_outputs/` = the paper's archived outputs of the example images |
| `data/`, `checkpoints/`, `output/` | created by the driver (git-ignored) |
