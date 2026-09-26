# Replicability: one command reproduces Fig. 9 (Graphics Replicability Stamp)

This directory reproduces **Fig. 9** of

> K. Madono, Y. Mingcheng, E. Simo-Serra, *Hint-AUC: Deterministic Region-based Hint Generation for Line Art
> Colorization Evaluation*, IEEE Transactions on Visualization and Computer Graphics, 2026, DOI 10.1109/TVCG.2026.3738401

with a single command and no arguments:

```bash
git clone https://github.com/MADONOKOUKI/DetFill-with-DHT.git
cd DetFill-with-DHT
bash replicability/run.sh
```

**Output:** `replicability/output/fig9.png` (plus the twelve individual panels and `output/summary.json` with the
run times of your machine).
**Compare with:** `replicability/expected/fig9_paper.png`, the figure exactly as printed in the paper, and
`replicability/expected/paper_panels/`, its 256×256 panels.

## What the script does, step by step

Fig. 9 ("Importance of deterministic hint sampling") compares, for two illustrations, the colorizations produced
from 10 % of the regions selected either in an area-independent order (the "random sample" of the paper) or by our
deterministic size-ordered rule ("region-based sample, top 10 %"). `run.sh` performs the complete pipeline of the
paper:

1. **Environment.** Creates a conda environment `detfill-grsi` from `replicability/environment.yml` (PyTorch 2.5.1
   with CUDA 12.4 wheels, which also run on CPU; NumPy 1.26.4; FilFinder 1.7.2 + astropy 5.3.4 for the skeleton
   longest path; torchmetrics 1.4.0, the version used for the paper's SSIM values) and installs this repository into it.
2. **Checkpoints.** Downloads the two paper models from the v1.0 release (`fetch_checkpoints.sh`; 1.5 GB in total)
   and verifies their SHA-256.
3. **Deterministic hint generation** (`hintauc.generate_hints`, paper settings): Felzenszwalb segmentation →
   Zhang–Suen thinning → 3×3 dilation → FilFinder longest path per region → region-mean colours, stored as 64×64
   scribble and dot hint maps. The dot is the in-region path pixel with the smallest total Manhattan distance to the
   other in-region path pixels (the rule behind the released stored maps).
4. **Region selection** at α = 10 %: k = ⌊0.1·n⌋ regions, either the k largest (Sec. IV-B) or the first k labels
   in ascending label order (the protocol of the paper's random-order comparison, Table III).
5. **DetFill inference** with the released checkpoints (`detfill_dot_illust_200ep.pth` for the dot row,
   `detfill_scribble_illust_200ep.pth` for the scribble row), seed 1234, 200 sampling steps, SketchKeras line art.
6. **Figure assembly** into the 2 × 6 panel.

The two input illustrations (Danbooru2021 ids 4731016 and 4942016) and their line drawings are shipped in `data/`
(see `data/NOTICE.txt`).

## What to expect

- The diffusion sampler is seeded, so the colorizations are deterministic on a given machine. Across GPU generations
  or on CPU the pixels differ only by floating-point noise; the panels look the same as the printed ones.
- Hint generation is deterministic except for the unseeded medial-axis tie-breaking inside FilFinder (paper,
  supplementary Sec. I), which can move a few hint pixels; the selected regions and the figure are unaffected.

## Requirements and run times

- Tested on Ubuntu 22.04 with an NVIDIA GPU (driver for CUDA 12.x) and on CPU only.
- **Miniconda or Anaconda** must be installed (https://docs.conda.io/en/latest/miniconda.html). Everything else is
  installed by `run.sh`.
- Internet access on the first run: conda/pip packages (~3 GB) and the two checkpoints (1.5 GB).
- Disk: ~8 GB. GPU memory: < 4 GB.

Measured on the authors' server (NVIDIA RTX A6000, 32-core CPU), after the one-time environment creation:

| Setting | Total time | Details |
|---|---|---|
| GPU | **about 3 minutes** | hint generation 1–2.5 min per image on the CPU; four DetFill runs of 18–26 s each |
| CPU only (`GRSI_GPU=-1`, 16 threads) | **about 30 minutes** | the four DetFill runs took 4–9 min each |

Optional environment variables: `GRSI_GPU=<id>` selects a GPU (`GRSI_GPU=-1` forces CPU), `GRSI_ENV=<name>` reuses
an existing conda environment, `GRSI_CKPT_DIR=<dir>` points at checkpoints downloaded by hand.

## Files

| Path | Role |
|---|---|
| `run.sh` | the one-command entry point (environment → checkpoints → `make_fig9.py`) |
| `environment.yml` | conda environment used by `run.sh` (and by `reproduce/examples/run_examples.sh`) |
| `fetch_checkpoints.sh` | downloads the two released checkpoints and verifies their SHA-256 |
| `make_fig9.py` | hint generation, region selection, DetFill inference, figure assembly |
| `data/` | the two test illustrations (`<id>.image.png`) and their SketchKeras sketches (`sketch/<id>.png`) |
| `expected/fig9_paper.png`, `expected/paper_panels/` | the figure and the 256×256 panels as printed in the paper |
| `output/` | created by the script: `fig9.png`, per-panel PNGs, `summary.json`, staged inputs, DetFill logs |

## Beyond this figure

`reproduce/examples/run_examples.sh` re-runs every experiment of the paper and the supplement on a few example
illustrations with the same environment and the released checkpoints (no arguments; about 45 minutes on a GPU in
the default mode), and `reproduce/README.md` explains how to rebuild every table from the released metric files and
how to re-run a whole table row on the 3,000 test images. The repository README has the overview table.
