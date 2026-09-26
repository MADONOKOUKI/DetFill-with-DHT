# Reproduce the paper's key figure with one command

This folder reproduces **Figure 9** of the paper — *deterministic vs. random hint selection and the resulting DetFill
colorizations* — from a fresh checkout, with a single command and no arguments. It is the entry point for the
Graphics Replicability Stamp.

```bash
git clone https://github.com/MADONOKOUKI/DetFill-with-DHT.git
cd DetFill-with-DHT
bash replicability/run.sh
```

- **Result:** `replicability/output/fig9.png`
- **Reference:** `replicability/expected/fig9_paper.png` (the figure as printed) and `expected/paper_panels/`
- **Time:** about 3 minutes on an RTX A6000 (4–5 minutes on an RTX 2080 Ti), about 30 minutes on 16 CPU threads, plus a one-time environment setup

## What the figure shows

Two illustrations are colorized from only 10 % of their regions. When the regions are chosen in an area-independent
order ("random sample"), the result depends on which regions happened to be picked; when they are chosen by the
deterministic size-ordered rule of the paper ("region-based sample, top 10 %"), the largest regions are always hinted
first and the colorization is faithful and repeatable. Both the dot-hint and the scribble-hint models are shown.

## What the script does

1. Creates a conda environment (`detfill-grsi`) from `environment.yml` and installs this repository into it.
2. Downloads the two paper checkpoints from the v1.0 release and verifies their checksums.
3. Generates the deterministic hints for the two illustrations with the `hintauc` library.
4. Selects 10 % of the regions in each of the two orders.
5. Runs DetFill (dot model and scribble model, fixed seed) on every combination.
6. Assembles the panels into `output/fig9.png`.

## What to expect

- The colorizations are deterministic on a given machine. On another GPU generation or on CPU they differ only by
  small floating-point noise; the panels look the same as the printed ones.
- The hint generator is deterministic except for a tie-breaking step inside its skeleton library that is not seeded;
  a few hint pixels may move between environments without changing the selected regions or the figure.

## Requirements

- Linux (tested on Ubuntu 22.04), Miniconda or Anaconda installed.
- An NVIDIA GPU with a CUDA 12 driver is recommended; without a GPU the script falls back to CPU automatically.
- Internet access on the first run (packages ≈ 3 GB, checkpoints 1.5 GB), ≈ 8 GB of disk, < 4 GB of GPU memory.
- Reference machines: NVIDIA RTX A6000 48 GB + 2 × AMD EPYC 9124 (32 threads, 377 GB RAM) and NVIDIA RTX 2080 Ti 11 GB + 2 × Intel Xeon Gold 6226R (32 threads, 187 GB RAM); Ubuntu 22.04, driver 535, CUDA 12.2.

| Optional variable | Effect |
|---|---|
| `GRSI_GPU=<id>` | GPU to use (`-1` forces CPU) |
| `GRSI_ENV=<name>` | reuse an existing conda environment |
| `GRSI_CKPT_DIR=<dir>` | use checkpoints downloaded by hand |

## Files

| Path | Role |
|---|---|
| `run.sh` | the one-command entry point |
| `environment.yml` | the conda environment (also used by `reproduce/examples/run_examples.sh`) |
| `fetch_checkpoints.sh` | downloads and verifies the two checkpoints |
| `make_fig9.py` | hint generation, region selection, inference, figure assembly |
| `data/` | the two test illustrations and their line drawings |
| `expected/` | the printed figure and its panels |
| `output/` | created by the script |

## Going further

- Every experiment of the paper on example images: `bash reproduce/examples/run_examples.sh`
- Every table rebuilt from the released metric files: `python reproduce/scripts/A1_tables_from_released_metrics.py`
- Full-scale re-runs and training: [reproduce/README.md](../reproduce/README.md)
