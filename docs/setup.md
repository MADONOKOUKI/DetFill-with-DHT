# Setup

Everything in this repository runs on Linux (tested on Ubuntu 22.04). An NVIDIA GPU with 4 GB or more is recommended
for DetFill; the `hintauc` library and all evaluation code also run on CPU.

## 1. The `hintauc` library (deterministic hints, metrics, Hint-AUC)

```bash
pip install hintauc                 # hint generation, the pixel metrics, Hint-AUC      (PyPI: https://pypi.org/project/hintauc/)
pip install "hintauc[perceptual]"   # + the perceptual metrics (LPIPS / OpenCLIP / DINOv2 / DreamSim)
```

- Python 3.9–3.12; wheels exist for all dependencies, no compiler is needed.
- Latest development version: `pip install git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git`
  (or `pip install -e .` inside a checkout).
- The perceptual metrics download their weights on first use into `~/.cache/hintauc` (override with `HINTAUC_CACHE_DIR`).
- Check the install: `python -c "import hintauc; print(hintauc.__version__)"` and `hintauc --help`.

## 2. DetFill and the reproduction scripts (conda environment)

```bash
conda env create -f replicability/environment.yml   # Python 3.9, PyTorch 2.5.1, CUDA 12.4 (runs on CPU too)
conda activate detfill-grsi
pip install -e .                                     # the library from this checkout
```

- `replicability/run.sh` and `reproduce/examples/run_examples.sh` create this environment themselves; `run.sh` also
  installs Miniconda into `~/miniconda3` when no conda is found (disable with `GRSI_NO_AUTO_CONDA=1`).
- The paper's original training environment is `detfill/environment.yml` (`conda env create -f detfill/environment.yml`).
- Without a GPU, pass `--gpu_ids -1` to `detfill/main.py`; the scripts detect the absence of CUDA automatically.

## 3. Checkpoints and data

- All checkpoints and data files are attached to the GitHub releases; the [Model zoo](../checkpoints/README.md) lists
  every file with its size, SHA-256 and the paper item it belongs to.
- `replicability/fetch_checkpoints.sh` downloads the two paper models and verifies their hashes;
  `reproduce/examples/run_examples.sh` downloads what each experiment needs.
- The illustrations come from [Danbooru2021](https://gwern.net/danbooru2021) and the natural images from an
  [ImageNet](https://www.image-net.org/download.php) subset; neither is redistributed. The split lists are in
  `reproduce/data/splits/` and `detfill/configs/{illust,real}/`, the data layout in [detfill/README.md](../detfill/README.md).

## 4. Hardware and run times

| Machine | GPU | CPU | RAM |
|---|---|---|---|
| A | NVIDIA RTX A6000, 48 GB | 2 × AMD EPYC 9124 (32 threads) | 377 GB |
| B | NVIDIA GeForce RTX 2080 Ti, 11 GB | 2 × Intel Xeon Gold 6226R (32 threads) | 187 GB |

Both run Ubuntu 22.04, driver 535, CUDA 12.2. Fig. 9 (`replicability/run.sh`): 3 min on A, 4–5 min on B, 30 min on
the CPUs of A (16 threads). The example suite and the full-scale estimates were measured on B; an A6000-class GPU is
roughly twice as fast. Full-scale DetFill inference: about 35 min per (hint ratio, line-art source) cell of 3,000 images
on A with the 96-channel scribble model, i.e. ≈ 14 GPU-hours for a Table II row.

## 5. Known pitfalls

- `torchmetrics` must be 1.4.0 for the published SSIM values (1.8.x differs by up to 0.04 per image); the environment
  file pins it.
- The DetFill test loader drops an incomplete last batch. The released configs use batch 5 (scribble) / 8 (dot), which
  divide 3,000; for small image sets set `TEST_BATCH=1` (`run_hauc_pipeline.sh`) or `BATCH_SIZE_OVERRIDE=1` (`main.py`).
- Sampling is seeded (`--seed 1234`) but not bit-exact across GPU generations or between GPU and CPU; metric
  differences of a few hundredths are expected, see [Detailed explanation](detail_explanation.md#what-is-exact-and-what-is-not).
- OpenCLIP weights are used in fp16 as in the archived runs (differences ≤ 8e-4); DreamSim runs on CPU-only hosts.
