# Setup

Everything in this repository runs on Linux (tested on Ubuntu 22.04). An NVIDIA GPU with 4 GB or more is recommended
for DetFill; the `hintauc` library and all evaluation code also run on CPU.

## 1. The `hintauc` library (deterministic hints, metrics, Hint-AUC)

```bash
pip install hintauc                 # hint generation, the pixel metrics, Hint-AUC      (PyPI: https://pypi.org/project/hintauc/)
pip install "hintauc[paper]"        # + torchvision and torchmetrics 1.4.0: the exact metric backend of the paper's numbers
pip install "hintauc[perceptual]"   # + the perceptual metrics (LPIPS / OpenCLIP / DINOv2 / DreamSim)
```

- Python 3.9–3.12; wheels exist for all dependencies, no compiler is needed.
- Latest development version: `pip install git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git`
  (or `pip install -e .` inside a checkout).
- The perceptual metrics download their weights on first use (DreamSim into `~/.cache/hintauc`, override with
  `HINTAUC_CACHE_DIR`; OpenCLIP and DINOv2 into their own Hugging Face caches; LPIPS ships with its package).
- On a minimal Ubuntu server image OpenCV needs two system libraries: `sudo apt-get install -y libgl1 libglib2.0-0`.
- Check the install: `hintauc demo` prints a metric table whose values are recorded in `examples/expected_numbers.json`
  (identical with and without PyTorch up to the SSIM fallback, see [Evaluate your own model](evaluate_your_model.md#which-numbers-to-expect)).
- Without PyTorch the evaluator resizes with Pillow's antialiased bilinear filter on float32 channels, which agrees with
  the paper's torchvision resize to about 1e-6 per pixel, and computes SSIM with scikit-image (a few 1e-4 from
  torchmetrics 1.4.0). `result["protocol"]["backend"]` records which backend produced a number.

## 2. DetFill and the reproduction scripts (conda environment)

```bash
conda env create -f replicability/environment.yml   # Python 3.9, PyTorch 2.5.1, CUDA 12.4 (runs on CPU too)
conda activate detfill-grsi
pip install -e .                                     # the library from this checkout
```

- `replicability/run.sh` and `reproduce/examples/run_examples.sh` create this environment themselves; `run.sh` also
  installs Miniconda into `~/miniconda3` when no conda is found (disable with `GRSI_NO_AUTO_CONDA=1`).
- The paper's inference and evaluation environment is `detfill/environment.yml` (`conda env create -f detfill/environment.yml`).
- Without a GPU, pass `--gpu_ids -1` to `detfill/main.py`; `replicability/run.sh` and `reproduce/examples/run_examples.sh`
  fall back to the CPU automatically when no CUDA device is found (slow: about 30 minutes for Fig. 9).

## 3. Checkpoints and data

- The released checkpoints and data files are attached to the GitHub releases; the [Model zoo](../checkpoints/README.md) lists
  every file with its size, SHA-256 and the paper item it belongs to.
- `replicability/fetch_checkpoints.sh` downloads the two paper models and verifies their hashes;
  `reproduce/examples/run_examples.sh` downloads what each experiment needs.
- The illustrations come from [Danbooru2021](https://gwern.net/danbooru2021) and the natural images from an
  [ImageNet](https://www.image-net.org/download.php) subset; neither is redistributed (except the nine illustrations
  that appear in the paper's figures, see `THIRD_PARTY_NOTICES.md`). `reproduce/scripts/fetch_originals.py` fetches
  illustrations by id from Danbooru and writes the dataset's 512 × 512 copies (bit-exact). The split lists are in
  `reproduce/data/splits/` and `detfill/configs/{illust,real}/`, the data layout in [detfill/README.md](../detfill/README.md).

## 4. Hardware and run times

| Machine | GPU | CPU | RAM |
|---|---|---|---|
| A | NVIDIA RTX A6000, 48 GB | 2 × AMD EPYC 9124 (32 threads) | 377 GB |
| B | NVIDIA GeForce RTX 2080 Ti, 11 GB | 2 × Intel Xeon Gold 6226R (32 threads) | 187 GB |

Both run Ubuntu 22.04, driver 535, CUDA 12.2. Fig. 9 (`replicability/run.sh`): 3 min on A, 4–5 min on B, 30 min on
the CPUs of A (16 threads), plus the one-time environment setup and downloads (10–20 min, ≈ 3 GB of packages and
1.5 GB of checkpoints). The example suite was measured on B. Full-scale DetFill inference: about 35 min per (hint
ratio, line-art source) cell of 3,000 images on A with the 96-channel scribble model, i.e. ≈ 14 GPU-hours for a
Table II row on A (about twice that on B).

## 5. Known pitfalls

- `torchmetrics` must be 1.4.0 for the published SSIM values (1.8.x differs by up to 0.04 per image); the environment
  file pins it.
- The DetFill test loader processes the last incomplete batch; `TEST_BATCH=1` (`run_hauc_pipeline.sh`) or
  `BATCH_SIZE_OVERRIDE=1` (`main.py`) only fixes the batch composition (the sampler is seeded per batch). Output
  directories carry a `run_manifest.json`; a run with a different checkpoint, seed or hint setting refuses to reuse them.
- `training.n_steps` in the configs counts micro-batches per process: 400,000 is 200 epochs of the 20,000-image
  training split on 10 GPUs; on one GPU it stops after 20 epochs, so pass `--max_steps 4000000` (or set `n_steps` to 0
  for an epoch-only budget). The budget is checked before every batch.
- Sampling is seeded (`--seed 1234`) but not bit-exact across GPU generations or between GPU and CPU; metric
  differences of a few hundredths are expected, see [Detailed explanation](detail_explanation.md#what-is-exact-and-what-is-not).
- OpenCLIP weights are used in fp16 as in the archived runs (differences ≤ 8e-4); DreamSim runs on CPU-only hosts.
