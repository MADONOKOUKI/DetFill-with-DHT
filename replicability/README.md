# Replicability (Graphics Replicability Stamp)

This directory reproduces **Fig. 9** of

> K. Madono, Y. Mingcheng, E. Simo-Serra, *Hint-AUC: Deterministic Region-based Hint Generation for Line Art
> Colorization Evaluation*, IEEE Transactions on Visualization and Computer Graphics, 2026.

with a single command and no arguments:

```bash
git clone https://github.com/MADONOKOUKI/DetFill-with-DHT.git
cd DetFill-with-DHT
bash replicability/run.sh
```

Output: `replicability/output/fig9.png` (plus the individual panels and `summary.json`).
Reference: `replicability/expected/fig9_paper.png` is the figure as printed in the paper.

## What is replicated

Fig. 9 ("Importance of deterministic hint sampling") compares, for two illustrations, colorizations produced from
10 % of the regions selected either in an area-independent order ("random sample") or by our deterministic
size-ordered rule ("region-based sample, top 10 %"). The script runs the full pipeline of the paper:

1. **DHT hint generation** (`hintauc.generate_hints`, paper settings): Felzenszwalb segmentation → Zhang–Suen
   thinning → 3×3 dilation → FilFinder longest path per region → region-mean colours, stored as 64×64 scribble and
   dot hint maps (the dot is the in-region path pixel with the smallest total Manhattan distance to the other
   in-region path pixels — the rule verified against the released stored maps, see `hint_generation/README.md`).
2. **Region selection** at α = 10 %: k = ⌊0.1·n⌋ regions, either the k largest (Sec. IV-B) or the first k labels in
   ascending label order (the protocol of the paper's random-order comparison, Table III).
3. **DetFill inference** with the released checkpoints (`detfill_dot_illust_200ep.pth` for the dot row,
   `detfill_scribble_illust_200ep.pth` for the scribble row), seed 1234, 200 sampling steps, SketchKeras line art.
4. Assembly of the 2 × 6 panel.

The two input illustrations and their line drawings are shipped in `data/` (see `data/NOTICE.txt`).
Because the diffusion sampler is seeded, the colorizations are deterministic on a given machine; across GPUs /
CPU the outputs differ only by floating-point noise. Hint generation is deterministic except for the unseeded
medial-axis tie-breaking inside the skeleton extraction (paper, supplementary Sec. I), which can move a few hint
pixels; the selected regions and the overall figure are unaffected.

## Requirements

- Tested on Ubuntu 22.04 with an NVIDIA GPU (CUDA 12.x driver) and on CPU only (see run times below).
- **Miniconda/Anaconda** must be installed (https://docs.conda.io/en/latest/miniconda.html); everything else is
  installed by `run.sh` into a new conda environment `detfill-grsi` from `replicability/environment.yml`
  (Python 3.11; PyTorch 2.5.1 wheels with CUDA 12.4 — they also run on CPU; NumPy 2.0.2, scikit-image 0.24,
  OpenCV 4.11, FilFinder 1.7.2 with astropy ≥ 6.1). These are the runtime pins of the paper's inference environment
  (`detfill/environment.yml`, Python 3.9) plus the hint-generation dependencies; Python 3.11 is used only because a
  NumPy-2-compatible astropy needs Python ≥ 3.10.
- Internet access on the first run: pip/conda packages (~3 GB) and the two checkpoints (1.5 GB, SHA-256 verified,
  downloaded from the GitHub release v1.0 by `fetch_checkpoints.sh`).
- Disk: ~8 GB. GPU memory: < 4 GB.

Run time (after the one-time environment creation), measured on the authors' server (NVIDIA RTX A6000, 32-core CPU):
- GPU: **about 3 minutes** in total (hint generation ≈ 1–2.5 min per image on the CPU; four DetFill runs of 18–26 s each).
- CPU only (`GRSI_GPU=-1`, 16 threads): the four DetFill runs take on the order of an hour in total (see `output/summary.json` for the exact per-run seconds of your machine).

Optional environment variables: `GRSI_GPU=<id>` to pick a GPU (`GRSI_GPU=-1` forces CPU), `GRSI_ENV=<name>` to use
an existing conda environment, `GRSI_CKPT_DIR=<dir>` if the checkpoints were downloaded manually.

## Files

| Path | Role |
|---|---|
| `run.sh` | the one-command entry point (environment → checkpoints → `make_fig9.py`) |
| `environment.yml` | conda environment used by `run.sh` (see Requirements) |
| `fetch_checkpoints.sh` | downloads the two released checkpoints and verifies their SHA-256 |
| `make_fig9.py` | hint generation, region selection, DetFill inference, figure assembly |
| `data/` | the two test illustrations (`<id>.image.png`) and their SketchKeras sketches (`sketch/<id>.png`) |
| `expected/fig9_paper.png`, `expected/paper_panels/` | the figure and the 256×256 panels as printed in the paper |
| `output/` | created by the script: `fig9.png`, per-panel PNGs, `summary.json`, staged inputs, DetFill logs |

## Beyond this figure

The same code reproduces the paper's tables given the Danbooru2021 test split (which we cannot redistribute):
`detfill/run_inference.sh` runs the Hint-AUC ratio grid with the released checkpoints and the released stored hint
maps (`test_split_hint_maps_64.tar.gz` in the v1.0 release), and `evaluation/` computes the per-ratio metrics and
Hint-AUC. Training code and configurations are in `detfill/` (`train.sh`); see the repository README.
