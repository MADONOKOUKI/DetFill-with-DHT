# DetFill-with-DHT: Deterministic Hints and Hint-AUC for Line Art Colorization

[![IEEE TVCG 2026](https://img.shields.io/badge/IEEE%20TVCG-2026-blue)](https://doi.org/10.1109/TVCG.2026.3738401)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTVCG.2026.3738401-blue)](https://doi.org/10.1109/TVCG.2026.3738401)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](docs/setup.md)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c)](docs/setup.md)
[![PyPI](https://img.shields.io/pypi/v/hintauc)](https://pypi.org/project/hintauc/)
[![Releases](https://img.shields.io/github/v/release/MADONOKOUKI/DetFill-with-DHT)](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases)
[![tests](https://github.com/MADONOKOUKI/DetFill-with-DHT/actions/workflows/tests.yml/badge.svg)](https://github.com/MADONOKOUKI/DetFill-with-DHT/actions/workflows/tests.yml)

Official implementation of **"Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization
Evaluation"**, IEEE Transactions on Visualization and Computer Graphics (TVCG), 2026.

[Koki Madono](https://madonokouki.github.io/) · [Yuan Mingcheng](https://esslab.jp/en/members/) · [Edgar Simo-Serra](https://esslab.jp/~ess/) — Waseda University

[Paper](https://doi.org/10.1109/TVCG.2026.3738401) · [Library on PyPI](https://pypi.org/project/hintauc/) · [Checkpoints and data](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases)

![Deterministic hint generation pipeline](assets/readme/dht_pipeline.png)

Hint-based line art colorization is usually evaluated with randomly sampled colour hints, so the scores change from
run to run. This repository provides **DHT**, a deterministic hint generator that turns an illustration into the same
scribble and dot hints every time; **Hint-AUC**, one score over the whole range of hint ratios, from no hints to fully
hinted; and **DetFill**, a pixel-space diffusion colorization model trained with these hints. All checkpoints, the
evaluation inputs and the scripts that reproduce the paper are released.

## Getting started

- **Replicability Stamp** — `bash replicability/run.sh` reproduces Fig. 9 of the paper from a fresh checkout with no
  arguments; see [replicability/README.md](replicability/README.md) and the submission sheet
  [replicability/GRSI_SUBMISSION.txt](replicability/GRSI_SUBMISSION.txt).
- **Quick start on CPU (a few minutes; `--fast` under a minute)** — `python examples/quickstart_cpu.py` writes hint
  images, a reference colorization at every hint ratio, the metric curves and the Hint-AUC to
  `examples/output/quickstart/`; the expected numbers are in the script's header.
- **Evaluate your own model** — see [Evaluate your own model](docs/evaluate_your_model.md): hand in images per hint
  ratio (`hintauc curve`) or plug in a Python function (`hintauc.evaluate_colorizer`); every result carries a protocol
  record (grid, metric settings, versions).
- **Installation** — see [Setup](docs/setup.md). The library is one command (`pip install hintauc`); DetFill and
  the reproduction scripts use a conda environment that the scripts create for you.
- **Model zoo** — see [Model zoo](checkpoints/README.md): every released checkpoint and data file with its size,
  SHA-256, download link and the paper item it belongs to.
- **Reproducing the paper** — see [Reproducing the paper](reproduce/README.md): Fig. 9 with one command (the
  Replicability-Stamp script), every table rebuilt from the released metrics, every experiment re-run on example
  images, full table rows, and training.
- **API reference** — see [API reference](docs/api.md): every function and command of the library with its role,
  arguments and an input/output example.
- **Added features (since the paper) for improving our library** — see [Added features](docs/added_features.md).
- **How it works** — see [Detailed explanation](docs/detail_explanation.md): each component, what is bit-exact and
  what is not.

## Quick start

```python
import hintauc

hints = hintauc.generate_hints("illustration.png", size=64)        # image -> deterministic hints
color, mask = hints.at_ratio(0.10, hint_type="scribble")           # hints of the largest 10 % of the regions
hints.save("out/illustration")                                     # files in the DetFill data layout

evaluator = hintauc.Evaluator(metrics=("psnr", "lpips", "dreamsim"))  # lpips / dreamsim need pip install "hintauc[perceptual]"
print(evaluator("colorized.png", "ground_truth.png"))              # the paper's metrics
```

- Command line: `hintauc generate image.png --ratio 0.1` and `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`.
- DetFill inference over the hint-ratio grid: `cd detfill && GPU=0 bash run_inference.sh scribble`
  (data layout and options in [detfill/README.md](detfill/README.md)).
- Fig. 9 of the paper from a fresh checkout: `bash replicability/run.sh` (about 3 minutes of compute on a GPU after a
  one-time environment setup and download of 10–20 minutes).

## Repository structure

```
docs/              setup, added features, detailed explanation
replicability/     one-command reproduction of Fig. 9 (Replicability-Stamp entry point)
reproduce/         reproduction package: scripts/, examples/, expected/ (released metrics), data/, paper_experiments/
hintauc/           the library: hints.py, metrics.py, auc.py, cli.py
detfill/           the DetFill model (BBDM fork): training, inference, configs
hint_generation/   original research scripts behind the library
evaluation/        original evaluation scripts
checkpoints/       model zoo: inventory of all released files with hashes
assets/            images for this page; representative image for the Replicability Stamp
```

## License

MIT License ([LICENSE](LICENSE)); third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
`detfill/` is built on [BBDM](https://github.com/xuekt98/BBDM) (MIT). The few Danbooru2021-derived images in this
repository and in the releases (example bundle, user-study stimuli, figure grids) are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and are provided for non-commercial research use only.

## Contributing

Contributions are welcome. See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing

GitHub's "Cite this repository" button uses `CITATION.cff`; the BibTeX entry:

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026},
  doi     = {10.1109/TVCG.2026.3738401}
}
```
