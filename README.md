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

[Paper](https://doi.org/10.1109/TVCG.2026.3738401) · [Library on PyPI](https://pypi.org/project/hintauc/) · [Checkpoints and data](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases) · [Documentation](https://madonokouki.github.io/projects/hintauc/docs/)

![Deterministic hint generation pipeline](assets/readme/dht_pipeline.png)

**DHT** turns an illustration into fixed scribble and dot hints, **Hint-AUC** scores a colorization model over the whole
range of hint ratios, and **DetFill** is the diffusion colorization model trained with these hints; the paper's
checkpoints, evaluation inputs, metric files and experiment scripts are released.

<details markdown="1">
<summary>Why deterministic hints, what is released, and what is exact</summary>

Hint-based line art colorization is usually evaluated with random colour hints, so scores change from run to run.
DHT derives the hints from the regions of the illustration (one scribble and one dot per region, largest regions
first as the hint ratio grows); Hint-AUC integrates a metric over hint ratios from no hints to fully hinted; DetFill is
a pixel-space diffusion model conditioned on line art and these hints. The
[coverage table](reproduce/README.md#coverage-what-each-paper-item-needs) says which paper items are rebuilt
exactly from the released per-ratio metrics, which are re-run from the checkpoints, and which are not provided. The
stored hint maps are the reference for every printed number: regeneration with the paper's FilFinder path can move a
few pixels between runs, the `geodesic` path is bit-reproducible
([details](docs/detail_explanation.md#what-is-exact-and-what-is-not)).

</details>

## Getting started

- **Demo, no data** — `pip install hintauc && hintauc demo` (about 10 s; the numbers are checked in CI).
- **Quick start on CPU** — `python examples/quickstart_cpu.py` (`--fast`: under a minute); expected numbers in the script header.
- **Evaluate your own model** — [docs/evaluate_your_model.md](docs/evaluate_your_model.md): images per ratio (`hintauc curve`) or a Python function (`hintauc.evaluate_colorizer`).
- **Install** — [docs/setup.md](docs/setup.md).
- **Model zoo** — [checkpoints/README.md](checkpoints/README.md): every released file, its hash and its paper item.
- **Reproduce the paper** — [reproduce/README.md](reproduce/README.md): Fig. 9 in one command, tables from the released metrics, the example suite, full table rows, training; a coverage table per paper item.
- **Replicability Stamp** — `bash replicability/run.sh` ([replicability/README.md](replicability/README.md), [submission sheet](replicability/GRSI_SUBMISSION.txt)).
- **Reference** — [API](docs/api.md) · [added features since the paper](docs/added_features.md) · [how it works](docs/detail_explanation.md).

## Quick start

```bash
pip install hintauc            # "hintauc[paper]" adds torchvision + torchmetrics 1.4.0, the paper's metric backend
hintauc demo                   # synthetic illustration -> hints -> oracle colorizer at 8 ratios -> Hint-AUC
```

```python
import hintauc

hints = hintauc.generate_hints("illustration.png", size=64)        # image -> deterministic hints
color, mask = hints.at_ratio(0.10, hint_type="scribble")           # hints of the largest 10 % of the regions
hints.save("out/illustration")                                     # files in the DetFill data layout

evaluator = hintauc.Evaluator(metrics=("psnr", "lpips", "dreamsim"))  # lpips / dreamsim: pip install "hintauc[perceptual]"
print(evaluator("colorized.png", "ground_truth.png"))              # the paper's metrics
```

- Command line: `hintauc generate image.png --ratio 0.1`, `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`,
  `hintauc curve pred_root gt_dir` (Hint-AUC from one directory of outputs per ratio), `hintauc demo`.
- DetFill inference over the ratio grid: `cd detfill && GPU=0 bash run_inference.sh scribble`
  ([detfill/README.md](detfill/README.md)).
- Fig. 9 from a fresh checkout: `bash replicability/run.sh` (3 minutes on a GPU after a one-time setup of 10–20 minutes).

## Repository structure

```
hintauc/           the library (hints, metrics, Hint-AUC, evaluation, CLI)
detfill/           the DetFill model: training, inference, configs (BBDM fork)
reproduce/         reproduction package: scripts, example suite, released metrics, data, paper launchers
replicability/     one-command Fig. 9 (Replicability Stamp entry point)
checkpoints/       model zoo
hint_generation/   original hint-generation scripts
evaluation/        original evaluation scripts
docs/              documentation
assets/            images
```

## License

MIT License ([LICENSE](LICENSE)); third-party components in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
`detfill/` is built on [BBDM](https://github.com/xuekt98/BBDM) (MIT). The Danbooru2021 illustrations are not
redistributed except the nine that appear in the paper's figures (listed there, non-commercial research use only);
all other images are fetched by id from Danbooru with `reproduce/scripts/fetch_originals.py`.

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
