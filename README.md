# DetFill-with-DHT

Code, models and reproduction package for
**Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation**
(K. Madono, Y. Mingcheng, E. Simo-Serra — IEEE TVCG 2026, DOI [10.1109/TVCG.2026.3738401](https://doi.org/10.1109/TVCG.2026.3738401)).

![Deterministic hint generation pipeline](assets/readme/dht_pipeline.png)

**What is in here**
- **DHT** — deterministic, region-based colour-hint generation: the same illustration always gives the same scribble / dot hints.
- **Hint-AUC** — an evaluation protocol that scores a colorization model over the whole range of hint ratios and integrates the metric curve into one number.
- **DetFill** — the pixel-space diffusion colorization model of the paper, with every checkpoint.
- A **reproduction package** that rebuilds the paper's tables, re-runs each experiment on example images and re-runs whole table rows.

This repository also contains minor fixes relative to the paper and a few added features; both are listed below.
Details of everything on this page: [detail_explanation.md](detail_explanation.md).

---

## Install

```bash
pip install git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git
# with the perceptual metrics (LPIPS / OpenCLIP / DINOv2 / DreamSim):
pip install "hintauc[perceptual] @ git+https://github.com/MADONOKOUKI/DetFill-with-DHT.git"
```

## Use the library

```python
import hintauc
hints = hintauc.generate_hints("illustration.png", size=64)      # image -> deterministic hints
color, mask = hints.at_ratio(0.10, hint_type="scribble")         # hints of the largest 10 % of the regions
scores = hintauc.Evaluator(metrics=("psnr", "lpips"))("out.png", "gt.png")   # paper metrics
```

Command line: `hintauc generate image.png --ratio 0.1`, `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`.
More: [examples/basic_usage.py](examples/basic_usage.py), [detail_explanation.md](detail_explanation.md#library).

## Reproduce the paper

Everything runs from this repository plus the files on the [GitHub releases](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases)
(downloaded automatically, SHA-256 verified). Requirements: Linux, Miniconda, ~10 GB disk, an NVIDIA GPU (CPU works but is slow).

| Goal | Command (no arguments) | Time |
|---|---|---|
| **Fig. 9** end to end — the Replicability-Stamp script | `bash replicability/run.sh` | GPU ≈ 3 min, CPU ≈ 30 min |
| **Every table** rebuilt from the released metric files (Tables II–V, supplementary tables, GLMM) | `python reproduce/scripts/A1_tables_from_released_metrics.py` → 322/324 cells match<br>`python reproduce/scripts/A2_userstudy_glmm.py` → all values match | < 2 min, CPU |
| **Every experiment** re-run on 12 example illustrations with the released checkpoints (Table II/III protocols, segmentation dependency, seed sensitivity, dense ratio curve, channel ablation, hint regeneration, 2024 models) | `bash reproduce/examples/run_examples.sh` | smoke ≈ 5 min, default ≈ 1 h, full = hours |
| **A whole table row** on the 3,000 test images | `DATA_ROOT=… bash reproduce/scripts/run_hauc_pipeline.sh` | ≈ 14 GPU-h per row |
| **Training** from scratch | `bash reproduce/scripts/B9_train_detfill.sh` | days, 10 GPUs |

- Outputs are compared automatically with the paper's numbers, the paper's archived images of the same examples and the authors' reference run.
- What is bit-exact and what is not: [reproduce/README.md → Known deviations](reproduce/README.md#known-deviations-and-gaps-honest-list).
- The scripts that were actually run for the paper (cluster paths included) are kept as example code in `reproduce/paper_experiments/`.

## Released checkpoints and data

All on the releases page; every file with size, SHA-256 and purpose in [checkpoints/README.md](checkpoints/README.md).

- **v1.0** — the paper's models: `detfill_scribble_illust_200ep.pth` (96 channels), `detfill_dot_illust_200ep.pth` (64 channels); the stored hint maps of the 3,000 test images.
- **v1.1** — scribble models trained with DanbooRegion / SLIC hints (segmentation-dependency study).
- **v1.2** — Diffusart retrained with our hints; natural-image (ImageNet) DetFill models and their test hint maps.
- **v1.3** — test-split line art (3 extractors), alternative-segmenter hint maps, the 12-image example bundle, the user-study stimuli.
- **legacy-2024** — models of the 2024 submission and the 32 / 64-channel models of the channel ablation.

## Minor fixes relative to the paper

- **Dot placement.** The text describes a truncated mean of the longest-path pixels; the data (training and test maps) use the *medoid* pixel. The library defaults to the data's rule; the text's rule is available as `dot_method="mean"`. Results are unaffected.
- **SSIM.** The published SSIM values need `torchmetrics==1.4.0` (pinned); newer versions changed the implementation.
- **Natural-image configs** corrected to 96 base channels (all archived ImageNet checkpoints are 96-channel).
- **Table II, DetFill scribble SSIM** is printed as 0.724; the value is 0.7235 (as in the supplementary α-grid table).
- **Line-art labels.** The loader's `sketch_type` 0/1/2 = sketch simplification / XDoG / SketchKeras; archived per-ratio files had 0 and 1 swapped in name only (all published values are means over the three).

## Added features (not in the paper)

- `hintauc`: pip-installable library and command line for hint generation, the seven metrics and Hint-AUC.
- Second longest-path implementation `path_method="geodesic"` — dependency-free and bit-exact (the paper used FilFinder).
- Dot placement options `medoid` / `mean` / `nearest_mean`; region tie-breaking option `tie_break="stable"`.
- Evaluation-protocol switch `hint_order: area | label` (Table II vs. Table III) in the DetFill data loader.
- Training option `include_full_hint` (also sample the fully hinted case during training).
- CPU inference (`--gpu_ids -1`) and a flat, user-configurable data layout (no cluster paths).
- Deterministic colour encoding of region ids in generated region maps.
- Other segmenters in the hint generator (`--segmenter slic | quickshift | watershed | danbooregion`).
- The replicability script, the reproduction package (table recomputation, generic inference + Hint-AUC pipeline, example suite) and the additional released checkpoints and data.

## Repository layout

| Directory | Contents |
|---|---|
| `replicability/` | one-command reproduction of Fig. 9 |
| `reproduce/` | reproduction package: `scripts/`, `examples/`, `expected/` (released metric files), `data/`, `paper_experiments/` |
| `hintauc/` | the library (hints, metrics, Hint-AUC, CLI) |
| `detfill/` | the DetFill model (BBDM fork): training, inference, configs |
| `hint_generation/`, `evaluation/` | the original research scripts behind the library |
| `checkpoints/` | inventory of all released files with hashes |
| `assets/` | README images, representative image for the Replicability Stamp |

## License and citation

MIT License; third-party components in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). `detfill/` is derived from
[BBDM](https://github.com/xuekt98/BBDM) (MIT). Released data files are derived from Danbooru2021 illustrations and are
provided for non-commercial research use only.

```bibtex
@article{madono2026hintauc,
  title   = {Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation},
  author  = {Madono, Koki and Mingcheng, Yuan and Simo-Serra, Edgar},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026},
  doi     = {10.1109/TVCG.2026.3738401}
}
```
