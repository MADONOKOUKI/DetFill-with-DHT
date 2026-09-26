# Third-party code and licenses

The repository is released under the MIT License (`LICENSE`), with the exceptions and attributions below.

| Location | Upstream | License | Notes |
|---|---|---|---|
| `detfill/` | [BBDM — Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM) (Li et al., 2023) | MIT (`detfill/LICENSE`) | DetFill is a fork adapted to sketch + deterministic-hint conditioning. |
| `reproduce/paper_experiments/coldiff_finetune/coldiff_v*_inference_official_fixed*.py` | [ColorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion) (Yan et al.) | **CC BY-NC-SA 4.0** (`reproduce/paper_experiments/coldiff_finetune/LICENSE`) | Modified copies of the official inference script (deterministic-hint input). They are distributed under the upstream license, **not** MIT, and only run inside the upstream code base with its weights. |
| `reproduce/scripts/sketch_tools/sketchkeras_*.py` | [sketchKeras](https://github.com/lllyasviel/sketchKeras) (lllyasviel) | Apache-2.0 (`sketchkeras_LICENSE`) | Our batch wrapper around the upstream helper; the `mod.h5` weights are downloaded from the upstream release. |
| `reproduce/scripts/sketch_tools/aki_edit_d.py`, `pysketchsimplify_*.py` | [Sketch Simplification](https://github.com/bobbens/sketch_simplification) (Simo-Serra et al.) | see upstream (weights: non-commercial research use) | Model definition and batch wrapper used to produce the "sketch simplification" line art; obtain the pretrained model from the upstream repository under its terms. |
| `reproduce/scripts/sketch_tools/XDoG_*.py` | XDoG (Winnemöller et al., 2012) | — | Our own NumPy/OpenCV implementation used for the XDoG line art. |
| `reproduce/paper_experiments/segmenter_dependency/r2-2_scripts/D_danbooregion_seg.py` | [DanbooRegion](https://github.com/lllyasviel/DanbooRegion) (Zhang et al., 2020) | upstream | Driver script only; it imports the upstream code and weights, which are not included. |
| `reproduce/paper_experiments/diffusart_retrain/` | our re-implementation of Diffusart (Carrillo et al., CVPRW 2023) | MIT (this repository) | Training/inference code written in our lab for the paper's Diffusart baseline and its deterministic-hint retraining; no upstream code is included. |
| `hintauc/`, `evaluation/`, `hint_generation/` | — | MIT | Uses FilFinder, scikit-image, LPIPS, OpenCLIP, DINOv2 (transformers), DreamSim and torchmetrics as pip dependencies; their weights are downloaded by the respective packages under their own licenses. |

Data: the released hint maps and metric files are derived from the Danbooru2021 test split; the images themselves are not redistributed.
