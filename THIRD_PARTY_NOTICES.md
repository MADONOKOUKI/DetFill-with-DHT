# Third-party code and licenses

The repository is released under the MIT License (`LICENSE`), with the exceptions and attributions below.

| Location | Upstream | License | Notes |
|---|---|---|---|
| `detfill/` | [BBDM — Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM) (Li et al., 2023) | MIT (`detfill/LICENSE`) | DetFill is a fork adapted to sketch + deterministic-hint conditioning. |
| `reproduce/paper_experiments/coldiff_finetune/coldiff_v*_inference_official_fixed*.py` | [ColorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion) (Yan et al.) | **CC BY-NC-SA 4.0** (`reproduce/paper_experiments/coldiff_finetune/LICENSE`) | Modified copies of the official inference script (deterministic-hint input). They are distributed under the upstream license, **not** MIT, and only run inside the upstream code base with its weights. |
| Release v1.5 assets `coldiff_*.safetensors` | [ColorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion) (Yan et al.) | **CC BY-NC-SA 4.0** | Our fine-tuned versions of the official ColorizeDiffusion v1/v2 weights. Distributed under the upstream license (non-commercial, share-alike, attribution), **not** MIT. |
| `reproduce/scripts/sketch_tools/sketchkeras_*.py` | [sketchKeras](https://github.com/lllyasviel/sketchKeras) (lllyasviel) | Apache-2.0 (`sketchkeras_LICENSE`) | Our batch wrapper around the upstream helper; the `mod.h5` weights are downloaded from the upstream release. |
| `reproduce/scripts/sketch_tools/aki_edit_d.py`, `pysketchsimplify_*.py` | [Sketch Simplification](https://github.com/bobbens/sketch_simplification) (Simo-Serra et al.) | MIT (code); pretrained weights for non-commercial research use only | Model definition and batch wrapper used to produce the "sketch simplification" line art; obtain the pretrained model from the upstream repository under its terms. |
| `reproduce/scripts/sketch_tools/XDoG_*.py` | XDoG (Winnemöller et al., 2012) | — | Our own NumPy/OpenCV implementation used for the XDoG line art. |
| `reproduce/paper_experiments/segmenter_dependency/r2-2_scripts/D_danbooregion_seg.py` | [DanbooRegion](https://github.com/lllyasviel/DanbooRegion) (Zhang et al., 2020) | upstream | Driver script only; it imports the upstream code and weights, which are not included. |
| `reproduce/paper_experiments/diffusart_retrain/` | our re-implementation of Diffusart (Carrillo et al., CVPRW 2023) | MIT (this repository) | Training/inference code written in our lab for the paper's Diffusart baseline and its deterministic-hint retraining; no upstream code is included. |
| `hintauc/`, `evaluation/`, `hint_generation/` | — | MIT | Uses FilFinder, scikit-image, LPIPS, OpenCLIP, DINOv2 (transformers), DreamSim and torchmetrics as pip dependencies; their weights are downloaded by the respective packages under their own licenses. |

## Runtime dependencies and pretrained weights

| Dependency | License | Used for |
|---|---|---|
| FilFinder (BSD-3), astropy (BSD-3), scikit-image (BSD-3), OpenCV (Apache-2.0), NumPy / SciPy / Pillow (BSD / HPND) | free | hint generation, pixel metrics |
| PyTorch, torchvision (BSD-3); torchmetrics (Apache-2.0); torch-fidelity (Apache-2.0) | free | DetFill, SSIM / MS-SSIM, FID / KID |
| LPIPS (BSD-2; AlexNet weights ship with the package) | free | LPIPS, LPIPS-VGG |
| OpenCLIP (MIT) with the LAION-2B ViT-B/32 weights (MIT) | free | OpenCLIP metric |
| DINOv2 through `transformers` (Apache-2.0; weights Apache-2.0) | free | DINO metric |
| DreamSim (MIT; weights MIT) | free | DreamSim metric |
| DISTS (`dists-pytorch`, MIT) | free | DISTS metric |
| taming-transformers (MIT), vendored under `detfill/model/VQGAN/taming` by BBDM | free | not used by the released configs |
| Contributor Covenant 2.1 (CC BY 4.0) | — | `CODE_OF_CONDUCT.md` |

All of them allow free non-commercial and academic use; weights are downloaded by the respective packages on first use.

## Data

The illustrations of the paper come from [Danbooru2021](https://gwern.net/danbooru2021) (Danbooru posts; the
Danbooru2021 ids are the post ids). The images themselves are **not redistributed**, with one exception: the
illustrations that appear in the figures of the paper. Everything else refers to the original distribution by id,
and `reproduce/scripts/fetch_originals.py` fetches an image by id from Danbooru and reproduces the dataset's 512 × 512
copy bit-exactly.

Released for non-commercial research use only, so that the paper can be replicated:

- derived data of the 3,000 test images, without the images: the 64 × 64 hint maps and region maps (v1.0, v1.3), the
  line art (v1.3) and the per-ratio metric files (`reproduce/expected/`);
- the nine paper-figure illustrations of the example bundle `examples_data.tar.gz` (v1.3; ids 514016, 4262016,
  1625016, 3421016, 4417016, 1019016, 1023016, 4731016, 4942016) with their line art and hint maps, the same two
  Fig. 9 illustrations in `replicability/data/` (see `replicability/data/NOTICE.txt`), the demonstration images in
  `assets/readme/` (ids 4262016 and 514016), the paper's archived outputs and the authors' example grids in
  `reproduce/examples/expected/` (these nine images only);
- the user-study stimuli (v1.3): line art, hint images and the colorizations of three methods, indexed by
  `index.csv`; the original illustrations of the study are not included (fetch them by id from `index.csv`).

Rights holders can request the removal of an image through the repository's issue tracker or the contact in
`replicability/GRSI_SUBMISSION.txt`.
The natural-image experiments use an [ImageNet](https://www.image-net.org/download.php) subset; only its 64 × 64 hint
maps are released (v1.2).
