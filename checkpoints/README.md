# Model zoo: released checkpoints and data

Everything needed to run or re-evaluate the paper's models is attached to the GitHub releases of this repository.
Each release carries a `SHA256SUMS.txt`; the tables below repeat the hashes so a downloaded file can be checked
with `sha256sum -c`. Sizes are approximate.

## v1.0 — the paper's models (Danbooru2021 illustrations) and their evaluation inputs

Release: https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0

| File | Size | SHA-256 | What it is |
|---|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 1.08 GB | `fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4` | DetFill **scribble-hint** model, 96 base channels, 200 epochs. Behind every scribble result of the paper (Table II, Table III, Fig. 9 scribble row, the supplementary α-grid and seed studies, the "Felzenszwalb" training row of the segmentation-dependency table). |
| `detfill_dot_illust_200ep.pth` | 0.48 GB | `b4779946f24b5f52cdf640418c73bef67925c845e50c8f36610aeac7d50f212b` | DetFill **dot-hint** model, 64 base channels, 200 epochs. Behind every dot result of the paper (Table II, Table III, Fig. 9 dot row). |
| `test_split_hint_maps_64.tar.gz` | 60 MB | `01c0aa92ae976d9e0a5c48a02881c7ed9db9f0e98d77fd14e44c396471ac0dfd` | The stored 64×64 deterministic hint maps (scribble and dot, colour + mask) and region maps of the 3,000 test images. These are the exact evaluation inputs of all reported numbers. |

Where the DetFill launchers expect the two models (`detfill/run_inference.sh` and `reproduce/scripts/run_hauc_pipeline.sh`
default to this path; `replicability/run.sh` and `reproduce/examples/run_examples.sh` download into their own
`checkpoints/` directories; any location works with `--resume_model <file>` / `CKPT=<file>`):

```bash
BASE=https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/download/v1.0
mkdir -p detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint detfill/results/dataset_name/BrownianBridge_dot_illust/checkpoint
curl -L $BASE/detfill_scribble_illust_200ep.pth -o detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth
curl -L $BASE/detfill_dot_illust_200ep.pth      -o detfill/results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth
```

Notes
- Both checkpoints hold the EMA weights saved by the training loop at epoch 200; DetFill is a pixel-space model, so no
  VQGAN or latent-diffusion weights are needed.
- Configuration: `detfill/configs/scribble_illust.yaml` (96 channels) for the scribble model, `detfill/configs/dot_illust.yaml`
  (64 channels) for the dot model. Inference is seeded (`--seed 1234`, the default of `detfill/main.py`).
- The hint maps in `test_split_hint_maps_64.tar.gz` are the reference inputs. Regenerating them from the images with the
  `hintauc` library gives the same maps up to the unseeded tie-breaking inside the skeleton extraction (see
  `hint_generation/README.md`), so use the stored maps whenever you want to compare with the printed numbers.
- Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded by their pip packages on first use and are not
  part of the releases.

## v1.1 — scribble models trained with other region segmenters

Release: https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.1 — the "DanbooRegion" and "SLIC" *training*
rows of the supplementary segmentation-dependency table (same protocol and training split as the v1.0 scribble model,
96 base channels, 200 epochs; only the segmenter that produced the training hints differs).

| File | Size | SHA-256 | Training segmenter |
|---|---|---|---|
| `detfill_scribble_illust_danbooregion_200ep.pth` | 1.08 GB | `1770cd61bb496060b4ff9bd8c2ec2e72b549fa99187a0429912100628cad9a78` | DanbooRegion |
| `detfill_scribble_illust_slic_200ep.pth` | 1.08 GB | `eb25667c1955584f46f34abf98db10eef6d702e4c1078fe3cff388f646cf4217` | SLIC |

Use them with `detfill/configs/scribble_illust.yaml` and `--resume_model <file>`; the evaluation data root must
contain hint maps from the matching segmenter (release v1.3, `test_split_hint_maps_64_{danbooregion,slic}.tar.gz`).

## v1.2 — Diffusart-retrain models, later natural-image (ImageNet) models and the ImageNet hint maps

Release: https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.2

| File | Size | SHA-256 | What it is |
|---|---|---|---|
| `diffusart_retrain_scribble_dethint_200ep_ema.pth` | 373 MB | `03ed65e385cab3947e52f9bbdf00c709d84af1ec7d05631624d2266dc40fb15b` | Diffusart (our re-implementation, `reproduce/paper_experiments/diffusart_retrain/code/`) retrained with the deterministic scribble hints, 200 epochs, EMA weights. The **Diffusart-retrain** scribble row of the supplementary "additional training comparisons" table; its PSNR 19.457 / LPIPS 0.177 are reproduced from the per-ratio record shipped in `reproduce/expected/`. |
| `diffusart_retrain_dot_dethint_200ep_ema.pth` | 373 MB | `8c78717ac73f111ca31e00b43f4c7686e618c4191ce5ea38a437f2ed6b8964d1` | Same for dot hints (the dot row of that table; same training batch). |
| `detfill_scribble_imagenet_96ch_200ep.pth` | 1.08 GB | `0f8514000110be43927b9031912b4331e2753e3028ea8a0ab680630857d1d329` | DetFill scribble model on natural images with **96 base channels**: a later training generation that is **not** behind the printed ImageNet tables (those come from the 64-channel models of v1.4). Config: `detfill/configs/scribble_real.yaml` with `model_channels: 96`. |
| `detfill_dot_imagenet_96ch_200ep.pth` | 1.08 GB | `1fdf881981c9a4e415711a8bdb6bab36c3e0251243e5d04544e7fc6b9811fd2e` | Same for dot hints (96 base channels, later generation; about 3 dB above the printed dot values at every hint ratio). Config: `detfill/configs/dot_real.yaml` with `model_channels: 96`. |
| `test_split_hint_maps_64_imagenet.tar.gz` | 40 MB | `870db22dd7526c2fb3fe7d9bd5d3b7d10b90c9fa776ac901c1b04ab52bf01847` | The stored 64×64 hint maps (scribble/dot colour + mask) and region maps of 2,999 of the 3,000 ImageNet test images in `detfill/configs/real/test.txt` (`n02667379_7324` has no stored map), i.e. the evaluation inputs of the supplementary ImageNet tables. |

The Diffusart-retrain models take the **64 × 64 deterministic hint maps of v1.0**, upsampled to 256 × 256 with
nearest-neighbour interpolation inside the inference script — this is how the paper's Diffusart-retrain record was
produced. Inference: `reproduce/paper_experiments/diffusart_retrain/code/infer_det_proposed.py` (PyTorch 2.5 and
`diffusers` 0.32; see the README in that directory).

Natural-image models: two training generations exist. The 64-channel models (release v1.4) reproduce the printed
ImageNet values (24 test images, SketchKeras line art: PSNR Hint-AUC 15.0 dot / 18.4 scribble against 15.7 / 18.0 printed),
whereas the 96-channel models released here score 2–3 dB higher at every hint ratio; the printed supplementary
ImageNet tables therefore come from the 64-channel models, and the 96-channel ones are a later, stronger
generation that is not in the paper. The `*_real.yaml` configs use the split-based layout (`dataset_path` plus the
relative lists in `detfill/configs/real/`, run `main.py` from `detfill/`); the ImageNet images themselves are not
redistributed.

## v1.3 — data for full-scale and example-based reproduction

Release: https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.3 (all files derived from the Danbooru2021
test split; the originals are not included — they are available from https://gwern.net/danbooru2021 by id; non-commercial
research use).

| File | Size | SHA-256 | Contents |
|---|---|---|---|
| `test_split_sketch_sketchkeras.tar.gz` | 441 MB | `6c121c523bca26546e87436514a5c8aae827e583dbde51c09e4596da7b4a895a` | SketchKeras line art of the 3,000 test images (sketch type 2) — the files used for every reported number |
| `test_split_sketch_pysimp.tar.gz` | 477 MB | `b05a78e6fc1b996b4010a9f2091c98a496c86e42c894d2042fa833090adbae59` | Sketch-simplification line art (sketch type 0) |
| `test_split_sketch_XDoG.tar.gz` | 56 MB | `0aa4529465f8ad4b0bef4d2d04206537ada194316a56bd22cbf1eed109731fe2` | XDoG line art (sketch type 1) |
| `test_split_hint_maps_64_danbooregion.tar.gz` | 37 MB | `d11217a11467ad4cb1aec85d4914be751d7e18d90d4d18d9dc6004b8e2693756` | DanbooRegion scribble hint maps + region maps of the test split (segmentation-dependency study, evaluation segmenter DanbooRegion) |
| `test_split_hint_maps_64_slic.tar.gz` | 38 MB | `01d6443c9069e23323e5afc98490c5a95bf34056f4d97da3b4111b1967ad9e51` | SLIC scribble hint maps + region maps of the test split |
| `test_split_hint_maps_64_felz.tar.gz` | 44 MB | `1933b97cd7c92e0651db4839da75e324cc4048bd6fced480c1f2bc512f2d0507` | Felzenszwalb maps regenerated for that study (main results use the v1.0 maps) |
| `examples_data.tar.gz` | 8 MB | `3d7f699da9f8de5a5def35a997e7bb42461abd56babd3ce6c30adfbee9f71c87` | 12 example illustrations with originals, three line-art versions, the paper's hint maps and the DanbooRegion/SLIC maps — input of `reproduce/examples/run_examples.sh` |
| `userstudy_stimuli.tar.gz` | 623 MB | `7bf1cc36999ef200edf48d8e2749338d990da3a4fb6323b021b26f007aef359d` | The images shown in the user study (192 images × 8 hint ratios: line art, hint image, DetFill / PaintsTorch / Diffusart colorizations, ground truth, `index.csv`); keyed like `reproduce/data/userstudy/glmm_trials.csv`. The ColorizeDiffusion-v2 stimuli were not preserved. |

## v1.4 — the natural-image models of the paper, the earlier-submission scribble model and the channel-ablation models

Release: https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.4. The two natural-image models here are the
ones behind the supplementary ImageNet tables; the other files are released for completeness and for the supplementary
channel ablation and are **not** behind the main tables (those are v1.0).

| File | Size | SHA-256 | What it is |
|---|---|---|---|
| `detfill_scribble_illust_32ch_200ep.pth` | 129 MB | `4eef804755b9adc4fc0b5234501666138a4a6ecddfbdb28245fd7fc32443ecda` | Scribble model with 32 base channels (supplementary channel-ablation figure, "C = 32"); use `scribble_illust.yaml` with `model_channels: 32` |
| `detfill_scribble_illust_64ch_200ep.pth` | 485 MB | `8eeee341331ed949cc216049c9fecebc70900d54fb709eee3b97a2b1fc34845b` | Scribble model with 64 base channels (channel ablation) |
| `detfill_scribble_illust_earlier_64ch_200ep.pth` | 485 MB | `7d74ebe7d6c8862286fe9944b7cf9d2448a6e473880c383b13af8a36e6ce0002` | Scribble model of the earlier submission (64 channels; its archived config sampled with 1,000 steps). Superseded by the 96-channel v1.0 model retrained for the revision |
| `detfill_scribble_imagenet_64ch_200ep.pth` | 485 MB | `84dc2e4363b0fd9964338b019a35cecf29e5ef4687d6b167b59f958503520f09` | **The natural-image scribble model behind the supplementary ImageNet tables** (64 base channels; `scribble_real.yaml`). Re-evaluated on 24 test images with SketchKeras line art: PSNR Hint-AUC 18.4 (printed: 18.0) |
| `detfill_dot_imagenet_64ch_200ep.pth` | 485 MB | `41089d46313943fdac66aaf5a022c0aa365cff47c0bbaed581436e040d122a9e` | **The natural-image dot model behind the supplementary ImageNet tables** (64 base channels; `dot_real.yaml`, 3-channel conditioning stage as released). Re-evaluated on 24 test images: PSNR Hint-AUC 15.0 (printed: 15.7) |
| `earlier_run_configs.tar.gz` | 2 KB | `1e3944bcb9aafa45d9359ee2ab12f0a2a1dffd03d404c65d6e8353224588b275` | The archived training configs of the earlier-submission scribble and natural-image models and of the 32/64-channel models |

## Which release do I need?

| I want to … | Download |
|---|---|
| reproduce Fig. 9 with one command | nothing by hand — `replicability/run.sh` fetches the two v1.0 models |
| run the example-based reproduction of every experiment | nothing by hand — `reproduce/examples/run_examples.sh` fetches v1.0, v1.1, v1.3 (and v1.4 in `full` mode) |
| recompute the paper's tables from the released metric files | nothing — the metric files are in the repository (`reproduce/expected/`) |
| re-run a whole table row on the 3,000 test images | v1.0 models + v1.0 hint maps + v1.3 line art + the Danbooru2021 originals (`reproduce/README.md`, part B) |
| re-run the segmentation-dependency study | v1.1 models + v1.3 segmenter hint maps |
| look at the user-study stimuli | v1.3 `userstudy_stimuli.tar.gz` |
