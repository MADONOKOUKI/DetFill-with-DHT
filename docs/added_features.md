# Added features (since the paper) for improving our library

Everything below was added after the paper's experiments and does not change any reported number.

- **`hintauc` pip library and command line** — hint generation, the seven metrics of the paper and Hint-AUC as a
  package (`pip install hintauc`; `hintauc generate`, `hintauc eval`).
- **Hint generation options** — a dependency-free, bit-exact longest path (`path_method="geodesic"`); dot placement
  rules `medoid` (the paper's stored maps) / `mean` / `nearest_mean`; deterministic region tie-breaking (`tie_break="stable"`);
  other segmenters (SLIC, Quickshift, Watershed, DanbooRegion) in the batch generator.
- **Evaluation protocols in the DetFill loader** — `hint_order: area | label` switches between the Table II and Table III
  region orders; the training option `include_full_hint` lets the fully hinted case appear during training.
- **More metrics, enabled by name** — MAE, MS-SSIM, CIEDE2000 colour difference (ΔE00), LPIPS-VGG and DISTS per image,
  and set-level FID / KID between two directories (`hintauc eval pred_dir gt_dir --set_metrics fid kid`). Hint-AUC
  works with any of them.
- **Runs everywhere** — DreamSim and the other metric backbones work on CPU-only machines and cache their weights under
  `~/.cache/hintauc`; DetFill inference runs on CPU (`--gpu_ids -1`); a flat, user-configurable data layout;
  deterministic region-id colours.
- **Reproduction tooling** — the one-command Fig. 9 script (`replicability/`), the reproduction package (`reproduce/`)
  with example-based re-runs of every experiment, and the additional releases (v1.1–v1.4) with all remaining
  checkpoints and data.

We also fixed a few minor issues in the original implementation to make the library more usable.

Details of each item: [Detailed explanation — Added features since the paper](detail_explanation.md#added-features-since-the-paper).
