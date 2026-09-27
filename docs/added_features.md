# Added features (since the paper) for improving our library

Everything below was added after the paper's experiments and does not change any reported number.

- **`hintauc` pip library and command line** — hint generation, the seven metrics of the paper and Hint-AUC as a
  package (`pip install hintauc`; `hintauc generate`, `hintauc eval`).
- **Hint generation options** — a dependency-free, bit-exact longest path (`path_method="geodesic"`); dot placement
  rules `medoid` (the paper's stored maps) / `mean` / `nearest_mean`; deterministic region tie-breaking (`tie_break="stable"`);
  other segmenters (SLIC and Quickshift in the library and the batch generator; DanbooRegion through `hint_generation/danbooregion_hints.py`).
- **Evaluation protocols in the DetFill loader** — `hint_order: area | label` switches between the Table II and Table III
  region orders; the training option `include_full_hint` lets the fully hinted case appear during training.
- **More metrics, enabled by name** — MAE, MS-SSIM, CIEDE2000 colour difference (ΔE00), LPIPS-VGG and DISTS per image,
  and set-level FID / KID between two directories (`hintauc eval pred_dir gt_dir --set_metrics fid kid`). Hint-AUC
  works with any of them.
- **Runs everywhere** — DreamSim and the other metric backbones work on CPU-only machines (DreamSim caches its weights
  under `~/.cache/hintauc`); DetFill inference runs on CPU (`--gpu_ids -1`); a flat, user-configurable data layout;
  deterministic region-id colours.
- **Evaluate your own model** — `hintauc.evaluate_colorizer` (plug in a Python function; the callable sees only the
  model's observations unless `oracle=True`) and `hintauc curve` (directories of images per hint ratio); every result
  carries a protocol record (grid, metric settings, the resize / SSIM backend, library versions, optional checkpoint
  hash) and can be plotted (`hintauc.plot_curves`, `eval_per_ratio.py --plot`).
- **Input checks before scoring** — every ratio directory must hold the same images (a missing prediction is an error,
  `--allow-missing` opts into the common subset); ratio grids are validated (no duplicates, strictly increasing, 0 to 1);
  saved predictions get collision-free directory names and a manifest; image writes that fail raise instead of
  returning success; float images are accepted only in [0, 1]; external region maps whose colours would collide under
  the loader's base-255 ids are re-encoded; `evaluate_colorizer` orders equal-area regions with the machine-independent
  stable tie-break by default (NumPy's default argsort orders them differently on AVX-512 CPUs).
- **One evaluation backend** — without PyTorch the evaluator now resizes float32 channels with Pillow's antialiased
  bilinear filter, which agrees with the paper's torchvision resize to about 1e-6 per pixel (before, a uint8 bicubic
  resize changed PSNR by up to 0.4 dB); `pip install "hintauc[paper]"` installs the exact backend.
- **`hintauc demo`** — a self-contained example on a synthetic illustration whose numbers are recorded and checked in CI.
- **Exact geodesic diameter** — `path_method="geodesic"` now searches all pixels of a skeleton component that contains a
  cycle (the diameter of such a component can end away from every endpoint).
- **Release gate** — the PyPI workflow installs the built wheel into fresh environments (with and without PyTorch) and
  runs the tests, the demo and the quick start before publishing; the number checker requires finite values and the
  expected backend.
- **Image decoding** — every input (file or array) is decoded to 8-bit RGB by one routine: palette, grayscale, LA,
  RGBA and CMYK files are read as the colours they show on both backends; 16-bit and float files are rejected.
- **Run identity** — `evaluate_colorizer` writes a manifest with the ratio grid, the prediction names and their
  ground-truth files, `oracle`, `tie_break` and the path method; `hintauc curve` scores exactly those files (stale
  directories of another run are refused, JPEG ground truths pair with their PNG predictions) and a `save_dir` that
  holds another run is refused unless `overwrite=True`. `eval_per_ratio.py` checks the image sets of all ratio
  directories, reuses rows only for unchanged files (SHA-256) and matching metric settings, and DetFill's
  `sample_to_eval` refuses to reuse outputs whose `run_manifest.json` (checkpoint hash, seed, hint settings, data)
  differs.
- **Training loop** — a failed training job re-raises after the emergency checkpoint; the step budget
  (`training.n_steps` / `--max_steps`, micro-batches per process) stops inside an epoch.
- **Batch hint generation** — `hint_generation/generate_hints.py` and `danbooregion_hints.py` use the library's
  generator: deterministic region ids, re-encoded DanbooRegion colours, checked writes and a resume that only skips
  complete outputs.
- **Example comparison** — `compare_with_expected.py` compares image by image on the ids present in both runs.
- **Reproduction tooling** — the one-command Fig. 9 script (`replicability/`), the reproduction package (`reproduce/`)
  with the example suite (eight experiments on example images), and the additional releases (v1.1–v1.4) with all remaining
  checkpoints and data.

We also fixed a few minor issues in the original implementation to make the library more usable.

Function-level documentation with examples: [API reference](api.md). Details of each item: [Detailed explanation — Added features since the paper](detail_explanation.md#added-features-since-the-paper).
