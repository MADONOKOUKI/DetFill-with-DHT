# Deterministic hint generation (DHT)

Everything in this directory produces the deterministic region-based hints used by the
paper. For most purposes the [`hintauc` library](../README.md#quick-start-pip) is the
easiest entry point (`hintauc.generate_hints(...)` implements the same pipeline as a
function); the scripts here are the batch tools used to build the datasets.

## The pipeline

For one color image:

1. **Region segmentation** — Felzenszwalb (`scale=100, sigma=0.5, min_size=100`, the
   paper's setting; SLIC / Quickshift variants for the robustness study).
2. **Region-id map** — every region gets a unique color; disconnected components of the
   same color are split (4-connectivity) and re-colored.
3. **Per-region scribble** — Zhang–Suen skeletonization → 3×3 dilation (1 iteration) →
   FilFinder longest path (branch/skeleton threshold 3 px, prune by length), clipped to
   the region.
4. **Colors** — each region is filled with its mean color; the scribble carries that
   color. The dot hint is a single pixel at the mean coordinate of the longest path.

Outputs per image (`<id>.image_*` naming, 64 px by default):

```
<id>.image_region64.png          # region-id map
<id>.image_scribble_mask64.png   # {0,255} scribble mask
<id>.image_scribble_col64.png    # scribble colors
<id>.image_flatten_img64.png     # region-mean color image
<id>.image_dot_mask64.png / _dot_col64.png
```

At evaluation time, the hints for hint ratio α are the top-α regions ordered by pixel
area (descending) — see `detfill/datasets/custom.py` (`sample_ratio`) or
`hintauc.HintResult.at_ratio()`.

## Files

| File | Role |
|---|---|
| `generate_hints.py` | Batch generator with an argparse CLI (SLIC / Quickshift segmenters; sharding; resume-safe). Contains the verbatim `make_scribbling` port. Recommended batch tool. |
| `danbooregion_hints.py` | DanbooRegion-segmenter variant (imports `make_scribbling` from the file above). |
| `canonical/hint_dot_generation.py` | Archival copy (verbatim) of the original script that built the paper dataset. Not meant to be run as-is. |
| `canonical/all_segmentations.py` | Archival copy of the segmentation stage (Felzenszwalb / SLIC / Quickshift / Watershed). |
| `requirements.txt` | Dependencies for the scripts in this directory. |

## Usage (tested)

```bash
pip install -r requirements.txt

python generate_hints.py \
    --segmenter slic --split test \
    --src_root /path/to/src \
    --txt_dir  /path/to/split_lists \
    --out_root /path/to/output
```

- `--src_root` layout: `<src_root>/segmentation_regions/felzenszwalb/<dir>/<id>.image.png`
  (ground-truth color images).
- `--txt_dir` contains `{train,valid,test}.txt`, one `<dir>/<id>.image.png` path
  per line (the paper's split lists ship in `../detfill/configs/illust/`).
- Output: `<out_root>/<segmenter>/<dir>/<id>.image_{region,scribble_mask,scribble_col}64.png`.
- `--limit N` processes only the first N ids (quick check); `--shard/--nshards` split
  the work across processes.

For Felzenszwalb hints (the paper's default) use the `hintauc` library:

```python
import hintauc
hints = hintauc.generate_hints("image.png", size=64)   # felzenszwalb by default
hints.save("out/image")                                # canonical file set
```

## Longest-path implementation

`generate_hints.py --path_method {filfinder,geodesic}` selects how the longest path of each
region skeleton is extracted. `filfinder` (default) is the paper pipeline (FilFinder2D 1.7.2 on
the 3x3-dilated skeleton). `geodesic` is a dependency-free, fully deterministic alternative
(`hintauc/longest_path.py`: geodesic diameter of the 8-connected skeleton, ties in raster
order); its outputs are written under `<out_root>/<segmenter>_geodesic/`. Hint maps from the
two methods differ slightly and must not be mixed in one evaluation; see
`compare_path_methods.py` and the top-level README.
