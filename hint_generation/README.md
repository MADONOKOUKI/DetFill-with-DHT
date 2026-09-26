# Deterministic hint generation (DHT): batch scripts

The scripts in this directory built the hint datasets of the paper. For new images the `hintauc` library is the
easiest entry point (`pip install hintauc`; `hintauc.generate_hints(...)` is the same pipeline as a function);
the scripts here are the batch tools with sharding and resume.

## The pipeline (one colour image → hint maps)

1. **Region segmentation** — Felzenszwalb (`scale=100, sigma=0.5, min_size=100`); SLIC, Quickshift, Watershed and
   DanbooRegion variants for the robustness study.
2. **Region-id map** — a unique colour per region; disconnected parts of one region are split and recoloured.
3. **Scribble per region** — Zhang–Suen skeleton → 3 × 3 dilation → longest path (FilFinder, branch/skeleton threshold
   3 px), clipped to the region; coloured with the region's mean colour.
4. **Dot per region** — the in-region path pixel with the smallest total Manhattan distance to the other in-region path
   pixels (`dot_method="medoid"`, the rule of the paper's stored maps).

Output files per image (64 px by default):

```
<id>.image_region64.png           region-id map
<id>.image_scribble_mask64.png    scribble mask          <id>.image_scribble_col64.png   scribble colours
<id>.image_dot_mask64.png         dot mask               <id>.image_dot_col64.png        dot colours
<id>.image_flatten_img64.png      region-mean colour image
```

At evaluation time the hints of the largest ⌊α·n⌋ regions are kept (`hintauc.HintResult.at_ratio`, the DetFill loader).

## Run

```bash
pip install -r requirements.txt
python generate_hints.py --segmenter slic --split test \
    --src_root /path/to/src --txt_dir /path/to/split_lists --out_root /path/to/output
```

- `--src_root`: `<src_root>/segmentation_regions/felzenszwalb/<dir>/<id>.image.png` (the colour images).
- `--txt_dir`: `{train,valid,test}.txt`, one `<dir>/<id>.image.png` per line (the paper's lists: `../detfill/configs/illust/`).
- Output: `<out_root>/<segmenter>/<dir>/<id>.image_{region,scribble_mask,scribble_col}64.png`.
- `--limit N` for a quick check; `--shard / --nshards` to split the work; `--path_method geodesic` for the
  dependency-free longest path (outputs go to `<segmenter>_geodesic/`; do not mix the two methods in one evaluation).

Felzenszwalb hints for a single image without the batch tool:

```python
import hintauc
hints = hintauc.generate_hints("image.png", size=64)   # Felzenszwalb by default
hints.save("out/image")                                # the file set above
```

## Files

| File | Role |
|---|---|
| `generate_hints.py` | batch generator (SLIC / Quickshift / Felzenszwalb; sharding; resume-safe); contains the verbatim `make_scribbling` port |
| `danbooregion_hints.py` | DanbooRegion-segmenter variant (needs the upstream DanbooRegion code and weights) |
| `compare_path_methods.py` | compares the FilFinder and geodesic longest paths on stored region maps |
| `canonical/hint_dot_generation_fixdot.py` | archival copy of the script that produced the paper's stored maps |
| `canonical/hint_dot_generation.py` | archival copy of the earlier version (dot = truncated mean of the path; not the rule of the released maps) |
| `canonical/all_segmentations.py` | archival copy of the segmentation stage |
| `requirements.txt` | dependencies of the scripts in this directory |
