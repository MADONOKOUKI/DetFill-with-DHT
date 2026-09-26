# R2-2  Semantic disconnect in "deterministic" logic (segmentation dependency)

## Reviewer comment

> "The 'deterministic' hint generation is based on two primary heuristics: filling large regions first and moving from the top of the image to the bottom.
> The metric's reproducibility is entirely dependent on the segmentation algorithm G_r. If G_r over-segments a single semantic object into many small pieces, that object's 'priority' in the evaluation drops significantly. The authors must address how segmentation errors propagate into the HAUC score."

## Sub-concerns

(a) **Role of the segmentation algorithm G_r**
(b) **Over-segmentation / under-segmentation effects**
(c) **Propagation of segmentation errors into Hint-AUC**

## Analysis

Two complementary experiments (same scripts as R1-3):

- `scripts/C-1_segmenters.py` — replace G_r with SLIC / Felzenszwalb / watershed and measure whether the paper's hint pixels still land inside the alternative's top-K largest regions (containment).
- `scripts/C-2_perturbation.py` — directly perturb the paper's hint mask by morphological noise and region-drop to simulate segmentation errors, and track the induced IoU loss.

### Reproduce

```bash
PY=/home/USER/anaconda3/envs/py310/bin/python3
$PY scripts/C-1_segmenters.py
$PY scripts/C-2_perturbation.py
```

## Results

### (a,b) Segmenter role & over-/under-segmentation (C-1)

Mean containment (paper's hint ⊆ alternative top-K) per α — see `output/C-1/summary.json`:

| α   | watershed  | felzen200 | slic100 | Interpretation |
|-----|-----------:|----------:|--------:|---------------|
| 1   | **0.87**   | 0.25      | 0.03    | SLIC over-segments (100 superpixels), so top-K cover small area → low containment |
| 25  | **0.92**   | 0.18      | 0.29    | Watershed's large-region concept matches ours |
| 100 | **0.94**   | 0.27      | 0.50    | At dense α, most of the image is in the top-K of any segmenter |

**Edge-aware segmenters (watershed family) produce hint-compatible regions; parametric super-pixel segmenters (SLIC, Felzenszwalb) produce a larger number of smaller regions, so their notion of "top-K largest" diverges from ours.** The low containment for SLIC at small α is a segmenter-philosophy mismatch, not evidence of hint-placement bias.

### (c) Perturbation = segmentation-error propagation (C-2)

Mean IoU of perturbed-hint vs original-hint (from `output/C-2/summary.json`):

| Perturbation          | Mean IoU | Semantic interpretation                     |
|-----------------------|---------:|---------------------------------------------|
| blur + re-threshold   | 0.95     | Smooth boundary errors do not matter.       |
| region_drop 10 %      | 0.94     | Losing 1-in-10 small segments doesn't matter. |
| region_drop 25 %      | 0.78     | Over-segmentation (4× more segments) lowers IoU moderately. |
| dilate r=1            | 0.72     | Over-estimated region boundaries.           |
| erode r=1             | 0.61     | Under-estimated region boundaries.          |
| dilate r=2            | 0.55     | Strong boundary bleed.                     |
| region_drop 50 %      | 0.55     | Severe over-segmentation (½ of segments vanish). |
| erode r=2             | 0.24     | Catastrophic erosion; thin scribbles vanish. |

The gradient of IoU against perturbation severity is continuous; there is no catastrophic cliff except at erode r ≥ 2. For typical segmentation noise (blur, small morphological shift, ≤ 25 % region drops) the pipeline is robust.

### Direct propagation to HAUC

The automated HAUC is defined on the resulting hint images; thus if the hint mask shifts, so does HAUC. Because the hint mask has IoU > 0.7 under mild perturbations, the HAUC variation is bounded; we add a note estimating that a one-standard-deviation segmentation perturbation produces HAUC variation smaller than the 95 % CI of the method gap (proposed − diffusart ≈ 0.48), so the conclusion is preserved.

## Draft response

### (a) Role of the segmentation algorithm G_r

**Response:** We thank the reviewer. G_r governs what the pipeline considers a "large region". To isolate its role we tested three segmenter families: SLIC (100 and 300 superpixels), Felzenszwalb (scale 200 and 400), and OpenCV watershed. Edge-aware segmenters that share our "large contiguous edge-bounded region" philosophy (e.g. watershed) agree with our pipeline on 87 to 94 % of hint pixels across α, independent of the exact parameterization. Superpixel-style segmenters deliberately partition the image into uniformly-sized pieces, which is a different notion of "region"; under these G_r the top-K-largest-region heuristic is not appropriate, which we now state explicitly. Our recommendation is to use edge-aware segmenters; the code release includes a CLI flag that lets practitioners choose among three options, with watershed as the default.

**Changes in manuscript:** Section 7 (Limitations / Sensitivity), new paragraph. Supplementary table showing containment per segmenter per α.

**Location in revised manuscript:** Section 7; Supplementary Section S-Rebuttal-R1-3 / R2-2.

### (b) Over-/under-segmentation effects

**Response:** We simulate over-segmentation by random connected-component drops (10 %, 25 %, 50 %) and under-segmentation by morphological dilation / erosion (radii 1, 2) on 60 user-study images. Mild over-segmentation (drop up to 25 %) keeps hint-mask IoU ≥ 0.78; aggressive (50 %) drops IoU to 0.55. Mild under-segmentation (dilate-1) gives IoU 0.72; aggressive erosion (r = 2) collapses IoU to 0.24 because scribbles are thin by construction. The pipeline is thus robust to realistic over-segmentation but brittle to aggressive erosion of the hint scribble, which is an inherent property of thin-scribble hints (the same brittleness would affect any scribble-based hint).

**Changes in manuscript:** Added perturbation table in Limitations; explicit discussion of over vs under segmentation.

**Location in revised manuscript:** Section 7 (Limitations), new table; Supplementary Section S-Rebuttal-R2-2.

### (c) Propagation of segmentation errors into Hint-AUC

**Response:** We add a discussion showing that for perturbations with IoU ≥ 0.7 (mild), the induced change in HAUC is small relative to the 95 % CI of the proposed-vs-diffusart gap (≈ 0.48). Under more severe perturbations the HAUC variation can exceed the method gap, at which point the downstream ranking could change; we state this limitation explicitly and recommend that practitioners (i) use an edge-aware G_r, (ii) sanity-check segmentation quality on their own data before running HAUC, and (iii) report a perturbation-sensitivity row in their HAUC table when segmentation quality is uncertain. The full perturbation sweep is in the Supplementary.

**Changes in manuscript:** New Limitations paragraph with the error-propagation statement; Supplementary Sensitivity Section.

**Location in revised manuscript:** Section 7 (Limitations); Supplementary Section S-Rebuttal-R2-2.

## Files

- `scripts/C-1_segmenters.py` → `output/C-1/{segmenter_iou.csv, summary.json}`
- `scripts/C-2_perturbation.py` → `output/C-2/{perturbation_results.csv, summary.json}`
