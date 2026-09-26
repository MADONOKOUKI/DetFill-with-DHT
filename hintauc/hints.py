"""Deterministic region-based hint generation (DHT).

Faithful port of the pipeline used to build the paper dataset
(`hint_generation/canonical/hint_dot_generation.py` and
`all_segmentations.py`; see also the cleaned R2-2 port
`hint_generation/generate_hints.py`):

    Felzenszwalb segmentation (scale=100, sigma=0.5, min_size=100)
      -> per-color connected-component split (4-connectivity)
      -> per-region Zhang-Suen skeletonization (skimage.morphology.skeletonize)
      -> 3x3 dilation (1 iteration)
      -> FilFinder2D longest path (branch/skel threshold 3 px, prune by length)
      -> scribble = longest path within the region
      -> region color = per-region mean color
      -> dot = single pixel at the mean coordinate of the longest path

Differences from the original scripts (documented, not behavioural for the
algorithm itself):
  * Region-id colors are assigned deterministically (base-255 encoding of a
    sequential id, channels restricted to 0..254) instead of rejection-sampled
    random colors.  Region colors are only identifiers; the base-255 encoding
    makes the id map bijective under the loader's ``c0*255^2 + c1*255 + c2``
    decoding used by ``detfill/datasets/custom.py``.
  * No files are written unless :meth:`HintResult.save` is called.

All image arrays follow the OpenCV BGR convention of the original pipeline
unless stated otherwise.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - import guard
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("hintauc.hints requires opencv-python (cv2)") from e

from skimage.morphology import skeletonize

from .longest_path import geodesic_longest_path
from skimage.segmentation import felzenszwalb

ImageLike = Union[str, "os.PathLike[str]", np.ndarray]

#: canonical hint resolution used for training / evaluation in the paper
DEFAULT_HINT_SIZE = 64

#: Felzenszwalb parameters used for the paper dataset (all_segmentations.py)
FELZENSZWALB_PARAMS = dict(scale=100, sigma=0.5, min_size=100)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_bgr(image: ImageLike) -> np.ndarray:
    """Load an image as HxWx3 uint8 BGR (OpenCV convention, as the original)."""
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    path = os.fspath(image)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


_ID_SCRAMBLE = 1000003  # coprime to 255**3 = (3*5*17)**3 -> bijective scrambling


def _id_to_color(region_id: int) -> np.ndarray:
    """Encode a region id as a BGR color, bijective under base-255 decoding.

    The dataloader (detfill/datasets/custom.py) decodes region colors as
    ``c0*255**2 + c1*255 + c2`` (base 255, a quirk kept for compatibility).
    Restricting every channel to 0..254 makes this decoding collision-free.
    Sequential ids are scrambled by a constant coprime to 255**3 (still
    bijective) so that region maps are visually distinguishable instead of
    near-identical dark shades.
    """
    if region_id >= 255 ** 3:
        raise ValueError("too many regions")
    v = (region_id * _ID_SCRAMBLE) % (255 ** 3)
    c0 = v // (255 * 255)
    c1 = (v // 255) % 255
    c2 = v % 255
    return np.array([c0, c1, c2], dtype=np.uint8)


def region_ids(region: np.ndarray) -> np.ndarray:
    """Decode a HxWx3 region-color map into a HxW int64 id map.

    Uses the same base-255 decoding as the training/evaluation dataloader.
    """
    r = region.astype(np.uint64)
    return (r[:, :, 0] * 255 * 255 + r[:, :, 1] * 255 + r[:, :, 2]).astype(np.int64)


def segment_regions(
    image: ImageLike,
    segmenter: str = "felzenszwalb",
    **seg_kwargs,
) -> np.ndarray:
    """Segment an image and return a region-color map at the input resolution.

    Each region receives a unique color (deterministic base-255 id encoding).
    ``seg_kwargs`` override the paper parameters
    (felzenszwalb: scale=100, sigma=0.5, min_size=100).
    """
    img = _load_bgr(image)
    if segmenter == "felzenszwalb":
        params = dict(FELZENSZWALB_PARAMS)
        params.update(seg_kwargs)
        labels = felzenszwalb(img, **params)
    elif segmenter == "slic":
        from skimage.segmentation import slic

        params = dict(n_segments=290, compactness=10, sigma=1, start_label=1)
        params.update(seg_kwargs)
        labels = slic(img, **params)
    elif segmenter == "quickshift":
        from skimage.segmentation import quickshift
        from skimage.util import img_as_float

        params = dict(kernel_size=3, max_dist=6, ratio=0.5)
        params.update(seg_kwargs)
        labels = quickshift(img_as_float(img), **params)
    else:
        raise ValueError(f"unknown segmenter: {segmenter}")

    region = np.zeros_like(img)
    for i, lab in enumerate(np.unique(labels)):
        region[labels == lab] = _id_to_color(i)
    return region


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class HintResult:
    """Deterministic hints for one image (arrays are BGR / uint8).

    Attributes mirror the canonical output files
    ``*_{region,scribble_mask,scribble_col,flatten_img,dot_mask,dot_col}<S>.png``.
    """

    region: np.ndarray          #: (S,S,3) region-id color map
    scribble_mask: np.ndarray   #: (S,S)   {0,255} longest-path scribbles
    scribble_color: np.ndarray  #: (S,S,3) region-mean color on scribble pixels
    flatten: np.ndarray         #: (S,S,3) region-mean color everywhere
    dot_mask: np.ndarray        #: (S,S)   {0,255} one pixel per region
    dot_color: np.ndarray       #: (S,S,3) region-mean color on dot pixels
    size: int = DEFAULT_HINT_SIZE
    failed_regions: int = 0     #: regions where no path could be extracted
    path_method: str = "filfinder"  #: 'filfinder' (paper) or 'geodesic' (dependency-free)
    dot_method: str = "medoid"      #: 'medoid' (paper data: L1 medoid of the in-region path), 'mean' or 'nearest_mean'
    _sorted_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _sorted_ids_stable: Optional[np.ndarray] = field(default=None, repr=False)

    # -- deterministic size-sorted ratio sampling (port of datasets/custom.py) --
    def _ids_sorted_by_area(self, tie_break: str = "default") -> np.ndarray:
        """Region labels by decreasing area.

        ``tie_break='default'`` reproduces the paper (NumPy's default
        ``argsort``, whose order among equal-area regions depends on the NumPy
        build); ``'stable'`` breaks ties by ascending label value (stable
        sort), which is independent of the NumPy version and hardware.
        """
        if tie_break not in ("default", "stable"):
            raise ValueError(f"tie_break must be 'default' or 'stable', got {tie_break!r}")
        cache = "_sorted_ids" if tie_break == "default" else "_sorted_ids_stable"
        if getattr(self, cache) is None:
            ids = region_ids(self.region)
            vals, counts = np.unique(ids.reshape(-1), return_counts=True)
            order = np.argsort(-counts) if tie_break == "default" else np.argsort(-counts, kind="stable")
            setattr(self, cache, vals[order])
        return getattr(self, cache)

    def at_ratio(
        self,
        alpha: float,
        hint_type: str = "scribble",
        resize_to: Optional[int] = None,
        tie_break: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Hints at hint ratio ``alpha`` in [0, 1].

        Deterministic size-sort selection: regions are ordered by pixel area
        (descending, background included) and the top ``int(n_regions*alpha)``
        regions keep their hints — exactly the evaluation-time behaviour of
        the paper (``detfill/datasets/custom.py`` with ``sample_ratio``).

        ``tie_break`` selects how equal-area regions are ordered: ``'default'``
        (paper: NumPy's default ``argsort``) or ``'stable'`` (ascending label
        value; version- and hardware-independent).  Not used for the reported
        results.

        Returns ``(hint_color, hint_mask)``; optionally upsampled with
        nearest-neighbour interpolation to ``resize_to``.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if hint_type == "scribble":
            color, mask = self.scribble_color, self.scribble_mask
        elif hint_type == "dot":
            color, mask = self.dot_color, self.dot_mask
        else:
            raise ValueError("hint_type must be 'scribble' or 'dot'")

        ids = region_ids(self.region)
        sorted_ids = self._ids_sorted_by_area(tie_break)
        n_keep = int(len(sorted_ids) * alpha)
        keep = sorted_ids[:n_keep]
        area = np.isin(ids, keep)

        hint_color = color * area[:, :, None]
        hint_mask = mask * area
        if resize_to:
            hint_color = cv2.resize(hint_color, (resize_to, resize_to),
                                    interpolation=cv2.INTER_NEAREST)
            hint_mask = cv2.resize(hint_mask, (resize_to, resize_to),
                                   interpolation=cv2.INTER_NEAREST)
        return hint_color, hint_mask

    def n_regions(self) -> int:
        return len(self._ids_sorted_by_area())

    def save(self, save_stem: Union[str, "os.PathLike[str]"]) -> Dict[str, str]:
        """Write the six canonical files ``<save_stem>_<kind><S>.png``."""
        stem = os.fspath(save_stem)
        s = str(self.size)
        paths = {
            "region": stem + "_region" + s + ".png",
            "scribble_mask": stem + "_scribble_mask" + s + ".png",
            "scribble_col": stem + "_scribble_col" + s + ".png",
            "flatten_img": stem + "_flatten_img" + s + ".png",
            "dot_mask": stem + "_dot_mask" + s + ".png",
            "dot_col": stem + "_dot_col" + s + ".png",
        }
        cv2.imwrite(paths["region"], self.region)
        cv2.imwrite(paths["scribble_mask"], self.scribble_mask)
        cv2.imwrite(paths["scribble_col"], self.scribble_color)
        cv2.imwrite(paths["flatten_img"], self.flatten)
        cv2.imwrite(paths["dot_mask"], self.dot_mask)
        cv2.imwrite(paths["dot_col"], self.dot_color)
        return paths


# --------------------------------------------------------------------------
# core generation (port of make_scribbling)
# --------------------------------------------------------------------------
def _split_disconnected_regions(region: np.ndarray) -> np.ndarray:
    """Assign fresh unique colors to disconnected components of a same color.

    Port of the connected-component split in ``make_scribbling`` (4-conn.);
    fresh colors are allocated deterministically, skipping any color already
    present in the map (works for arbitrary user-provided region maps too).
    """
    region = region.copy()
    ids = region_ids(region)
    used = set(int(v) for v in np.unique(ids))
    candidate = 0

    def alloc_color() -> np.ndarray:
        nonlocal candidate
        while True:
            col = _id_to_color(candidate)
            candidate += 1
            v = int(col[0]) * 255 * 255 + int(col[1]) * 255 + int(col[2])
            if v not in used:
                used.add(v)
                return col

    for val in np.unique(ids):
        mask = (ids == val).astype(np.uint8) * 255
        n_labels, labeled = cv2.connectedComponents(mask, connectivity=4)
        # label 0 = background of the mask, label 1 = first (kept) component
        for j in range(2, n_labels):
            region[labeled == j] = alloc_color()
    return region


def _require_filfinder():
    """Import check for the FilFinder backend (paper setting).

    Raises a clear error instead of letting every region silently count as a
    failed path extraction when the optional dependency is missing.
    """
    try:
        import fil_finder  # noqa: F401
        import astropy  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "path_method='filfinder' needs the fil_finder and astropy packages "
            "(pip install 'fil_finder>=1.7' astropy), or use path_method='geodesic'.") from e


def _longest_path(skeleton_dilated: np.ndarray):
    """FilFinder longest-path extraction (parameters of the original script)."""
    from fil_finder import FilFinder2D
    import astropy.units as u

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = FilFinder2D(skeleton_dilated, distance=250 * u.pc, mask=skeleton_dilated)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=3 * u.pix, skel_thresh=3 * u.pix,
                              prune_criteria="length")
    return fil.skeleton_longpath


def generate_hints(
    image: ImageLike,
    size: int = DEFAULT_HINT_SIZE,
    segmenter: str = "felzenszwalb",
    region_map: Optional[ImageLike] = None,
    verbose: bool = False,
    path_method: str = "filfinder",
    dot_method: str = "medoid",
    **seg_kwargs,
) -> HintResult:
    """Generate deterministic region-based hints for ``image``.

    Parameters
    ----------
    image:
        Path or HxWx3 uint8 array (BGR).  Segmentation runs at the input
        resolution; hints are produced at ``size`` x ``size`` (paper: 64).
    size:
        Hint resolution (the paper trains/evaluates with 64).
    segmenter:
        'felzenszwalb' (paper default), 'slic', or 'quickshift'.
    region_map:
        Optional precomputed region-color map (path or array) at any
        resolution; when given, ``segmenter`` is ignored.
    path_method:
        How the longest path of each region skeleton is extracted.
        ``'filfinder'`` (default) reproduces the paper: 3x3 dilation followed
        by FilFinder2D (medial axis, pruning, longest path); note that
        FilFinder's medial-axis step breaks pixel ties with an unseeded
        random generator, so regeneration is not bit-exact.
        ``'geodesic'`` is a dependency-free, fully deterministic alternative
        (:func:`hintauc.longest_path.geodesic_longest_path`): the longest
        shortest path on the 8-connected Zhang--Suen skeleton, all ties broken
        in raster order.  The two methods give slightly different scribbles;
        hint maps produced with different methods must not be mixed within
        one evaluation.
    dot_method:
        Where the dot of a region is placed.  ``'medoid'`` (default) is the rule
        of the stored hint maps used for every experiment in the paper
        (verified against the released test-split maps: identical dot
        positions given identical scribbles): the in-region longest-path pixel
        with the smallest total Manhattan (L1) distance to the other in-region
        path pixels, ties broken in raster order; a single-pixel region whose
        path was pruned away gets the pixel itself.  Every dot therefore lies on
        its own scribble and inside its region.  ``'mean'``: the mean
        row/column of the whole longest path truncated to integers (the rule of
        the January-2024 generation script and the wording of the paper's
        Sec. IV-A); it is not projected onto the region and can fall outside
        it.  ``'nearest_mean'``: the in-region path pixel nearest to that mean.
    """
    if path_method == "filfinder":
        _require_filfinder()
    if path_method not in ("filfinder", "geodesic"):
        raise ValueError(f"path_method must be 'filfinder' or 'geodesic', got {path_method!r}")
    if dot_method not in ("medoid", "mean", "nearest_mean"):
        raise ValueError(f"dot_method must be 'medoid', 'mean' or 'nearest_mean', got {dot_method!r}")
    img_full = _load_bgr(image)
    if region_map is None:
        region_full = segment_regions(img_full, segmenter=segmenter, **seg_kwargs)
    else:
        region_full = _load_bgr(region_map)

    # --- resize to hint resolution (original: img linear, region nearest) ---
    img = cv2.resize(img_full, (size, size))
    region = cv2.resize(region_full, (size, size), interpolation=cv2.INTER_NEAREST)

    region = _split_disconnected_regions(region)
    cand_vals = np.unique(region.reshape(-1, 3), axis=0)

    scribble_img = np.zeros(region.shape)           # region-mean color (flatten)
    scribbles_single = np.zeros((size, size))       # longest-path scribbles
    dot_mask = np.zeros((size, size))
    failed = 0

    iterator = range(len(cand_vals))
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="regions")
        except ImportError:
            pass

    for i in iterator:
        region_mask = np.all(region == cand_vals[i], axis=2)
        indices = np.argwhere(region_mask)
        if indices.size == 0:
            continue

        skeleton_tmp = np.zeros((size, size))
        skeleton_tmp[indices[:, 0], indices[:, 1]] = 1
        skeleton_s = skeletonize(skeleton_tmp)

        # region-mean color, painted over the whole region (flatten image)
        mval = img[indices[:, 0], indices[:, 1]].mean(axis=0).astype(np.uint8)
        scribble_img[indices[:, 0], indices[:, 1]] = mval

        if path_method == "filfinder":
            # 3x3 dilation then FilFinder longest path (original parameters)
            kernel = np.ones((3, 3), np.uint8)
            skeleton_d = cv2.dilate(skeleton_s.astype(np.uint8), kernel, iterations=1) * 255
            try:
                longpath = _longest_path(skeleton_d)
            except Exception:
                failed += 1
                continue
        else:
            # dependency-free geodesic diameter of the (undilated) skeleton
            longpath = geodesic_longest_path(skeleton_s).mask.astype(np.uint8)
            if not longpath.any():
                failed += 1
                continue

        # keep the path inside the (undilated) region
        scribbles_single += longpath * skeleton_tmp

        idx_scr = np.argwhere(longpath == 1)                              # whole longest path (may leave the region)
        idx_in = np.argwhere((longpath == 1) & (skeleton_tmp == 1))       # the part inside the region (= the scribble)
        if dot_method == "medoid":
            # Rule of the paper's stored hint maps (main_hg_illust_64.py, Oct 2024 "fixdot" dataset): the in-region
            # path pixel with the smallest total Manhattan distance to the other in-region path pixels; ties -> first
            # in raster order. A single-pixel region without a path pixel gets the pixel itself (also as scribble).
            if idx_in.size:
                d = np.abs(idx_in[:, None, :] - idx_in[None, :, :]).sum(axis=(1, 2))
                mx, my = (int(v) for v in idx_in[int(np.argmin(d))])
                dot_mask[mx, my] = 1
            elif indices.shape[0] == 1:
                mx, my = (int(v) for v in indices[0])
                dot_mask[mx, my] = 1
                scribbles_single[mx, my] = 1
        elif idx_scr.size:
            if dot_method == "mean":
                # truncated mean of the whole path (hint_dot_generation_20240114_illust_64.py, Jan 2024); not
                # projected onto the region, so it can fall outside it. NOT the rule of the paper's stored maps.
                mx, my = int(idx_scr[:, 0].mean()), int(idx_scr[:, 1].mean())
            else:  # "nearest_mean": in-region path pixel nearest to the (float) mean of the whole path
                cand = idx_in if idx_in.size else idx_scr
                cy, cx = idx_scr[:, 0].mean(), idx_scr[:, 1].mean()
                d2 = (cand[:, 0] - cy) ** 2 + (cand[:, 1] - cx) ** 2
                mx, my = (int(v) for v in cand[int(np.argmin(d2))])
            dot_mask[mx, my] = 1

    scribbles_single = np.clip(scribbles_single, 0, 1)
    scribble_mask = (scribbles_single * 255).astype(np.uint8)
    scribble_color = (scribble_img * scribbles_single[:, :, None]).astype(np.uint8)
    flatten = scribble_img.astype(np.uint8)
    dot_color = (scribble_img * dot_mask[:, :, None]).astype(np.uint8)
    dot_mask_u8 = (dot_mask * 255).astype(np.uint8)

    return HintResult(
        region=region,
        scribble_mask=scribble_mask,
        scribble_color=scribble_color,
        flatten=flatten,
        dot_mask=dot_mask_u8,
        dot_color=dot_color,
        size=size,
        failed_regions=failed,
        path_method=path_method,
        dot_method=dot_method,
    )
