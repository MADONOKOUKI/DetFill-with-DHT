"""Colorization evaluation metrics.

Faithful port of ``evaluation/eval_single_run.py`` (the script behind the
paper's per-ratio metrics):

  * MSE / PSNR / SSIM on [0,1] tensors resized to 256x256
  * LPIPS  — AlexNet backbone, inputs in [-1, 1]
  * OpenCLIP — ViT-B-32 / laion2b_s34b_b79k, cosine similarity mapped to [0,1]
  * DINO   — facebook/dinov2-base (transformers), mean-pooled last hidden
             state, cosine similarity mapped to [0,1]
  * DreamSim — dreamsim(pretrained=True) distance

Metrics added after the paper (not used for any published number; enable them by name):

  * MAE      — mean absolute error on [0,1] tensors (256x256)
  * MS-SSIM  — multi-scale SSIM (torchmetrics), data_range 1
  * DeltaE   — mean CIEDE2000 colour difference in CIELAB (scikit-image); a colour-fidelity measure
  * LPIPS-VGG — LPIPS with the VGG backbone
  * DISTS    — Deep Image Structure and Texture Similarity (DISTS_pytorch)
  * FID / KID — set-level Fréchet / Kernel Inception Distance between a directory of predictions and a
               directory of ground truths (:func:`evaluate_set`; torchmetrics + torch-fidelity)

The perceptual metrics require the optional dependencies
(``pip install hintauc[perceptual]``); their models are loaded lazily on
first use and cached on the :class:`Evaluator`.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ._images import as_uint8_image

ImageLike = Union[str, "os.PathLike[str]", np.ndarray]

DEFAULT_METRICS: Tuple[str, ...] = (
    "mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim",
)
#: metrics added after the paper (opt-in)
EXTRA_METRICS: Tuple[str, ...] = ("mae", "ms_ssim", "deltae", "lpips_vgg", "dists")
ALL_METRICS: Tuple[str, ...] = DEFAULT_METRICS + EXTRA_METRICS
#: set-level metrics computed by :func:`evaluate_set` (directory vs directory)
SET_METRICS: Tuple[str, ...] = ("fid", "kid")
PIXEL_METRICS = {"mse", "psnr", "ssim", "mae", "ms_ssim", "deltae"}
#: direction of each metric (True = lower is better), used by hint_auc callers
LOWER_IS_BETTER = {
    "mse": True, "psnr": False, "ssim": False, "lpips": True,
    "openclip": False, "dino": False, "dreamsim": True,
    "mae": True, "ms_ssim": False, "deltae": True, "lpips_vgg": True, "dists": True,
    "fid": True, "kid": True,
}


def _require(pkg: str, extra: str = "perceptual"):
    raise ImportError(
        f"metric requires the optional dependency '{pkg}'. "
        f"Install with: pip install hintauc[{extra}]"
    )


class Evaluator:
    """Reusable evaluator; loads metric models once and caches them.

    >>> ev = Evaluator(metrics=("mse", "psnr", "ssim"))
    >>> scores = ev(pred_image, gt_image)
    """

    def __init__(
        self,
        metrics: Sequence[str] = DEFAULT_METRICS,
        device: Optional[str] = None,
        resize: int = 256,
    ):
        self.metrics = tuple(m.lower() for m in metrics)
        unknown = set(self.metrics) - set(ALL_METRICS)
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)} (available: {ALL_METRICS}; "
                             f"set-level metrics {SET_METRICS} are computed by evaluate_set)")
        self.resize = resize
        self._device = device
        self._bundle: Dict[str, object] = {}
        self._torch = None

    # ------------------------------------------------------------------
    @property
    def torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self._torch = False
        return self._torch

    @property
    def device(self):
        torch = self.torch
        if not torch:
            return None
        if self._device is not None:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    @staticmethod
    def _rgb_uint8(image: ImageLike) -> np.ndarray:
        """Decode any input to an HxWx3 uint8 RGB array.

        Files are decoded with Pillow and converted to RGB, so palette (P), grayscale (L, LA), RGBA and CMYK files
        give the colours a viewer shows; 16-bit and floating-point files are rejected (the protocol is 8-bit; convert
        them first). Arrays are taken as RGB, uint8 or float in [0, 1] (:func:`hintauc._images.as_uint8_image`).
        """
        if isinstance(image, np.ndarray):
            arr = as_uint8_image(image, "image")
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            return np.ascontiguousarray(arr[:, :, :3])
        from PIL import Image
        with Image.open(os.fspath(image)) as im:
            if im.mode in ("I", "F") or im.mode.startswith("I;16") or im.mode in ("RGB;16", "BGR;16"):
                raise ValueError(f"{image}: {im.mode} images (16-bit or float) are not supported; the protocol uses "
                                 "8-bit images, convert the file first")
            return np.array(im.convert("RGB"), dtype=np.uint8)      # a writable copy (torch.from_numpy needs one)

    def _to_pil(self, image: ImageLike):
        """The input as an RGB PIL image (files and arrays go through :meth:`_rgb_uint8`). The arrays returned by
        ``hintauc.generate_hints`` / ``HintResult.at_ratio`` are BGR (OpenCV): pass ``arr[:, :, ::-1]``."""
        from PIL import Image
        return Image.fromarray(self._rgb_uint8(image))

    # -- backends -------------------------------------------------------------------------------------
    @property
    def torchvision(self):
        """torchvision module when torch *and* torchvision import, else False (the paper's resize backend)."""
        if not hasattr(self, "_tv"):
            self._tv = False
            if self.torch:
                try:
                    import torchvision  # noqa: F401
                    import torchvision.transforms.functional  # noqa: F401
                    self._tv = torchvision
                except ImportError:
                    self._tv = False
        return self._tv

    def resize_backend(self) -> str:
        """``'torchvision'`` (the paper's evaluation: antialiased bilinear resize of float tensors) or ``'pillow'``
        (the same operation with Pillow's antialiased bilinear filter on float32 channels; agrees with torchvision
        to about 1e-3 per pixel, see docs/evaluate_your_model.md)."""
        return "torchvision" if self.torchvision else "pillow"

    def ssim_backend(self) -> str:
        """``'torchmetrics'`` (the paper's SSIM; pin 1.4.0) or ``'scikit-image'`` (fallback, Gaussian window)."""
        if self.torch:
            try:
                import torchmetrics  # noqa: F401
                return "torchmetrics"
            except ImportError:
                pass
        return "scikit-image"

    def backend(self) -> Dict[str, object]:
        """The evaluation backend actually in use, for protocol records: resize and SSIM implementation + versions."""
        def ver(name):
            try:
                return __import__(name).__version__
            except Exception:
                return None
        return {"resize": self.resize_backend(), "resize_filter": "bilinear, antialiased, float32 in [0, 1]",
                "ssim": self.ssim_backend(), "pillow": ver("PIL"), "torchvision": ver("torchvision"),
                "torchmetrics": ver("torchmetrics"), "scikit-image": ver("skimage")}

    def _read_01(self, image: ImageLike) -> np.ndarray:
        """HxWx3 float32 in [0,1], resized to (resize, resize).

        Every input is first decoded to 8-bit RGB by :meth:`_rgb_uint8` (files through Pillow's ``convert("RGB")``,
        so palette, grayscale, RGBA and CMYK files are read as the colours they show; arrays as RGB). The paper's
        evaluator (eval_single_run.py) converts to float in [0, 1] and applies torchvision's antialiased bilinear
        resize; that is the ``'torchvision'`` backend, used whenever torch and torchvision are installed. Without
        them the ``'pillow'`` backend applies Pillow's antialiased bilinear filter to the same float32 channels
        (not a uint8 resize, which would round and use a different filter). :meth:`backend` reports which one is
        in use.
        """
        rgb = self._rgb_uint8(image)
        tv = self.torchvision
        if tv:
            torch = self.torch
            import torchvision.transforms.functional as TVF
            img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            if self.resize:
                try:
                    img = TVF.resize(img, [self.resize, self.resize], antialias=True)
                except TypeError:
                    img = TVF.resize(img, [self.resize, self.resize])
            return img.permute(1, 2, 0).numpy()
        from PIL import Image
        arr = rgb.astype(np.float32) / 255.0
        if self.resize:
            size = (self.resize, self.resize)
            arr = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(arr[:, :, c])).resize(size, Image.BILINEAR),
                                       dtype=np.float32) for c in range(3)], axis=-1)
        return arr

    # ------------------------------------------------------------------
    def _lazy(self, key: str, builder):
        if key not in self._bundle:
            self._bundle[key] = builder()
        return self._bundle[key]

    def _lpips_model(self, net: str = "alex"):
        def build():
            try:
                import lpips
            except ImportError:
                _require("lpips")
            return lpips.LPIPS(net=net).to(self.device).eval()
        return self._lazy(f"lpips_{net}", build)

    def _dists_model(self):
        def build():
            try:
                from DISTS_pytorch import DISTS
            except ImportError:
                _require("dists-pytorch")
            return DISTS().to(self.device).eval()
        return self._lazy("dists", build)

    def _openclip(self):
        def build():
            try:
                import open_clip
            except ImportError:
                _require("open_clip_torch")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            return model.to(self.device).eval(), preprocess
        return self._lazy("openclip", build)

    def _dino(self):
        def build():
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError:
                _require("transformers")
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
            return model, processor
        return self._lazy("dino", build)

    def _dreamsim(self):
        def build():
            try:
                from dreamsim import dreamsim
            except ImportError:
                _require("dreamsim")
            cache = os.environ.get("HINTAUC_CACHE_DIR", os.path.expanduser("~/.cache/hintauc/dreamsim"))
            os.makedirs(cache, exist_ok=True)
            # pass the device explicitly: dreamsim defaults to CUDA and fails on CPU-only machines
            model, preprocess = dreamsim(pretrained=True, device=str(self.device), cache_dir=cache)
            return model.to(self.device).eval(), preprocess
        return self._lazy("dreamsim", build)

    # ------------------------------------------------------------------
    def __call__(self, pred: ImageLike, gt: ImageLike) -> Dict[str, float]:
        """Compute the configured metrics between one prediction and its GT."""
        out: Dict[str, float] = {}
        wanted = set(self.metrics)

        if wanted & PIXEL_METRICS:
            x = self._read_01(pred)
            y = self._read_01(gt)
            if "mse" in wanted:
                out["mse"] = float(np.mean((x - y) ** 2))
            if "psnr" in wanted:
                mse = float(np.mean((x - y) ** 2))
                out["psnr"] = float(10.0 * np.log10(1.0 / (mse + 1e-10)))
            if "ssim" in wanted:
                out["ssim"] = self._ssim(x, y)
            if "mae" in wanted:
                out["mae"] = float(np.mean(np.abs(x - y)))
            if "ms_ssim" in wanted:
                out["ms_ssim"] = self._ms_ssim(x, y)
            if "deltae" in wanted:
                from skimage.color import deltaE_ciede2000, rgb2lab
                out["deltae"] = float(np.mean(deltaE_ciede2000(rgb2lab(x), rgb2lab(y))))

        heavy = wanted & {"lpips", "lpips_vgg", "openclip", "dino", "dreamsim", "dists"}
        if heavy:
            torch = self.torch
            if not torch:
                _require("torch")
            device = self.device
            pred_pil = self._to_pil(pred)
            gt_pil = self._to_pil(gt)

            with torch.no_grad():
                if wanted & {"lpips", "lpips_vgg", "dists"}:
                    x = self._read_01(pred)
                    y = self._read_01(gt)
                    x01 = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
                    y01 = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0).to(device)
                    if "lpips" in wanted:
                        out["lpips"] = float(self._lpips_model("alex")(x01 * 2 - 1, y01 * 2 - 1).item())
                    if "lpips_vgg" in wanted:
                        out["lpips_vgg"] = float(self._lpips_model("vgg")(x01 * 2 - 1, y01 * 2 - 1).item())
                    if "dists" in wanted:
                        out["dists"] = float(self._dists_model()(x01, y01).item())

                if "openclip" in wanted:
                    model, preprocess = self._openclip()
                    fx = model.encode_image(preprocess(pred_pil).unsqueeze(0).to(device))
                    fy = model.encode_image(preprocess(gt_pil).unsqueeze(0).to(device))
                    fx = fx / fx.norm(dim=-1, keepdim=True)
                    fy = fy / fy.norm(dim=-1, keepdim=True)
                    sim = torch.nn.functional.cosine_similarity(fx, fy, dim=-1).item()
                    out["openclip"] = float((sim + 1.0) / 2.0)

                if "dino" in wanted:
                    model, processor = self._dino()
                    o1 = model(**processor(images=pred_pil, return_tensors="pt").to(device)
                               ).last_hidden_state.mean(dim=1)
                    o2 = model(**processor(images=gt_pil, return_tensors="pt").to(device)
                               ).last_hidden_state.mean(dim=1)
                    sim = torch.nn.functional.cosine_similarity(o1, o2, dim=-1).item()
                    out["dino"] = float((sim + 1.0) / 2.0)

                if "dreamsim" in wanted:
                    model, preprocess = self._dreamsim()
                    xt = preprocess(pred_pil).to(device)
                    yt = preprocess(gt_pil).to(device)
                    out["dreamsim"] = float(model(xt, yt).detach().cpu().item())

        return out

    # ------------------------------------------------------------------
    def _ms_ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        if not torch:
            _require("torch")
        from torchmetrics.functional.image import (
            multiscale_structural_similarity_index_measure as ms_ssim_fn)
        xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        yt = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0)
        return float(ms_ssim_fn(xt, yt, data_range=1.0).item())

    def _ssim(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        if torch:
            try:
                from torchmetrics.functional.image import (
                    structural_similarity_index_measure as ssim_fn)
                xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
                yt = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0)
                return float(ssim_fn(xt, yt, data_range=1.0).item())
            except ImportError:
                pass
        import warnings
        from skimage.metrics import structural_similarity
        warnings.warn("torchmetrics is not installed: SSIM falls back to scikit-image with an 11x11 Gaussian window "
                      "(sigma 1.5); the published values used torchmetrics 1.4.0 (pip install torchmetrics==1.4.0).",
                      stacklevel=2)
        return float(structural_similarity(x, y, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                           sigma=1.5, use_sample_covariance=False))


# --------------------------------------------------------------------------
# convenience functions
# --------------------------------------------------------------------------
def evaluate_pair(
    pred: ImageLike,
    gt: ImageLike,
    metrics: Sequence[str] = ("mse", "psnr", "ssim"),
    device: Optional[str] = None,
    resize: int = 256,
) -> Dict[str, float]:
    """One-shot evaluation of a single prediction/GT pair."""
    return Evaluator(metrics=metrics, device=device, resize=resize)(pred, gt)


def _list_images(d: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(exts))
    return [os.path.join(d, n) for n in names]


def _examples(names, n=3):
    names = sorted(names)
    return ", ".join(names[:n]) + (" ..." if len(names) > n else "")


def list_pairs(pred_dir: str, gt_dir: str, pairing: str = "sorted", allow_missing: bool = False,
               limit: int = 0) -> List[Tuple[str, str, str]]:
    """The (prediction, ground truth, name) triples that an evaluation of two directories would score.

    ``pairing='name'`` pairs files with identical basenames; ``'sorted'`` pairs the two sorted listings position by
    position (the pairing of the paper pipeline, whose loader iterates in sorted order). Both are strict: a ground
    truth without a prediction, a prediction without a ground truth, or (sorted) two listings of different length
    raise ``ValueError`` naming the files, because a missing prediction silently changes the evaluated set. With
    ``allow_missing=True`` the common files are used and a warning lists what was dropped.
    """
    preds = _list_images(pred_dir)
    gts = _list_images(gt_dir)
    if not preds:
        raise ValueError(f"no images in {pred_dir}")
    if not gts:
        raise ValueError(f"no images in {gt_dir}")
    if pairing == "name":
        pred_by_name = {os.path.basename(p): p for p in preds}
        gt_by_name = {os.path.basename(g): g for g in gts}
        missing = sorted(set(gt_by_name) - set(pred_by_name))
        extra = sorted(set(pred_by_name) - set(gt_by_name))
        if missing or extra:
            msg = (f"{pred_dir}: {len(missing)} ground-truth image(s) have no prediction ({_examples(missing)}) and "
                   f"{len(extra)} prediction(s) have no ground truth ({_examples(extra)})")
            if not allow_missing:
                raise ValueError(msg + "; pass allow_missing=True (CLI: --allow-missing) to evaluate the common files only")
            import warnings
            warnings.warn(msg + "; evaluating the common files only", stacklevel=2)
        names = sorted(set(pred_by_name) & set(gt_by_name))
        pairs = [(pred_by_name[n], gt_by_name[n], n) for n in names]
    elif pairing == "sorted":
        if len(preds) != len(gts):
            msg = f"pred/gt counts differ ({len(preds)} in {pred_dir} vs {len(gts)} in {gt_dir})"
            if not allow_missing:
                raise ValueError(msg + "; use pairing='name' for files that share names, or align the directories")
            import warnings
            warnings.warn(msg + "; pairing the first min(n) files by sorted order", stacklevel=2)
        pairs = [(p, g, os.path.basename(p)) for p, g in zip(preds, gts)]
        if [os.path.basename(p) for p, _, _ in pairs] != [os.path.basename(g) for _, g, _ in pairs]:
            import warnings
            warnings.warn("prediction and ground-truth file names differ; pairing by sorted order (pass pairing='name' when the files share names)", stacklevel=2)
    else:
        raise ValueError("pairing must be 'sorted' or 'name'")
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("no image pairs found")
    return pairs


def evaluate_pairs(pairs: Sequence[Tuple[str, str, str]], evaluator: Evaluator) -> Dict[str, float]:
    """Mean metrics over explicit (prediction, ground truth, name) triples."""
    if not pairs:
        raise ValueError("no image pairs")
    sums: Dict[str, float] = {}
    for pred, gt, _ in pairs:
        for k, v in evaluator(pred, gt).items():
            sums[k] = sums.get(k, 0.0) + v
    return {k: v / len(pairs) for k, v in sums.items()}


def evaluate_dirs(
    pred_dir: str,
    gt_dir: str,
    evaluator: Optional[Evaluator] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    pairing: str = "sorted",
    limit: int = 0,
    allow_missing: bool = False,
) -> Dict[str, float]:
    """Mean metrics over a directory of predictions and a directory of GTs.

    pairing='sorted': pair by lexically sorted basename order (the pairing
    used by the paper pipeline, matching the dataloader iteration order).
    pairing='name':   pair files with identical basenames.
    Missing or unmatched files are an error unless ``allow_missing=True`` (see :func:`list_pairs`).
    """
    ev = evaluator or Evaluator(metrics=metrics)
    return evaluate_pairs(list_pairs(pred_dir, gt_dir, pairing=pairing, allow_missing=allow_missing, limit=limit), ev)


# --------------------------------------------------------------------------
# set-level metrics (added after the paper)
# --------------------------------------------------------------------------
def evaluate_set(
    pred_dir: str,
    gt_dir: str,
    metrics: Sequence[str] = SET_METRICS,
    device: Optional[str] = None,
    resize: int = 256,
    kid_subset_size: Optional[int] = None,
) -> Dict[str, float]:
    """FID / KID between the images of ``pred_dir`` and ``gt_dir`` (Inception-v3 pool features, 299x299).

    Requires ``torchmetrics`` and ``torch-fidelity`` (``pip install hintauc[perceptual]``). FID needs a few hundred
    images per side to be meaningful; KID (unbiased) is preferable for small sets and returns the mean of the
    polynomial-kernel MMD over subsets of ``kid_subset_size`` images (default: min(50, n)).
    """
    metrics = tuple(m.lower() for m in metrics)
    unknown = set(metrics) - set(SET_METRICS)
    if unknown:
        raise ValueError(f"unknown set metrics: {sorted(unknown)} (available: {SET_METRICS})")
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        _require("torchmetrics[image] / torch-fidelity")
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, gts = _list_images(pred_dir), _list_images(gt_dir)
    if not preds or not gts:
        raise ValueError("empty directory")
    n = min(len(preds), len(gts))
    ev = Evaluator(metrics=("mse",), device=device, resize=resize)   # reuse its image reader

    def batches(paths, bs=32):
        for i in range(0, len(paths), bs):
            arr = np.stack([ev._read_01(p) for p in paths[i:i + bs]])          # B,H,W,3 in [0,1]
            yield torch.from_numpy(arr).permute(0, 3, 1, 2).to(dev)

    out: Dict[str, float] = {}
    if "fid" in metrics:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
        for b in batches(gts):
            fid.update(b, real=True)
        for b in batches(preds):
            fid.update(b, real=False)
        out["fid"] = float(fid.compute().item())
    if "kid" in metrics:
        subset = kid_subset_size or max(2, min(50, n))
        kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size=subset).to(dev)
        for b in batches(gts):
            kid.update(b, real=True)
        for b in batches(preds):
            kid.update(b, real=False)
        mean, std = kid.compute()
        out["kid"] = float(mean.item())
        out["kid_std"] = float(std.item())
    out["n_pred"], out["n_gt"] = len(preds), len(gts)
    return out
