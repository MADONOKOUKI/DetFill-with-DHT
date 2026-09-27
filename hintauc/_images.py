"""Input validation shared by the hint generator and the evaluator.

Images are ``uint8`` arrays with values 0..255. Floating-point arrays are accepted only when every value lies in
[0, 1]; they are scaled by 255 and rounded. Anything else is rejected instead of being cast silently (a silent
``astype(uint8)`` turns 0.5 into 0 and makes a wrong prediction look perfect).
"""
from __future__ import annotations

import numpy as np


def as_uint8_image(arr, what: str = "image") -> np.ndarray:
    """Return ``arr`` as a ``uint8`` array (HxW or HxWxC), or raise ``ValueError`` with a precise message."""
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(f"{what}: expected an HxW or HxWxC array, got shape {a.shape}")
    if a.ndim == 3 and a.shape[2] not in (1, 3, 4):
        raise ValueError(f"{what}: expected 1, 3 or 4 channels, got shape {a.shape}")
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255
    if np.issubdtype(a.dtype, np.floating):
        if a.size and not np.isfinite(a).all():
            raise ValueError(f"{what}: floating-point image contains NaN or inf")
        lo = float(a.min()) if a.size else 0.0
        hi = float(a.max()) if a.size else 0.0
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"{what}: floating-point images must be in [0, 1] (got min {lo:.4g}, max {hi:.4g}); "
                             "pass uint8 (0..255) or scale to [0, 1] explicitly")
        return np.rint(a * 255.0).astype(np.uint8)
    if np.issubdtype(a.dtype, np.integer):
        if a.size and (int(a.min()) < 0 or int(a.max()) > 255):
            raise ValueError(f"{what}: integer image values must be in 0..255 (got min {int(a.min())}, max {int(a.max())})")
        return a.astype(np.uint8)
    raise ValueError(f"{what}: unsupported dtype {a.dtype}")
