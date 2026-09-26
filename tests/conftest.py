"""Shared helpers for the test-suite: a small synthetic illustration with flat colour regions."""
import numpy as np
import pytest


def synthetic_illustration(seed: int = 0, size: int = 128) -> np.ndarray:
    """HxWx3 uint8 image: light background + three flat rectangles (clear Felzenszwalb regions)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 235, np.uint8)
    boxes = [((10, 10), (60, 70)), ((70, 20), (120, 60)), ((30, 80), (100, 120))]
    for (y0, x0), (y1, x1) in boxes:
        img[y0:y1, x0:x1] = rng.integers(20, 220, size=3, dtype=np.uint8)
    return img


@pytest.fixture
def illustration(tmp_path):
    import cv2
    p = tmp_path / "illustration.png"
    cv2.imwrite(str(p), synthetic_illustration())
    return p
