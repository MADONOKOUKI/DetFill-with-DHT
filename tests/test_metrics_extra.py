"""CPU-only tests for the metrics added after the paper (mae, ms_ssim, deltae) and the metric registry."""
import numpy as np
import pytest

from hintauc.metrics import ALL_METRICS, DEFAULT_METRICS, EXTRA_METRICS, LOWER_IS_BETTER, SET_METRICS, Evaluator


def _img(seed, shift=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return np.roll(base, shift, axis=1) if shift else base


def test_registry_is_consistent():
    assert set(DEFAULT_METRICS) == {"mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"}
    assert set(EXTRA_METRICS) == {"mae", "ms_ssim", "deltae", "lpips_vgg", "dists"}
    assert set(ALL_METRICS) == set(DEFAULT_METRICS) | set(EXTRA_METRICS)
    for m in ALL_METRICS + SET_METRICS:
        assert m in LOWER_IS_BETTER
    with pytest.raises(ValueError):
        Evaluator(metrics=("fid",))          # set-level metric: not a per-image one


def test_identical_images_are_perfect():
    ev = Evaluator(metrics=("mae", "deltae", "mse", "psnr"))
    a = _img(0)
    s = ev(a, a.copy())
    assert s["mae"] == 0.0 and s["mse"] == 0.0 and s["deltae"] == pytest.approx(0.0, abs=1e-6)
    assert s["psnr"] > 90


def test_shifted_image_is_worse():
    ev = Evaluator(metrics=("mae", "deltae"))
    a, b = _img(1), _img(1, shift=8)
    s = ev(a, b)
    assert s["mae"] > 0.05 and s["deltae"] > 1.0


def test_ms_ssim_bounds():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    ev = Evaluator(metrics=("ms_ssim",), device="cpu")
    a = _img(2)
    assert ev(a, a.copy())["ms_ssim"] == pytest.approx(1.0, abs=1e-4)
    v = ev(a, _img(3))["ms_ssim"]
    assert -1.0 <= v < 0.9
