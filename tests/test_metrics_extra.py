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


def test_float_predictions_are_not_silently_truncated():
    ev = Evaluator(metrics=("mse", "psnr"))
    gt = np.zeros((16, 16, 3), np.uint8)
    half = np.full((16, 16, 3), 0.5, np.float32)             # a mid-gray prediction against black
    s = ev(half, gt)
    assert s["mse"] == pytest.approx((128 / 255) ** 2, abs=1e-6) and s["psnr"] < 10
    with pytest.raises(ValueError):
        ev(np.full((16, 16, 3), 128.0, np.float32), gt)        # floats outside [0, 1]
    with pytest.raises(ValueError):
        ev(np.zeros((16, 16, 3), np.int32) - 1, gt)


def test_pillow_and_torchvision_resize_backends_agree():
    """The Pillow fallback resizes float32 channels with the antialiased bilinear filter, like torchvision."""
    pytest.importorskip("torch"); pytest.importorskip("torchvision")
    ev_tv = Evaluator(metrics=("mse",))
    ev_pil = Evaluator(metrics=("mse",)); ev_pil._tv = False
    assert ev_tv.resize_backend() == "torchvision" and ev_pil.resize_backend() == "pillow"
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    img = np.ascontiguousarray(np.repeat(np.repeat(img[::4, ::4], 4, axis=0), 4, axis=1))   # smooth-ish content
    a, b = ev_tv._read_01(img), ev_pil._read_01(img)
    assert a.shape == b.shape == (256, 256, 3) and float(np.abs(a - b).max()) < 1e-4
    ev_pil.metrics = ("mse", "psnr")
    ev_tv.metrics = ("mse", "psnr")
    shifted = np.roll(img, 5, axis=1)
    sa, sb = ev_tv(img, shifted), ev_pil(img, shifted)
    assert sa["psnr"] == pytest.approx(sb["psnr"], abs=1e-3)


def test_ms_ssim_bounds():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    ev = Evaluator(metrics=("ms_ssim",), device="cpu")
    a = _img(2)
    assert ev(a, a.copy())["ms_ssim"] == pytest.approx(1.0, abs=1e-4)
    v = ev(a, _img(3))["ms_ssim"]
    assert -1.0 <= v < 0.9
