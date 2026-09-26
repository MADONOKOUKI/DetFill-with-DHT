"""The DetFill evaluation loader on a synthetic flat data layout (no checkpoint, no download)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import hintauc  # noqa: E402
from conftest import synthetic_illustration  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = ("1016", "2016", "3016")


def _flat_layout(root):
    import cv2
    for d in ("segmentations/originals", "sketch/XDoG", "sketch/pysimp", "sketch/sketchkeras",
              "hint_from_regions_64_rev", "hint_from_regions_256"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for k, idx in enumerate(IDS):
        img = synthetic_illustration(seed=k)
        cv2.imwrite(os.path.join(root, "segmentations/originals", f"{idx}.image.png"), img)
        line_art = 255 - cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        for t in ("XDoG", "pysimp", "sketchkeras"):
            cv2.imwrite(os.path.join(root, "sketch", t, f"{idx}.png"), line_art)
        h = hintauc.generate_hints(img, size=64, path_method="geodesic")
        paths = h.save(os.path.join(root, "hint_from_regions_64_rev", f"{idx}.image"))
        os.replace(paths["region"], os.path.join(root, "hint_from_regions_256", f"{idx}.image_region64.png"))
    return root


_CLASS = None


def _dataset_class():
    """Import detfill/datasets/custom.py once (the module-level registry rejects a second registration)."""
    global _CLASS
    if _CLASS is None:
        detfill = os.path.join(ROOT, "detfill")
        if detfill not in sys.path:
            sys.path.insert(0, detfill)
        for m in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
            del sys.modules[m]                    # never pick up an unrelated 'datasets' package
        import datasets.custom as custom          # noqa: WPS433
        _CLASS = custom.HintColorizationDataset
    return _CLASS


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    return _flat_layout(str(tmp_path_factory.mktemp("flat")))


def _config(root, **kw):
    from types import SimpleNamespace
    cfg = dict(image_size=128, domain="illust", hint_type="scribble", scratch_root=root, channels=3)
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def test_flat_layout_item_shapes(layout):
    ds = _dataset_class()(_config(layout), stage="test", sample_ratio=0.5, sketch_type=2)
    assert len(ds) == len(IDS)
    (x, name), (cond, name2), (weight, _) = ds[0]
    assert name == name2 and name.split(".")[0] in IDS     # image name '<id>.image' -> output '<id>.image.png'
    assert tuple(x.shape) == (5, 128, 128) and tuple(cond.shape) == (5, 128, 128)
    assert x.dtype == torch.float32 and float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert 0.0 <= float(weight) <= 1.0


@pytest.mark.parametrize("hint_type", ["scribble", "dot"])
def test_hint_ratio_controls_the_mask_channel(layout, hint_type):
    cls = _dataset_class()
    none = cls(_config(layout, hint_type=hint_type), stage="test", sample_ratio=0.0, sketch_type=2)
    full = cls(_config(layout, hint_type=hint_type), stage="test", sample_ratio=1.0, sketch_type=2)
    (_, _), (cond0, _), (w0, _) = none[1]
    (_, _), (cond1, _), (w1, _) = full[1]
    assert float(cond0[1].max()) == 0.0 and float(w0) == 0.0            # no hint pixels at ratio 0
    assert float(cond1[1].max()) > 0.0 and float(w1) > 0.0              # hints present at ratio 1


def test_region_orders_and_validation(layout):
    cls = _dataset_class()
    area = cls(_config(layout, hint_order="area"), stage="test", sample_ratio=0.5, sketch_type=2)
    label = cls(_config(layout, hint_order="label"), stage="test", sample_ratio=0.5, sketch_type=2)
    (_, _), (ca, _), _ = area[2]
    (_, _), (cl, _), _ = label[2]
    assert tuple(ca.shape) == tuple(cl.shape) == (5, 128, 128)
    with pytest.raises(ValueError):
        cls(_config(layout, hint_order="random"), stage="test", sample_ratio=0.5, sketch_type=2)


def test_placeholder_roots_are_rejected(layout):
    cls = _dataset_class()
    with pytest.raises(ValueError):                    # the released configs' placeholders select no layout
        cls(_config(layout, scratch_root="/path/to/dataset", dataset_path="/path/to/dataset"), stage="test", sample_ratio=0.5, sketch_type=2)


def test_sketch_types_are_all_readable(layout):
    ds = _dataset_class()(_config(layout), stage="test", sample_ratio=0.1, sketch_type=None)
    for i in range(len(ds)):
        (x, _), (cond, _), _ = ds[i]
        assert tuple(cond.shape) == (5, 128, 128)
    for t in (0, 1, 2):
        d = _dataset_class()(_config(layout), stage="test", sample_ratio=0.1, sketch_type=t)
        d[0]
