"""Deterministic hint generation: shape, determinism, the size-ordered selection and the DetFill file layout."""
import numpy as np
import pytest

import hintauc
from conftest import synthetic_illustration


@pytest.mark.parametrize("path_method", ["geodesic", "filfinder"])
def test_generate_hints_is_deterministic(illustration, path_method):
    if path_method == "filfinder":
        pytest.importorskip("fil_finder")
    h1 = hintauc.generate_hints(illustration, size=64, path_method=path_method)
    h2 = hintauc.generate_hints(illustration, size=64, path_method=path_method)
    for name in ("region", "scribble_mask", "scribble_color", "dot_mask", "dot_color", "flatten"):
        assert np.array_equal(getattr(h1, name), getattr(h2, name)), name
    assert h1.region.shape == (64, 64, 3) and h1.dot_mask.shape == (64, 64) and h1.scribble_color.shape == (64, 64, 3)
    assert set(np.unique(h1.dot_mask)) <= {0, 255} and set(np.unique(h1.scribble_mask)) <= {0, 255}


def test_one_dot_per_region_and_colours_only_on_hints(illustration):
    h = hintauc.generate_hints(illustration, size=64, path_method="geodesic")
    n = h.n_regions()
    assert n >= 4                                   # background + three rectangles
    dots = int((h.dot_mask > 0).sum())
    assert n - h.failed_regions <= dots <= n        # the paper's rule: one dot per region
    assert (h.dot_color[h.dot_mask == 0] == 0).all()
    assert (h.scribble_color[h.scribble_mask == 0] == 0).all()
    assert (h.scribble_mask > 0).sum() >= dots      # a scribble contains its dot


def test_at_ratio_is_nested_and_bounded(illustration):
    h = hintauc.generate_hints(illustration, size=64, path_method="geodesic")
    previous = np.zeros((64, 64), bool)
    for alpha in hintauc.DEFAULT_ALPHAS:
        color, mask = h.at_ratio(alpha, hint_type="scribble")
        assert color.shape == (64, 64, 3) and mask.shape == (64, 64)
        assert (color[mask == 0] == 0).all()
        current = mask > 0
        assert not (previous & ~current).any()      # regions selected at a lower ratio stay selected
        previous = current
    _, m0 = h.at_ratio(0.0)
    assert m0.sum() == 0
    _, m1 = h.at_ratio(1.0)
    assert np.array_equal(m1, h.scribble_mask)
    _, mdot = h.at_ratio(1.0, hint_type="dot", resize_to=256)
    assert mdot.shape == (256, 256) and (mdot > 0).sum() == 16 * (h.dot_mask > 0).sum()
    with pytest.raises(ValueError):
        h.at_ratio(1.5)
    with pytest.raises(ValueError):
        h.at_ratio(0.5, hint_type="stroke")


def test_stable_tie_break_gives_same_pixel_counts(illustration):
    h = hintauc.generate_hints(illustration, size=64, path_method="geodesic")
    for alpha in (0.25, 0.5):
        _, a = h.at_ratio(alpha)
        _, b = h.at_ratio(alpha, tie_break="stable")
        assert (a > 0).sum() > 0 and a.shape == b.shape


def test_save_writes_the_detfill_layout(illustration, tmp_path):
    import cv2
    h = hintauc.generate_hints(illustration, size=64, path_method="geodesic")
    paths = h.save(tmp_path / "123.image")
    suffixes = sorted(str(p).split("123.image_")[1] for p in paths.values())
    assert suffixes == sorted(["region64.png", "scribble_mask64.png", "scribble_col64.png",
                               "flatten_img64.png", "dot_mask64.png", "dot_col64.png"])
    for p in paths.values():
        assert cv2.imread(p) is not None, p
    assert np.array_equal(cv2.imread(paths["region"]), h.region)          # PNG round trip is lossless
    assert np.array_equal(cv2.imread(paths["dot_mask"], cv2.IMREAD_GRAYSCALE), h.dot_mask)


def test_save_creates_missing_directories(illustration, tmp_path):
    import cv2
    h = hintauc.generate_hints(illustration, size=16, path_method="geodesic")
    paths = h.save(tmp_path / "not" / "yet" / "there" / "img")
    assert len(paths) == 6 and all(cv2.imread(p) is not None for p in paths.values())


def test_float_inputs_follow_the_documented_policy(illustration):
    import cv2
    img = cv2.imread(str(illustration))
    ref = hintauc.generate_hints(img, size=32, path_method="geodesic")
    same = hintauc.generate_hints(img.astype(np.float32) / 255.0, size=32, path_method="geodesic")   # [0, 1] floats
    assert np.array_equal(ref.region, same.region) and np.array_equal(ref.scribble_mask, same.scribble_mask)
    with pytest.raises(ValueError):
        hintauc.generate_hints(img.astype(np.float32), size=32, path_method="geodesic")               # floats 0..255
    with pytest.raises(ValueError):
        hintauc.generate_hints(np.zeros((8, 8, 5), np.uint8), size=8, path_method="geodesic")


def test_external_region_map_colours_that_collide_are_reencoded():
    region = np.zeros((16, 16, 3), np.uint8)
    region[:, :8] = (0, 255, 0)                     # both decode to 65025 under the base-255 rule
    region[:, 8:] = (1, 0, 0)
    with pytest.raises(ValueError):
        hintauc.region_ids(region)
    assert len(np.unique(hintauc.region_ids(hintauc.reencode_region_map(region)))) == 2
    img = np.zeros((16, 16, 3), np.uint8); img[:, :8] = (200, 30, 30); img[:, 8:] = (30, 30, 200)
    h = hintauc.generate_hints(img, size=16, region_map=region, path_method="geodesic")
    assert h.n_regions() == 2 and int((h.dot_mask > 0).sum()) == 2
    _, m = h.at_ratio(0.5)
    assert (m > 0).sum() >= 1                       # half of two regions = one region's hints


def test_array_input_matches_file_input(illustration):
    import cv2
    from_file = hintauc.generate_hints(illustration, size=64, path_method="geodesic")
    from_array = hintauc.generate_hints(cv2.imread(str(illustration)), size=64, path_method="geodesic")
    assert np.array_equal(from_file.region, from_array.region)
    assert np.array_equal(from_file.scribble_mask, from_array.scribble_mask)


def test_other_hint_sizes(illustration):
    h = hintauc.generate_hints(illustration, size=32, path_method="geodesic")
    assert h.region.shape == (32, 32, 3) and h.size == 32
    assert synthetic_illustration().shape == (128, 128, 3)
