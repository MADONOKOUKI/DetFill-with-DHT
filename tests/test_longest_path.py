"""Unit tests for hintauc.longest_path (pure NumPy; run: python -m pytest tests/ or python tests/test_longest_path.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hintauc.longest_path import geodesic_longest_path


def test_line():
    s = np.zeros((7, 9), bool); s[3, 1:8] = True
    r = geodesic_longest_path(s)
    assert r.length == 6 and r.mask.sum() == 7 and (r.mask == s).all()


def test_l_shape_keeps_corner():
    s = np.zeros((9, 9), bool); s[1:8, 1] = True; s[7, 1:8] = True
    r = geodesic_longest_path(s)
    assert r.mask.sum() == 13 and abs(r.length - 12) < 1e-9


def test_t_shape_picks_two_long_arms():
    s = np.zeros((9, 11), bool); s[1, 1:10] = True; s[1:5, 5] = True
    r = geodesic_longest_path(s)
    assert r.mask.sum() == 9 and abs(r.length - 8) < 1e-9
    assert r.start == (1, 1) and r.end == (1, 9)


def test_diagonal_weights():
    s = np.zeros((6, 6), bool); np.fill_diagonal(s, True)
    r = geodesic_longest_path(s)
    assert abs(r.length - 5 * np.sqrt(2)) < 1e-9 and r.mask.sum() == 6


def test_ring_without_endpoints():
    s = np.zeros((9, 9), bool); s[1, 1:8] = True; s[7, 1:8] = True; s[1:8, 1] = True; s[1:8, 7] = True
    r = geodesic_longest_path(s)
    assert 11 <= r.mask.sum() <= 13 and not (r.mask & ~s).any()


def test_single_and_empty():
    s = np.zeros((5, 5), bool); s[2, 2] = True
    r = geodesic_longest_path(s)
    assert r.mask.sum() == 1 and r.length == 0 and r.start == (2, 2) == r.end
    r = geodesic_longest_path(np.zeros((5, 5), bool))
    assert r.mask.sum() == 0 and r.start is None


def test_deterministic_tie_break():
    s = np.zeros((11, 11), bool); s[5, 1:6] = True; s[0:5, 5] = True; s[6:11, 5] = True
    r1 = geodesic_longest_path(s); r2 = geodesic_longest_path(s.copy())
    assert (r1.mask == r2.mask).all() and (r1.start, r1.end) == (r2.start, r2.end) == ((0, 5), (10, 5))


def test_path_is_subset_of_skeleton_random():
    rng = np.random.default_rng(0)
    from skimage.morphology import skeletonize
    for _ in range(20):
        m = np.zeros((32, 32), bool)
        for _ in range(3):
            r0, c0 = rng.integers(0, 24, 2); m[r0:r0 + rng.integers(3, 9), c0:c0 + rng.integers(3, 9)] = True
        sk = skeletonize(m)
        r = geodesic_longest_path(sk)
        assert not (r.mask & ~sk).any()
        if sk.any():
            assert r.mask.any()


def test_medoid_dot_inside_region():
    """generate_hints(dot_method='medoid') puts every dot on its own scribble (inside the region)."""
    import numpy as np, cv2
    from hintauc import generate_hints, region_ids
    rng = np.random.default_rng(1)
    img = np.zeros((128, 128, 3), np.uint8)
    # crescent-like region on a two-colour background
    yy, xx = np.mgrid[0:128, 0:128]
    ring = ((yy - 64) ** 2 + (xx - 64) ** 2 < 50 ** 2) & ((yy - 64) ** 2 + (xx - 84) ** 2 > 40 ** 2)
    img[ring] = (200, 30, 30); img[~ring] = (30, 30, 200)
    region = np.zeros_like(img); region[ring] = (1, 2, 3); region[~ring] = (4, 5, 6)
    res = generate_hints(img, size=64, region_map=region, path_method="geodesic", dot_method="medoid")
    ids = region_ids(res.region)
    dots = np.argwhere(res.dot_mask > 0)
    assert len(dots) >= 1
    for r, c in dots:
        assert res.scribble_mask[r, c] > 0          # dot lies on a scribble pixel
    assert res.dot_method == "medoid"


def test_stable_tie_break_is_label_order():
    import numpy as np
    from hintauc import generate_hints
    img = np.full((64, 64, 3), 120, np.uint8)
    region = np.zeros((64, 64, 3), np.uint8)
    for (r, c, col) in [(0, 0, (9, 0, 0)), (0, 16, (3, 0, 0)), (16, 0, (7, 0, 0)), (16, 16, (1, 0, 0))]:
        region[r:r + 16, c:c + 16] = col      # four equal-area blocks, scrambled labels
    region[32:, :] = (200, 0, 0)              # one large region
    res = generate_hints(img, size=64, region_map=region, path_method="geodesic")
    ids_stable = res._ids_sorted_by_area("stable")
    assert list(ids_stable[1:]) == sorted(ids_stable[1:])   # ties by ascending label after the largest
    c1, m1 = res.at_ratio(0.4, tie_break="stable"); c2, m2 = res.at_ratio(0.4, tie_break="stable")
    assert (m1 == m2).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
