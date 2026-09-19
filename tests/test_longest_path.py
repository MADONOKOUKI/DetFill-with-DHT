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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
