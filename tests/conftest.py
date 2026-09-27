"""Shared helpers for the test-suite: a small synthetic illustration with flat colour regions."""
import pytest

from hintauc.demo import synthetic_illustration  # noqa: F401  (re-exported for the tests)


@pytest.fixture
def illustration(tmp_path):
    import cv2
    p = tmp_path / "illustration.png"
    cv2.imwrite(str(p), synthetic_illustration())
    return p
