"""Evaluate-your-own-model entry points: callable path, directory path, and input validation."""
import json
import shutil

import numpy as np
import pytest

import hintauc
from conftest import synthetic_illustration


def _samples(tmp_path):
    import cv2
    gt = synthetic_illustration()
    gt_path = tmp_path / "img.png"
    cv2.imwrite(str(gt_path), gt)
    line_art = 255 - cv2.Canny(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY), 50, 150)
    sk_path = tmp_path / "img_sketch.png"
    cv2.imwrite(str(sk_path), line_art)
    return [(str(sk_path), str(gt_path))]


def test_reference_colorizer_gives_a_rising_curve(tmp_path):
    res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, _samples(tmp_path), metrics=("mse", "psnr"),
                                     path_method="geodesic", save_dir=tmp_path / "pred", oracle=True)
    assert res["n_images"] == 1 and set(res["hint_auc"]) == {"mse", "psnr"}
    assert res["alphas"] == list(hintauc.DEFAULT_ALPHAS) and res["protocol"]["paper_grid"] is True
    pa = res["per_alpha"]
    assert pa[1.0]["psnr"] > pa[0.0]["psnr"]                 # more hints -> closer to the ground truth
    assert np.isfinite(res["hint_auc"]["psnr"]) and pa[0.0]["psnr"] <= res["hint_auc"]["psnr"] <= pa[1.0]["psnr"]
    assert (tmp_path / "pred" / "1.00" / "img.png").exists() and (tmp_path / "pred" / "0.00" / "img.png").exists()
    assert res["protocol"]["versions"]["hintauc"] == hintauc.__version__
    assert res["protocol"]["backend"]["resize"] in ("torchvision", "pillow") and res["names"] == ["img.png"]
    manifest = json.load(open(tmp_path / "pred" / "manifest.json"))
    assert manifest["dirs"]["0.01"] == 0.01 and manifest["names"] == ["img.png"] and manifest["n_images"] == 1


def test_callback_output_is_validated(tmp_path):
    with pytest.raises(ValueError):
        hintauc.evaluate_colorizer(lambda s: np.zeros((3, 3), np.uint8), _samples(tmp_path),
                                   alphas=(0.0, 1.0), metrics=("mse",), path_method="geodesic")
    with pytest.raises(ValueError):
        hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, _samples(tmp_path),
                                   alphas=(0.1, 1.0), metrics=("mse",), path_method="geodesic", oracle=True)


def test_callback_sees_only_observations_unless_oracle(tmp_path):
    seen = []

    def spy(sample):
        seen.append(sorted(sample))
        h, w = sample["line_art"].shape
        return np.zeros((h, w, 3), np.uint8)

    hintauc.evaluate_colorizer(spy, _samples(tmp_path), alphas=(0.0, 1.0), metrics=("mse",), path_method="geodesic")
    assert seen[0] == sorted(hintauc.evaluate.OBSERVED_KEYS)
    assert "ground_truth" not in seen[0] and "hints" not in seen[0]
    with pytest.raises(KeyError):                       # the oracle baseline refuses to run without oracle=True
        hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, _samples(tmp_path), alphas=(0.0, 1.0),
                                   metrics=("mse",), path_method="geodesic")


def test_alpha_grid_is_validated(tmp_path):
    white = lambda s: np.full(s["line_art"].shape + (3,), 255, np.uint8)
    for bad in [(0, 0.5, 0.5, 1), (0, 0.5, 0.1, 1), (), (0.0,), (0, float("nan"), 1), (0, 1.5)]:
        with pytest.raises(ValueError):
            hintauc.evaluate_colorizer(white, _samples(tmp_path), alphas=bad, metrics=("mse",), path_method="geodesic")
    assert hintauc.check_alphas((0, .5, 1)) == [0.0, 0.5, 1.0]
    assert hintauc.check_alphas((0.2, 0.7), full_range=False) == [0.2, 0.7]
    with pytest.raises(ValueError):
        hintauc.trapz([0.0, 0.5, 0.5, 1.0], [1, 1, 1, 1])


def test_fine_alphas_get_distinct_directories(tmp_path):
    white = lambda s: np.full(s["line_art"].shape + (3,), 255, np.uint8)
    res = hintauc.evaluate_colorizer(white, _samples(tmp_path), alphas=(0, .001, .004, 1), metrics=("mse",),
                                     path_method="geodesic", save_dir=tmp_path / "fine")
    assert sorted(d.name for d in (tmp_path / "fine").iterdir() if d.is_dir()) == ["0.00", "0.001", "0.004", "1.00"]
    assert [hintauc.alpha_dir_name(a) for a in hintauc.DEFAULT_ALPHAS] == \
        ["0.00", "0.01", "0.03", "0.05", "0.10", "0.25", "0.50", "1.00"]
    values = [res["per_alpha"][a]["mse"] for a in res["alphas"]]
    assert max(values) - min(values) < 1e-12                       # the colorizer ignores the hints: a flat curve
    assert res["hint_auc"]["mse"] == pytest.approx(values[0])      # whose integral over [0, 1] is the constant


def test_duplicate_sample_names_are_rejected(tmp_path):
    s = _samples(tmp_path)
    with pytest.raises(ValueError):
        hintauc.evaluate_colorizer(lambda x: np.zeros(x["line_art"].shape + (3,), np.uint8), s + s,
                                   alphas=(0.0, 1.0), metrics=("mse",), path_method="geodesic")


def test_saved_predictions_reproduce_the_curve_through_the_directory_path(tmp_path, capsys):
    from hintauc.cli import main
    samples = _samples(tmp_path)
    res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, samples, alphas=(0.0, 0.5, 1.0),
                                     metrics=("mse", "psnr"), path_method="geodesic", save_dir=tmp_path / "pred",
                                     oracle=True)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    shutil.copy(samples[0][1], gt_dir / "img.png")
    main(["curve", str(tmp_path / "pred"), str(gt_dir), "--metrics", "mse", "psnr", "--json", str(tmp_path / "curve.json")])
    out = json.load(open(tmp_path / "curve.json"))
    assert out["alphas"] == [0.0, 0.5, 1.0]
    assert abs(out["hint_auc"]["psnr"] - res["hint_auc"]["psnr"]) < 1e-6
    assert out["protocol"]["paper_grid"] is False and out["n_images"] == 1 and out["images"] == ["img.png"]


def test_directory_evaluation_rejects_misaligned_inputs(tmp_path):
    import cv2
    from hintauc.metrics import evaluate_dirs
    pred, gt = tmp_path / "p", tmp_path / "g"
    pred.mkdir(); gt.mkdir()
    for i in range(3):
        cv2.imwrite(str(gt / f"{i}.png"), synthetic_illustration(i))
    for i in range(2):
        cv2.imwrite(str(pred / f"{i}.png"), synthetic_illustration(i))
    with pytest.raises(ValueError):
        evaluate_dirs(str(pred), str(gt), metrics=("mse",), pairing="sorted")       # 2 vs 3 files
    with pytest.raises(ValueError):
        evaluate_dirs(str(pred), str(gt), metrics=("mse",), pairing="name")         # one ground truth has no prediction
    with pytest.warns(UserWarning):
        scores = evaluate_dirs(str(pred), str(gt), metrics=("mse",), pairing="name", allow_missing=True)
    assert scores["mse"] == 0.0                                                         # the two common names
    with pytest.raises(ValueError):
        evaluate_dirs(str(pred), str(tmp_path), metrics=("mse",), pairing="name")    # nothing in common


def test_curve_refuses_a_missing_prediction_at_one_ratio(tmp_path):
    """A prediction missing at one ratio must not silently shrink that ratio's evaluated set (review finding)."""
    import cv2
    from hintauc import evaluate_hint_curve
    gt = tmp_path / "gt"; gt.mkdir()
    black = np.zeros((16, 16, 3), np.uint8); white = np.full((16, 16, 3), 255, np.uint8)
    for n in ("a.png", "b.png"):
        cv2.imwrite(str(gt / n), black)
    dirs = {}
    for a in (0.0, 0.5, 1.0):
        d = tmp_path / hintauc.alpha_dir_name(a); d.mkdir(); dirs[a] = str(d)
        cv2.imwrite(str(d / "a.png"), black)
        if a != 0.5:
            cv2.imwrite(str(d / "b.png"), white)              # b is missing at ratio 0.5
    with pytest.raises(ValueError):
        evaluate_hint_curve(dirs, str(gt), metrics=("mse",), pairing="name")
    with pytest.warns(UserWarning):
        res = evaluate_hint_curve(dirs, str(gt), metrics=("mse",), pairing="name", allow_missing=True)
    assert res["names"] == ["a.png"] and res["hint_auc"]["mse"] == 0.0     # the common subset, at every ratio
    cv2.imwrite(str(tmp_path / "0.50" / "b.png"), white)
    res = evaluate_hint_curve(dirs, str(gt), metrics=("mse",), pairing="name")
    assert res["n_images"] == 2 and res["hint_auc"]["mse"] == pytest.approx(0.5)
