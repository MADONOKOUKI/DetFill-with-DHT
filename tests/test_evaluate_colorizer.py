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
                                     path_method="geodesic", save_dir=tmp_path / "pred")
    assert res["n_images"] == 1 and set(res["hint_auc"]) == {"mse", "psnr"}
    assert res["alphas"] == list(hintauc.DEFAULT_ALPHAS) and res["protocol"]["paper_grid"] is True
    pa = res["per_alpha"]
    assert pa[1.0]["psnr"] > pa[0.0]["psnr"]                 # more hints -> closer to the ground truth
    assert np.isfinite(res["hint_auc"]["psnr"]) and pa[0.0]["psnr"] <= res["hint_auc"]["psnr"] <= pa[1.0]["psnr"]
    assert (tmp_path / "pred" / "1.00" / "img.png").exists() and (tmp_path / "pred" / "0.00" / "img.png").exists()
    assert res["protocol"]["versions"]["hintauc"] == hintauc.__version__


def test_callback_output_is_validated(tmp_path):
    with pytest.raises(ValueError):
        hintauc.evaluate_colorizer(lambda s: np.zeros((3, 3), np.uint8), _samples(tmp_path),
                                   alphas=(0.0, 1.0), metrics=("mse",), path_method="geodesic")
    with pytest.raises(ValueError):
        hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, _samples(tmp_path),
                                   alphas=(0.1, 1.0), metrics=("mse",), path_method="geodesic")


def test_saved_predictions_reproduce_the_curve_through_the_directory_path(tmp_path, capsys):
    from hintauc.cli import main
    samples = _samples(tmp_path)
    res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, samples, alphas=(0.0, 0.5, 1.0),
                                     metrics=("mse", "psnr"), path_method="geodesic", save_dir=tmp_path / "pred")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    shutil.copy(samples[0][1], gt_dir / "img.png")
    main(["curve", str(tmp_path / "pred"), str(gt_dir), "--metrics", "mse", "psnr", "--json", str(tmp_path / "curve.json")])
    out = json.load(open(tmp_path / "curve.json"))
    assert out["alphas"] == [0.0, 0.5, 1.0]
    assert abs(out["hint_auc"]["psnr"] - res["hint_auc"]["psnr"]) < 1e-6
    assert out["protocol"]["paper_grid"] is False


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
    scores = evaluate_dirs(str(pred), str(gt), metrics=("mse",), pairing="name")     # the two common names
    assert scores["mse"] == 0.0
    with pytest.raises(ValueError):
        evaluate_dirs(str(pred), str(tmp_path), metrics=("mse",), pairing="name")    # nothing in common
