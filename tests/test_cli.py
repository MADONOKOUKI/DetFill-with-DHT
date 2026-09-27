"""The command line: hint generation and evaluation on synthetic data, without any download."""
import json

import numpy as np
import pytest

from hintauc.cli import main


def test_generate_command_writes_the_canonical_files(illustration, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    main(["generate", str(illustration), "-o", str(out / "img.image"), "--path_method", "geodesic"])
    for kind in ("region", "scribble_mask", "scribble_col", "dot_mask", "dot_col", "flatten_img"):
        assert (out / f"img.image_{kind}64.png").exists(), kind
    main(["generate", str(illustration), "-o", str(out / "half.image"), "--path_method", "geodesic",
          "--ratio", "0.5", "--hint_type", "dot"])


def test_eval_command_on_identical_and_shifted_images(tmp_path, capsys):
    import cv2
    rng = np.random.default_rng(1)
    pred, gt = tmp_path / "pred", tmp_path / "gt"
    pred.mkdir(); gt.mkdir()
    for i in range(3):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(gt / f"{i}.png"), img)
        cv2.imwrite(str(pred / f"{i}.png"), img if i else np.roll(img, 8, axis=1))
    main(["eval", str(pred), str(gt), "--metrics", "mse", "psnr", "--resize", "64"])
    out = capsys.readouterr().out
    assert "mse" in out and "psnr" in out
    try:
        data = json.loads(out)
        assert data["mse"] >= 0
    except json.JSONDecodeError:
        pass                                    # table output is acceptable too


def test_generate_creates_missing_output_directories(illustration, tmp_path):
    rc = main(["generate", str(illustration), "-o", str(tmp_path / "new" / "deeper" / "img.image"),
               "--path_method", "geodesic", "--size", "16", "--ratio", "0.5"])
    assert rc == 0
    files = sorted(p.name for p in (tmp_path / "new" / "deeper").iterdir())
    assert len(files) == 8 and "img.image_region16.png" in files and "img.image_scribble_col16_r50.png" in files


def test_curve_reads_percent_directories_and_rejects_collisions(tmp_path, capsys):
    import cv2
    gt = tmp_path / "gt"; gt.mkdir()
    cv2.imwrite(str(gt / "a.png"), np.zeros((8, 8, 3), np.uint8))
    root = tmp_path / "pred"
    for name in ("0%", "1%", "3%", "5%", "10%", "25%", "50%", "100%"):
        (root / name).mkdir(parents=True)
        cv2.imwrite(str(root / name / "a.png"), np.zeros((8, 8, 3), np.uint8))
    rc = main(["curve", str(root), str(gt), "--metrics", "mse"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["alphas"] == [0.0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0] and out["n_images"] == 1
    (root / "0.10").mkdir()                                     # same ratio as "10%"
    cv2.imwrite(str(root / "0.10" / "a.png"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(SystemExit):
        main(["curve", str(root), str(gt), "--metrics", "mse"])


def test_demo_runs_without_data(tmp_path, capsys):
    rc = main(["demo", "--out", str(tmp_path / "demo"), "--json"])
    res = json.loads(capsys.readouterr().out)
    assert rc == 0 and res["n_images"] == 1 and res["alphas"] == list(np.round(np.array([0, .01, .03, .05, .1, .25, .5, 1]), 2))
    assert res["per_alpha"]["1.0"]["psnr"] > res["per_alpha"]["0.0"]["psnr"]
    assert (tmp_path / "demo" / "result.json").exists() and (tmp_path / "demo" / "pred" / "1.00" / "000000.png").exists()
    assert (tmp_path / "demo" / "pred" / "manifest.json").exists()


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
