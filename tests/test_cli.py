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


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
