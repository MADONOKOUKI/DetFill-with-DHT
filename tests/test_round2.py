"""Second-round regressions: image decoding, run manifests, stale directories, the number checker, the per-ratio
evaluation script, the example comparison and the DetFill provenance / training-budget helpers (all CPU, no downloads)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import hintauc
from hintauc.metrics import Evaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rgb(seed=0, size=24):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------------------------- decoding
def test_files_of_every_mode_decode_to_the_colours_they_show(tmp_path):
    from PIL import Image
    ev = Evaluator(metrics=("mse",), resize=0)
    rgb = _rgb(0)
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    pal = Image.fromarray(rgb).quantize(64)            # palette PNG
    pal.save(tmp_path / "palette.png")
    pal_rgb = np.asarray(pal.convert("RGB"))
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    Image.fromarray(gray).save(tmp_path / "gray.png")
    la = Image.merge("LA", (Image.fromarray(gray), Image.new("L", gray.shape[::-1], 200)))
    la.save(tmp_path / "la.png")
    Image.fromarray(rgb).convert("CMYK").save(tmp_path / "cmyk.jpg", quality=100)
    cmyk_rgb = np.asarray(Image.open(tmp_path / "cmyk.jpg").convert("RGB"))
    Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])).save(tmp_path / "rgba.png")
    assert ev(str(tmp_path / "rgb.png"), rgb)["mse"] == 0.0
    assert ev(str(tmp_path / "palette.png"), pal_rgb)["mse"] == 0.0
    assert ev(str(tmp_path / "gray.png"), np.stack([gray] * 3, -1))["mse"] == 0.0
    assert ev(str(tmp_path / "la.png"), np.stack([gray] * 3, -1))["mse"] == 0.0
    assert ev(str(tmp_path / "cmyk.jpg"), cmyk_rgb)["mse"] == 0.0
    assert ev(str(tmp_path / "rgba.png"), rgb)["mse"] == 0.0


def test_sixteen_bit_files_are_rejected(tmp_path):
    from PIL import Image
    arr16 = (np.arange(24 * 24, dtype=np.uint16).reshape(24, 24) * 100)
    Image.fromarray(arr16, mode="I;16").save(tmp_path / "deep.png")
    with pytest.raises(ValueError):
        Evaluator(metrics=("mse",), resize=0)(str(tmp_path / "deep.png"), _rgb(1))


@pytest.mark.skipif(not Evaluator().torchvision, reason="torchvision not installed")
def test_both_backends_decode_files_identically(tmp_path):
    from PIL import Image
    rgb = _rgb(2, 48)
    Image.fromarray(rgb).quantize(32).save(tmp_path / "p.png")
    ev_tv = Evaluator(metrics=("mse", "psnr"))
    ev_pil = Evaluator(metrics=("mse", "psnr")); ev_pil._tv = False
    a, b = ev_tv._read_01(str(tmp_path / "p.png")), ev_pil._read_01(str(tmp_path / "p.png"))
    assert float(np.abs(a - b).max()) < 1e-4


# -------------------------------------------------------------------------------- manifests / save_dir
def _samples(tmp_path, ext="png"):
    import cv2
    gt = hintauc.synthetic_illustration()
    gt_path = tmp_path / f"img.{ext}"
    if ext == "jpg":
        cv2.imwrite(str(gt_path), gt, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(str(gt_path), gt)
    sk_path = tmp_path / "img_sketch.png"
    cv2.imwrite(str(sk_path), 255 - cv2.Canny(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY), 50, 150))
    return [(str(sk_path), str(gt_path))]


def test_save_dir_of_another_run_is_refused_unless_overwritten(tmp_path):
    black = lambda s: np.zeros(s["line_art"].shape + (3,), np.uint8)
    white = lambda s: np.full(s["line_art"].shape + (3,), 255, np.uint8)
    out = tmp_path / "pred"
    hintauc.evaluate_colorizer(black, _samples(tmp_path), alphas=(0.0, 0.5, 1.0), metrics=("mse",),
                               path_method="geodesic", save_dir=out)
    with pytest.raises(ValueError):                                  # a different grid into the same directory
        hintauc.evaluate_colorizer(white, _samples(tmp_path), alphas=(0.0, 1.0), metrics=("mse",),
                                   path_method="geodesic", save_dir=out)
    assert (out / "0.50").exists()                                   # nothing was touched
    res = hintauc.evaluate_colorizer(white, _samples(tmp_path), alphas=(0.0, 1.0), metrics=("mse",),
                                     path_method="geodesic", save_dir=out, overwrite=True)
    assert sorted(d.name for d in out.iterdir() if d.is_dir()) == ["0.00", "1.00"]
    manifest = hintauc.read_manifest(out)
    assert manifest["alphas"] == [0.0, 1.0] and manifest["oracle"] is False and manifest["tie_break"] == "stable"
    assert manifest["sources"] == {"img.png": "img.png"} and res["manifest"].endswith("manifest.json")


def test_curve_follows_the_manifest_and_rejects_stale_directories(tmp_path, capsys):
    import cv2
    from hintauc.cli import main
    out, gt_dir = tmp_path / "pred", tmp_path / "gt"
    samples = _samples(tmp_path, ext="jpg")                          # a JPEG ground truth, PNG predictions
    res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, samples, alphas=(0.0, 0.5, 1.0), metrics=("mse", "psnr"),
                                     path_method="geodesic", save_dir=out, oracle=True)
    gt_dir.mkdir()
    import shutil
    shutil.copy(samples[0][1], gt_dir / "img.jpg")
    rc = main(["curve", str(out), str(gt_dir), "--metrics", "mse", "psnr", "--json", str(tmp_path / "c.json")])
    js = json.load(open(tmp_path / "c.json"))
    assert rc == 0 and abs(js["hint_auc"]["psnr"] - res["hint_auc"]["psnr"]) < 1e-6
    assert js["protocol"]["run"]["oracle"] is True and js["protocol"]["pairing"] == "manifest"
    capsys.readouterr()
    (out / "0.25").mkdir()                                           # a directory of another run
    cv2.imwrite(str(out / "0.25" / "img.png"), np.zeros((128, 128, 3), np.uint8))
    rc = main(["curve", str(out), str(gt_dir), "--metrics", "mse"])
    assert rc == 1 and "manifest does not list" in capsys.readouterr().err
    rc = main(["curve", str(out), str(gt_dir), "--metrics", "mse", "--ignore-manifest", "--allow-missing"])
    assert rc == 1                                                   # without the manifest the JPEG name cannot pair


# ------------------------------------------------------------------------------------ check_expected.py
def _run_checker(tmp_path, hint_auc, backend_args=()):
    result = {"hint_auc": hint_auc, "protocol": {"backend": {"resize": "pillow", "ssim": "scikit-image"}}}
    p = tmp_path / "r.json"; p.write_text(json.dumps(result))
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "check_expected.py"), str(p), "demo", *backend_args],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_number_checker_rejects_nan_and_wrong_backend(tmp_path):
    expected = json.load(open(os.path.join(ROOT, "examples", "expected_numbers.json")))["demo"]["hint_auc"]["pillow"]
    assert _run_checker(tmp_path, expected)[0] == 0
    rc, out = _run_checker(tmp_path, dict(expected, mse=float("nan")))
    assert rc == 1 and "FAIL" in out
    rc, out = _run_checker(tmp_path, {k: v for k, v in expected.items() if k != "psnr"})
    assert rc == 1                                                   # a missing metric is a failure
    rc, out = _run_checker(tmp_path, expected, ("--backend", "torch"))
    assert rc == 1 and "expected torch" in out


# --------------------------------------------------------------------------------- eval_per_ratio.py
def _pipeline_tree(tmp_path, drop_b_at=None):
    import cv2
    root = tmp_path / "results" / "scribble"
    gt = tmp_path / "gt"; gt.mkdir(parents=True)
    black, white = np.zeros((16, 16, 3), np.uint8), np.full((16, 16, 3), 255, np.uint8)
    for n in ("a.image.png", "b.image.png"):
        cv2.imwrite(str(gt / n), black)
    for r in ("0.0", "0.5", "1.0"):
        d = root / "2" / r / "200"; d.mkdir(parents=True)
        cv2.imwrite(str(d / "a.image.png"), black)
        if r != drop_b_at:
            cv2.imwrite(str(d / "b.image.png"), white)
        g = root / "2" / r / "ground_truth"; g.mkdir()
        for n in ("a.image.png", "b.image.png"):
            cv2.imwrite(str(g / n), black)
    return root, gt


def _eval(root, gt, out, *extra, gt_from_results=False):
    cmd = [sys.executable, os.path.join(ROOT, "reproduce", "scripts", "eval_per_ratio.py"), "--results_root", str(root),
           "--out_dir", str(out), "--sketches", "2", "--metrics", "mse", "--gpu", "-1", *extra]
    cmd += ["--gt_from_results"] if gt_from_results else ["--gt_dir", str(gt)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_eval_script_refuses_a_missing_prediction_and_verifies_resume(tmp_path):
    import cv2
    root, gt = _pipeline_tree(tmp_path, drop_b_at="0.5")
    proc = _eval(root, gt, tmp_path / "out")
    assert proc.returncode != 0 and "do not hold the same images" in proc.stderr
    proc = _eval(root, gt, tmp_path / "out", "--allow-missing")
    assert proc.returncode == 0, proc.stderr
    h = json.load(open(tmp_path / "out" / "hauc.json"))
    assert h["per_sketch"]["2"]["mse"] == pytest.approx(0.0)         # the common subset (a only) at every ratio
    root2, gt2 = _pipeline_tree(tmp_path / "second")
    proc = _eval(root2, gt2, tmp_path / "out2")
    assert proc.returncode == 0 and json.load(open(tmp_path / "out2" / "hauc.json"))["per_sketch"]["2"]["mse"] == pytest.approx(0.5)
    for r in ("0.0", "0.5", "1.0"):                                  # the predictions change on disk
        cv2.imwrite(str(root2 / "2" / r / "200" / "b.image.png"), np.zeros((16, 16, 3), np.uint8))
    proc = _eval(root2, gt2, tmp_path / "out2")
    assert proc.returncode == 0 and "reused" in proc.stdout
    assert json.load(open(tmp_path / "out2" / "hauc.json"))["per_sketch"]["2"]["mse"] == pytest.approx(0.0)
    rows = list(open(tmp_path / "out2" / "per_image.csv"))
    assert "pred_sha256" in rows[0]
    proc = _eval(root2, gt2, tmp_path / "out2", "--metrics", "mse", "psnr")   # other metric settings, same out_dir
    assert proc.returncode != 0 and "different settings" in proc.stderr
    proc = _eval(root2, gt2, tmp_path / "out2", "--metrics", "mse", "psnr", "--fresh")
    assert proc.returncode == 0


def test_eval_script_ground_truth_from_results(tmp_path):
    root, gt = _pipeline_tree(tmp_path)
    proc = _eval(root, gt, tmp_path / "out", gt_from_results=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    h = json.load(open(tmp_path / "out" / "hauc.json"))
    assert h["per_sketch"]["2"]["mse"] == pytest.approx(0.5) and "ground_truth" in h["protocol"]["gt"]


# ------------------------------------------------------------------------------ compare_with_expected.py
def test_example_comparison_uses_common_images(tmp_path):
    ids9 = [str(i) for i in range(9)]
    per_image = {"0.0": {i: {"psnr": 10.0 + int(i), "lpips": 0.5, "dreamsim": 0.3} for i in ids9}}
    ref = {"ids": ids9, "models": {"m": {"per_ratio_mean": {"0.0": {"psnr": 14.0, "lpips": 0.5, "dreamsim": 0.3}},
                                         "per_image": per_image}}}
    ids4 = ids9[:4]
    run = {"ids": ids4, "models": {"m": {"per_ratio_mean": {"0.0": {"psnr": 11.5, "lpips": 0.5, "dreamsim": 0.3}},
                                         "per_image": {"0.0": {i: per_image["0.0"][i] for i in ids4}}}}}
    (tmp_path / "out" / "metrics").mkdir(parents=True); (tmp_path / "exp").mkdir()
    json.dump(run, open(tmp_path / "out" / "metrics" / "E1.json", "w")); json.dump(ref, open(tmp_path / "exp" / "E1.json", "w"))
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "reproduce", "examples", "compare_with_expected.py"),
                           "--out", str(tmp_path / "out"), "--expected", str(tmp_path / "exp")], capture_output=True, text=True)
    assert proc.returncode == 0 and "0 outside tolerance" in proc.stdout and "over 4 common image(s)" in proc.stdout


# --------------------------------------------------------------------------- DetFill provenance / budget
def test_run_manifest_and_step_budget(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "detfill"))
    from runners.provenance import check_run_manifest, step_budget_reached, write_run_manifest
    m = {"checkpoint_sha256": "aaa", "seed": 1234, "hint_type": "scribble", "hint_order": "area", "sample_ratio": 0.1,
         "sketch_type": 2, "sample_step": 200, "domain": "illust", "model_name": "BB", "config_sha256": "c", "data_root": "/d"}
    d = str(tmp_path / "cell")
    check_run_manifest(d, m, has_outputs=False)                      # nothing there yet
    write_run_manifest(d, m)
    check_run_manifest(d, dict(m), has_outputs=True)                 # same conditions: fine
    with pytest.raises(RuntimeError):
        check_run_manifest(d, dict(m, checkpoint_sha256="bbb"), has_outputs=True)
    check_run_manifest(d, dict(m, checkpoint_sha256="bbb"), has_outputs=False)   # no outputs: nothing to protect
    assert step_budget_reached(400000, 400000) and not step_budget_reached(399999, 400000)
    assert not step_budget_reached(10, None) and not step_budget_reached(10, 0)


def test_training_exception_is_reraised_after_the_emergency_save(tmp_path):
    pytest.importorskip("torch")
    sys.path.insert(0, os.path.join(ROOT, "detfill"))
    from runners.BaseRunner import BaseRunner

    class Stub:
        is_main_process = True
        saved = 0

        def get_checkpoint_states(self, stage):
            assert stage == "exception"
            self.saved += 1
            return {"w": 1}, {"o": 1}
    import argparse
    stub = Stub(); stub.config = argparse.Namespace(result=argparse.Namespace(ckpt_path=str(tmp_path)))
    with pytest.raises(RuntimeError):
        BaseRunner.handle_training_exception(stub, RuntimeError("loss is NaN"))
    assert stub.saved == 1 and (tmp_path / "last_model.pth").exists()


# ---------------------------------------------------------------------------- batch hint generation
def test_batch_generator_colours_are_deterministic_and_writes_are_checked(tmp_path):
    pytest.importorskip("skimage")
    sys.path.insert(0, os.path.join(ROOT, "hint_generation"))
    import generate_hints as gh
    seg = np.zeros((32, 32), np.int64); seg[:, 16:] = 1; seg[16:, :] = 2
    a, b = gh.colorize_regions(seg), gh.colorize_regions(seg)
    assert np.array_equal(a, b) and len(np.unique(a.reshape(-1, 3), axis=0)) == 3
    img = np.zeros((32, 32, 3), np.uint8); img[:, 16:] = (200, 30, 30); img[16:, :] = (30, 30, 200)
    region64, mask64, col64 = gh.make_scribbling(img, a, path_method="geodesic")
    assert region64.shape == (64, 64, 3) and mask64.shape == (64, 64) and (mask64 > 0).sum() > 0
    prefix = str(tmp_path / "new" / "dir" / "x.image")
    gh.write_outputs(prefix, region64, mask64, col64)
    assert gh.outputs_complete(prefix)
    os.remove(prefix + "_scribble_col64.png")
    assert not gh.outputs_complete(prefix)                           # a partial output is not "done"
