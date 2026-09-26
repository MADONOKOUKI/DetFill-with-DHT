"""Quick start on CPU: deterministic hints, a reference colorizer over the hint-ratio grid, Hint-AUC.

    python examples/quickstart_cpu.py            # paper settings (FilFinder longest path), about 4-5 minutes on 2 CPU threads
    python examples/quickstart_cpu.py --fast     # dependency-free longest path, under a minute

Uses the two illustrations shipped with the repository (replicability/data, Danbooru2021 ids 4731016 and 4942016)
and their SketchKeras line art. The "model" is hintauc.hint_fill_colorizer, a reference baseline that fills every
hinted region with its hint colour, so the curve shows how a score rises with the hint ratio. Outputs, under
examples/output/quickstart/ by default:

    hints_<id>_<alpha>.png     hint visualisations at 1 %, 10 % and 100 %
    pred/<alpha>/<id>.png      the reference colorizer's outputs (the layout of `hintauc curve`)
    curve.png, result.json     the metric curves, the Hint-AUC and the protocol record

Expected result with the default settings (metrics on 256 x 256, mean over the two images): PSNR rises from
8.3 dB without hints to 16.6 dB with all hints, PSNR Hint-AUC 14.19 (MSE 0.0437, SSIM 0.546); with --fast the Hint-AUC
is 14.29. `hintauc curve` on pred/ returns exactly the same numbers. Your own model: docs/evaluate_your_model.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

try:
    import hintauc
except ImportError:                     # running from a checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import hintauc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IDS = ("4731016", "4942016")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "output", "quickstart"))
    ap.add_argument("--fast", action="store_true", help="geodesic longest path instead of FilFinder (seconds instead of a minute)")
    ap.add_argument("--metrics", nargs="+", default=["mse", "psnr", "ssim"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = os.path.join(ROOT, "replicability", "data")
    samples = [(os.path.join(data, "sketch", f"{i}.png"), os.path.join(data, f"{i}.image.png")) for i in IDS]
    path_method = "geodesic" if a.fast else "filfinder"

    t0 = time.time()
    print(f"[1/3] hints ({path_method}) + reference colorizer at {len(hintauc.DEFAULT_ALPHAS)} hint ratios ...", flush=True)
    res = hintauc.evaluate_colorizer(hintauc.hint_fill_colorizer, samples, metrics=a.metrics, path_method=path_method,
                                     save_dir=os.path.join(a.out, "pred"), verbose=True)
    for i in IDS:                                                 # hint visualisations
        h = hintauc.generate_hints(os.path.join(data, f"{i}.image.png"), size=64, path_method=path_method)
        for alpha in (0.01, 0.10, 1.00):
            color, mask = h.at_ratio(alpha, hint_type="scribble", resize_to=256)
            vis = np.full_like(color, 255)
            vis[mask > 0] = color[mask > 0]
            cv2.imwrite(os.path.join(a.out, f"hints_{i}_{int(alpha * 100)}.png"), vis)
    print("[2/3] per-ratio means:")
    for alpha in res["alphas"]:
        print("   alpha %.2f  " % alpha + "  ".join(f"{m} {res['per_alpha'][alpha][m]:.4f}" for m in a.metrics))
    print("      Hint-AUC  " + "  ".join(f"{m} {res['hint_auc'][m]:.4f}" for m in a.metrics))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    if hintauc.plot_curves(res["per_alpha"], os.path.join(a.out, "curve.png"), title="reference colorizer, 2 images"):
        print(f"      curve: {os.path.join(a.out, 'curve.png')}")

    print("[3/3] the same numbers from the saved images (`hintauc curve pred/ gt/`):", flush=True)
    gt_dir = os.path.join(a.out, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    for i in IDS:
        cv2.imwrite(os.path.join(gt_dir, f"{i}.image.png"), cv2.imread(os.path.join(data, f"{i}.image.png")))
    out = subprocess.run([sys.executable, "-m", "hintauc.cli", "curve", os.path.join(a.out, "pred"), gt_dir,
                          "--metrics", *a.metrics, "--json", os.path.join(a.out, "curve.json")],
                         capture_output=True, text=True, check=True)
    print("   ", json.loads(out.stdout)["hint_auc"])
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
