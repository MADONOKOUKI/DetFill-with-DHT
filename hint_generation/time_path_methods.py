"""Time hint generation with the two longest-path implementations on the same machine.

    python hint_generation/time_path_methods.py --label "Apple M2 (MacBook Air, 16 GB)" \
        --out hint_generation/time_path_methods_m2.json

Runs the whole ``hintauc.generate_hints(image, size=64)`` call (segmentation, skeletons, longest paths, dots) on the
paper's two Fig. 9 illustrations (replicability/data) with ``path_method="filfinder"`` (the paper's path) and
``path_method="geodesic"`` (added after the paper), and records the wall-clock seconds of every run, the median
speed-up, the region / scribble / dot counts and the library versions. Thread count is fixed with --threads (set it
before NumPy is imported: the script re-executes itself with the thread variables when they differ). Needs
``pip install "hintauc"`` with fil_finder and astropy (the paper used fil_finder 1.7.2 and astropy 5.3.4).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
import warnings

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "replicability", "data"))
    ap.add_argument("--images", nargs="+", default=["4942016.image.png", "4731016.image.png"])
    ap.add_argument("--runs_filfinder", type=int, default=2)
    ap.add_argument("--runs_geodesic", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--label", default="", help="machine description recorded in the output (CPU, RAM)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if any(os.environ.get(v) != str(args.threads) for v in THREAD_VARS):
        env = dict(os.environ, **{v: str(args.threads) for v in THREAD_VARS})
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    warnings.filterwarnings("ignore")
    import astropy
    import cv2
    import fil_finder
    import numpy
    import skimage
    import hintauc

    cv2.setNumThreads(args.threads)
    runs = {"filfinder": args.runs_filfinder, "geodesic": args.runs_geodesic}
    res = {
        "what": "wall-clock seconds of hintauc.generate_hints(image, size=64, path_method=...) per run",
        "machine": {"label": args.label, "platform": platform.platform(), "processor": platform.processor(),
                    "threads": args.threads},
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__, "opencv": cv2.__version__,
                     "scikit-image": skimage.__version__, "fil_finder": getattr(fil_finder, "__version__", "?"),
                     "astropy": astropy.__version__, "hintauc": hintauc.__version__},
        "images": {},
    }
    for name in args.images:
        img = cv2.imread(os.path.join(args.data_dir, name))
        if img is None:
            sys.exit(f"cannot read {os.path.join(args.data_dir, name)}")
        r = res["images"][name] = {"height": int(img.shape[0]), "width": int(img.shape[1])}
        for method, n in runs.items():
            seconds = []
            for _ in range(n):
                t = time.perf_counter()
                h = hintauc.generate_hints(img, size=64, path_method=method)
                seconds.append(time.perf_counter() - t)
            r[method] = {"seconds": [round(s, 3) for s in seconds], "median": round(statistics.median(seconds), 3),
                         "n_regions": int(h.n_regions()), "scribble_pixels": int((h.scribble_mask > 0).sum()),
                         "dots": int((h.dot_mask > 0).sum()), "failed_regions": int(h.failed_regions)}
            print(f"{name} {img.shape[1]}x{img.shape[0]} {method:9s} median {r[method]['median']:7.2f} s "
                  f"runs {r[method]['seconds']} regions {r[method]['n_regions']}", flush=True)
        r["speedup_median"] = round(r["filfinder"]["median"] / r["geodesic"]["median"], 2)
        print(f"{name} speed-up (median filfinder / median geodesic): {r['speedup_median']}x", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
