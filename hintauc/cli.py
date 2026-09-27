"""Command-line interface: ``hintauc generate`` / ``hintauc curve`` / ``hintauc eval`` / ``hintauc demo``."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _ratio_dirs(pred_root: str):
    """Map <pred_root>/<ratio>/ directories to hint ratios.

    A name ending in ``%`` is a percentage (``1%`` -> 0.01, ``100%`` -> 1.0); a bare number above 1 is read as a
    percentage too (``10`` -> 0.10); anything else is the ratio itself (``0.10``). Two directories that map to the
    same ratio are an error rather than a silent overwrite.
    """
    preds, sources = {}, {}
    for d in sorted(os.listdir(pred_root)):
        full = os.path.join(pred_root, d)
        if not os.path.isdir(full):
            continue
        try:
            if d.endswith("%"):
                a = float(d[:-1]) / 100.0
            else:
                a = float(d)
                if a > 1.0:
                    a = a / 100.0
        except ValueError:
            continue
        a = round(a, 8)
        if a in preds:
            raise SystemExit(f"error: directories {sources[a]!r} and {d!r} both denote hint ratio {a:g}")
        preds[a], sources[a] = full, d
    return preds


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="hintauc",
        description="Deterministic hint generation (DHT) and Hint-AUC evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate deterministic hints for an image")
    g.add_argument("image", help="input image (ground-truth color image)")
    g.add_argument("-o", "--out_stem", default=None,
                   help="output stem (default: <image without extension>); missing directories are created")
    g.add_argument("--size", type=int, default=64, help="hint resolution (default 64)")
    g.add_argument("--segmenter", default="felzenszwalb",
                   choices=["felzenszwalb", "slic", "quickshift"])
    g.add_argument("--ratio", type=float, default=None,
                   help="also write hints masked to this hint ratio (0..1)")
    g.add_argument("--hint_type", default="scribble", choices=["scribble", "dot"])
    g.add_argument("--path_method", default="filfinder", choices=["filfinder", "geodesic"],
                   help="longest-path extraction: 'filfinder' (paper; a few pixels may differ between runs) or "
                        "'geodesic' (dependency-free, bit-reproducible)")
    g.add_argument("--tie_break", default="default", choices=["default", "stable"],
                   help="order of equal-area regions when selecting by --ratio: 'default' (paper: NumPy argsort) or 'stable' (ascending label; version-independent)")
    g.add_argument("--dot_method", default="medoid", choices=["medoid", "mean", "nearest_mean"],
                   help="dot placement: 'medoid' (paper's stored maps: in-region path pixel with the smallest total distance to the others), 'mean' (truncated mean of the path, may leave the region) or 'nearest_mean' (in-region path pixel nearest to the mean)")
    g.add_argument("-v", "--verbose", action="store_true")

    from .metrics import DEFAULT_METRICS as _DEFAULT_METRICS
    c = sub.add_parser("curve", help="Hint-AUC from per-ratio prediction directories (<pred_root>/<ratio>/*.png)")
    c.add_argument("pred_root", help="directory with one sub-directory per hint ratio (0.00, 0.01, ..., 1.00; or 0%%, 1%%, ...)")
    c.add_argument("gt", help="ground-truth directory (files named like the predictions)")
    c.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))
    c.add_argument("--pairing", default="name", choices=["name", "sorted"])
    c.add_argument("--allow-missing", action="store_true",
                   help="evaluate the images common to all ratio directories instead of stopping when one is missing")
    c.add_argument("--resize", type=int, default=256)
    c.add_argument("--device", default=None)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--json", default=None, help="write the result (with the list of evaluated images) to this file")
    c.add_argument("--plot", default=None, help="write the metric curves to this PNG (needs matplotlib)")

    e = sub.add_parser("eval", help="evaluate colorization(s) against ground truth")
    e.add_argument("pred", help="predicted image or directory")
    e.add_argument("gt", help="ground-truth image or directory")
    e.add_argument("--metrics", nargs="+",
                   default=["mse", "psnr", "ssim"],
                   help="per-image metrics: mse psnr ssim lpips openclip dino dreamsim (paper) "
                        "and mae ms_ssim deltae lpips_vgg dists (added after the paper)")
    e.add_argument("--set_metrics", nargs="*", default=None,
                   help="set-level metrics between the two directories: fid kid (needs torch-fidelity)")
    e.add_argument("--pairing", default="sorted", choices=["sorted", "name"])
    e.add_argument("--allow-missing", action="store_true", help="skip unmatched files instead of stopping")
    e.add_argument("--resize", type=int, default=256)
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=0)

    d = sub.add_parser("demo", help="self-contained example on a synthetic illustration (no data needed)")
    d.add_argument("--out", default=None, help="write the images, result.json and curve.png here")
    d.add_argument("--path_method", default="geodesic", choices=["geodesic", "filfinder"])
    d.add_argument("--metrics", nargs="+", default=["mse", "psnr", "ssim"])
    d.add_argument("--json", action="store_true", help="print the full result as JSON instead of the table")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        from .hints import generate_hints, write_image

        res = generate_hints(args.image, size=args.size,
                             segmenter=args.segmenter, verbose=args.verbose,
                             path_method=args.path_method, dot_method=args.dot_method)
        stem = args.out_stem or os.path.splitext(args.image)[0]
        try:
            paths = res.save(stem)
            if args.ratio is not None:
                color, mask = res.at_ratio(args.ratio, hint_type=args.hint_type, tie_break=args.tie_break)
                pct = int(round(args.ratio * 100))
                write_image(f"{stem}_{args.hint_type}_col{res.size}_r{pct}.png", color)
                write_image(f"{stem}_{args.hint_type}_mask{res.size}_r{pct}.png", mask)
                paths[f"ratio_{pct}"] = f"{stem}_{args.hint_type}_*{res.size}_r{pct}.png"
        except OSError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        info = {"n_regions": res.n_regions(), "failed_regions": res.failed_regions,
                "path_method": res.path_method, "dot_method": res.dot_method, "outputs": paths}
        print(json.dumps(info, indent=1))
        return 0

    if args.cmd == "curve":
        from .auc import evaluate_hint_curve
        from .evaluate import plot_curves, protocol_record
        from .metrics import Evaluator
        preds = _ratio_dirs(args.pred_root)
        if not preds:
            p.error(f"no ratio sub-directories found under {args.pred_root}")
        alphas = sorted(preds)
        if alphas[0] != 0.0 or alphas[-1] != 1.0:
            print(f"[warn] the ratio grid {alphas} does not span [0, 1]; the Hint-AUC is not comparable with the paper", flush=True)
        ev = Evaluator(metrics=args.metrics, device=args.device, resize=args.resize)
        try:
            res = evaluate_hint_curve(preds, args.gt, evaluator=ev, pairing=args.pairing, limit=args.limit,
                                      allow_missing=args.allow_missing)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        out = {"alphas": res["alphas"], "n_images": res["n_images"], "per_alpha": res["per_alpha"],
               "hint_auc": res["hint_auc"],
               "protocol": protocol_record(ev, res["alphas"], pairing=args.pairing, allow_missing=args.allow_missing,
                                           pred_root=os.path.abspath(args.pred_root))}
        print(json.dumps(out, indent=1))
        if args.json:
            out["images"] = res["names"]
            with open(args.json, "w") as f:
                json.dump(out, f, indent=1)
        if args.plot and not plot_curves(res["per_alpha"], args.plot):
            print("[warn] matplotlib not installed; no plot written", flush=True)
        return 0

    if args.cmd == "eval":
        from .metrics import Evaluator, evaluate_dirs

        ev = Evaluator(metrics=args.metrics, device=args.device, resize=args.resize)
        try:
            if os.path.isdir(args.pred):
                scores = evaluate_dirs(args.pred, args.gt, evaluator=ev, pairing=args.pairing, limit=args.limit,
                                       allow_missing=args.allow_missing)
                if args.set_metrics is not None:
                    from .metrics import evaluate_set
                    scores.update(evaluate_set(args.pred, args.gt, metrics=args.set_metrics or ("fid", "kid"),
                                               device=args.device, resize=args.resize))
            else:
                scores = ev(args.pred, args.gt)
        except ValueError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(json.dumps(scores, indent=1))
        return 0

    if args.cmd == "demo":
        from .demo import run_demo
        res = run_demo(out=args.out, path_method=args.path_method, metrics=args.metrics, verbose=not args.json)
        if args.json:
            print(json.dumps(res, indent=1))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
