"""Command-line interface: ``hintauc generate`` / ``hintauc eval``."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="hintauc",
        description="Deterministic hint generation (DHT) and Hint-AUC evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate deterministic hints for an image")
    g.add_argument("image", help="input image (ground-truth color image)")
    g.add_argument("-o", "--out_stem", default=None,
                   help="output stem (default: <image without extension>)")
    g.add_argument("--size", type=int, default=64, help="hint resolution (default 64)")
    g.add_argument("--segmenter", default="felzenszwalb",
                   choices=["felzenszwalb", "slic", "quickshift"])
    g.add_argument("--ratio", type=float, default=None,
                   help="also write hints masked to this hint ratio (0..1)")
    g.add_argument("--hint_type", default="scribble", choices=["scribble", "dot"])
    g.add_argument("--path_method", default="filfinder", choices=["filfinder", "geodesic"],
                   help="longest-path extraction: 'filfinder' (paper) or 'geodesic' (dependency-free, deterministic)")
    g.add_argument("--tie_break", default="default", choices=["default", "stable"],
                   help="order of equal-area regions when selecting by --ratio: 'default' (paper: NumPy argsort) or 'stable' (ascending label; version-independent)")
    g.add_argument("--dot_method", default="medoid", choices=["medoid", "mean", "nearest_mean"],
                   help="dot placement: 'medoid' (paper's stored maps: in-region path pixel with the smallest total distance to the others), 'mean' (truncated mean of the path, may leave the region) or 'nearest_mean' (in-region path pixel nearest to the mean)")
    g.add_argument("-v", "--verbose", action="store_true")

    from .metrics import DEFAULT_METRICS as _DEFAULT_METRICS
    c = sub.add_parser("curve", help="Hint-AUC from per-ratio prediction directories (<pred_root>/<alpha>/*.png)")
    c.add_argument("pred_root", help="directory with one sub-directory per hint ratio (e.g. 0.00, 0.01, ..., 1.00)")
    c.add_argument("gt", help="ground-truth directory (files named like the predictions)")
    c.add_argument("--metrics", nargs="+", default=list(_DEFAULT_METRICS))
    c.add_argument("--pairing", default="name", choices=["name", "sorted"])
    c.add_argument("--resize", type=int, default=256)
    c.add_argument("--device", default=None)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--json", default=None, help="write the result to this file")
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
    e.add_argument("--resize", type=int, default=256)
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=0)

    args = p.parse_args(argv)

    if args.cmd == "generate":
        from .hints import generate_hints
        import cv2

        res = generate_hints(args.image, size=args.size,
                             segmenter=args.segmenter, verbose=args.verbose,
                             path_method=args.path_method, dot_method=args.dot_method)
        stem = args.out_stem or os.path.splitext(args.image)[0]
        paths = res.save(stem)
        if args.ratio is not None:
            color, mask = res.at_ratio(args.ratio, hint_type=args.hint_type, tie_break=args.tie_break)
            pct = int(round(args.ratio * 100))
            cv2.imwrite(f"{stem}_{args.hint_type}_col{res.size}_r{pct}.png", color)
            cv2.imwrite(f"{stem}_{args.hint_type}_mask{res.size}_r{pct}.png", mask)
            paths[f"ratio_{pct}"] = f"{stem}_{args.hint_type}_*{res.size}_r{pct}.png"
        info = {"n_regions": res.n_regions(), "failed_regions": res.failed_regions,
                "path_method": res.path_method, "dot_method": res.dot_method, "outputs": paths}
        print(json.dumps(info, indent=1))
        return 0

    if args.cmd == "curve":
        from .auc import evaluate_hint_curve
        from .evaluate import plot_curves, protocol_record
        from .metrics import Evaluator
        preds = {}
        for d in sorted(os.listdir(args.pred_root)):
            full = os.path.join(args.pred_root, d)
            if not os.path.isdir(full):
                continue
            try:
                a = float(d.rstrip("%"))
            except ValueError:
                continue
            preds[a / 100.0 if a > 1.0 else a] = full
        if not preds:
            p.error(f"no ratio sub-directories found under {args.pred_root}")
        alphas = sorted(preds)
        if alphas[0] != 0.0 or alphas[-1] != 1.0:
            print(f"[warn] the ratio grid {alphas} does not span [0, 1]; the Hint-AUC is not comparable with the paper", flush=True)
        ev = Evaluator(metrics=args.metrics, device=args.device, resize=args.resize)
        res = evaluate_hint_curve(preds, args.gt, evaluator=ev, pairing=args.pairing, limit=args.limit)
        out = {"alphas": alphas, "per_alpha": res["per_alpha"], "hint_auc": res["hint_auc"],
               "protocol": protocol_record(ev, alphas, pairing=args.pairing, pred_root=os.path.abspath(args.pred_root))}
        text = json.dumps(out, indent=1)
        print(text)
        if args.json:
            with open(args.json, "w") as f:
                f.write(text)
        if args.plot and not plot_curves(res["per_alpha"], args.plot):
            print("[warn] matplotlib not installed; no plot written", flush=True)
        return

    if args.cmd == "eval":
        from .metrics import Evaluator, evaluate_dirs

        ev = Evaluator(metrics=args.metrics, device=args.device, resize=args.resize)
        if os.path.isdir(args.pred):
            scores = evaluate_dirs(args.pred, args.gt, evaluator=ev,
                                   pairing=args.pairing, limit=args.limit)
            if args.set_metrics is not None:
                from .metrics import evaluate_set
                scores.update(evaluate_set(args.pred, args.gt, metrics=args.set_metrics or ("fid", "kid"),
                                           device=args.device, resize=args.resize))
        else:
            scores = ev(args.pred, args.gt)
        print(json.dumps(scores, indent=1))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
