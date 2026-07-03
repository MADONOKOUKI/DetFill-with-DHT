#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_summary.py — Aggregate per_image.csv → per_ratio_summary.csv (303 rows).

Standalone summary builder, used by B_run_parallel.sh's watcher AFTER all 3
parallel eval processes finish. Works on any per_image.csv produced by
eval_curve.py.
"""
import argparse
import sys
from pathlib import Path


SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}


def build_summary(per_image_path, per_ratio_path, metrics):
    import pandas as pd
    df = pd.read_csv(per_image_path)
    if df.empty:
        print(f"[summary] {per_image_path} is empty; nothing to do")
        return

    grouped = df.groupby(["sketch", "ratio"], dropna=False)
    rows = []
    for (sk, rt), g in grouped:
        row = {
            "sketch": int(sk),
            "sketch_name": SKETCH_NAMES.get(int(sk), "?"),
            "ratio": rt,
            "n": len(g),
        }
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"]  = float(g[m].std())
        rows.append(row)

    out = pd.DataFrame(rows)
    out["__rf"] = out["ratio"].astype(float)
    out.sort_values(["sketch", "__rf"], inplace=True)
    out.drop(columns="__rf", inplace=True)

    cols = ["sketch", "sketch_name", "ratio", "n"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_std"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(per_ratio_path, index=False)
    print(f"[summary] wrote {per_ratio_path}  ({len(out)} rows)")

    # quick sanity: how many (sketch, ratio) combos and any with n != 3000?
    by_sketch = out.groupby("sketch").agg(n_dirs=("ratio", "count"),
                                          n_total=("n", "sum"),
                                          n_min=("n", "min"),
                                          n_max=("n", "max"))
    print("[summary] per-sketch coverage:")
    print(by_sketch.to_string())
    odd = out[out["n"] != 3000]
    if len(odd):
        print(f"[summary] WARN: {len(odd)} rows with n != 3000:")
        print(odd[["sketch", "sketch_name", "ratio", "n"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_image", required=True)
    p.add_argument("--per_ratio", required=True)
    p.add_argument("--metrics", nargs="+",
                   default=["psnr", "ssim", "lpips", "dreamsim"])
    args = p.parse_args()

    per_img = Path(args.per_image).expanduser().resolve()
    per_rat = Path(args.per_ratio).expanduser().resolve()
    if not per_img.exists():
        print(f"[error] not found: {per_img}", file=sys.stderr)
        sys.exit(1)
    per_rat.parent.mkdir(parents=True, exist_ok=True)
    build_summary(per_img, per_rat, args.metrics)


if __name__ == "__main__":
    main()
