#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

# Paper setting:
EXPECTED_ALPHA = [0.00, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]

DEFAULT_METRICS = ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_single_row_csv(path: str) -> Dict[str, str]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if row is None:
            raise RuntimeError(f"empty csv: {path}")
        return row


def trapz(xs: List[float], ys: List[float]) -> float:
    """Trapezoidal rule: 1/2 * Σ (y_{i+1}+y_i)*(x_{i+1}-x_i)"""
    auc = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        auc += 0.5 * (ys[i + 1] + ys[i]) * dx
    return auc


def parse_alpha_csv_pairs(items: List[str]) -> Dict[float, str]:
    """
    Parse repeated args like:
      --alpha_csv 0.00=/path/to/a0.csv
      --alpha_csv 0.01:/path/to/a1.csv
    """
    out: Dict[float, str] = {}
    for s in items:
        if "=" in s:
            k, v = s.split("=", 1)
        elif ":" in s:
            k, v = s.split(":", 1)
        else:
            raise ValueError(f"invalid --alpha_csv '{s}'. Use alpha=path (or alpha:path).")

        a = float(k)
        if a in out:
            raise ValueError(f"duplicate alpha specified: {a}")
        out[a] = os.path.abspath(os.path.expanduser(v))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="",
                   help="出力行のラベル（例: illust_dot_sketch2_det）")
    p.add_argument("--alpha_csv", action="append", required=True,
                   help="alpha=csv_path を8回指定（例: --alpha_csv 0.00=...csv）")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help="CSV列名として読むメトリクス（例: mse psnr ssim ...）")
    p.add_argument("--out_csv", type=str, default="",
                   help="summary CSV 出力先（未指定なら ./hint_auc_manual_summary.csv）")
    p.add_argument("--out_curve_csv", type=str, default="",
                   help="任意：alphaごとの点を出すCSV（デバッグ用）")
    p.add_argument("--check_hint_ratio", action="store_true",
                   help="CSV内の hint_ratio が alpha と整合しているか軽く警告（手指定が正なら不要）")
    p.add_argument("--hint_ratio_scale", choices=["auto", "ratio", "percent"], default="auto",
                   help="check_hint_ratio 用: hint_ratioが 0..1 か 0..100 か。autoは値で推定")
    args = p.parse_args()

    alpha_to_csv = parse_alpha_csv_pairs(args.alpha_csv)

    # ---- strict: must match EXPECTED_ALPHA exactly ----
    exp_set = set(EXPECTED_ALPHA)
    got_set = set(alpha_to_csv.keys())
    missing = sorted(exp_set - got_set)
    extra = sorted(got_set - exp_set)

    if missing or extra:
        msg = []
        if missing:
            msg.append(f"missing alpha: {missing}")
        if extra:
            msg.append(f"extra alpha: {extra}")
        raise RuntimeError("alpha set mismatch. " + " / ".join(msg) +
                           f"\nExpected: {EXPECTED_ALPHA}")

    # ---- read all points ----
    metrics = [m.strip() for m in args.metrics if m.strip()]
    points: List[Tuple[float, Dict[str, float], Dict[str, str]]] = []

    # for hint_ratio check (optional)
    hint_ratio_vals = []

    for a in EXPECTED_ALPHA:
        csv_path = alpha_to_csv[a]
        if not os.path.isfile(csv_path):
            raise RuntimeError(f"csv not found for alpha={a}: {csv_path}")

        row = read_single_row_csv(csv_path)
        vals: Dict[str, float] = {}
        for m in metrics:
            if m in row:
                vals[m] = to_float(row[m])
        points.append((a, vals, row))

        hr = to_float(row.get("hint_ratio", ""))
        if math.isfinite(hr):
            hint_ratio_vals.append(hr)

    # ---- optional: check hint_ratio consistency ----
    if args.check_hint_ratio and hint_ratio_vals:
        hr_max = max(hint_ratio_vals)
        if args.hint_ratio_scale == "percent":
            scale = 0.01
        elif args.hint_ratio_scale == "ratio":
            scale = 1.0
        else:
            # auto: if max > 1.5 => percent
            scale = 0.01 if hr_max > 1.5 else 1.0

        # warn if some point is far
        for a, _, row in points:
            hr = to_float(row.get("hint_ratio", ""))
            if not math.isfinite(hr):
                continue
            est_a = hr * scale
            if abs(est_a - a) > 1e-6:
                print(f"[WARN] hint_ratio mismatch? alpha={a} but csv hint_ratio={hr} (scale={scale} => {est_a})")

    # ---- compute AUC for each metric ----
    xs = [a for a, _, _ in points]
    summary = {
        "name": args.name,
        "n_points": str(len(xs)),
        "alpha_min": f"{min(xs):.8f}",
        "alpha_max": f"{max(xs):.8f}",
    }

    # also keep some meta (optional)
    # take from alpha=0.00 row
    _, _, row0 = points[0]
    summary["example_run"] = row0.get("run", "")
    summary["example_run_dir"] = row0.get("run_dir", "")
    summary["example_sample_dir"] = row0.get("sample_dir", "")

    for m in metrics:
        ys = []
        ok = True
        for _, vals, _ in points:
            v = vals.get(m, float("nan"))
            if not math.isfinite(v):
                ok = False
                break
            ys.append(v)
        summary[f"HintAUC_{m}"] = f"{trapz(xs, ys):.10g}" if ok else ""

    # ---- print ----
    label = args.name if args.name else "(no name)"
    print(f"[Hint-AUC] {label}")
    for m in metrics:
        k = f"HintAUC_{m}"
        if summary.get(k, "") != "":
            print(f"  {k:14s}: {summary[k]}")
        else:
            print(f"  {k:14s}: (missing values in some CSVs)")

    # ---- write summary CSV ----
    out_csv = args.out_csv.strip() or os.path.abspath("hint_auc_manual_summary.csv")
    out_csv = os.path.abspath(os.path.expanduser(out_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    keys = ["name", "n_points", "alpha_min", "alpha_max",
            "example_run", "example_run_dir", "example_sample_dir"] + [f"HintAUC_{m}" for m in metrics]

    # append if exists, else create with header
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            w.writeheader()
        w.writerow({k: summary.get(k, "") for k in keys})

    print(f"[OK] wrote/append summary: {out_csv}")

    # ---- optional: write curve CSV ----
    if args.out_curve_csv.strip():
        out_curve = os.path.abspath(os.path.expanduser(args.out_curve_csv.strip()))
        os.makedirs(os.path.dirname(out_curve), exist_ok=True)
        ckeys = ["name", "alpha", "csv_path", "run", "run_dir", "hint_ratio"] + metrics
        with open(out_curve, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ckeys)
            w.writeheader()
            for a, vals, row in points:
                csv_path = alpha_to_csv[a]
                out_row = {
                    "name": args.name,
                    "alpha": f"{a:.8f}",
                    "csv_path": csv_path,
                    "run": row.get("run", ""),
                    "run_dir": row.get("run_dir", ""),
                    "hint_ratio": row.get("hint_ratio", ""),
                }
                for m in metrics:
                    if m in vals and math.isfinite(vals[m]):
                        out_row[m] = vals[m]
                    else:
                        out_row[m] = ""
                w.writerow(out_row)
        print(f"[OK] wrote curve: {out_curve}")


if __name__ == "__main__":
    main()
