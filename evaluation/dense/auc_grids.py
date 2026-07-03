#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auc_grids.py — 7-metric Hint-AUC grid-sensitivity script (supersedes the original 4-metric variant).

Identical logic, but reports Hint-AUC for ALL SEVEN paper metrics
(MSE, PSNR, SSIM, LPIPS, OpenCLIP, DINO, DreamSim) instead of the 4 in the
original. The original 4-metric script is left untouched (used elsewhere).

For the deterministic-model (R3-2/R2-3) Hint-AUC, the canonical value per
metric is the "Original front-loaded" row.

Reads per_ratio_summary.csv (with <metric>_mean columns) and interpolates each
metric's mean curve onto candidate alpha grids; trapezoidal Hint-AUC over [0,1].
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]
# higher-is-better (up) vs lower-is-better (down)
DIRECTION = {"mse": "down", "lpips": "down", "dreamsim": "down",
             "psnr": "up", "ssim": "up", "openclip": "up", "dino": "up"}
ARROW = {"up": "↑", "down": "↓"}
SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}

GRIDS = [
    ("Original front-loaded",     [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]),
    ("Uniform (N=5)",             list(np.linspace(0.0, 1.0,  5))),
    ("Uniform (N=8)",             list(np.linspace(0.0, 1.0,  8))),
    ("Uniform (N=11)",            list(np.linspace(0.0, 1.0, 11))),
    ("Uniform (N=21)",            list(np.linspace(0.0, 1.0, 21))),
    ("Low-α reduced",        [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]),
    ("High-α shifted",       [0.0, 0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 1.00]),
    ("Dense reference (N=101)",   list(np.linspace(0.0, 1.0, 101))),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True,
                   help="per_ratio_summary.csv from eval_curve (7-metric)")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def trapz_auc(alphas, values):
    a = np.asarray(alphas)
    v = np.asarray(values)
    order = np.argsort(a)
    return float(np.trapz(v[order], a[order]))


def build_per_sketch_table(df):
    rows = []
    for grid_name, grid_alphas in GRIDS:
        grid_alphas = sorted(set(round(a, 4) for a in grid_alphas))
        N = len(grid_alphas)
        for sk in sorted(df["sketch"].unique()):
            sub = df[df["sketch"] == sk].sort_values("ratio_f")
            xs = sub["ratio_f"].values
            for m in METRICS:
                col = f"{m}_mean"
                if col not in sub.columns:
                    raise SystemExit(
                        f"[ERR] column '{col}' not in summary; "
                        f"run B_eval with --metrics {' '.join(METRICS)}")
                ys = sub[col].values
                ys_at_grid = np.interp(grid_alphas, xs, ys)
                auc = trapz_auc(grid_alphas, ys_at_grid)
                rows.append({
                    "grid": grid_name, "N": N, "sketch": int(sk),
                    "sketch_name": SKETCH_NAMES.get(int(sk), "?"),
                    "metric": m, "auc": auc,
                })
    return pd.DataFrame(rows)


def build_mean_table(per_sketch):
    mean = (per_sketch
            .groupby(["grid", "N", "metric"], as_index=False)["auc"]
            .mean())
    wide = mean.pivot_table(index=["grid", "N"], columns="metric",
                            values="auc").reset_index()
    wide.columns.name = None
    order = [g for g, _ in GRIDS]
    wide["__rank"] = wide["grid"].map({g: i for i, g in enumerate(order)})
    wide.sort_values("__rank", inplace=True)
    wide.drop(columns="__rank", inplace=True)
    wide.reset_index(drop=True, inplace=True)

    orig_row = wide[wide["grid"] == "Original front-loaded"].iloc[0]
    for m in METRICS:
        wide[f"d_{m}"] = wide[m] - orig_row[m]

    def _max_rel(row):
        diffs = []
        for m in METRICS:
            denom = abs(orig_row[m]) if abs(orig_row[m]) > 1e-12 else 1.0
            diffs.append(abs(row[m] - orig_row[m]) / denom)
        return max(diffs)
    wide["max_rel_diff"] = wide.apply(_max_rel, axis=1)
    return wide


# ─── output formatters (data-driven over METRICS) ──────────────────────────
def fmt_metric(val, m):
    if m == "psnr":
        return f"{val:.2f}"
    if m in ("ssim", "openclip", "dino"):
        return f"{val:.3f}"
    return f"{val:.4f}"  # mse, lpips, dreamsim


def fmt_metric_d(val, m):
    sign = "+" if val >= 0 else ""
    if m == "psnr":
        return f"{sign}{val:.2f}"
    if m in ("ssim", "openclip", "dino"):
        return f"{sign}{val:.3f}"
    return f"{sign}{val:.4f}"


def to_markdown(mean_table):
    header = (["Grid", "N"]
              + [f"{m.upper()} {ARROW[DIRECTION[m]]}" for m in METRICS]
              + [f"Δ {m.upper()}" for m in METRICS]
              + ["max |Δ|/orig"])
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for _, r in mean_table.iterrows():
        row = ([r["grid"], str(int(r["N"]))]
               + [fmt_metric(r[m], m) for m in METRICS]
               + [fmt_metric_d(r[f"d_{m}"], m) for m in METRICS]
               + [f"{r['max_rel_diff']*100:.2f}%"])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def to_latex(mean_table):
    colspec = "l c " + "r" * len(METRICS) + " c"
    head_metrics = " & ".join(
        f"{m.upper()} $\\{'uparrow' if DIRECTION[m]=='up' else 'downarrow'}$"
        for m in METRICS)
    pre = (
        "\\begin{table*}[t]\n  \\centering\n"
        "  \\caption{Sensitivity of the seven-metric Hint-AUC to the choice of "
        "$\\alpha$ grid. Each AUC is the trapezoidal integral over $[0,1]$ of the "
        "metric--hint curve interpolated onto the listed grid, averaged over the "
        "three sketch types. Higher is better for PSNR/SSIM/OpenCLIP/DINO; lower "
        "is better for MSE/LPIPS/DreamSim.}\n"
        "  \\label{tab:S-R2-1-B-2-7m}\n  \\small\n"
        f"  \\begin{{tabular}}{{{colspec}}}\n    \\toprule\n"
        f"    Grid & $N$ & {head_metrics} & "
        "$\\max\\!\\big|\\Delta\\big|/\\mathrm{orig}$ \\\\\n    \\midrule\n"
    )
    body = []
    for _, r in mean_table.iterrows():
        vals = " & ".join(fmt_metric(r[m], m) for m in METRICS)
        body.append(
            f"    {r['grid']:<26s} & {int(r['N']):>3d} & {vals} & "
            f"{r['max_rel_diff']*100:>5.2f}\\% \\\\")
    post = "    \\bottomrule\n  \\end{tabular}\n\\end{table*}\n"
    return pre + "\n".join(body) + "\n" + post


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    df["ratio_f"] = df["ratio"].astype(float)
    df = df.sort_values(["sketch", "ratio_f"]).reset_index(drop=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sketch = build_per_sketch_table(df)
    mean = build_mean_table(per_sketch)

    per_sketch.to_csv(out_dir / "auc_grid_sensitivity__per_sketch.csv", index=False)
    mean.to_csv(out_dir / "auc_grid_sensitivity__mean.csv", index=False)
    md = to_markdown(mean)
    tex = to_latex(mean)
    (out_dir / "auc_grid_sensitivity__markdown.md").write_text(md + "\n")
    (out_dir / "auc_grid_sensitivity__latex.tex").write_text(tex)

    print("=== mean (sketch-averaged) 7-metric AUC table ===")
    print(md)
    print(f"\n[csv] {out_dir / 'auc_grid_sensitivity__mean.csv'}")


if __name__ == "__main__":
    main()
