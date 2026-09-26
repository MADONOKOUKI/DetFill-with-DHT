#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_auc_grid_sensitivity.py — recompute Hint-AUC under alternative α grids.

Reads per_ratio_summary.csv (dense 101-point sweep) and interpolates each
metric's mean curve onto a list of candidate α grids; reports the trapezoidal
Hint-AUC over [0,1] for each (grid × sketch × metric).

The grids span both *N* (5/8/11/21/101) and *distribution* (front-loaded,
uniform, low-α reduced, high-α shifted) so that the table answers both the
reviewer's N-sensitivity and distribution-sensitivity questions.

Outputs:
  - <out_dir>/auc_grid_sensitivity__per_sketch.csv (per sketch breakdown)
  - <out_dir>/auc_grid_sensitivity__mean.csv       (sketch-averaged main table)
  - <out_dir>/auc_grid_sensitivity__latex.tex      (LaTeX tabular for paste-in)
  - <out_dir>/auc_grid_sensitivity__markdown.md    (markdown table for review)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["psnr", "ssim", "lpips", "dreamsim"]
SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}

# (display_name, list_of_alphas).  Names mirror the reviewer's phrasing.
GRIDS = [
    ("Original front-loaded",     [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]),
    ("Uniform (N=5)",             list(np.linspace(0.0, 1.0,  5))),
    ("Uniform (N=8)",             list(np.linspace(0.0, 1.0,  8))),
    ("Uniform (N=11)",            list(np.linspace(0.0, 1.0, 11))),
    ("Uniform (N=21)",            list(np.linspace(0.0, 1.0, 21))),
    ("Low-α reduced",             [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]),
    ("High-α shifted",            [0.0, 0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 1.00]),
    ("Dense reference (N=101)",   list(np.linspace(0.0, 1.0, 101))),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True,
                   help="per_ratio_summary.csv from B_eval_dense_curve")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def trapz_auc(alphas, values):
    """Trapezoidal AUC over the alpha range covered by the grid.
    NumPy's trapz works for the irregular grids we have."""
    a = np.asarray(alphas)
    v = np.asarray(values)
    order = np.argsort(a)
    return float(np.trapz(v[order], a[order]))


def build_per_sketch_table(df):
    """For every (grid, sketch, metric) -> AUC and N."""
    rows = []
    for grid_name, grid_alphas in GRIDS:
        grid_alphas = sorted(set(round(a, 4) for a in grid_alphas))
        N = len(grid_alphas)
        for sk in sorted(df["sketch"].unique()):
            sub = df[df["sketch"] == sk].sort_values("ratio_f")
            xs = sub["ratio_f"].values
            for m in METRICS:
                ys = sub[f"{m}_mean"].values
                ys_at_grid = np.interp(grid_alphas, xs, ys)
                auc = trapz_auc(grid_alphas, ys_at_grid)
                rows.append({
                    "grid": grid_name,
                    "N": N,
                    "sketch": int(sk),
                    "sketch_name": SKETCH_NAMES.get(int(sk), "?"),
                    "metric": m,
                    "auc": auc,
                })
    return pd.DataFrame(rows)


def build_mean_table(per_sketch):
    """Average AUC across sketches; one row per (grid, N), wide on metrics.
    Adds signed Δ vs 'Original front-loaded' and worst-case |Δ|/orig per row."""
    mean = (per_sketch
            .groupby(["grid", "N", "metric"], as_index=False)["auc"]
            .mean())
    wide = mean.pivot_table(index=["grid", "N"], columns="metric",
                            values="auc").reset_index()
    wide.columns.name = None

    # preserve canonical order
    order = [g for g, _ in GRIDS]
    wide["__rank"] = wide["grid"].map({g: i for i, g in enumerate(order)})
    wide.sort_values("__rank", inplace=True)
    wide.drop(columns="__rank", inplace=True)
    wide.reset_index(drop=True, inplace=True)

    # Δ from "Original front-loaded"
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


# ─── output formatters ─────────────────────────────────────────────────────
def fmt_metric(val, m):
    if m in ("psnr",):           return f"{val:.2f}"
    if m in ("ssim",):           return f"{val:.3f}"
    return f"{val:.4f}"  # LPIPS / DreamSim

def fmt_metric_d(val, m):
    sign = "+" if val >= 0 else ""
    if m in ("psnr",):
        return f"{sign}{val:.2f}"
    if m in ("ssim",):
        return f"{sign}{val:.3f}"
    return f"{sign}{val:.4f}"


def to_markdown(mean_table):
    """Compact markdown table for review."""
    lines = []
    header = ["Grid", "N",
              "PSNR ↑", "SSIM ↑", "LPIPS ↓", "DreamSim ↓",
              "Δ PSNR", "Δ SSIM", "Δ LPIPS", "Δ DreamSim",
              "max |Δ|/orig"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for _, r in mean_table.iterrows():
        row = [
            r["grid"],
            str(int(r["N"])),
            fmt_metric(r["psnr"], "psnr"),
            fmt_metric(r["ssim"], "ssim"),
            fmt_metric(r["lpips"], "lpips"),
            fmt_metric(r["dreamsim"], "dreamsim"),
            fmt_metric_d(r["d_psnr"], "psnr"),
            fmt_metric_d(r["d_ssim"], "ssim"),
            fmt_metric_d(r["d_lpips"], "lpips"),
            fmt_metric_d(r["d_dreamsim"], "dreamsim"),
            f"{r['max_rel_diff']*100:.2f}%",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def to_latex(mean_table):
    """LaTeX tabular for direct paste into the manuscript."""
    pre = r"""\begin{table*}[t]
  \centering
  \caption{%
    Sensitivity of Hint-AUC to the choice of $\alpha$ grid.
    Each AUC is computed by linear interpolation of the dense 101-point
    curve (Fig.~\ref{fig:S-R2-1-B-1}) onto the listed grid, followed by
    trapezoidal integration over $[0, 1]$, and is averaged over the three
    sketch types.
    Higher is better for PSNR/SSIM; lower is better for LPIPS/DreamSim.
    The right-most column is the worst-case relative deviation of any
    metric from the canonical \emph{Original front-loaded} grid.%
  }
  \label{tab:S-R2-1-B-2}
  \small
  \begin{tabular}{l c rrrr c}
    \toprule
    Grid & $N$ &
      PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & DreamSim $\downarrow$ &
      $\max\!\big|\Delta\big|/\mathrm{orig}$ \\
    \midrule
"""
    body_lines = []
    for _, r in mean_table.iterrows():
        body_lines.append(
            f"    {r['grid']:<26s} & {int(r['N']):>3d} & "
            f"{fmt_metric(r['psnr'],'psnr'):>6s} & "
            f"{fmt_metric(r['ssim'],'ssim'):>6s} & "
            f"{fmt_metric(r['lpips'],'lpips'):>7s} & "
            f"{fmt_metric(r['dreamsim'],'dreamsim'):>7s} & "
            f"{r['max_rel_diff']*100:>5.2f}\\% \\\\"
        )
    post = r"""    \bottomrule
  \end{tabular}
\end{table*}
"""
    return pre + "\n".join(body_lines) + "\n" + post


def main():
    args = parse_args()
    df = pd.read_csv(args.summary)
    df["ratio_f"] = df["ratio"].astype(float)
    df = df.sort_values(["sketch", "ratio_f"]).reset_index(drop=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sketch = build_per_sketch_table(df)
    mean = build_mean_table(per_sketch)

    # CSV outputs
    per_sketch.to_csv(out_dir / "auc_grid_sensitivity__per_sketch.csv", index=False)
    mean.to_csv(out_dir / "auc_grid_sensitivity__mean.csv", index=False)

    # Markdown + LaTeX
    md = to_markdown(mean)
    tex = to_latex(mean)
    (out_dir / "auc_grid_sensitivity__markdown.md").write_text(md + "\n")
    (out_dir / "auc_grid_sensitivity__latex.tex").write_text(tex)

    print("=== mean (sketch-averaged) AUC table ===")
    print(md)
    print()
    print(f"[csv] {out_dir / 'auc_grid_sensitivity__per_sketch.csv'}")
    print(f"[csv] {out_dir / 'auc_grid_sensitivity__mean.csv'}")
    print(f"[md ] {out_dir / 'auc_grid_sensitivity__markdown.md'}")
    print(f"[tex] {out_dir / 'auc_grid_sensitivity__latex.tex'}")


if __name__ == "__main__":
    main()
