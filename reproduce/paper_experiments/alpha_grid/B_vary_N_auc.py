#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_vary_N_auc.py — Vary the number of $\alpha$ grid points (uniform spacing)
and recompute Hint-AUC by interpolating the dense 101-point curve.

This script answers ONE question, narrowly:
    "How does Hint-AUC change when N (the number of $\alpha$ grid points)
     is varied, with the points distributed uniformly on [0, 1]?"

It also reports two reference rows for context:
  - the paper's *Original front-loaded* grid ($N = 8$)
  - the *Dense reference* ($N = 101$)
so that every uniform grid can be compared against both.

Inputs:
  per_ratio_summary.csv  (produced by B_build_summary.py)
                         — gives per-(sketch, ratio) means for 4 metrics.

Outputs (under --out_dir):
  vary_N_auc__mean.csv          sketch-averaged main table
  vary_N_auc__per_sketch.csv    per-sketch breakdown
  vary_N_auc__verification.txt  step-by-step intermediate values
                                (the α points, the interpolated metric values,
                                and the trapezoidal sum, for every grid×sketch
                                — so any reported number can be cross-checked)
  vary_N_auc__markdown.md       human-readable table
  vary_N_auc__latex.tex         LaTeX tabular ready for paste-in

Reproducibility check:
  By default the script uses N=5,8,11,21,101 with uniform spacing
  via np.linspace(0,1,N), plus the *Original front-loaded* paper grid
  $\{0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00\}$ as a reference row.
  All numbers in the supplementary table B.2 for the relevant rows
  should reproduce exactly when running with default arguments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["psnr", "ssim", "lpips", "dreamsim"]
SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}
PAPER_GRID = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]


# ─── core: trapezoidal AUC and grid construction ───────────────────────────
def trapz_auc(alphas, values):
    """Trapezoidal integral of `values` over the (sorted) `alphas` axis."""
    a = np.asarray(alphas, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(a)
    return float(np.trapz(v[order], a[order]))


def uniform_grid(N):
    """np.linspace(0, 1, N) deduplicated and rounded to 6 decimals."""
    return sorted(set(round(a, 6) for a in np.linspace(0.0, 1.0, N)))


def grid_collection(n_values, include_original=True, include_dense=True):
    """Returns ordered list of (display_name, grid_alphas).
    If N=101 appears in n_values, it is labelled as 'Dense reference (N=101)'
    rather than 'Uniform (N=101)' so we don't double-list the same grid."""
    grids = []
    if include_original:
        grids.append(("Original front-loaded (paper, N=8)", PAPER_GRID))
    has_101_in_n = 101 in n_values
    for N in n_values:
        label = "Dense reference (N=101)" if N == 101 else f"Uniform (N={N})"
        grids.append((label, uniform_grid(N)))
    if include_dense and not has_101_in_n:
        grids.append(("Dense reference (N=101)", uniform_grid(101)))
    return grids


# ─── pipeline ──────────────────────────────────────────────────────────────
def load_dense(summary_path):
    df = pd.read_csv(summary_path)
    df["ratio_f"] = df["ratio"].astype(float)
    df = df.sort_values(["sketch", "ratio_f"]).reset_index(drop=True)
    # sanity: every sketch should have 101 unique ratios
    counts = df.groupby("sketch")["ratio_f"].nunique()
    if not (counts == 101).all():
        raise ValueError(f"Expected 101 unique α per sketch, got\n{counts}")
    return df


def build_dense_lookup(df):
    """{sketch: {metric: (alphas_array, means_array)}} for fast interpolation."""
    out = {}
    for sk in sorted(df["sketch"].unique()):
        sub = df[df["sketch"] == sk].sort_values("ratio_f")
        xs = sub["ratio_f"].values
        out[int(sk)] = {m: (xs, sub[f"{m}_mean"].values) for m in METRICS}
    return out


def compute_one(grid_alphas, dense_for_sketch, metric):
    """Returns (interpolated_values, auc) for one (grid, sketch, metric)."""
    xs, ys_dense = dense_for_sketch[metric]
    grid_arr = np.array(grid_alphas, dtype=float)
    ys_at_grid = np.interp(grid_arr, xs, ys_dense)
    auc = trapz_auc(grid_arr, ys_at_grid)
    return ys_at_grid, auc


def run(n_values, summary_path, out_dir,
        include_original=True, include_dense=True):
    df = load_dense(summary_path)
    dense = build_dense_lookup(df)
    grids = grid_collection(n_values,
                            include_original=include_original,
                            include_dense=include_dense)

    # ─── per-sketch rows + verification trace ───────────────────────────
    per_sketch_rows = []
    verification_lines = []
    verification_lines.append("# Verification trace — Vary-N Hint-AUC computation")
    verification_lines.append("")
    verification_lines.append(f"Source CSV: {summary_path}")
    verification_lines.append(f"Sketches : {sorted(dense.keys())}")
    verification_lines.append(f"Metrics  : {METRICS}")
    verification_lines.append("")
    verification_lines.append("Each AUC below is obtained by")
    verification_lines.append("  (a) selecting the α grid,")
    verification_lines.append("  (b) reading metric values at those α "
                              "(linear interpolation of the dense 101-point curve),")
    verification_lines.append("  (c) trapezoidal integration: "
                              "AUC = Σᵢ (αᵢ₊₁ − αᵢ) (yᵢ + yᵢ₊₁) / 2.")
    verification_lines.append("")

    for grid_name, grid in grids:
        verification_lines.append(f"## {grid_name}")
        verification_lines.append(f"α grid (N={len(grid)}): {grid}")
        verification_lines.append("")
        for sk in sorted(dense.keys()):
            sk_name = SKETCH_NAMES.get(sk, "?")
            verification_lines.append(f"  sketch {sk} ({sk_name}):")
            for m in METRICS:
                ys_at_grid, auc = compute_one(grid, dense[sk], m)
                per_sketch_rows.append({
                    "grid": grid_name,
                    "N": len(grid),
                    "sketch": sk,
                    "sketch_name": sk_name,
                    "metric": m,
                    "auc": auc,
                })
                verification_lines.append(
                    f"    {m:>8s}: interp = ["
                    + ", ".join(f"{v:.4f}" for v in ys_at_grid)
                    + f"]  →  AUC = {auc:.6f}"
                )
            verification_lines.append("")
        verification_lines.append("")
    per_sketch_df = pd.DataFrame(per_sketch_rows)

    # ─── sketch-averaged main table ─────────────────────────────────────
    mean = (per_sketch_df
            .groupby(["grid", "N", "metric"], as_index=False)["auc"]
            .mean())
    main = mean.pivot_table(index=["grid", "N"], columns="metric",
                            values="auc").reset_index()
    main.columns.name = None

    grid_order = [g for g, _ in grids]
    main["__rank"] = main["grid"].map({g: i for i, g in enumerate(grid_order)})
    main.sort_values("__rank", inplace=True)
    main.drop(columns="__rank", inplace=True)
    main.reset_index(drop=True, inplace=True)

    # Δ vs paper grid
    paper_label = "Original front-loaded (paper, N=8)"
    if include_original:
        orig = main[main["grid"] == paper_label].iloc[0]
        for m in METRICS:
            main[f"d_{m}"] = main[m] - orig[m]

        def max_rel(row):
            return max(abs((row[m] - orig[m]) / orig[m]) if orig[m] else 0
                       for m in METRICS)
        main["max_rel_diff"] = main.apply(max_rel, axis=1)

    # ─── output files ───────────────────────────────────────────────────
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sketch_path = out_dir / "vary_N_auc__per_sketch.csv"
    main_path       = out_dir / "vary_N_auc__mean.csv"
    verif_path      = out_dir / "vary_N_auc__verification.txt"
    md_path         = out_dir / "vary_N_auc__markdown.md"
    tex_path        = out_dir / "vary_N_auc__latex.tex"

    per_sketch_df.to_csv(per_sketch_path, index=False)
    main.to_csv(main_path, index=False)
    verif_path.write_text("\n".join(verification_lines) + "\n")
    md_path.write_text(_to_markdown(main, include_original) + "\n")
    tex_path.write_text(_to_latex(main, include_original))

    return main, per_sketch_df, [per_sketch_path, main_path, verif_path,
                                  md_path, tex_path]


# ─── formatters ────────────────────────────────────────────────────────────
def _fmt(val, m):
    if m == "psnr":  return f"{val:.2f}"
    if m == "ssim":  return f"{val:.3f}"
    return f"{val:.4f}"

def _fmt_d(val, m):
    s = "+" if val >= 0 else ""
    if m == "psnr":  return f"{s}{val:.2f}"
    if m == "ssim":  return f"{s}{val:.3f}"
    return f"{s}{val:.4f}"


def _to_markdown(main, include_original):
    header = ["Grid", "N", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "DreamSim ↓"]
    if include_original:
        header += ["Δ PSNR", "Δ SSIM", "Δ LPIPS", "Δ DreamSim", "max |Δ|/orig"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for _, r in main.iterrows():
        row = [r["grid"], str(int(r["N"]))]
        for m in METRICS:
            row.append(_fmt(r[m], m))
        if include_original:
            for m in METRICS:
                row.append(_fmt_d(r[f"d_{m}"], m))
            row.append(f"{r['max_rel_diff']*100:.2f}%")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _to_latex(main, include_original):
    pre = r"""\begin{table}[t]
  \centering
  \caption{Sensitivity of Hint-AUC to the number of $\alpha$ grid points
  (uniform spacing). AUC is computed by linear interpolation of the dense
  101-point curve onto each grid and trapezoidal integration over $[0,1]$,
  averaged over the three sketch types.}
  \label{tab:S-R2-1-B-vary-N}
  \small
  \begin{tabular}{l c rrrr c}
    \toprule
    Grid & $N$ &
      PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & DreamSim $\downarrow$ &
      $\max\!\big|\Delta\big|/\mathrm{orig}$ \\
    \midrule
"""
    body = []
    for _, r in main.iterrows():
        line = (
            f"    {r['grid']:<40s} & {int(r['N']):>3d} & "
            f"{_fmt(r['psnr'],'psnr'):>6s} & "
            f"{_fmt(r['ssim'],'ssim'):>6s} & "
            f"{_fmt(r['lpips'],'lpips'):>7s} & "
            f"{_fmt(r['dreamsim'],'dreamsim'):>7s} & "
        )
        if include_original:
            line += f"{r['max_rel_diff']*100:>5.2f}\\% \\\\"
        else:
            line += "--- \\\\"
        body.append(line)
    post = r"""    \bottomrule
  \end{tabular}
\end{table}
"""
    return pre + "\n".join(body) + "\n" + post


# ─── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True,
                   help="per_ratio_summary.csv path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_values", default="5,8,11,21,101",
                   help="comma-separated list of N (default 5,8,11,21,101). "
                        "N=101 is automatically labelled 'Dense reference'.")
    p.add_argument("--no_original", action="store_true",
                   help="skip the paper's front-loaded N=8 reference row")
    p.add_argument("--no_dense", action="store_true",
                   help="skip the N=101 dense reference row")
    return p.parse_args()


def main():
    args = parse_args()
    n_values = [int(s) for s in args.n_values.split(",") if s.strip()]
    main_table, per_sketch, paths = run(
        n_values=n_values,
        summary_path=args.summary,
        out_dir=args.out_dir,
        include_original=not args.no_original,
        include_dense=not args.no_dense,
    )

    print("=== Vary-N Hint-AUC (sketch-averaged) ===")
    print(_to_markdown(main_table, include_original=not args.no_original))
    print()
    print("Files:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
