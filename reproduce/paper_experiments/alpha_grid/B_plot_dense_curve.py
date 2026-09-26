#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_plot_dense_curve.py — R2-1 dense hint-ratio curve plotter.

Reads per_ratio_summary.csv produced by B_eval_dense_curve.py and produces
a 2x2 panel figure (PSNR / SSIM / LPIPS / DreamSim) with one line per sketch
type, ±1σ shaded band per line, and the paper's 8-point α grid marked as
vertical dotted lines.

Usage:
  python B_plot_dense_curve.py \
      --summary /path/to/per_ratio_summary.csv \
      --out_pdf /path/to/dense_curve_2x2.pdf \
      [--out_png /path/to/dense_curve_2x2.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style ──────────────────────────────────────────────────────────────────
SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}
SKETCH_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}  # matplotlib tab10
SKETCH_MARKERS = {0: "o", 1: "s", 2: "^"}

# Paper's α grid (front-loaded, used in original Hint-AUC)
PAPER_GRID = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]

# Panel layout: (metric_name_in_csv, display_label, direction)
PANELS = [
    ("psnr",      "PSNR (dB)",     "↑"),
    ("ssim",      "SSIM",          "↑"),
    ("lpips",     "LPIPS",         "↓"),
    ("dreamsim",  "DreamSim",      "↓"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, required=True,
                   help="per_ratio_summary.csv path")
    p.add_argument("--out_pdf", type=str, required=True)
    p.add_argument("--out_png", type=str, default="",
                   help="optional PNG output for quick preview")
    p.add_argument("--show_band", action="store_true", default=True,
                   help="show ±1σ band (default on)")
    p.add_argument("--no_band", dest="show_band", action="store_false",
                   help="disable ±1σ band")
    p.add_argument("--show_paper_grid", action="store_true", default=True)
    p.add_argument("--no_paper_grid", dest="show_paper_grid", action="store_false")
    p.add_argument("--figsize", nargs=2, type=float, default=[10.0, 7.5],
                   help="figure size in inches (default 10x7.5)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_summary(csv_path):
    df = pd.read_csv(csv_path)
    df["ratio_f"] = df["ratio"].astype(float)
    df = df.sort_values(["sketch", "ratio_f"]).reset_index(drop=True)
    return df


def plot_metric(ax, df, metric_key, label, direction, show_band, show_paper_grid):
    """Plot one panel: x = ratio, y = metric_mean, lines per sketch."""
    for sk in sorted(df["sketch"].unique()):
        sub = df[df["sketch"] == sk]
        x = sub["ratio_f"].values
        mean = sub[f"{metric_key}_mean"].values
        std  = sub[f"{metric_key}_std"].values

        color  = SKETCH_COLORS[sk]
        marker = SKETCH_MARKERS[sk]
        name   = SKETCH_NAMES[sk]

        if show_band:
            ax.fill_between(x, mean - std, mean + std,
                            alpha=0.12, color=color, linewidth=0)

        ax.plot(x, mean, color=color, linewidth=1.5, alpha=0.95,
                label=name, marker=None)

        # marker only on paper grid points (so points read as "here are the
        # paper's 8 sample sites" without cluttering the dense curve)
        for pg in PAPER_GRID:
            idx = np.argmin(np.abs(x - pg))
            if abs(x[idx] - pg) < 1e-6:
                ax.scatter([x[idx]], [mean[idx]],
                           s=22, color=color, marker=marker,
                           edgecolors="white", linewidths=0.6, zorder=4)

    # paper grid as vertical dotted lines
    if show_paper_grid:
        for pg in PAPER_GRID:
            ax.axvline(pg, color="gray", linestyle=":", linewidth=0.6,
                       alpha=0.5, zorder=1)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("hint ratio α", fontsize=10)
    ax.set_ylabel(f"{label} {direction}", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)


def main():
    args = parse_args()
    df = load_summary(args.summary)
    print(f"[load] {len(df)} rows from {args.summary}")
    print(f"[load] sketches: {sorted(df.sketch.unique())}, "
          f"ratios: {df['ratio_f'].min():.2f} .. {df['ratio_f'].max():.2f}")

    fig, axes = plt.subplots(2, 2, figsize=args.figsize, dpi=args.dpi)
    axes = axes.flatten()

    for ax, (metric, label, direction) in zip(axes, PANELS):
        plot_metric(ax, df, metric, label, direction,
                    show_band=args.show_band,
                    show_paper_grid=args.show_paper_grid)

    # single shared legend at figure level (top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.99),
               frameon=False, fontsize=11)

    fig.suptitle("", y=0.97)  # placeholder; legend takes that space
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # annotate paper grid in figure footer
    if args.show_paper_grid:
        pg_str = ", ".join(f"{g:g}" for g in PAPER_GRID)
        fig.text(0.5, 0.005,
                 f"vertical dotted lines = paper's α grid: {{{pg_str}}}",
                 ha="center", fontsize=8, color="gray")

    out_pdf = Path(args.out_pdf).expanduser().resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[save] {out_pdf}")

    if args.out_png:
        out_png = Path(args.out_png).expanduser().resolve()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        print(f"[save] {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    main()
