"""A-5  Alpha-grid sensitivity of the AUC-style summary.

R2-1 says the paper's grid {0, .01, .03, .05, .10, .25, .50, 1.00} is
front-loaded and may skew HAUC. We recompute a human win-rate AUC under:
  (a) the paper grid (linear x-axis)
  (b) uniform grid {0, .05, .10, ..., 1.0}  — interpolation
  (c) log-spaced grid
  (d) uniform-index grid (treat alphas as ordinal)
  (e) back-loaded grid emphasizing alpha >= 0.25
Compare final method rankings & AUC gap (proposed - diffusart).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
from ab_loader import load_all, ALPHAS, METHODS

OUT = HERE.parent / "output" / "A-5"; OUT.mkdir(parents=True, exist_ok=True)


def winrate(df, m, a):
    sub = df[(df.alpha == a) & ((df.method_A == m) | (df.method_B == m))]
    if len(sub) == 0:
        return np.nan
    wins = ((sub.method_A == m) & (sub.A_won == 1)) | (
        (sub.method_B == m) & (sub.A_won == 0)
    )
    return wins.mean()


def interp_curve(wrs, alphas, x):
    """Piecewise-linear interpolate win-rate at x from known (alpha, wr)."""
    return np.interp(x, alphas, wrs)


def auc(y, x):
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def main():
    df = load_all()
    alphas_frac = np.array(ALPHAS, dtype=float) / 100.0
    curves = {m: np.array([winrate(df, m, a) for a in ALPHAS]) for m in METHODS}

    grids = {
        "paper_front_loaded": alphas_frac,
        "uniform_index": np.arange(len(ALPHAS), dtype=float),
        "uniform_linear_0_to_1_step_0.05": np.arange(0, 1.0001, 0.05),
        "log_spaced": np.r_[0.0, np.logspace(-2, 0, 20)],
        "back_loaded": np.array([0.0, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]),
        "uniform_linear_0_to_1_step_0.1": np.arange(0, 1.0001, 0.1),
        "dense_low": np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]),
    }

    rows = []
    rank_table = {}
    for gname, g in grids.items():
        wrs_interp = {}
        for m in METHODS:
            if gname == "uniform_index":
                # use index->winrate directly, no interp
                wrs_interp[m] = curves[m]
            else:
                wrs_interp[m] = interp_curve(curves[m], alphas_frac, g)
        aucs = {m: auc(wrs_interp[m], g) for m in METHODS}
        order = sorted(METHODS, key=lambda m: -aucs[m])
        rank_table[gname] = dict(
            order=order,
            aucs=aucs,
            prop_minus_diff=aucs["proposed"] - aucs["diffusart"],
            prop_minus_paint=aucs["proposed"] - aucs["painttorch"],
        )
        for m in METHODS:
            rows.append(dict(grid=gname, method=m, auc=aucs[m]))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "auc_per_grid.csv", index=False)
    with open(OUT / "ranking_per_grid.json", "w") as fh:
        json.dump(rank_table, fh, indent=2)

    print(tbl.pivot(index="grid", columns="method", values="auc").to_string())
    print("\n=== ranking per grid ===")
    print(json.dumps(rank_table, indent=2))

    # sensitivity: does the top method ever change? does prop-diff gap swing?
    tops = {gname: v["order"][0] for gname, v in rank_table.items()}
    gaps_diff = {gname: v["prop_minus_diff"] for gname, v in rank_table.items()}
    summary = {
        "tops_per_grid": tops,
        "top_changes": len(set(tops.values())),
        "proposed_minus_diffusart_range": [min(gaps_diff.values()), max(gaps_diff.values())],
    }
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
