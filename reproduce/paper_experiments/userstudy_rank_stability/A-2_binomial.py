"""A-2  Per-alpha statistical significance of pairwise preference.

For each (pair, alpha), run an exact binomial two-sided test that the
population preference is 50-50. Report point win-rate, N, p-value,
Wilson 95% CI, and a Holm-Bonferroni corrected p across all 24 cells.

This addresses R1-1 ("demonstrating instability is statistically
significant") and R1-2 ("confidence intervals or hypothesis testing").
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
from ab_loader import load_all, ALPHAS, METHODS

OUT = HERE.parent / "output" / "A-2"; OUT.mkdir(parents=True, exist_ok=True)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def main():
    df = load_all()
    rows = []
    for pair, sub_pair in df.groupby("pair"):
        mA, mB = pair.split("_vs_")
        for alpha in ALPHAS:
            sub = sub_pair[sub_pair.alpha == alpha]
            k = int(sub.A_won.sum())
            n = int(len(sub))
            wr = k / n if n else np.nan
            # two-sided exact binomial
            res = stats.binomtest(k, n, p=0.5, alternative="two-sided")
            lo, hi = wilson_ci(k, n)
            rows.append(
                dict(
                    alpha=alpha,
                    pair=pair,
                    method_A=mA,
                    method_B=mB,
                    k=k,
                    n=n,
                    winrate_A=wr,
                    wilson_lo=lo,
                    wilson_hi=hi,
                    p_value=res.pvalue,
                )
            )
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_value"].values)
    t["significant_0.05_holm"] = t["p_holm"] < 0.05
    t.to_csv(OUT / "per_pair_alpha_binomial.csv", index=False)
    print(t.to_string(index=False))

    # alpha-level ranking table of method (winner) and significance
    summary = {}
    for alpha in ALPHAS:
        rank = {}
        for m in METHODS:
            pair_rows = t[(t.alpha == alpha) & ((t.method_A == m) | (t.method_B == m))]
            wins = 0; total = 0
            for _, r in pair_rows.iterrows():
                if r.method_A == m:
                    wins += r.k; total += r.n
                else:
                    wins += r.n - r.k; total += r.n
            rank[m] = {"winrate": wins / total if total else None,
                       "k": wins, "n": total}
        order = sorted(rank, key=lambda m: -rank[m]["winrate"])
        summary[alpha] = {"ranking": order, "winrates": rank}
    with open(OUT / "alpha_level_ranking.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
