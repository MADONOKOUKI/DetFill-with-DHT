"""A-1  Bootstrap ranking stability (per-user resampling).

For each alpha and each method m, aggregate per-user #wins and #trials of m
(over both pairs m participates in). Bootstrap over users to obtain 95% CI
of win-rate, and to estimate how often the three-way ranking of methods
at that alpha changes under resampling. Addresses R1-1 / R1-2 directly.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
from ab_loader import load_all, ALPHAS, METHODS

OUT = HERE.parent / "output" / "A-1"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)
N_BOOT = 10000


def per_user_counts(df: pd.DataFrame):
    """Return dict[alpha][method] = (wins, trials) per user (arrays)."""
    users = sorted(df.user.unique())
    counts = {}
    for alpha in ALPHAS:
        sub = df[df.alpha == alpha]
        counts[alpha] = {}
        for m in METHODS:
            mask = (sub.method_A == m) | (sub.method_B == m)
            sb = sub[mask]
            wins = ((sb.method_A == m) & (sb.A_won == 1)) | (
                (sb.method_B == m) & (sb.A_won == 0)
            )
            sb = sb.assign(_win=wins.astype(int))
            agg = sb.groupby("user")["_win"].agg(["sum", "count"])
            w = np.array([agg.loc[u, "sum"] if u in agg.index else 0 for u in users])
            n = np.array([agg.loc[u, "count"] if u in agg.index else 0 for u in users])
            counts[alpha][m] = (w, n)
    return users, counts


def main():
    df = load_all()
    users, counts = per_user_counts(df)
    n_users = len(users)

    rows = []
    rank_dist_all = {}
    for alpha in ALPHAS:
        wr_point = {}
        boot_w = {m: None for m in METHODS}
        for m in METHODS:
            w, n = counts[alpha][m]
            wr_point[m] = w.sum() / n.sum()
        # Bootstrap: resample user indices once per iter, apply to all methods.
        idx = RNG.integers(0, n_users, size=(N_BOOT, n_users))
        rank_counter = {}
        for m in METHODS:
            w, n = counts[alpha][m]
            ws = w[idx].sum(axis=1)
            ns = n[idx].sum(axis=1)
            boot_w[m] = ws / np.maximum(ns, 1)
        for b in range(N_BOOT):
            order = tuple(sorted(METHODS, key=lambda m: -boot_w[m][b]))
            rank_counter[order] = rank_counter.get(order, 0) + 1
        for m in METHODS:
            arr = boot_w[m]
            rows.append(
                dict(
                    alpha=alpha, method=m,
                    point=float(wr_point[m]),
                    mean=float(arr.mean()),
                    lo=float(np.quantile(arr, 0.025)),
                    hi=float(np.quantile(arr, 0.975)),
                )
            )
        rank_dist_all[alpha] = [
            {"order": list(o), "count": c, "prob": c / N_BOOT}
            for o, c in sorted(rank_counter.items(), key=lambda kv: -kv[1])
        ]

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "winrate_ci.csv", index=False)
    with open(OUT / "rank_distribution.json", "w") as fh:
        json.dump(rank_dist_all, fh, indent=2)

    # overall ranking = mean across alphas (unweighted) using point values
    overall = {m: float(table[table.method == m]["point"].mean()) for m in METHODS}
    overall_top = max(overall, key=lambda m: overall[m])
    per_alpha_flip = {}
    for alpha in ALPHAS:
        flip = 0
        for b in range(N_BOOT):
            wrs = {m: boot_w[m][b] for m in METHODS}  # last boot_w is last alpha; redo per alpha
        # redo properly:
        # compute boot_w again for this alpha using the same RNG stream is expensive;
        # instead, re-derive from stored rank_dist_all.
        # top != overall_top counts:
        flip_prob = sum(
            entry["prob"] for entry in rank_dist_all[alpha] if entry["order"][0] != overall_top
        )
        per_alpha_flip[alpha] = flip_prob

    summary = {
        "overall_winrate": overall,
        "overall_top": overall_top,
        "prob_top_method_is_not_overall_top_at_alpha": per_alpha_flip,
    }
    with open(OUT / "rank_flip.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(table.to_string(index=False))
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
