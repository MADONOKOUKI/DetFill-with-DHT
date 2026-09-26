"""A-3  Bradley-Terry latent-quality estimation per alpha with bootstrap CI.

At each alpha, fit log-odds latent scores s_m such that
    P(m beats n) = sigmoid(s_m - s_n).
Scores identifiable up to a constant; we fix s_{painttorch} = 0.
CIs via user-level bootstrap (5000 reps).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common"))
from ab_loader import load_all, ALPHAS, METHODS

OUT = HERE.parent / "output" / "A-3"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
N_BOOT = 5000
ANCHOR = "painttorch"


def fit_bt(df_alpha, methods=METHODS, anchor=ANCHOR):
    """Return {method: score}. anchor score fixed at 0."""
    others = [m for m in methods if m != anchor]
    idx = {m: i for i, m in enumerate(others)}

    ks = {}
    ns = {}
    for (mA, mB), sub in df_alpha.groupby(["method_A", "method_B"]):
        ks[(mA, mB)] = int(sub.A_won.sum())
        ns[(mA, mB)] = int(len(sub))

    def score(m, theta):
        if m == anchor:
            return 0.0
        return theta[idx[m]]

    def nll(theta):
        ll = 0.0
        for (mA, mB), n in ns.items():
            k = ks[(mA, mB)]
            diff = score(mA, theta) - score(mB, theta)
            # log-sigmoid w/ stability
            ll += k * (-np.logaddexp(0.0, -diff)) + (n - k) * (-np.logaddexp(0.0, diff))
        return -ll

    x0 = np.zeros(len(others))
    res = minimize(nll, x0, method="BFGS")
    out = {anchor: 0.0}
    for m in others:
        out[m] = float(res.x[idx[m]])
    return out


def main():
    df = load_all()
    users = sorted(df.user.unique())
    n_users = len(users)
    user_df = {u: df[df.user == u] for u in users}

    # Point estimates
    point = {a: fit_bt(df[df.alpha == a]) for a in ALPHAS}

    # Bootstrap
    boot = {a: {m: [] for m in METHODS} for a in ALPHAS}
    for b in range(N_BOOT):
        picks = RNG.integers(0, n_users, size=n_users)
        sub_b = pd.concat([user_df[users[i]] for i in picks], ignore_index=True)
        for a in ALPHAS:
            s = fit_bt(sub_b[sub_b.alpha == a])
            for m in METHODS:
                boot[a][m].append(s[m])

    rows = []
    for a in ALPHAS:
        for m in METHODS:
            arr = np.array(boot[a][m])
            rows.append(dict(alpha=a, method=m,
                             point=point[a][m],
                             mean=float(arr.mean()),
                             lo=float(np.quantile(arr, 0.025)),
                             hi=float(np.quantile(arr, 0.975))))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "bt_scores_ci.csv", index=False)
    print(tbl.to_string(index=False))

    # significance: (proposed - diffusart) > 0 ?
    sig = {}
    for a in ALPHAS:
        pp = np.array(boot[a]["proposed"])
        dd = np.array(boot[a]["diffusart"])
        diff = pp - dd
        sig[a] = {
            "mean_diff": float(diff.mean()),
            "ci_lo": float(np.quantile(diff, 0.025)),
            "ci_hi": float(np.quantile(diff, 0.975)),
            "p_gt_0": float((diff > 0).mean()),
            "p_lt_0": float((diff < 0).mean()),
        }
    with open(OUT / "proposed_vs_diffusart_bt_diff.json", "w") as fh:
        json.dump(sig, fh, indent=2)
    print("\nproposed - diffusart latent score diff, per alpha:")
    print(json.dumps(sig, indent=2))


if __name__ == "__main__":
    main()
