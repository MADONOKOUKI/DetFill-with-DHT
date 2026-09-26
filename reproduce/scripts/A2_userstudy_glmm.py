#!/usr/bin/env python3
"""Reproduce the user-study results (main paper Tables IV and V and the GLMM statistics of Sec. VII)
from the released trial data ``reproduce/data/userstudy/glmm_trials.csv``.

Data: 32 participants x 192 forced-choice trials = 6,144 rows (anonymised participant ids ``sid``).
Columns: sid, trialIndex, trialId, imageId, ratio (hint ratio in %), pairIdx, leftMethod, rightMethod,
winnerMethod, loserMethod, responseTimeMs.  Methods: proposed (DetFill), painttorch, diffusart, colordiffv2.

Outputs (written to --out_dir):
  table4_pairwise_preference.csv   row method preferred over column method (%), Table IV
  table5_detfill_by_ratio.csv      DetFill preferred over each baseline per hint ratio (%), Table V
  glmm_summary.txt                 logistic mixed model (random intercepts: participant, image), Sec. VII
The GLMM follows Evaluation_paper/glmm_userstudy_analysis.py (statsmodels BinomialBayesMixedGLM, variational
Bayes); the variational fit is deterministic but its z/chi-square values can differ in the last digits between
statsmodels versions. Tables IV/V are exact counts and must match the paper to the last digit.

Usage:  python reproduce/scripts/A2_userstudy_glmm.py [--csv ...] [--out_dir ...] [--skip_glmm]
"""
import argparse, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data", "userstudy", "glmm_trials.csv")
METHODS = ["proposed", "painttorch", "diffusart", "colordiffv2"]
LABEL = {"proposed": "DetFill", "painttorch": "PaintsTorch", "diffusart": "Diffusart", "colordiffv2": "ColDiffv2"}
RATIOS = [0, 1, 3, 5, 10, 25, 50, 100]
# values printed in the paper (for the automatic check)
PAPER_TABLE4 = {("proposed", "painttorch"): 85.0, ("proposed", "diffusart"): 67.4, ("proposed", "colordiffv2"): 67.8,
                ("diffusart", "painttorch"): 79.2, ("colordiffv2", "painttorch"): 75.2, ("colordiffv2", "diffusart"): 55.1}
PAPER_TABLE5 = {"painttorch": [33.6, 83.6, 85.9, 89.8, 96.9, 94.5, 96.1, 99.2],
                "diffusart": [19.5, 54.7, 53.1, 78.9, 71.1, 79.7, 90.6, 91.4],
                "colordiffv2": [64.8, 55.5, 68.0, 64.1, 64.1, 63.3, 85.9, 76.6]}


def pref_rate(df, a, b):
    """Percentage of trials between a and b in which a was chosen."""
    m = ((df.leftMethod == a) & (df.rightMethod == b)) | ((df.leftMethod == b) & (df.rightMethod == a))
    sub = df[m]
    return 100.0 * (sub.winnerMethod == a).mean(), len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "output", "userstudy"))
    ap.add_argument("--skip_glmm", action="store_true", help="only the count tables (no statsmodels)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"trials: {len(df)}  participants: {df.sid.nunique()}  images: {df.imageId.nunique()}  ratios: {sorted(df.ratio.unique())}")

    # ---- Table IV: pairwise preference (row over column) ----
    rows = []
    ok = True
    for a in METHODS:
        r = {"method": LABEL[a]}
        for b in METHODS:
            if a == b:
                r[LABEL[b]] = "-"
                continue
            p, n = pref_rate(df, a, b)
            r[LABEL[b]] = f"{p:.1f}"
            if (a, b) in PAPER_TABLE4:
                flag = abs(p - PAPER_TABLE4[(a, b)]) < 0.05
                ok &= flag
                print(f"Table IV  {LABEL[a]:>11} > {LABEL[b]:<11}: {p:5.1f}%  (n={n}, paper {PAPER_TABLE4[(a, b)]})  {'OK' if flag else 'MISMATCH'}")
        rows.append(r)
    t4 = pd.DataFrame(rows).set_index("method")
    t4.to_csv(os.path.join(args.out_dir, "table4_pairwise_preference.csv"))
    print("\nTable IV (row preferred over column, %):\n" + t4.to_string())

    # ---- Table V: DetFill vs each baseline per hint ratio ----
    t5 = {}
    for b in ["painttorch", "diffusart", "colordiffv2"]:
        vals = []
        for r in RATIOS:
            p, n = pref_rate(df[df.ratio == r], "proposed", b)
            vals.append(p)
        t5[LABEL[b]] = vals
        flags = [abs(v - e) < 0.05 for v, e in zip(vals, PAPER_TABLE5[b])]
        ok &= all(flags)
        print(f"Table V   DetFill > {LABEL[b]:<11}: " + " ".join(f"{v:5.1f}" for v in vals) + f"   {'OK' if all(flags) else 'MISMATCH ' + str(flags)}")
    t5 = pd.DataFrame(t5, index=[f"{r}%" for r in RATIOS]).T
    t5.to_csv(os.path.join(args.out_dir, "table5_detfill_by_ratio.csv"))
    print("\nTable V (DetFill preferred, % per hint ratio):\n" + t5.to_string())
    print("\nCOUNT TABLES:", "all values match the paper" if ok else "SOME VALUES DIFFER FROM THE PAPER")

    if args.skip_glmm:
        return
    # ---- GLMM (Sec. VII): logistic mixed model with random intercepts for participant and image ----
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = df[(df.leftMethod == "proposed") | (df.rightMethod == "proposed")].copy()
    d["choice_target"] = (d.winnerMethod == "proposed").astype(int)
    d["opponent"] = np.where(d.leftMethod == "proposed", d.rightMethod, d.leftMethod)
    vc = {"sid_re": "0 + C(sid)", "img_re": "0 + C(imageId)"}
    lines = []
    m0 = BinomialBayesMixedGLM.from_formula("choice_target ~ 1", vc_formulas=vc, data=d)
    r0 = m0.fit_vb(minim_opts={"maxiter": 500})
    b0, s0 = float(r0.fe_mean[0]), float(r0.fe_sd[0]); z0 = b0 / s0
    p0 = 2 * (1 - 0.5 * (1 + math.erf(abs(z0) / math.sqrt(2))))
    lines.append(f"[Model A] intercept only: beta0 = {b0:.4f} +- {s0:.4f}, z = {z0:.2f}, two-sided p ~ {p0:.3g}; "
                 f"P(DetFill chosen) = {1/(1+math.exp(-b0)):.3f}   (paper: z = 30.18, p < 1e-16)")
    m1 = BinomialBayesMixedGLM.from_formula("choice_target ~ C(opponent, Sum) * C(ratio, Sum)", vc_formulas=vc, data=d)
    r1 = m1.fit_vb(minim_opts={"maxiter": 800})
    names = list(m1.exog_names); z = np.asarray(r1.fe_mean) / np.asarray(r1.fe_sd)
    from scipy.stats import chi2
    def wald(mask, label, paper):
        stat = float(np.sum(z[mask] ** 2)); dof = int(mask.sum())
        lines.append(f"[Model B] {label}: diag-Wald chi2({dof}) = {stat:.2f}, p ~ {chi2.sf(stat, dof):.3g}   (paper: {paper})")
    names_arr = np.array(names)
    wald(np.array([n.startswith("C(ratio") for n in names]), "hint-ratio main effect", "chi2(7) = 618.33, p = 2.75e-129")
    wald(np.array([":C(ratio" in n for n in names]), "opponent x ratio interaction", "chi2(14) = 324.04, p = 1.12e-60")
    txt = "\n".join(lines) + "\n\n" + str(r1.summary())
    open(os.path.join(args.out_dir, "glmm_summary.txt"), "w").write(txt)
    print("\n" + "\n".join(lines))
    print(f"\nwritten: {args.out_dir}")


if __name__ == "__main__":
    main()
