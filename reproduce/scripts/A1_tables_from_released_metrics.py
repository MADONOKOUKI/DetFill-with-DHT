#!/usr/bin/env python3
"""Recompute the paper's Hint-AUC tables from the released per-ratio metric files.

Every table below is rebuilt from the raw per-ratio means in ``reproduce/expected/`` with the
same rule as the paper: Hint-AUC = trapezoidal integral of metric(alpha) over alpha in [0, 1]
(``hintauc.auc.trapz``), computed per sketch source and then reported as
mean +- sample SD (ddof=1) over the three sketch sources (XDoG, sketch simplification, SketchKeras).

  [1] Table II, DetFill scribble row (7 metrics)          <- alpha_grid/B_dense_curve/*per_ratio_summary*.csv
  [2] Supp. Table "alpha-grid sensitivity" (5 grids)      <- same dense sweep, re-integrated on each grid
  [3] Supp. Table "segmentation dependency" (3x3, 3 metr) <- segmenter_dependency/hauc_7m/*/per_ratio_summary.csv
  [4] Table III, DetFill scribble row (fixed random order)<- table3_scribble/hauc7_scribble.json
  [5] Supp. Table "additional training", Diffusart-retrain<- additional_training/diffusart_retrain_R3-2/hauc_runs_v3_*/hauc_summary.json (PSNR/SSIM/LPIPS)
  [6] Supp. Table "size-sort vs random" (10 seeds, N=300) <- seed_sensitivity/metrics_felz96/seed*_alpha_*/per_ratio_summary.csv
                                                             (+ seed_sensitivity/aggregated_sort.csv for the size-sort row)

Each recomputed cell is compared with the number printed in the paper (tolerance: half a unit of the
printed precision). Usage:  python reproduce/scripts/A1_tables_from_released_metrics.py [--expected DIR] [--out DIR]
Only numpy and pandas are needed.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
try:
    from hintauc.auc import trapz, DEFAULT_ALPHAS
except Exception:  # standalone fallback (identical arithmetic)
    DEFAULT_ALPHAS = (0.00, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00)
    def trapz(xs, ys):
        return float(sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0 for i in range(1, len(xs))))

METRICS = ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]
# sketch_type index in the DetFill loader: 0 = sketch simplification (pysimp), 1 = XDoG, 2 = SketchKeras
# (the archived summary files name 0 "XDoG" and 1 "pysimp" - a label swap in the old evaluation script;
# every published value is a mean/SD over the three sources, so the swap changes nothing).
SKETCH = {0: "type 0 (pysimp)", 1: "type 1 (XDoG)", 2: "type 2 (SketchKeras)"}
FRONT = list(DEFAULT_ALPHAS)
REPORT = []          # (table, row, metric, recomputed, paper, ok)


def tol(paper_str):
    """half a unit of the printed precision (+1%) e.g. '0.0148' -> 0.0000505"""
    d = len(paper_str.split(".")[1]) if "." in paper_str else 0
    return 0.505 * 10 ** (-d)


def check(table, row, metric, val, paper_str):
    ok = None
    if paper_str is not None:
        ok = abs(val - float(paper_str)) <= tol(paper_str)
    REPORT.append((table, row, metric, val, paper_str, ok))
    return ok


def fmt(v, m):
    return f"{v:.3f}" if m == "psnr" else f"{v:.4f}"


def curve_auc(df, metric, alphas):
    """Hint-AUC per sketch from a per_ratio_summary dataframe (ratio in [0,1]); linear interpolation
    onto `alphas` (exact when the grid point was measured)."""
    out = {}
    for sk, g in df.groupby("sketch"):
        g = g.sort_values("ratio")
        xs, ys = g["ratio"].to_numpy(float), g[f"{metric}_mean"].to_numpy(float)
        if xs[0] > 0 or xs[-1] < 1:
            raise ValueError(f"sketch {sk}: curve must span [0,1], got [{xs[0]}, {xs[-1]}]")
        out[int(sk)] = trapz(list(alphas), list(np.interp(alphas, xs, ys)))
    return out


def mean_sd(d):
    v = np.array(list(d.values()), float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def load_summary(paths):
    dfs = [pd.read_csv(p) for p in paths]
    df = dfs[0]
    for o in dfs[1:]:
        df = df.merge(o, on=["sketch", "sketch_name", "ratio", "n"], how="outer")
    return df


def row_table(name, rows, header, out_dir, fname):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    md = "\n".join(lines)
    print(f"\n### {name}\n{md}")
    with open(os.path.join(out_dir, fname), "w") as f:
        f.write(f"# {name}\n\n{md}\n")
    return md


# ---------------------------------------------------------------------------------------------
def table2_scribble(E, out_dir):
    paths = sorted(glob.glob(os.path.join(E, "alpha_grid", "B_dense_curve", "*per_ratio_summary*.csv")))
    df = load_summary(paths)
    paper = {"mse": ("0.015", "0.005"), "psnr": ("19.723", "1.41"), "ssim": ("0.724", "0.086"), "lpips": ("0.183", "0.047"),
             "openclip": ("0.954", "0.010"), "dino": ("0.951", "0.011"), "dreamsim": ("0.084", "0.021")}
    rows, sk_rows = [], []
    for m in METRICS:
        aucs = curve_auc(df, m, FRONT)
        mu, sd = mean_sd(aucs)
        ok = check("Table II (DetFill, scribble)", "mean", m, mu, paper[m][0])
        ok2 = check("Table II (DetFill, scribble)", "sd", m, sd, paper[m][1])
        rows.append([m, f"{fmt(mu, m)} ± {fmt(sd, m)}", f"{paper[m][0]}±{paper[m][1]}", "OK" if (ok and ok2) else "MISMATCH"])
        sk_rows.append([m] + [fmt(aucs[k], m) for k in (0, 1, 2)])
    row_table("Table II — DetFill (scribble hints), Hint-AUC over the front-loaded grid, mean ± SD over 3 sketch sources",
              rows, ["metric", "recomputed", "paper", "check"], out_dir, "table2_detfill_scribble.md")
    row_table("Table II — per-sketch Hint-AUC (DetFill scribble)", sk_rows, ["metric", SKETCH[0], SKETCH[1], SKETCH[2]],
              out_dir, "table2_detfill_scribble_per_sketch.md")
    return df


def alpha_grid_table(df, out_dir):
    def step_grid(step):
        g = list(np.round(np.arange(0.0, 1.0 + 1e-9, step), 6))
        if abs(g[-1] - 1.0) > 1e-9:
            g.append(1.0)
        return g
    grids = [("Uniform, Δα=0.02 (dense ref.)", step_grid(0.02)), ("Front-loaded (paper grid)", FRONT),
             ("Uniform, Δα=0.04", step_grid(0.04)), ("Uniform-like, nominal Δα=0.08", step_grid(0.08)),
             ("Uniform-like, nominal Δα=0.16", step_grid(0.16))]
    paper = {
        "Uniform, Δα=0.02 (dense ref.)": ["0.0148±0.0049", "19.779±1.420", "0.7254±0.0858", "0.1807±0.0466", "0.9549±0.0099", "0.9514±0.0112", "0.0823±0.0207"],
        "Front-loaded (paper grid)": ["0.0149±0.0050", "19.723±1.414", "0.7235±0.0860", "0.1829±0.0472", "0.9543±0.0101", "0.9509±0.0114", "0.0839±0.0212"],
        "Uniform, Δα=0.04": ["0.0154±0.0050", "19.745±1.416", "0.7246±0.0857", "0.1817±0.0465", "0.9546±0.0099", "0.9512±0.0112", "0.0833±0.0206"],
        "Uniform-like, nominal Δα=0.08": ["0.0168±0.0051", "19.653±1.408", "0.7226±0.0856", "0.1844±0.0461", "0.9537±0.0098", "0.9506±0.0112", "0.0864±0.0204"],
        "Uniform-like, nominal Δα=0.16": ["0.0199±0.0055", "19.410±1.389", "0.7174±0.0853", "0.1915±0.0456", "0.9513±0.0098", "0.9489±0.0112", "0.0945±0.0201"],
    }
    rows = []
    for name, g in grids:
        cells, allok = [], True
        for m, p in zip(METRICS, paper[name]):
            mu, sd = mean_sd(curve_auc(df, m, g))
            pm, ps = p.split("±")
            ok = check("Supp alpha-grid", name, m, mu, pm) and check("Supp alpha-grid", name, m + " sd", sd, ps)
            allok &= ok
            cells.append(f"{fmt(mu, m)}±{fmt(sd, m)}" + ("" if ok else f" (paper {p})"))
        rows.append([name, len(g)] + cells + ["OK" if allok else "MISMATCH"])
    row_table("Supp. Table — sensitivity of Hint-AUC to the α grid (DetFill scribble; mean ± SD over 3 sketch sources)",
              rows, ["α grid", "N"] + METRICS + ["check"], out_dir, "supp_alpha_grid_sensitivity.md")


def segdep_table(E, df_front, out_dir):
    combos = [("DanbooRegion", "Felzenszwalb", "danbooregion_model__on_felz"), ("DanbooRegion", "DanbooRegion", "danbooregion_model__on_danbooregion"),
              ("DanbooRegion", "SLIC", "danbooregion_model__on_slic"), ("SLIC", "Felzenszwalb", "slic_model__on_felz"),
              ("SLIC", "DanbooRegion", "slic_model__on_danbooregion"), ("SLIC", "SLIC", "slic_model__on_slic"),
              ("Felzenszwalb", "Felzenszwalb†", None), ("Felzenszwalb", "DanbooRegion", "proposed_model__on_danbooregion"),
              ("Felzenszwalb", "SLIC", "proposed_model__on_slic")]
    paper = {("DanbooRegion", "Felzenszwalb"): ["0.0160±0.0045", "0.1960±0.0469", "0.0943±0.0221"],
             ("DanbooRegion", "DanbooRegion"): ["0.0172±0.0049", "0.2045±0.0470", "0.0990±0.0223"],
             ("DanbooRegion", "SLIC"): ["0.0269±0.0044", "0.2756±0.0540", "0.1554±0.0331"],
             ("SLIC", "Felzenszwalb"): ["0.0203±0.0059", "0.2182±0.0445", "0.1012±0.0189"],
             ("SLIC", "DanbooRegion"): ["0.0209±0.0061", "0.2321±0.0487", "0.1104±0.0209"],
             ("SLIC", "SLIC"): ["0.0269±0.0063", "0.2370±0.0436", "0.1214±0.0214"],
             ("Felzenszwalb", "Felzenszwalb†"): ["0.0149±0.0050", "0.1829±0.0472", "0.0839±0.0212"],
             ("Felzenszwalb", "DanbooRegion"): ["0.0164±0.0051", "0.1999±0.0487", "0.0945±0.0225"],
             ("Felzenszwalb", "SLIC"): ["0.0243±0.0049", "0.2615±0.0563", "0.1397±0.0324"]}
    rows3, rows7 = [], []
    for tr, ev, d in combos:
        if d is None:
            df = df_front
        else:
            cands = glob.glob(os.path.join(E, "segmenter_dependency", "hauc_7m", d, "per_ratio_summary.csv")) + \
                    glob.glob(os.path.join(E, "segmenter_dependency", "hauc_7m", d, "auc", "per_ratio_summary.csv"))
            df = pd.read_csv(cands[0])
        cells3, cells7, allok = [], [], True
        for m in METRICS:
            mu, sd = mean_sd(curve_auc(df, m, FRONT))
            cells7.append(f"{fmt(mu, m)}±{fmt(sd, m)}")
            if m in ("mse", "lpips", "dreamsim"):
                p = paper[(tr, ev)][["mse", "lpips", "dreamsim"].index(m)]
                pm, ps = p.split("±")
                ok = check("Supp segmentation dependency", f"{tr}->{ev}", m, mu, pm) and \
                     check("Supp segmentation dependency", f"{tr}->{ev}", m + " sd", sd, ps)
                allok &= ok
                cells3.append(f"{mu:.4f}±{sd:.4f}" + ("" if ok else f" (paper {p})"))
        rows3.append([tr, ev] + cells3 + ["OK" if allok else "MISMATCH"])
        rows7.append([tr, ev] + cells7)
    row_table("Supp. Table — segmentation-dependency cross-evaluation (train seg. × eval seg.), MSE/LPIPS/DreamSim Hint-AUC, mean ± SD over 3 sketch sources",
              rows3, ["train seg.", "eval seg.", "MSE", "LPIPS", "DreamSim", "check"], out_dir, "supp_segmenter_dependency_3metric.md")
    row_table("Supp. — segmentation-dependency cross-evaluation, all 7 metrics", rows7, ["train seg.", "eval seg."] + METRICS,
              out_dir, "supp_segmenter_dependency_7metric.md")


def table3_scribble(E, out_dir):
    p = os.path.join(E, "table3_scribble", "hauc7_scribble.json")
    j = json.load(open(p))
    paper = {"mse": "0.021±0.007", "lpips": "0.194±0.045", "dreamsim": "0.091±0.020"}
    rows = []
    for m in METRICS:
        aucs = {k: j["per_sketch"][k]["hauc"][m] for k in ("sk0", "sk1", "sk2")}
        mu, sd = mean_sd(aucs)
        assert abs(mu - j["avg_over_sketches"][m]) < 1e-9 and abs(sd - j["std_over_sketches"][m]) < 1e-9
        note = ""
        if m in paper:
            pm, ps = paper[m].split("±")
            ok = check("Table III (DetFill, scribble)", "mean", m, mu, pm) and check("Table III (DetFill, scribble)", "sd", m, sd, ps)
            note = "OK" if ok else f"MISMATCH (paper {paper[m]})"
        rows.append([m, f"{fmt(mu, m)} ± {fmt(sd, m)}", paper.get(m, "(not printed)"), note])
    row_table("Table III — DetFill (scribble hints), fixed random region order (label order), mean ± SD over 3 sketch sources",
              rows, ["metric", "recomputed", "paper", "check"], out_dir, "table3_detfill_scribble.md")


def diffusart_retrain(E, out_dir):
    """Supp. 'additional training comparisons', Diffusart-retrain scribble row.

    The only per-ratio record of that run that was recovered is the PSNR/SSIM/LPIPS summary of the
    inference run (``hauc_runs_v3``: Diffusart trained with the deterministic hints, 200 epochs, scribble,
    3 sketch sources x 8 ratios x 3000 images). MSE/OpenCLIP/DINO/DreamSim of that row are not covered here."""
    p = os.path.join(E, "additional_training", "diffusart_retrain_R3-2", "hauc_runs_v3_scribble_psnr_ssim_lpips", "hauc_summary.json")
    paper = {"psnr": "19.457±1.06", "ssim": "0.717±0.066", "lpips": "0.177±0.031"}
    rows = []
    if not os.path.exists(p):
        rows.append(["scribble", "file missing", "", "", ""])
    else:
        j = json.load(open(p))
        assert [int(r) for r in j["ratios"]] == [int(a * 100) for a in FRONT]
        cells, allok = [], True
        for m in ("psnr", "ssim", "lpips"):
            aucs = {}
            for cell, per_ratio in j["cells"].items():
                xs = [int(r) / 100.0 for r in per_ratio]
                ys = [per_ratio[r][m] for r in per_ratio]
                order = np.argsort(xs)
                aucs[cell] = trapz([xs[i] for i in order], [ys[i] for i in order])
                assert abs(aucs[cell] - j["hauc"][cell][m]) < 1e-9      # matches the stored per-sketch Hint-AUC
            mu, sd = mean_sd(aucs)
            pm, ps = paper[m].split("±")
            ok = check("Supp additional training (Diffusart-retrain)", "scribble", m, mu, pm) and \
                 check("Supp additional training (Diffusart-retrain)", "scribble", m + " sd", sd, ps)
            allok &= ok
            cells.append(f"{fmt(mu, m)}±{fmt(sd, m)}" + ("" if ok else f" (paper {paper[m]})"))
        rows.append(["scribble"] + cells + ["OK" if allok else "MISMATCH (see NOTE.md)"])
    row_table("Supp. Table — Diffusart retrained with deterministic hints (Diffusart-retrain), scribble; PSNR/SSIM/LPIPS Hint-AUC, mean ± SD over 3 sketch sources",
              rows, ["hint", "psnr", "ssim", "lpips", "check"], out_dir, "supp_diffusart_retrain.md")


def seed_table(E, out_dir):
    root = os.path.join(E, "seed_sensitivity", "metrics_felz96")
    alphas_pct = [0, 1, 3, 5, 10, 25, 50, 100]
    per_seed = {}   # seed -> {alpha_pct: {metric: mean}}
    for d in sorted(glob.glob(os.path.join(root, "seed*_alpha_*"))):
        tag = os.path.basename(d)
        seed = int(tag.split("_")[0][4:]); a = int(tag.split("_alpha_")[1])
        r = pd.read_csv(os.path.join(d, "per_ratio_summary.csv")).iloc[0]
        assert int(r["n"]) == 300 and int(r["sketch"]) == 2, tag
        per_seed.setdefault(seed, {})[a] = {m: float(r[f"{m}_mean"]) for m in METRICS}
    seeds = sorted(per_seed)
    # alpha 0 % (no hint) and 100 % (all hints) do not depend on the seed: measured once (seed 1)
    for s in seeds:
        for a in (0, 100):
            per_seed[s].setdefault(a, per_seed[1][a])
    paper = {  # size-sort row, random row (mean±sd), HAUC sort, HAUC random
        "mse": (["0.1003", "0.0501", "0.0310", "0.0239", "0.0166", "0.0113", "0.0089", "0.0075"],
                ["0.1003", "0.0706±0.0017", "0.0535±0.0011", "0.0439±0.0006", "0.0316±0.0005", "0.0183±0.0003", "0.0117±0.0002", "0.0075"], "0.0119", "0.0172±0.0001"),
        "lpips": (["0.4405", "0.3315", "0.2673", "0.2357", "0.1954", "0.1496", "0.1206", "0.0991"],
                  ["0.4405", "0.3592±0.0028", "0.3062±0.0021", "0.2764±0.0012", "0.2322±0.0010", "0.1721±0.0009", "0.1316±0.0005", "0.0991"], "0.1402", "0.1551±0.0004"),
        "dreamsim": (["0.3463", "0.2518", "0.1709", "0.1346", "0.0969", "0.0652", "0.0499", "0.0399"],
                     ["0.3463", "0.2575±0.0022", "0.1934±0.0018", "0.1632±0.0012", "0.1222±0.0010", "0.0780±0.0010", "0.0546±0.0002", "0.0399"], "0.0651", "0.0734±0.0003"),
        "psnr": (["10.43", "13.74", "15.81", "16.96", "18.57", "20.34", "21.43", "22.30"],
                 ["10.43", "12.27±0.08", "13.56±0.07", "14.41±0.05", "15.85±0.04", "18.29±0.05", "20.32±0.03", "22.30"], "20.71", "19.45±0.02"),
        "ssim": (["0.5317", "0.6213", "0.6691", "0.6940", "0.7278", "0.7705", "0.8015", "0.8284"],
                 ["0.5317", "0.5812±0.0025", "0.6206±0.0020", "0.6456±0.0013", "0.6864±0.0008", "0.7477±0.0010", "0.7921±0.0006", "0.8284"], "0.7842", "0.7687±0.0004"),
        "openclip": (["0.8767", "0.9050", "0.9286", "0.9397", "0.9518", "0.9629", "0.9686", "0.9729"],
                     ["0.8767", "0.9015±0.0013", "0.9209±0.0008", "0.9303±0.0008", "0.9434±0.0007", "0.9584±0.0006", "0.9670±0.0003", "0.9729"], "0.9636", "0.9608±0.0003"),
        "dino": (["0.8992", "0.9229", "0.9367", "0.9434", "0.9521", "0.9612", "0.9664", "0.9709"],
                 ["0.8992", "0.9122±0.0012", "0.9231±0.0008", "0.9304±0.0007", "0.9411±0.0005", "0.9554±0.0007", "0.9646±0.0003", "0.9709"], "0.9627", "0.9588±0.0003"),
    }
    sort_csv = os.path.join(E, "seed_sensitivity", "aggregated_sort.csv")
    sort_vals = None
    if os.path.exists(sort_csv):        # long format: alpha_pct, metric, n, mean, std  (N=300, sketchkeras)
        sdf = pd.read_csv(sort_csv)
        sort_vals = {m: {int(r.alpha_pct): float(r["mean"]) for _, r in sdf[sdf.metric == m].iterrows()} for m in METRICS}
    agg_csv = os.path.join(E, "seed_sensitivity", "aggregated_n300_10seeds_96ch.csv")
    if os.path.exists(agg_csv):         # the aggregation used for the paper figure; must equal the per-seed files
        adf = pd.read_csv(agg_csv)
        for _, r in adf.iterrows():
            v = per_seed[int(r.seed)][int(r.alpha_pct)][r.metric]
            assert abs(v - float(r["mean"])) < 1e-9, ("aggregated csv differs from per-seed files", r.seed, r.alpha_pct, r.metric)
        print(f"[seed table] aggregated_n300_10seeds_96ch.csv is consistent with the {len(seeds)} per-seed summaries")
    rows = []
    for m in METRICS:
        dec = 2 if m == "psnr" else 4
        f = lambda v: f"{v:.{dec}f}"
        # random: per-alpha mean ± SD over seeds; HAUC per seed then mean ± SD
        vals = np.array([[per_seed[s][a][m] for a in alphas_pct] for s in seeds])   # seeds x 8
        mu, sd = vals.mean(0), vals.std(0, ddof=1)
        haucs = np.array([trapz([a / 100 for a in alphas_pct], list(vals[i])) for i in range(len(seeds))])
        cells, allok = [], True
        p_sort, p_rand, p_hs, p_hr = paper[m]
        for j, a in enumerate(alphas_pct):
            c = f(mu[j]) if a in (0, 100) else f"{f(mu[j])}±{f(sd[j])}"
            if p_rand is not None:
                pm = p_rand[j].split("±")[0]
                ok = check("Supp seed table (random)", m, f"alpha={a}", mu[j], pm)
                if "±" in p_rand[j]:
                    ok = check("Supp seed table (random)", m, f"alpha={a} sd", sd[j], p_rand[j].split("±")[1]) and ok
                allok &= ok
                if not ok:
                    c += f" (paper {p_rand[j]})"
            cells.append(c)
        hr = f"{f(haucs.mean())}±{f(haucs.std(ddof=1))}"
        if p_hr is not None:
            ok = check("Supp seed table (random)", m, "HAUC", haucs.mean(), p_hr.split("±")[0]) and \
                 check("Supp seed table (random)", m, "HAUC sd", haucs.std(ddof=1), p_hr.split("±")[1])
            allok &= ok
            if not ok:
                hr += f" (paper {p_hr})"
        rows.append([m, f"random ({len(seeds)} seeds)"] + cells + [hr, "OK" if allok else ("MISMATCH" if p_rand else "n/a")])
        # size-sort (deterministic) row
        if sort_vals is not None:
            svals = [sort_vals[m][a] for a in alphas_pct]
            hs = trapz([a / 100 for a in alphas_pct], svals)
            cells, allok = [], True
            for j, a in enumerate(alphas_pct):
                c = f(svals[j])
                if p_sort is not None:
                    ok = check("Supp seed table (size-sort)", m, f"alpha={a}", svals[j], p_sort[j]); allok &= ok
                    if not ok:
                        c += f" (paper {p_sort[j]})"
                cells.append(c)
            hc = f(hs)
            if p_hs is not None:
                ok = check("Supp seed table (size-sort)", m, "HAUC", hs, p_hs); allok &= ok
                if not ok:
                    hc += f" (paper {p_hs})"
            rows.append([m, "size-sort (proposed)"] + cells + [hc, "OK" if allok else ("MISMATCH" if p_sort else "n/a")])
        else:
            rows.append([m, "size-sort (proposed)"] + ["(aggregated_sort.csv not found)"] * 8 + ["", ""])
    row_table("Supp. Table — size-sort vs. random region selection (DetFill 96ch scribble, SketchKeras line art, N=300 test images); random = mean ± SD over hint-sampling seeds",
              rows, ["metric", "selection"] + [f"{a}%" for a in alphas_pct] + ["HAUC", "check"], out_dir, "supp_seed_sort_vs_random.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default=os.path.join(HERE, "..", "expected"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "output", "tables"))
    a = ap.parse_args()
    E, out = os.path.abspath(a.expected), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    df = table2_scribble(E, out)
    alpha_grid_table(df, out)
    segdep_table(E, df, out)
    table3_scribble(E, out)
    diffusart_retrain(E, out)
    seed_table(E, out)
    rep = pd.DataFrame(REPORT, columns=["table", "row", "metric", "recomputed", "paper", "match"])
    rep.to_csv(os.path.join(out, "check_report.csv"), index=False)
    n_ok = int(rep["match"].fillna(False).astype(bool).sum()); n_cmp = int(rep["match"].notna().sum())
    print(f"\n==== {n_ok}/{n_cmp} compared cells match the paper (tolerance: half unit of printed precision). "
          f"Details: {os.path.join(out, 'check_report.csv')}")
    bad = rep[rep["match"] == False]
    if len(bad):
        print(bad.to_string())
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
