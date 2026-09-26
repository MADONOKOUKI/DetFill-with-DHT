#!/usr/bin/env python3
"""Evaluate DetFill (or any method's) per-ratio outputs and write ``per_ratio_summary.csv`` + Hint-AUC.

Input layout (what ``detfill/main.py --sample_to_eval`` writes):
    <results_root>/<sketch_type>/<ratio>/200/<id>.image.png      e.g. .../scribble/2/0.1/200/4731016.image.png
Ground truth:
    <gt_dir>/<id>.image.png                                       (Danbooru2021 originals, any size; resized to 256 like the paper)

Metrics (``hintauc.metrics.Evaluator`` = port of the paper's eval_single_run.py):
    mse psnr ssim (256x256, [0,1]) | lpips (AlexNet) | openclip (ViT-B-32 laion2b) | dino (dinov2-base) | dreamsim
Output:
    <out_dir>/per_image.csv          long form: sketch, sketch_name, ratio, image_name, <metrics...>
    <out_dir>/per_ratio_summary.csv  sketch x ratio: n, <metric>_mean, <metric>_std   (same columns as reproduce/expected/**)
    <out_dir>/hauc.json              Hint-AUC per sketch + mean/SD over sketch types over the ratios found (paper grid by default)

Example (Table II scribble row with the released checkpoint, after B4_table2_inference.sh):
    python reproduce/scripts/eval_per_ratio.py \
        --results_root detfill/results/dataset_name/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble \
        --gt_dir <scratch_root>/segmentations/originals --out_dir reproduce/output/table2_scribble --gpu 0
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from hintauc.metrics import Evaluator, DEFAULT_METRICS      # noqa: E402
from hintauc.auc import trapz, DEFAULT_ALPHAS                # noqa: E402

# sketch_type index -> line-art source, as in the DetFill dataset loader (detfill/datasets/custom.py:
# sketch_cands = [pysimp, XDoG, sketchkeras]). NOTE: the archived per_ratio_summary files of the paper
# experiments (reproduce/expected/**) carry the names "XDoG" for 0 and "pysimp" for 1 (label swap of the
# evaluation script); all published numbers are means/SDs over the three sources, so nothing changes.
SKETCH_NAME = {0: "pysimp", 1: "XDoG", 2: "sketchkeras"}


def list_ratios(root, sketch):
    d = os.path.join(root, str(sketch))
    if not os.path.isdir(d):
        return []
    out = []
    for r in os.listdir(d):
        try:
            float(r)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(d, r, "200")):
            out.append(r)
    return sorted(out, key=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sketches", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ratios", nargs="*", default=None, help="subset (as written in the dir names, e.g. 0.0 0.01 ...); default: all found")
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--gpu", default="0", help="CUDA device index, or -1 for CPU")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images per cell (smoke test)")
    ap.add_argument("--epoch_dir", default="200")
    a = ap.parse_args()
    if a.gpu != "-1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    device = "cpu" if a.gpu == "-1" else None
    os.makedirs(a.out_dir, exist_ok=True)
    ev = Evaluator(metrics=a.metrics, device=device)

    per_image_path = os.path.join(a.out_dir, "per_image.csv")
    done = set()
    if os.path.exists(per_image_path):            # resumable
        with open(per_image_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["sketch"]), row["ratio"], row["image_name"]))
    fh = open(per_image_path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["sketch", "sketch_name", "ratio", "image_name"] + list(a.metrics))

    t0 = time.time()
    for sk in a.sketches:
        ratios = a.ratios or list_ratios(a.results_root, sk)
        if not ratios:
            print(f"[warn] no ratio dirs under {a.results_root}/{sk}")
        for r in ratios:
            d = os.path.join(a.results_root, str(sk), r, a.epoch_dir)
            names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
            if a.limit:
                names = names[: a.limit]
            todo = [n for n in names if (sk, r, n) not in done]
            for i, n in enumerate(todo):
                gt = os.path.join(a.gt_dir, n)
                if not os.path.exists(gt):
                    raise FileNotFoundError(f"ground truth missing for {n}: {gt}")
                s = ev(os.path.join(d, n), gt)
                w.writerow([sk, SKETCH_NAME.get(sk, str(sk)), r, n] + [f"{s[m]:.8f}" for m in a.metrics])
                if (i + 1) % 200 == 0:
                    fh.flush()
                    print(f"  sk{sk} ratio {r}: {i + 1}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
            print(f"[done] sk{sk} ratio {r}: {len(names)} images ({len(todo)} new)", flush=True)
    fh.close()

    # ---- aggregate ----
    import pandas as pd
    df = pd.read_csv(per_image_path)
    df["ratio_f"] = df["ratio"].astype(float)
    agg = df.groupby(["sketch", "sketch_name", "ratio_f"])
    rows = []
    for (sk, name, r), g in agg:
        row = {"sketch": sk, "sketch_name": name, "ratio": r, "n": len(g)}
        for m in a.metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=0)
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values(["sketch", "ratio"])
    summ.to_csv(os.path.join(a.out_dir, "per_ratio_summary.csv"), index=False)

    hauc = {"alphas": None, "per_sketch": {}, "mean_over_sketches": {}, "sd_over_sketches": {}}
    grid = [float(x) for x in DEFAULT_ALPHAS]
    for sk, g in summ.groupby("sketch"):
        g = g.sort_values("ratio")
        xs = g["ratio"].to_numpy(float)
        have = [x for x in grid if np.any(np.isclose(xs, x))]
        if len(have) < 2 or have[0] != 0.0 or have[-1] != 1.0:
            print(f"[warn] sketch {sk}: paper grid incomplete (found {have}); Hint-AUC skipped")
            continue
        hauc["alphas"] = have
        hauc["per_sketch"][int(sk)] = {}
        for m in a.metrics:
            ys = [float(g.loc[np.isclose(xs, x), f"{m}_mean"].iloc[0]) for x in have]
            hauc["per_sketch"][int(sk)][m] = trapz(have, ys)
    if hauc["per_sketch"]:
        for m in a.metrics:
            v = np.array([hauc["per_sketch"][k][m] for k in hauc["per_sketch"]])
            hauc["mean_over_sketches"][m] = float(v.mean())
            hauc["sd_over_sketches"][m] = float(v.std(ddof=1)) if len(v) > 1 else None
    json.dump(hauc, open(os.path.join(a.out_dir, "hauc.json"), "w"), indent=2)
    print("\nper_ratio_summary.csv written; Hint-AUC over", hauc["alphas"])
    for m in a.metrics:
        if m in hauc["mean_over_sketches"]:
            sd = hauc["sd_over_sketches"][m]
            print(f"  {m:9s} {hauc['mean_over_sketches'][m]:.4f}" + (f" ± {sd:.4f} (SD over sketch types)" if sd is not None else ""))


if __name__ == "__main__":
    main()
