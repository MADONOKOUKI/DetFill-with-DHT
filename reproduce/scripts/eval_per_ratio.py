#!/usr/bin/env python3
"""Evaluate DetFill (or any method's) per-ratio outputs and write ``per_ratio_summary.csv`` + Hint-AUC.

Input layout (what ``detfill/main.py --sample_to_eval`` writes):
    <results_root>/<sketch_type>/<ratio>/200/<id>.image.png      e.g. .../scribble/2/0.1/200/4731016.image.png
Ground truth:
    <gt_dir>/<id>.image.png                                       (Danbooru2021 originals, any size; resized to 256 like the paper)
    or --gt_from_results: <results_root>/<sketch_type>/<ratio>/ground_truth/<id>.image.png, the 256 x 256 copies the
    sampler writes from its own loader (used for the natural-image models, whose originals live in a nested layout)

Metrics (``hintauc.metrics.Evaluator`` = port of the paper's eval_single_run.py):
    mse psnr ssim (256x256, [0,1]) | lpips (AlexNet) | openclip (ViT-B-32 laion2b) | dino (dinov2-base) | dreamsim
Output:
    <out_dir>/per_image.csv          long form: sketch, sketch_name, ratio, image_name, <metrics...>, pred_sha256, gt_sha256
    <out_dir>/per_ratio_summary.csv  sketch x ratio: n, <metric>_mean, <metric>_std   (same columns as reproduce/expected/**)
    <out_dir>/hauc.json              Hint-AUC per sketch + mean/SD over sketch types over the ratios found (paper grid by default)
    <out_dir>/run_manifest.json      metrics, resize, evaluation backend, hintauc version, input roots

Before anything is scored, the image sets are checked: every ground truth must exist and, per sketch type, every
ratio directory must hold the same image names (a prediction missing at one ratio would change the evaluated set of
that ratio alone); ``--allow-missing`` evaluates the common subset instead. The script is resumable: rows of an
existing per_image.csv are reused only when the prediction and ground-truth files still have the recorded SHA-256,
and only when the metric list and evaluation backend match run_manifest.json (``--fresh`` discards the old CSV).

Example (Table II scribble row with the released checkpoint, after run_hauc_pipeline.sh / detfill/run_inference.sh):
    python reproduce/scripts/eval_per_ratio.py \\
        --results_root detfill/results/dataset_name/BrownianBridge_scribble_illust/sample_to_eval/illust/scribble \\
        --gt_dir <scratch_root>/segmentations/originals --out_dir reproduce/output/table2_scribble --gpu 0
"""
import argparse, csv, json, math, os, statistics, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from hintauc.metrics import Evaluator, DEFAULT_METRICS      # noqa: E402
from hintauc.auc import trapz, DEFAULT_ALPHAS                # noqa: E402
from hintauc.evaluate import file_sha256, plot_curves, protocol_record   # noqa: E402

# sketch_type index -> line-art source, as in the DetFill dataset loader (detfill/datasets/custom.py:
# sketch_cands = [pysimp, XDoG, sketchkeras]). NOTE: the archived per_ratio_summary files of the paper
# experiments (reproduce/expected/**) carry the names "XDoG" for 0 and "pysimp" for 1 (label swap of the
# evaluation script); all published numbers are means/SDs over the three sources, so nothing changes.
SKETCH_NAME = {0: "pysimp", 1: "XDoG", 2: "sketchkeras"}
MANIFEST = "run_manifest.json"


def list_ratios(root, sketch, epoch_dir):
    d = os.path.join(root, str(sketch))
    if not os.path.isdir(d):
        return []
    out = []
    for r in os.listdir(d):
        try:
            float(r)
        except ValueError:
            continue
        if os.path.isdir(os.path.join(d, r, epoch_dir)):
            out.append(r)
    return sorted(out, key=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--gt_dir", default=None, help="directory with the ground-truth images (named like the outputs)")
    ap.add_argument("--gt_from_results", action="store_true",
                    help="use the ground_truth/ copies next to each ratio's outputs instead of --gt_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sketches", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ratios", nargs="*", default=None, help="subset (as written in the dir names, e.g. 0.0 0.01 ...); default: all found")
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--gpu", default="0", help="CUDA device index, or -1 for CPU")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images per cell (smoke test)")
    ap.add_argument("--epoch_dir", default="200")
    ap.add_argument("--ckpt", default=None, help="checkpoint file whose SHA-256 is recorded in hauc.json")
    ap.add_argument("--plot", action="store_true", help="also write curves.png (mean over the evaluated sketch types)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="evaluate the images common to all ratios of a sketch type instead of stopping on a missing prediction")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing per_image.csv / run_manifest.json and score everything again")
    a = ap.parse_args()
    if bool(a.gt_dir) == bool(a.gt_from_results):
        ap.error("give exactly one of --gt_dir and --gt_from_results")
    if a.gpu != "-1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    device = "cpu" if a.gpu == "-1" else None
    os.makedirs(a.out_dir, exist_ok=True)
    ev = Evaluator(metrics=a.metrics, device=device)
    gt_source = os.path.abspath(a.gt_dir) if a.gt_dir else "<results_root>/<sketch>/<ratio>/ground_truth"

    # ---- run manifest: an existing evaluation is only continued under the same metric settings -------------------
    manifest = {"metrics": list(a.metrics), "resize": ev.resize, "backend": ev.backend(), "epoch_dir": a.epoch_dir,
                "results_root": os.path.abspath(a.results_root), "gt": gt_source}
    from hintauc import __version__ as hintauc_version
    manifest["hintauc"] = hintauc_version
    man_path = os.path.join(a.out_dir, MANIFEST)
    per_image_path = os.path.join(a.out_dir, "per_image.csv")
    if a.fresh:
        for p in (man_path, per_image_path):
            if os.path.exists(p):
                os.remove(p)
    if os.path.exists(man_path):
        old = json.load(open(man_path))
        same = all(old.get(k) == manifest[k] for k in ("metrics", "resize", "epoch_dir", "results_root", "gt")) and \
            all(old.get("backend", {}).get(k) == manifest["backend"][k] for k in ("resize", "ssim"))
        if not same:
            sys.exit(f"error: {a.out_dir} holds an evaluation with different settings ({man_path}: metrics {old.get('metrics')}, "
                     f"backend {old.get('backend', {}).get('resize')}/{old.get('backend', {}).get('ssim')}, gt {old.get('gt')}); "
                     "use another --out_dir or --fresh")
    json.dump(manifest, open(man_path, "w"), indent=1)

    # ---- preflight: the image sets ---------------------------------------------------------------------------------
    cells = {}          # (sk, r) -> {"dir": ..., "gt_dir": ..., "names": [...]}
    for sk in a.sketches:
        ratios = a.ratios or list_ratios(a.results_root, sk, a.epoch_dir)
        if not ratios:
            print(f"[warn] no ratio dirs under {a.results_root}/{sk}")
        for r in ratios:
            d = os.path.join(a.results_root, str(sk), r, a.epoch_dir)
            if not os.path.isdir(d):
                sys.exit(f"error: missing output directory {d}")
            names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
            if a.limit:
                names = names[: a.limit]
            gt_dir = os.path.join(a.results_root, str(sk), r, "ground_truth") if a.gt_from_results else a.gt_dir
            cells[(sk, r)] = {"dir": d, "gt_dir": gt_dir, "names": names}
    problems = []
    for sk in a.sketches:
        sets = {r: set(c["names"]) for (s, r), c in cells.items() if s == sk}
        if not sets:
            continue
        common = set.intersection(*sets.values())
        for r, names in sorted(sets.items(), key=lambda kv: float(kv[0])):
            extra, missing = names - common, common ^ names
            if names != common:
                problems.append(f"sketch {sk} ratio {r}: {len(names)} images, {len(names - common)} not in every ratio "
                                f"(e.g. {', '.join(sorted(names - common)[:3])})")
        if any(names != common for names in sets.values()):
            if not a.allow_missing:
                pass
            else:
                for (s, r), c in cells.items():
                    if s == sk:
                        c["names"] = [n for n in c["names"] if n in common]
    missing_gt = []
    for (sk, r), c in cells.items():
        for n in c["names"]:
            if not os.path.isfile(os.path.join(c["gt_dir"], n)):
                missing_gt.append(f"sketch {sk} ratio {r}: {os.path.join(c['gt_dir'], n)}")
    if missing_gt:
        sys.exit("error: ground truth missing for " + f"{len(missing_gt)} image(s), e.g. " + "; ".join(missing_gt[:3]))
    if problems:
        msg = "the ratio directories do not hold the same images:\n  " + "\n  ".join(problems)
        if not a.allow_missing:
            sys.exit("error: " + msg + "\n(pass --allow-missing to evaluate the images common to all ratios of each sketch type)")
        print("[warn] " + msg + "\n[warn] evaluating the common subset (--allow-missing)", flush=True)
    sets_by_sketch = {sk: set(c["names"]) for (sk, _), c in cells.items()}
    if len({frozenset(v) for v in sets_by_sketch.values()}) > 1:
        print("[warn] the sketch types are evaluated on different image sets: " +
              ", ".join(f"sketch {sk}: {len(v)}" for sk, v in sets_by_sketch.items()), flush=True)

    # ---- resume: reuse rows whose files are unchanged ----------------------------------------------------------------
    fieldnames = ["sketch", "sketch_name", "ratio", "image_name"] + list(a.metrics) + ["pred_sha256", "gt_sha256"]
    old_rows = {}
    if os.path.exists(per_image_path):
        with open(per_image_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "pred_sha256" in reader.fieldnames and all(m in reader.fieldnames for m in a.metrics):
                for row in reader:
                    old_rows[(int(row["sketch"]), row["ratio"], row["image_name"])] = row
            else:
                print("[note] the existing per_image.csv has no file hashes or other metrics; everything is scored again", flush=True)
    rows, reused, t0 = [], 0, time.time()
    for (sk, r), c in sorted(cells.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        new_here = 0
        for i, n in enumerate(c["names"]):
            pred, gt = os.path.join(c["dir"], n), os.path.join(c["gt_dir"], n)
            ph, gh = file_sha256(pred), file_sha256(gt)
            old = old_rows.get((sk, r, n))
            if old is not None and old.get("pred_sha256") == ph and old.get("gt_sha256") == gh:
                rows.append(old); reused += 1
                continue
            s = ev(pred, gt)
            rows.append({"sketch": sk, "sketch_name": SKETCH_NAME.get(sk, str(sk)), "ratio": r, "image_name": n,
                         **{m: f"{s[m]:.8f}" for m in a.metrics}, "pred_sha256": ph, "gt_sha256": gh})
            new_here += 1
            if new_here % 200 == 0:
                print(f"  sk{sk} ratio {r}: {new_here} new  ({time.time() - t0:.0f}s)", flush=True)
                _write_rows(per_image_path, fieldnames, rows)
        print(f"[done] sk{sk} ratio {r}: {len(c['names'])} images ({new_here} new)", flush=True)
    _write_rows(per_image_path, fieldnames, rows)
    print(f"{len(rows)} rows ({reused} reused from the previous run, files unchanged)")

    # ---- aggregate (no pandas) --------------------------------------------------------------------------------------
    groups = {}
    for row in rows:
        groups.setdefault((int(row["sketch"]), row["sketch_name"], float(row["ratio"])), []).append(row)
    summ = []
    for (sk, name, r), g in sorted(groups.items()):
        rec = {"sketch": sk, "sketch_name": name, "ratio": r, "n": len(g)}
        for m in a.metrics:
            vals = [float(x[m]) for x in g]
            rec[f"{m}_mean"] = statistics.fmean(vals)
            rec[f"{m}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summ.append(rec)
    with open(os.path.join(a.out_dir, "per_ratio_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()) if summ else ["sketch", "sketch_name", "ratio", "n"])
        w.writeheader()
        for rec in summ:
            w.writerow(rec)

    hauc = {"alphas": None, "per_sketch": {}, "mean_over_sketches": {}, "sd_over_sketches": {}, "n_images": {}}
    grid = [float(x) for x in DEFAULT_ALPHAS]
    for sk in sorted({rec["sketch"] for rec in summ}):
        g = sorted((rec for rec in summ if rec["sketch"] == sk), key=lambda rec: rec["ratio"])
        xs = [rec["ratio"] for rec in g]
        have = [x for x in grid if any(math.isclose(v, x) for v in xs)]
        if len(have) < 2 or have[0] != 0.0 or have[-1] != 1.0:
            print(f"[warn] sketch {sk}: paper grid incomplete (found {have}); Hint-AUC skipped")
            continue
        hauc["alphas"] = have
        hauc["per_sketch"][int(sk)] = {}
        hauc["n_images"][int(sk)] = min(rec["n"] for rec in g)
        for m in a.metrics:
            ys = [next(rec[f"{m}_mean"] for rec in g if math.isclose(rec["ratio"], x)) for x in have]
            hauc["per_sketch"][int(sk)][m] = trapz(have, ys)
    if hauc["per_sketch"]:
        for m in a.metrics:
            v = np.array([hauc["per_sketch"][k][m] for k in hauc["per_sketch"]])
            hauc["mean_over_sketches"][m] = float(v.mean())
            hauc["sd_over_sketches"][m] = float(v.std(ddof=1)) if len(v) > 1 else None
    hauc["protocol"] = protocol_record(ev, alphas=hauc["alphas"] or grid, sketch_types=list(a.sketches),
                                       results_root=os.path.abspath(a.results_root), gt=gt_source,
                                       allow_missing=a.allow_missing, rows_reused=reused,
                                       checkpoint_sha256=file_sha256(a.ckpt) if a.ckpt else None, checkpoint=a.ckpt)
    if a.plot and summ:
        per_alpha = {}
        for r in sorted({rec["ratio"] for rec in summ}):
            recs = [rec for rec in summ if rec["ratio"] == r]
            per_alpha[float(r)] = {m: statistics.fmean(rec[f"{m}_mean"] for rec in recs) for m in a.metrics}
        plot_curves(per_alpha, os.path.join(a.out_dir, "curves.png"), title="mean over sketch types")
    json.dump(hauc, open(os.path.join(a.out_dir, "hauc.json"), "w"), indent=2)
    print("\nper_ratio_summary.csv written; Hint-AUC over", hauc["alphas"])
    for m in a.metrics:
        if m in hauc["mean_over_sketches"]:
            sd = hauc["sd_over_sketches"][m]
            print(f"  {m:9s} {hauc['mean_over_sketches'][m]:.4f}" + (f" ± {sd:.4f} (SD over sketch types)" if sd is not None else ""))


def _write_rows(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
