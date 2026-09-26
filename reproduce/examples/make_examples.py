#!/usr/bin/env python3
"""Example-based reproduction of every experiment of the paper on a handful of test illustrations.

Each experiment below runs the released checkpoints on the example images shipped in ``examples_data``
(release v1.3), writes a labelled image grid, computes the seven metrics of the paper against the ground
truth, and — where the paper's own outputs for the same images are archived in ``expected/`` — compares the
new outputs with them. Everything is driven by the same ``detfill/main.py`` inference entry point and the same
``hintauc`` evaluator as the full-scale experiments; only the number of images differs.

    E1  Table II protocol      : size-ordered deterministic hints, scribble (96-ch) and dot (64-ch) models, ratio grid
    E2  Table III protocol     : same images, fixed random (ascending-label) region order
    E3  Segmentation dependency: scribble models trained on Felzenszwalb / DanbooRegion / SLIC hints, each evaluated
                                 with hints from each of the three segmenters (supplementary 3x3 table and figures)
    E4  Seed sensitivity       : size-ordered selection vs. seeded uniformly random selection (supplementary table)
    E5  Dense hint-ratio curve : per-image metric curves on a fine ratio grid (supplementary alpha-grid study)
    E8  Channel ablation       : 32 / 64 / 96 base-channel scribble models (supplementary channel ablation)
    E9  Hint-map regeneration  : deterministic hint generation re-run with the hintauc library vs. the stored maps
    E11 Earlier checkpoint     : the scribble model of the earlier submission on the same images (release v1.4)
    (E6 Diffusart-retrain and E7 natural-image models are run by the authors only; see README)

Usage (normally via run_examples.sh):
    python reproduce/examples/make_examples.py --data <examples_data dir> --ckpt_dir <checkpoints> --gpu 0 \
        --mode quick|full|smoke --out reproduce/examples/output
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from hintauc.auc import trapz  # noqa: E402

ALL_IDS = ["514016", "4262016", "1625016", "3421016", "4417016", "1019016", "1023016", "4731016", "4942016",
           "100016", "1001016", "10016"]
PAPER_GRID = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]
SKETCH_DIR = {0: "pysimp", 1: "XDoG", 2: "sketchkeras"}
METRICS = ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]
CKPT = {  # key -> (release asset file, released config template, model_channels override, sampling steps override)
    "scribble": ("detfill_scribble_illust_200ep.pth", "scribble_illust.yaml", None, None),
    "dot": ("detfill_dot_illust_200ep.pth", "dot_illust.yaml", None, None),
    "scribble_danbooregion": ("detfill_scribble_illust_danbooregion_200ep.pth", "scribble_illust.yaml", None, None),
    "scribble_slic": ("detfill_scribble_illust_slic_200ep.pth", "scribble_illust.yaml", None, None),
    "scribble_32ch": ("detfill_scribble_illust_32ch_200ep.pth", "scribble_illust.yaml", 32, None),
    "scribble_64ch": ("detfill_scribble_illust_64ch_200ep.pth", "scribble_illust.yaml", 64, None),
    # the archived config of the earlier scribble model used 1000 sampling steps (all other models: 200)
    "earlier_scribble": ("detfill_scribble_illust_earlier_64ch_200ep.pth", "scribble_illust.yaml", 64, 1000),
}
MODES = {
    "smoke": dict(ids=ALL_IDS[:1], ratios=[0.10, 1.00], exps=["E1s", "E9"]),
    "quick": dict(ids=ALL_IDS[:4], ratios=[0.01, 0.10, 0.50, 1.00], exps=["E1", "E2", "E3", "E4", "E9"]),
    "full": dict(ids=ALL_IDS, ratios=PAPER_GRID, exps=["E1", "E2", "E3", "E4", "E5", "E8", "E9", "E11"]),
}


# ----------------------------------------------------------------------------------------------------------
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def rdir(t):
    return str(float(t))  # main.py names the ratio directory str(float(ratio))


def write_config(template, data_root, out_path, hint_order="area", model_channels=None, sample_step=None):
    """Copy a released config, pointing the data paths at ``data_root`` (flat evaluation layout)."""
    import re
    s = open(os.path.join(ROOT, "detfill", "configs", template)).read()
    s = re.sub(r"^(\s*dataset_path:).*$", rf"\1 '{data_root}'", s, flags=re.M)
    s = re.sub(r"^(\s*scratch_root:).*$", rf"\1 '{data_root}'", s, flags=re.M)
    if re.search(r"^\s*#?\s*hint_order:", s, flags=re.M):
        s = re.sub(r"^(\s*)#?\s*hint_order:.*$", rf"\1hint_order: '{hint_order}'", s, flags=re.M)
    else:
        s = re.sub(r"^(\s*)hint_type:(.*)$", rf"\1hint_type:\2\n\1hint_order: '{hint_order}'", s, flags=re.M)
    if model_channels is not None:
        s = re.sub(r"^(\s*model_channels:).*$", rf"\1 {model_channels}", s, flags=re.M)
    if sample_step is not None:
        s = re.sub(r"^(\s*sample_step:).*$", rf"\1 {sample_step}", s, flags=re.M)
    # test batch size 1: the runner's test loader drops the last incomplete batch (drop_last=True), so with the
    # released batch sizes (scribble 5, dot 8) a handful of example images would be silently skipped
    s = re.sub(r"^(\s*batch_size:) \d+\s*$", r"\1 1", s, flags=re.M)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(s)
    return out_path


def run_detfill(model_key, data_root, ratios, sketch_type, out_tag, args, hint_order="area"):
    """Run detfill/main.py for every ratio; returns {ratio: dir with <id>.image.png}."""
    asset, template, mc, steps = CKPT[model_key]
    ckpt = os.path.join(args.ckpt_dir, asset)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint missing: {ckpt} (run fetch step / see checkpoints/README.md)")
    hint = "dot" if "dot" in model_key else "scribble"
    result_path = os.path.join(args.out, "runs", out_tag)
    cfg = write_config(template, data_root, os.path.join(result_path, "config.yaml"), hint_order, mc, steps)
    outs = {}
    for r in ratios:
        # main.py names the last directory after the number of sampling steps (200 for the released configs)
        d = os.path.join(result_path, "dataset_name", f"BrownianBridge_{hint}_illust", "sample_to_eval", "illust",
                         hint, str(sketch_type), rdir(r), str(steps or 200))
        ids = sorted(os.path.basename(p)[:-len(".image.png")] for p in glob.glob(os.path.join(data_root, "segmentations", "originals", "*.image.png")))
        if all(os.path.exists(os.path.join(d, f"{i}.image.png")) for i in ids):
            log(f"  [{out_tag}] ratio {r}: cached")
        else:
            cmd = [sys.executable, "main.py", "--config", cfg, "--resume_model", ckpt, "--sample_to_eval", "--save_top",
                   "--gpu_ids", str(args.gpu), "--sample_ratio", f"{r:.2f}", "--sketch_type", str(sketch_type),
                   "--result_path", result_path]
            log(f"  [{out_tag}] ratio {r}: {' '.join(cmd[1:6])} ...")
            t0 = time.time()
            env = dict(os.environ, BATCH_SIZE_OVERRIDE="1")   # belt and braces: the runner honours this override too
            with open(os.path.join(result_path, f"log_r{r}_sk{sketch_type}.txt"), "w") as lf:
                subprocess.run(cmd, cwd=os.path.join(ROOT, "detfill"), stdout=lf, stderr=subprocess.STDOUT, check=True, env=env)
            log(f"  [{out_tag}] ratio {r}: {time.time() - t0:.0f}s")
        outs[r] = d
    return outs


# ----------------------------------------------------------------------------------------------------------
_EVAL = None


def evaluator(args):
    global _EVAL
    if _EVAL is None:
        from hintauc.metrics import Evaluator
        _EVAL = Evaluator(metrics=METRICS, device=("cpu" if str(args.gpu) == "-1" else None))
    return _EVAL


def score(pred, gt, args):
    return evaluator(args)(pred, gt)


def load_rgb(p, size=256):
    return Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)


def hint_vis(data_root, hid, hint, size=256, sub="hint_from_regions_64_rev"):
    """Visualise a stored 64x64 hint map (colour where mask>0, white elsewhere)."""
    col = np.array(Image.open(os.path.join(data_root, sub, f"{hid}.image_{hint}_col64.png")).convert("RGB"))
    msk = np.array(Image.open(os.path.join(data_root, sub, f"{hid}.image_{hint}_mask64.png")).convert("L")) > 0
    vis = np.full_like(col, 255)
    vis[msk] = col[msk]
    return Image.fromarray(vis).resize((size, size), Image.NEAREST)


def grid(rows, col_titles, row_titles, path, cell=192, title=None):
    """rows: list of lists of PIL images (same length). Writes a labelled grid PNG."""
    pad, top, left = 4, 28, 110
    W = left + len(col_titles) * (cell + pad)
    H = top + (24 if title else 0) + len(rows) * (cell + pad)
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    y0 = 24 if title else 0
    if title:
        dr.text((6, 4), title, fill="black")
    for j, t in enumerate(col_titles):
        dr.text((left + j * (cell + pad) + 4, y0 + 6), t, fill="black")
    for i, (rt, row) in enumerate(zip(row_titles, rows)):
        y = y0 + top + i * (cell + pad)
        dr.text((4, y + cell // 2 - 6), rt, fill="black")
        for j, img in enumerate(row):
            if img is not None:
                im.paste(img.resize((cell, cell)), (left + j * (cell + pad), y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return path


def compare_with_paper(new_dir, paper_dir, ids, args):
    """PSNR between the re-generated output and the paper's archived output of the same image (identity check)."""
    out = {}
    for i in ids:
        a, b = os.path.join(new_dir, f"{i}.image.png"), os.path.join(paper_dir, f"{i}.image.png")
        if os.path.exists(a) and os.path.exists(b):
            x = np.asarray(load_rgb(a), np.float32) / 255; y = np.asarray(load_rgb(b), np.float32) / 255
            mse = float(np.mean((x - y) ** 2))
            out[i] = 10 * np.log10(1 / (mse + 1e-10))
    return out


def hauc_from_curve(per_ratio):
    xs = sorted(per_ratio)
    return {m: trapz(xs, [per_ratio[x][m] for x in xs]) for m in METRICS}


# ----------------------------------------------------------------------------------------------------------
def exp_E1(args, ids, ratios, models=("scribble", "dot"), tag="E1_tableII_protocol", hint_order="area"):
    """Table II (size-ordered) or Table III (label order) protocol on the example images."""
    data_root = stage_subset(args, ids, tag)
    gt = {i: os.path.join(data_root, "segmentations", "originals", f"{i}.image.png") for i in ids}
    report = {"protocol": hint_order, "ratios": ratios, "ids": ids, "sketch_type": 2, "models": {}}
    for mk in models:
        hint = "dot" if "dot" in mk else "scribble"
        outs = run_detfill(mk, data_root, ratios, 2, f"{tag}/{mk}", args, hint_order)
        per_ratio, per_image = {}, {}
        for r, d in outs.items():
            vals = {i: score(os.path.join(d, f"{i}.image.png"), gt[i], args) for i in ids}
            per_image[r] = vals
            per_ratio[r] = {m: float(np.mean([v[m] for v in vals.values()])) for m in METRICS}
        entry = {"per_ratio_mean": {str(r): v for r, v in per_ratio.items()},
                 "per_image": {str(r): v for r, v in per_image.items()}}
        if set(ratios) == set(PAPER_GRID):
            entry["hint_auc_over_examples"] = hauc_from_curve(per_ratio)
        # comparison with the paper's archived outputs (scribble, size order) when shipped
        paper_root = os.path.join(HERE, "expected", "paper_outputs", f"{tag}_{mk}")
        if os.path.isdir(paper_root):
            ident = {}
            for r, d in outs.items():
                pd = os.path.join(paper_root, rdir(r))
                if os.path.isdir(pd):
                    ident[str(r)] = compare_with_paper(d, pd, ids, args)
            entry["psnr_vs_paper_outputs_dB"] = ident
        report["models"][mk] = entry
        # grid: GT | sketch | hint(α) ... | output(α) ...
        rows, rtitles = [], []
        for i in ids:
            row = [load_rgb(gt[i]), load_rgb(os.path.join(data_root, "sketch", "sketchkeras", f"{i}.png"))]
            for r in ratios:
                row.append(hint_vis_selected(data_root, i, hint, r, hint_order))
                row.append(load_rgb(os.path.join(outs[r], f"{i}.image.png")))
            rows.append(row); rtitles.append(i)
        cols = ["ground truth", "line art (SketchKeras)"]
        for r in ratios:
            cols += [f"hints alpha={int(r*100)}%", f"{mk} alpha={int(r*100)}%"]
        grid(rows, cols, rtitles, os.path.join(args.out, "grids", f"{tag}_{mk}.png"),
             title=f"{tag}: {mk} model, hint order = {hint_order}")
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    return report


def stage_subset(args, ids, tag):
    """Flat evaluation layout containing only ``ids`` (symlinks into examples_data)."""
    root = os.path.join(args.out, "stage", tag)
    if os.path.isdir(root):
        shutil.rmtree(root)
    for sub in ["segmentations/originals", "hint_from_regions_64_rev", "hint_from_regions_256"] + [f"sketch/{s}" for s in SKETCH_DIR.values()]:
        os.makedirs(os.path.join(root, sub))
    for i in ids:
        os.symlink(os.path.join(args.data, "segmentations", "originals", f"{i}.image.png"), os.path.join(root, "segmentations", "originals", f"{i}.image.png"))
        for s in SKETCH_DIR.values():
            os.symlink(os.path.join(args.data, "sketch", s, f"{i}.png"), os.path.join(root, "sketch", s, f"{i}.png"))
        for p in glob.glob(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_*64.png")):
            os.symlink(p, os.path.join(root, "hint_from_regions_64_rev", os.path.basename(p)))
        os.symlink(os.path.join(args.data, "hint_from_regions_256", f"{i}.image_region64.png"), os.path.join(root, "hint_from_regions_256", f"{i}.image_region64.png"))
    return root


def exp_E3(args, ids, ratios):
    """Segmentation dependency: 3 training segmenters x 3 evaluation segmenters (scribble hints)."""
    tag = "E3_segmentation_dependency"
    models = {"Felzenszwalb": "scribble", "DanbooRegion": "scribble_danbooregion", "SLIC": "scribble_slic"}
    report = {"ids": ids, "ratios": ratios, "cells": {}}
    roots = {}
    for eval_seg in ["Felzenszwalb", "DanbooRegion", "SLIC"]:
        root = stage_subset(args, ids, f"{tag}/eval_{eval_seg}")
        if eval_seg != "Felzenszwalb":   # swap the hint maps for the alternative segmenter's maps
            sub = os.path.join(args.data, "segmenters", eval_seg.lower())
            for p in glob.glob(os.path.join(root, "hint_from_regions_64_rev", "*")) + glob.glob(os.path.join(root, "hint_from_regions_256", "*")):
                os.remove(p)
            for i in ids:
                for suf in ["scribble_col64", "scribble_mask64"]:
                    os.symlink(os.path.join(sub, "hint_from_regions_64_rev", f"{i}.image_{suf}.png"), os.path.join(root, "hint_from_regions_64_rev", f"{i}.image_{suf}.png"))
                os.symlink(os.path.join(sub, "hint_from_regions_256", f"{i}.image_region64.png"), os.path.join(root, "hint_from_regions_256", f"{i}.image_region64.png"))
        roots[eval_seg] = root
    gt = {i: os.path.join(args.data, "segmentations", "originals", f"{i}.image.png") for i in ids}
    outs = {}
    for train_seg, mk in models.items():
        for eval_seg, root in roots.items():
            o = run_detfill(mk, root, ratios, 2, f"{tag}/train_{train_seg}__eval_{eval_seg}", args)
            outs[(train_seg, eval_seg)] = o
            cell = {}
            for r, d in o.items():
                vals = {i: score(os.path.join(d, f"{i}.image.png"), gt[i], args) for i in ids}
                cell[str(r)] = {"mean": {m: float(np.mean([v[m] for v in vals.values()])) for m in METRICS}, "per_image": vals}
            report["cells"][f"train_{train_seg}__eval_{eval_seg}"] = cell
    # grids: one per image: rows = training segmenter, cols = [hint(eval seg) | outputs per ratio] per eval seg
    for i in ids:
        rows, rtitles = [], []
        for train_seg in models:
            row = []
            for eval_seg, root in roots.items():
                row.append(hint_vis(root, i, "scribble"))
                for r in ratios:
                    row.append(load_rgb(os.path.join(outs[(train_seg, eval_seg)][r], f"{i}.image.png")))
            rows.append(row); rtitles.append(f"trained: {train_seg}")
        cols = []
        for eval_seg in roots:
            cols += [f"{eval_seg} hints"] + [f"alpha={int(r*100)}%" for r in ratios]
        grid(rows, cols, rtitles, os.path.join(args.out, "grids", f"{tag}_{i}.png"), cell=160,
             title=f"{tag}: image {i} — rows: training segmenter, column groups: evaluation segmenter")
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    return report


def seeded_random_maps(args, ids, seed, ratio, root_out, all_ids_sorted):
    """Reproduce the paper's seeded uniformly random region selection (gen_seeded_random_hints.py) for the example ids."""
    import cv2
    os.makedirs(os.path.join(root_out, "hint_from_regions_64_rev"), exist_ok=True)
    for i in ids:
        idx = all_ids_sorted.index(i)          # the paper seeded the generator with the image's index in the sorted test list
        region = cv2.imread(os.path.join(args.data, "hint_from_regions_256", f"{i}.image_region64.png"))
        region = cv2.resize(region, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.uint64)
        id_maps = region[:, :, 0] * 255 * 255 + region[:, :, 1] * 255 + region[:, :, 2]
        cand = np.unique(id_maps.reshape(-1))
        rng = np.random.default_rng([20260611, seed, idx])
        perm = rng.permutation(cand)
        keep = perm[: int(len(cand) * ratio)]
        area = np.isin(id_maps, keep).astype(np.uint8)
        col = np.array(Image.open(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_scribble_col64.png")).convert("RGB"))
        msk_img = Image.open(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_scribble_mask64.png"))
        msk = np.array(msk_img.convert("L"))
        Image.fromarray(col * area[:, :, None]).save(os.path.join(root_out, "hint_from_regions_64_rev", f"{i}.image_scribble_col64.png"))
        Image.fromarray(msk * area).convert(msk_img.mode).save(os.path.join(root_out, "hint_from_regions_64_rev", f"{i}.image_scribble_mask64.png"))


def exp_E4(args, ids, ratios, seeds=(1, 2, 3)):
    """Seed sensitivity: size-ordered selection vs seeded random selection at the same ratios (scribble model, SketchKeras)."""
    tag = "E4_seed_sensitivity"
    ratios = [r for r in ratios if 0 < r < 1] or [0.05, 0.25]
    all_ids_sorted = sorted(l.strip() for l in open(os.path.join(args.data, "..", "test_image_ids.txt")) if l.strip()) \
        if os.path.exists(os.path.join(args.data, "..", "test_image_ids.txt")) else sorted(l.strip() for l in open(os.path.join(HERE, "test_image_ids.txt")))
    gt = {i: os.path.join(args.data, "segmentations", "originals", f"{i}.image.png") for i in ids}
    report = {"ids": ids, "ratios": ratios, "seeds": list(seeds), "size_sort": {}, "random": {}}
    base = stage_subset(args, ids, f"{tag}/size_sort")
    sort_out = run_detfill("scribble", base, ratios, 2, f"{tag}/size_sort", args)
    for r, d in sort_out.items():
        vals = {i: score(os.path.join(d, f"{i}.image.png"), gt[i], args) for i in ids}
        report["size_sort"][str(r)] = {"mean": {m: float(np.mean([v[m] for v in vals.values()])) for m in METRICS}, "per_image": vals}
    rnd_out = {}
    for s in seeds:
        for r in ratios:
            root = stage_subset(args, ids, f"{tag}/seed{s}_r{r}")
            for p in glob.glob(os.path.join(root, "hint_from_regions_64_rev", "*scribble*")):
                os.remove(p)
            seeded_random_maps(args, ids, s, r, root, all_ids_sorted)
            # pre-baked hints -> run at ratio 1.0 so the loader keeps every provided region (paper procedure)
            o = run_detfill("scribble", root, [1.0], 2, f"{tag}/seed{s}_r{r}", args)
            rnd_out[(s, r)] = (root, o[1.0])
            vals = {i: score(os.path.join(o[1.0], f"{i}.image.png"), gt[i], args) for i in ids}
            report["random"].setdefault(str(r), {})[f"seed{s}"] = {"mean": {m: float(np.mean([v[m] for v in vals.values()])) for m in METRICS}, "per_image": vals}
    for r in ratios:
        means = np.array([[report["random"][str(r)][f"seed{s}"]["mean"][m] for m in METRICS] for s in seeds])
        report["random"][str(r)]["mean_over_seeds"] = dict(zip(METRICS, means.mean(0).tolist()))
        report["random"][str(r)]["sd_over_seeds"] = dict(zip(METRICS, means.std(0, ddof=1).tolist()))
    for i in ids:
        rows, rtitles = [], []
        row = [load_rgb(gt[i])]
        for r in ratios:
            row += [hint_vis_masked(base, i, r, args), load_rgb(os.path.join(sort_out[r], f"{i}.image.png"))]
        rows.append(row); rtitles.append("size-ordered")
        for s in seeds:
            row = [None]
            for r in ratios:
                root, d = rnd_out[(s, r)]
                row += [hint_vis(root, i, "scribble"), load_rgb(os.path.join(d, f"{i}.image.png"))]
            rows.append(row); rtitles.append(f"random seed {s}")
        cols = ["ground truth"]
        for r in ratios:
            cols += [f"hints alpha={int(r*100)}%", f"output alpha={int(r*100)}%"]
        grid(rows, cols, rtitles, os.path.join(args.out, "grids", f"{tag}_{i}.png"), title=f"{tag}: image {i} (scribble model, SketchKeras line art)")
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    return report


def hint_vis_selected(root, i, hint, ratio, hint_order="area"):
    """Visualise the hints the loader keeps at ``ratio``: largest regions first (area) or ascending label order (label)."""
    import cv2
    region = cv2.imread(os.path.join(root, "hint_from_regions_256", f"{i}.image_region64.png"))
    region = cv2.resize(region, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.uint64)
    id_maps = region[:, :, 0] * 255 * 255 + region[:, :, 1] * 255 + region[:, :, 2]
    vals, counts = np.unique(id_maps, return_counts=True)
    order = vals[np.argsort(-counts)] if hint_order == "area" else np.sort(vals)
    keep = order[: int(len(vals) * ratio)]
    area = np.isin(id_maps, keep)
    col = np.array(Image.open(os.path.join(root, "hint_from_regions_64_rev", f"{i}.image_{hint}_col64.png")).convert("RGB"))
    msk = np.array(Image.open(os.path.join(root, "hint_from_regions_64_rev", f"{i}.image_{hint}_mask64.png")).convert("L")) > 0
    vis = np.full_like(col, 255)
    sel = msk & area
    vis[sel] = col[sel]
    return Image.fromarray(vis).resize((256, 256), Image.NEAREST)


def hint_vis_masked(root, i, ratio, args=None):
    """Size-ordered scribble selection at ``ratio`` (the DetFill loader's Table II rule)."""
    return hint_vis_selected(root, i, "scribble", ratio, "area")


def exp_E5(args, ids):
    """Dense hint-ratio curve for a few images (scribble model)."""
    tag = "E5_dense_ratio_curve"
    ratios = [round(x, 2) for x in np.linspace(0, 1, 21)] + [0.01, 0.03, 0.05]
    ratios = sorted(set(ratios))
    root = stage_subset(args, ids, tag)
    gt = {i: os.path.join(args.data, "segmentations", "originals", f"{i}.image.png") for i in ids}
    outs = run_detfill("scribble", root, ratios, 2, tag, args)
    curve = {str(r): {i: score(os.path.join(d, f"{i}.image.png"), gt[i], args) for i in ids} for r, d in outs.items()}
    report = {"ids": ids, "ratios": ratios, "per_image": curve,
              "hint_auc_dense": hauc_from_curve({r: {m: float(np.mean([curve[str(r)][i][m] for i in ids])) for m in METRICS} for r in ratios}),
              "hint_auc_paper_grid": hauc_from_curve({r: {m: float(np.mean([curve[str(r)][i][m] for i in ids])) for m in METRICS} for r in PAPER_GRID})}
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, m in zip(axes, ["psnr", "lpips", "dreamsim"]):
            for i in ids:
                ax.plot(ratios, [curve[str(r)][i][m] for r in ratios], marker=".", label=i)
            ax.set_xlabel("hint ratio α"); ax.set_title(m); ax.grid(alpha=.3)
        axes[0].legend(fontsize=7); fig.tight_layout()
        os.makedirs(os.path.join(args.out, "grids"), exist_ok=True)
        fig.savefig(os.path.join(args.out, "grids", f"{tag}.png"), dpi=120)
    except Exception as e:  # matplotlib optional
        log("E5 plot skipped:", e)
    return report


def exp_E8(args, ids, ratios):
    """Channel ablation: 32 / 64 / 96 base channels (scribble)."""
    tag = "E8_channel_ablation"
    ratios = ratios or [0.10]
    root = stage_subset(args, ids, tag)
    gt = {i: os.path.join(args.data, "segmentations", "originals", f"{i}.image.png") for i in ids}
    report = {"ids": ids, "ratios": ratios, "models": {}}
    outs = {}
    for mk, name in [("scribble_32ch", "32 channels"), ("scribble_64ch", "64 channels"), ("scribble", "96 channels (paper)")]:
        o = run_detfill(mk, root, ratios, 2, f"{tag}/{mk}", args)
        outs[name] = o
        report["models"][name] = {str(r): {"mean": {m: float(np.mean([score(os.path.join(d, f"{i}.image.png"), gt[i], args)[m] for i in ids])) for m in METRICS}} for r, d in o.items()}
    rows, rtitles = [], []
    for i in ids:
        row = [load_rgb(gt[i]), hint_vis_masked(root, i, ratios[0], args)]
        for name, o in outs.items():
            row.append(load_rgb(os.path.join(o[ratios[0]], f"{i}.image.png")))
        rows.append(row); rtitles.append(i)
    grid(rows, ["ground truth", f"hints alpha={int(ratios[0]*100)}%"] + list(outs), rtitles, os.path.join(args.out, "grids", f"{tag}.png"), title=f"{tag} (scribble, alpha={int(ratios[0]*100)}%)")
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    return report


def _n_regions(h):
    """Number of regions found by the library."""
    return int(h.n_regions())


def exp_E9(args, ids):
    """Deterministic hint generation re-run with the library vs the stored maps (CPU)."""
    tag = "E9_hint_regeneration"
    import hintauc
    report = {"ids": ids, "per_image": {}}
    rows, rtitles = [], []
    for i in ids:
        img = os.path.join(args.data, "segmentations", "originals", f"{i}.image.png")
        t0 = time.time()
        h = hintauc.generate_hints(img, size=64)
        dt = time.time() - t0
        stored_m = np.array(Image.open(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_scribble_mask64.png")).convert("L")) > 0
        stored_d = np.array(Image.open(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_dot_mask64.png")).convert("L")) > 0
        _, new_m = h.at_ratio(1.0, hint_type="scribble")
        _, new_d = h.at_ratio(1.0, hint_type="dot")
        new_m = np.asarray(new_m) > 0; new_d = np.asarray(new_d) > 0
        iou = float((new_m & stored_m).sum() / max(1, (new_m | stored_m).sum()))
        report["per_image"][i] = {"seconds": dt, "regions_stored": int(len(np.unique(np.array(Image.open(os.path.join(args.data, "hint_from_regions_256", f"{i}.image_region64.png")).convert("RGB")).reshape(-1, 3), axis=0))),
                                  "regions_regenerated": _n_regions(h),
                                  "scribble_pixels_stored": int(stored_m.sum()), "scribble_pixels_regenerated": int(new_m.sum()),
                                  "scribble_mask_iou_vs_stored": iou,
                                  "dot_pixels_stored": int(stored_d.sum()), "dot_pixels_regenerated": int(new_d.sum())}
        def vis(mask, col):
            v = np.full((64, 64, 3), 255, np.uint8); c = np.array(col.convert("RGB")) if col is not None else np.zeros((64, 64, 3), np.uint8)
            v[mask] = c[mask] if col is not None else (0, 0, 0)
            return Image.fromarray(v).resize((256, 256), Image.NEAREST)
        stored_col = Image.open(os.path.join(args.data, "hint_from_regions_64_rev", f"{i}.image_scribble_col64.png"))
        new_col, _ = h.at_ratio(1.0, hint_type="scribble")
        new_col = np.asarray(new_col)
        if new_col.ndim == 3 and new_col.shape[2] == 3:
            new_col = new_col[:, :, ::-1]          # the library works in OpenCV BGR order
        rows.append([load_rgb(img), vis(stored_m, stored_col), vis(new_m, Image.fromarray(np.ascontiguousarray(new_col)))])
        rtitles.append(f"{i}\nIoU {iou:.2f}")
    grid(rows, ["illustration", "stored scribble map (paper)", "regenerated with hintauc"], rtitles, os.path.join(args.out, "grids", f"{tag}.png"), title=tag)
    json.dump(report, open(os.path.join(args.out, "metrics", f"{tag}.json"), "w"), indent=2)
    return report


def exp_E11(args, ids, ratios):
    """The scribble model of the earlier submission (release v1.4) on the same images."""
    return exp_E1(args, ids, ratios, models=("earlier_scribble",), tag="E11_earlier_checkpoints")


# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="examples_data directory (release v1.3 asset)")
    ap.add_argument("--ckpt_dir", required=True, help="directory with the release checkpoints (asset file names)")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--gpu", default="0", help="CUDA device index, -1 = CPU")
    ap.add_argument("--mode", default="quick", choices=list(MODES))
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiments, e.g. E1 E9")
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    args.data = os.path.abspath(args.data); args.ckpt_dir = os.path.abspath(args.ckpt_dir); args.out = os.path.abspath(args.out)
    if str(args.gpu) != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu); args.gpu = "0"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for d in ["grids", "metrics"]:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    cfg = MODES[args.mode]
    ids = args.ids or cfg["ids"]; ratios = cfg["ratios"]; exps = args.only or cfg["exps"]
    t_start = time.time(); done = {}
    for e in exps:
        log(f"==== {e} ====")
        t0 = time.time()
        if e == "E1s":
            done[e] = exp_E1(args, ids, ratios, models=("scribble",))
        elif e == "E1":
            done[e] = exp_E1(args, ids, ratios)
        elif e == "E2":
            done[e] = exp_E1(args, ids, ratios, models=("scribble",), tag="E2_tableIII_protocol", hint_order="label")
        elif e == "E3":
            done[e] = exp_E3(args, [i for i in ids if i in ("1019016", "1023016")] or ids[:2], [r for r in ratios if 0 < r < 1][:2] or [0.05, 0.25])
        elif e == "E4":
            done[e] = exp_E4(args, ids[:3], ratios)
        elif e == "E5":
            done[e] = exp_E5(args, ids[:2])
        elif e == "E8":
            done[e] = exp_E8(args, [i for i in ids if i in ("3421016", "4417016")] or ids[:2], [0.10])
        elif e == "E9":
            done[e] = exp_E9(args, ids)
        elif e == "E11":
            done[e] = exp_E11(args, ids[:4], ratios)
        else:
            sys.exit(f"unknown experiment id {e!r}; available: E1s E1 E2 E3 E4 E5 E8 E9 E11 (E6 = Diffusart-retrain and E7 = natural-image models are not automated, see README)")
        log(f"==== {e} done in {time.time() - t0:.0f}s")
    json.dump({"mode": args.mode, "ids": ids, "ratios": ratios, "experiments": list(done), "seconds": time.time() - t_start},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    log(f"all done in {(time.time() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
