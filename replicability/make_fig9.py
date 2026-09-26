#!/usr/bin/env python3
"""Replicate Fig. 9 of the paper ("Importance of deterministic hint sampling") — no arguments needed.

Pipeline (paper settings throughout):
  1. DHT hint generation: hintauc.generate_hints(GT image, size=64)
     -> 64x64 region-label map + dot / scribble hints (Felzenszwalb -> skeleton -> longest path, region-mean colours;
        dot = L1 medoid of the in-region path pixels, the rule of the paper's stored hint maps)
  2. Two selections of 10 % of the regions (k = floor(0.1 * n), n = number of region labels):
       "Region-based sample (top 10 %)": the k largest regions (paper Sec. IV-B; identical to hintauc.HintResult.at_ratio)
       "Random sample (10 %)"          : k regions in an area-independent fixed order (ascending label order,
                                         the protocol of the paper's random-order comparison / Table III)
  3. DetFill inference with the released checkpoints (row 1: dot model, row 2: scribble model),
     fixed seed 1234 (main.py default), 200 sampling steps, SketchKeras line art shipped in data/.
  4. output/fig9.png  (columns: input sketch | random hints | colorization | region-based hints | colorization | GT)

Environment variables (all optional): GRSI_GPU (GPU id, "-1" = CPU; default: first GPU if available),
GRSI_CKPT_DIR (directory holding detfill_{dot,scribble}_illust_200ep.pth; default: replicability/checkpoints).
"""
import glob, json, os, re, shutil, subprocess, sys, time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import hintauc                                   # noqa: E402
from hintauc.hints import region_ids             # noqa: E402

ALPHA = 0.10                                     # hint ratio used in Fig. 9
SKETCH_TYPE = 2                                  # 0 = sketch simplification, 1 = XDoG, 2 = SketchKeras (used in Fig. 9)
ROWS = [("4731016", "dot"), ("4942016", "scribble")]   # (image id, hint type) as in the paper
CKPT = {"scribble": "detfill_scribble_illust_200ep.pth", "dot": "detfill_dot_illust_200ep.pth"}
OUT = os.path.join(HERE, "output")
DATA = os.path.join(HERE, "data")


def pick_device():
    if os.environ.get("GRSI_GPU"):
        return os.environ["GRSI_GPU"]
    try:
        import torch
        return "0" if torch.cuda.is_available() else "-1"
    except Exception:
        return "-1"


def write_hint_files(h, keep, ids, hint_type, dst_dir, image_id):
    """Write the canonical 64x64 hint files restricted to the selected regions."""
    area = np.isin(ids, keep)
    if hint_type == "dot":
        mask, color = h.dot_mask, h.dot_color
    else:
        mask, color = h.scribble_mask, h.scribble_color
    mask_sel = (mask * area).astype(np.uint8)
    color_sel = (color * area[:, :, None]).astype(np.uint8)
    os.makedirs(dst_dir, exist_ok=True)
    cv2.imwrite(os.path.join(dst_dir, f"{image_id}.image_{hint_type}_mask64.png"), mask_sel)
    cv2.imwrite(os.path.join(dst_dir, f"{image_id}.image_{hint_type}_col64.png"), color_sel)
    # visualisation (hint colours on a grey background, nearest-neighbour upsampled), as in the paper figures
    vis = np.full_like(color_sel, 128)
    vis[mask_sel > 0] = color_sel[mask_sel > 0]
    return cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST), int(mask_sel.astype(bool).sum())


def stage_dataset(stage, image_id, hint_type, h, keep, ids):
    """Create the flat evaluation layout expected by detfill/datasets/custom.py (one image)."""
    for sub in ["segmentations/originals", f"sketch/{['pysimp', 'XDoG', 'sketchkeras'][SKETCH_TYPE]}",
                "hint_from_regions_64_rev", "hint_from_regions_256"]:
        os.makedirs(os.path.join(stage, sub), exist_ok=True)
    shutil.copy(os.path.join(DATA, f"{image_id}.image.png"), os.path.join(stage, "segmentations/originals", f"{image_id}.image.png"))
    shutil.copy(os.path.join(DATA, "sketch", f"{image_id}.png"),
                os.path.join(stage, "sketch", ["pysimp", "XDoG", "sketchkeras"][SKETCH_TYPE], f"{image_id}.png"))
    cv2.imwrite(os.path.join(stage, "hint_from_regions_256", f"{image_id}.image_region64.png"), h.region)
    return write_hint_files(h, keep, ids, hint_type, os.path.join(stage, "hint_from_regions_64_rev"), image_id)


def make_config(hint_type, stage, dst):
    src = os.path.join(ROOT, "detfill", "configs", f"{hint_type}_illust.yaml")
    txt = open(src).read()
    txt = re.sub(r"dataset_path:\s*'[^']*'", f"dataset_path: '{stage}'", txt)
    txt = re.sub(r"scratch_root:\s*'[^']*'", f"scratch_root: '{stage}'", txt)
    txt = re.sub(r"(test:\s*\n\s*batch_size:)\s*\d+", r"\1 1", txt)          # 1 image per stage
    with open(dst, "w") as f:
        f.write(txt)


def run_detfill(hint_type, cfg, ckpt, result_path, gpu):
    cmd = [sys.executable, "main.py", "--config", cfg, "--resume_model", ckpt, "--sample_to_eval", "--save_top",
           "--gpu_ids", gpu, "--sample_ratio", "1.00", "--sketch_type", str(SKETCH_TYPE), "--result_path", result_path]
    print("  $", " ".join(cmd), flush=True)
    log = os.path.join(result_path, "detfill_stdout.log")
    os.makedirs(result_path, exist_ok=True)
    with open(log, "w") as lf:
        subprocess.run(cmd, cwd=os.path.join(ROOT, "detfill"), check=True, stdout=lf, stderr=subprocess.STDOUT)


def find_output(result_path, image_id):
    hits = [p for p in glob.glob(os.path.join(result_path, "**", f"{image_id}.image.png"), recursive=True)
            if "ground_truth" not in p and "condition" not in p]
    if not hits:
        raise RuntimeError(f"no DetFill output for {image_id} under {result_path}")
    return sorted(hits)[-1]


def fit(im, height):
    return im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)


def main():
    t0 = time.time()
    gpu = pick_device()
    ckpt_dir = os.environ.get("GRSI_CKPT_DIR", os.path.join(HERE, "checkpoints"))
    for k, f in CKPT.items():
        if not os.path.isfile(os.path.join(ckpt_dir, f)):
            sys.exit(f"checkpoint missing: {os.path.join(ckpt_dir, f)} (run fetch_checkpoints.sh)")
    os.makedirs(OUT, exist_ok=True)
    print(f"device: {'CPU' if gpu == '-1' else 'GPU ' + gpu}   output: {OUT}", flush=True)

    summary = {"alpha": ALPHA, "sketch_type": SKETCH_TYPE, "seed": 1234, "device": gpu, "rows": []}
    panels = []
    for image_id, hint_type in ROWS:
        print(f"[{image_id}] DHT hint generation ({hint_type})", flush=True)
        gt_path = os.path.join(DATA, f"{image_id}.image.png")
        h = hintauc.generate_hints(gt_path, size=64)                   # paper defaults (Felzenszwalb, FilFinder path, mean dot)
        ids = region_ids(h.region)
        vals, counts = np.unique(ids.reshape(-1), return_counts=True)
        n = len(vals); k = int(n * ALPHA)
        n_dot, n_scr = int((h.dot_mask > 0).sum()), int((h.scribble_mask > 0).sum())
        print(f"[{image_id}] {n} regions, {n_dot} dot px, {n_scr} scribble px, {h.failed_regions} failed extractions", flush=True)
        if n_dot == 0 or n_scr == 0 or h.failed_regions >= n:
            sys.exit("hint generation produced no hints - the FilFinder backend is not working in this environment "
                     "(check that fil_finder and astropy are installed); aborting instead of running DetFill on empty hints.")
        keep = {"region": vals[np.argsort(-counts)][:k],              # largest k regions (== HintResult.at_ratio)
                "random": np.sort(vals)[:k]}                           # first k labels in ascending label order
        row = {"image": image_id, "hint_type": hint_type, "n_regions": int(n), "k_selected": int(k),
               "failed_regions": int(h.failed_regions)}
        imgs = {}
        for variant in ("random", "region"):
            stage = os.path.join(OUT, "stage", f"{hint_type}_{variant}")
            vis, n_hint_px = stage_dataset(stage, image_id, hint_type, h, keep[variant], ids)
            if n_hint_px == 0:
                sys.exit(f"no hint pixels selected for {image_id}/{hint_type}/{variant}; aborting")
            cv2.imwrite(os.path.join(OUT, f"{image_id}_{hint_type}_{variant}_hints.png"), vis)
            cfg = os.path.join(OUT, "configs", f"{hint_type}_{variant}.yaml"); os.makedirs(os.path.dirname(cfg), exist_ok=True)
            make_config(hint_type, stage, cfg)
            result_path = os.path.join(OUT, "detfill", f"{hint_type}_{variant}")
            t1 = time.time()
            print(f"[{image_id}] DetFill ({hint_type}, {variant}: {k}/{n} regions, {n_hint_px} hint px)", flush=True)
            run_detfill(hint_type, cfg, os.path.join(ckpt_dir, CKPT[hint_type]), result_path, gpu)
            out = find_output(result_path, image_id)
            shutil.copy(out, os.path.join(OUT, f"{image_id}_{hint_type}_{variant}_colorization.png"))
            row[f"{variant}_hint_pixels_64"] = n_hint_px
            row[f"{variant}_seconds"] = round(time.time() - t1, 1)
            imgs[variant] = (Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), Image.open(out).convert("RGB"))
        summary["rows"].append(row)
        gt = Image.open(gt_path).convert("RGB")
        sketch = Image.open(os.path.join(DATA, "sketch", f"{image_id}.png")).convert("RGB")
        aspect = gt.width / gt.height
        def shaped(im):                                                # display at the GT aspect ratio, as in the paper
            return im.resize((round(300 * aspect), 300), Image.LANCZOS)
        panels.append([shaped(sketch), shaped(imgs["random"][0]), shaped(imgs["random"][1]),
                       shaped(imgs["region"][0]), shaped(imgs["region"][1]), fit(gt, 300)])

    # ---- assemble Fig. 9 ----
    titles = ["Input sketch", "Random sample 10%", "Colorization", "Region-based top 10%", "Colorization", "GT"]
    pad, top, H = 8, 34, 300
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    W = max(sum(im.width for im in r) + pad * (len(r) + 1) for r in panels)
    canvas = Image.new("RGB", (W, top + len(panels) * (H + pad) + pad), "white")
    d = ImageDraw.Draw(canvas)
    for r, ims in enumerate(panels):
        x, y = pad, top + r * (H + pad)
        for c, im in enumerate(ims):
            canvas.paste(im, (x, y))
            if r == 0:
                d.text((x + im.width // 2, 10), titles[c], fill="black", font=font, anchor="mt")
            x += im.width + pad
    fig = os.path.join(OUT, "fig9.png"); canvas.save(fig)
    summary["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDONE  {fig}\ncompare with {os.path.join(HERE, 'expected', 'fig9_paper.png')}\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
