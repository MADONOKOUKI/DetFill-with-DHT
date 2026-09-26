#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_eval_dense_curve.py — R2-1 dense hint-ratio sweep (LPIPS, DreamSim, SSIM, PSNR)

For each (sketch_type, hint_ratio) directory under R2-1 results, compute four
quality metrics between generated images (`200/`) and ground truth (`ground_truth/`),
matched by basename. Outputs:

  - <out_dir>/per_image.csv         long form, ~909k rows (audit / debug)
  - <out_dir>/per_ratio_summary.csv 303 rows (sketch × ratio, mean+std+n) ← the headline file

Resumable: if per_image.csv exists, rows already present (sketch, ratio, image)
are skipped on restart.

Reused logic (metric bundle, IO conventions) is adapted from
  /home/USER/gitlab/labrepo/main/Evaluation_paper/eval_single_run.py
but functions are inlined here because that script runs argparse at import time.
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

# ─── arg parse first (set CUDA_VISIBLE_DEVICES before torch import) ────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str,
                   default="/home/USER/gitlab/labrepo/main/BBDM_region_revision/tvcg_mr/R2-1/results/illust/scribble",
                   help="root containing {0,1,2}/<ratio>/{200,ground_truth}/")
    p.add_argument("--sketches", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ratios", nargs="*", default=None,
                   help="empty = all dirs found; else explicit list e.g. '0.0 0.5 1.0'")
    p.add_argument("--metrics", nargs="+",
                   default=["psnr", "ssim", "lpips", "dreamsim"],
                   help="subset of: mse psnr ssim lpips openclip dino dreamsim")
    p.add_argument("--per_image_name", type=str, default="per_image.csv",
                   help="CSV filename inside out_dir (default per_image.csv). "
                        "Use a different name when running extra metrics in parallel "
                        "with an existing per_image.csv.")
    p.add_argument("--gpu", type=str, default="0",
                   help="value for CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader workers (NFS friendly, keep small)")
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="debug: only first N pairs per ratio (0 = no limit)")
    p.add_argument("--no_resume", action="store_true",
                   help="ignore existing per_image.csv (overwrite)")
    p.add_argument("--no_summary", action="store_true",
                   help="skip building per_ratio_summary.csv at end "
                        "(useful when this process is one of several writing per_image.csv concurrently)")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# ─── heavy imports ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader
from PIL import Image


SKETCH_NAMES = {0: "XDoG", 1: "pysimp", 2: "sketchkeras"}
METRIC_LIST = ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]


# ─── tensor IO (from eval_single_run.py) ──────────────────────────────────
def read_tensor_01(path: str, resize: int = 256) -> torch.Tensor:
    img = torchvision.io.read_image(path)
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.float() / 255.0
    if resize and resize > 0:
        try:
            img = TVF.resize(img, [resize, resize], antialias=True)
        except TypeError:
            img = TVF.resize(img, [resize, resize])
    return img


# ─── metric bundle ──────────────────────────────────────────────────────────
def make_metric_bundle(metrics, device):
    metrics = set(m.lower() for m in metrics)
    bundle = {"_metrics": metrics}

    if "ssim" in metrics:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        bundle["ssim_fn"] = ssim_fn

    if "lpips" in metrics:
        import lpips
        bundle["lpips_model"] = lpips.LPIPS(net="alex").to(device).eval()

    if "dreamsim" in metrics:
        from dreamsim import dreamsim
        model_ds, pre_ds = dreamsim(pretrained=True)
        bundle["dreamsim_model"] = model_ds.to(device).eval()
        bundle["dreamsim_preprocess"] = pre_ds

    if "openclip" in metrics:
        import open_clip
        model_oc, _, pre_oc = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        bundle["openclip_model"] = model_oc.to(device).eval()
        bundle["openclip_preprocess"] = pre_oc

    if "dino" in metrics:
        from transformers import AutoImageProcessor, AutoModel
        bundle["dino_processor"] = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        bundle["dino_model"] = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    return bundle


@torch.no_grad()
def compute_batch_scores(gen_t, gt_t, gen_pils, gt_pils, bundle, device):
    """
    gen_t, gt_t: (B, 3, H, W) float [0,1] on device
    gen_pils, gt_pils: list[PIL.Image] (RGB)
    returns dict {metric_name: list[float] (length B)}
    """
    metrics = bundle["_metrics"]
    out = {}
    B = gen_t.size(0)

    if "mse" in metrics or "psnr" in metrics:
        mse_per = ((gen_t - gt_t) ** 2).flatten(1).mean(1)         # (B,)
        if "mse" in metrics:
            out["mse"] = mse_per.cpu().tolist()
        if "psnr" in metrics:
            psnr_per = 10.0 * torch.log10(1.0 / (mse_per + 1e-10))
            out["psnr"] = psnr_per.cpu().tolist()

    if "ssim" in metrics:
        ssim_fn = bundle["ssim_fn"]
        try:
            # torchmetrics returns scalar by default; per-sample needs reduction='none'
            vals = ssim_fn(gen_t, gt_t, data_range=1.0, reduction="none")
            out["ssim"] = vals.cpu().tolist() if vals.dim() > 0 else [float(vals.item())] * B
        except TypeError:
            # fallback: per-image loop
            vals = [float(ssim_fn(gen_t[i:i+1], gt_t[i:i+1], data_range=1.0).item()) for i in range(B)]
            out["ssim"] = vals

    if "lpips" in metrics:
        a = gen_t * 2.0 - 1.0
        b = gt_t * 2.0 - 1.0
        vals = bundle["lpips_model"](a, b).flatten()
        out["lpips"] = vals.cpu().tolist()

    if "dreamsim" in metrics:
        ds_pre = bundle["dreamsim_preprocess"]
        gen_ds = torch.cat([ds_pre(p) for p in gen_pils], dim=0).to(device)
        gt_ds  = torch.cat([ds_pre(p) for p in gt_pils],  dim=0).to(device)
        vals = bundle["dreamsim_model"](gen_ds, gt_ds).flatten()
        out["dreamsim"] = vals.cpu().tolist()

    if "openclip" in metrics:
        pre_oc = bundle["openclip_preprocess"]
        model_oc = bundle["openclip_model"]
        gen_oc = torch.stack([pre_oc(p) for p in gen_pils]).to(device)
        gt_oc  = torch.stack([pre_oc(p) for p in gt_pils]).to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            fx = model_oc.encode_image(gen_oc)
            fy = model_oc.encode_image(gt_oc)
        fx = fx / fx.norm(dim=-1, keepdim=True)
        fy = fy / fy.norm(dim=-1, keepdim=True)
        sims = torch.nn.functional.cosine_similarity(fx, fy, dim=-1)
        out["openclip"] = ((sims + 1.0) / 2.0).float().cpu().tolist()

    if "dino" in metrics:
        proc = bundle["dino_processor"]
        model_dino = bundle["dino_model"]
        in_gen = proc(images=gen_pils, return_tensors="pt").to(device)
        in_gt  = proc(images=gt_pils,  return_tensors="pt").to(device)
        f_gen = model_dino(**in_gen).last_hidden_state.mean(dim=1)
        f_gt  = model_dino(**in_gt).last_hidden_state.mean(dim=1)
        sims = torch.nn.functional.cosine_similarity(f_gen, f_gt, dim=-1)
        out["dino"] = ((sims + 1.0) / 2.0).cpu().tolist()

    return out


# ─── dataset (parallel IO via DataLoader) ──────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs, resize):
        self.pairs = pairs        # list of (sketch, ratio, name, gen_path, gt_path)
        self.resize = resize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sk, rt, nm, gp, gtp = self.pairs[idx]
        gen_t = read_tensor_01(gp, self.resize)
        gt_t  = read_tensor_01(gtp, self.resize)
        gen_pil = Image.open(gp).convert("RGB")
        gt_pil  = Image.open(gtp).convert("RGB")
        return gen_t, gt_t, gen_pil, gt_pil, sk, rt, nm


def pair_collate(batch):
    """Custom collate so we can keep PIL list (default collate fails on PIL)."""
    gen_ts  = torch.stack([b[0] for b in batch])
    gt_ts   = torch.stack([b[1] for b in batch])
    gen_pils = [b[2] for b in batch]
    gt_pils  = [b[3] for b in batch]
    sks    = [b[4] for b in batch]
    rts    = [b[5] for b in batch]
    nms    = [b[6] for b in batch]
    return gen_ts, gt_ts, gen_pils, gt_pils, sks, rts, nms


# ─── walk results dir, find pairs ──────────────────────────────────────────
def list_pairs_for(sketch, ratio_str, results_root):
    """returns list of (sketch, ratio_str, name, gen_path, gt_path)"""
    rdir = Path(results_root) / str(sketch) / ratio_str
    gen_dir = rdir / "200"
    gt_dir  = rdir / "ground_truth"
    if not gen_dir.is_dir() or not gt_dir.is_dir():
        return []

    # generated names: {id}.image.png
    gen_files = {p.name: str(p) for p in gen_dir.glob("*.image.png")}
    gt_files  = {p.name: str(p) for p in gt_dir.glob("*.image.png")}
    common = sorted(gen_files.keys() & gt_files.keys())

    return [(sketch, ratio_str, n, gen_files[n], gt_files[n]) for n in common]


def list_ratios_for(sketch, results_root):
    sdir = Path(results_root) / str(sketch)
    if not sdir.is_dir():
        return []
    ratios = []
    for d in sdir.iterdir():
        if d.is_dir():
            try:
                _ = float(d.name)   # accept "0.0", "0.5", "1.0", "0.42", etc.
                ratios.append(d.name)
            except ValueError:
                continue
    # sort by float value
    ratios.sort(key=lambda x: float(x))
    return ratios


# ─── resume support ─────────────────────────────────────────────────────────
def load_done_keys(per_image_path):
    """Load (sketch, ratio, name) tuples already present in per_image.csv."""
    done = set()
    if not os.path.exists(per_image_path):
        return done
    with open(per_image_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((int(row["sketch"]), row["ratio"], row["image_name"]))
            except (KeyError, ValueError):
                continue
    return done


# ─── summary ────────────────────────────────────────────────────────────────
def build_summary(per_image_path, per_ratio_path, metrics):
    """Aggregate per_image to (sketch × ratio): mean, std, n per metric."""
    import pandas as pd
    df = pd.read_csv(per_image_path)
    if df.empty:
        print("[summary] per_image.csv is empty; nothing to aggregate")
        return

    grouped = df.groupby(["sketch", "ratio"], dropna=False)
    rows = []
    for (sk, rt), g in grouped:
        row = {
            "sketch": int(sk),
            "sketch_name": SKETCH_NAMES.get(int(sk), "?"),
            "ratio": rt,
            "n": len(g),
        }
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"]  = float(g[m].std())
        rows.append(row)

    out = pd.DataFrame(rows)
    # Sort by (sketch ASC, ratio float ASC)
    out["__rf"] = out["ratio"].astype(float)
    out.sort_values(["sketch", "__rf"], inplace=True)
    out.drop(columns="__rf", inplace=True)

    cols = ["sketch", "sketch_name", "ratio", "n"]
    for m in metrics:
        cols += [f"{m}_mean", f"{m}_std"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(per_ratio_path, index=False)
    print(f"[summary] wrote {per_ratio_path}  ({len(out)} rows)")


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / args.per_image_name
    # per_ratio_summary name follows per_image_name (per_image_extra.csv → per_ratio_summary_extra.csv)
    if args.per_image_name == "per_image.csv":
        per_ratio_path = out_dir / "per_ratio_summary.csv"
    else:
        stem = Path(args.per_image_name).stem.replace("per_image", "per_ratio_summary")
        per_ratio_path = out_dir / f"{stem}.csv"

    print(f"[cfg] results_root = {args.results_root}")
    print(f"[cfg] sketches     = {args.sketches}")
    print(f"[cfg] metrics      = {args.metrics}")
    print(f"[cfg] gpu          = {args.gpu}")
    print(f"[cfg] batch_size   = {args.batch_size}")
    print(f"[cfg] workers      = {args.workers}")
    print(f"[cfg] resize       = {args.resize}")
    print(f"[cfg] out_dir      = {out_dir}")
    print(f"[cfg] limit        = {args.limit}")
    print(f"[cfg] no_resume    = {args.no_resume}")

    # build work list
    tasks = []  # list of (sketch, ratio_str)
    for sk in args.sketches:
        if args.ratios:
            ratios = list(args.ratios)
        else:
            ratios = list_ratios_for(sk, args.results_root)
        for rt in ratios:
            tasks.append((sk, rt))

    print(f"[plan] {len(tasks)} (sketch, ratio) tasks queued")

    # resume keys
    if args.no_resume and per_image_path.exists():
        per_image_path.unlink()
    done_keys = load_done_keys(per_image_path) if not args.no_resume else set()
    print(f"[resume] {len(done_keys)} pairs already done in per_image.csv")

    # device + bundle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    bundle = make_metric_bundle(args.metrics, device)
    print(f"[bundle] loaded: {sorted(bundle['_metrics'])}")

    # CSV writer (append mode if resume)
    fieldnames = ["sketch", "sketch_name", "ratio", "image_name"] + list(args.metrics)
    write_header = not per_image_path.exists()
    fimg = open(per_image_path, "a", newline="")
    img_writer = csv.DictWriter(fimg, fieldnames=fieldnames)
    if write_header:
        img_writer.writeheader()
        fimg.flush()

    total_processed = 0
    start = time.time()

    try:
        for ti, (sk, rt) in enumerate(tasks):
            pairs_all = list_pairs_for(sk, rt, args.results_root)
            pairs = [p for p in pairs_all if (p[0], p[1], p[2]) not in done_keys]
            if args.limit and args.limit > 0:
                pairs = pairs[: args.limit]
            sk_name = SKETCH_NAMES.get(sk, "?")
            if not pairs:
                print(f"[task {ti+1}/{len(tasks)}] skip sketch={sk}({sk_name}) ratio={rt}: 0 to do (have {len(pairs_all)})")
                continue
            print(f"[task {ti+1}/{len(tasks)}] sketch={sk}({sk_name}) ratio={rt}: {len(pairs)} / {len(pairs_all)} pairs")

            ds = PairDataset(pairs, resize=args.resize)
            loader = DataLoader(ds,
                                batch_size=args.batch_size,
                                num_workers=args.workers,
                                shuffle=False,
                                collate_fn=pair_collate,
                                pin_memory=False)

            t0 = time.time()
            n_done = 0
            for gen_t, gt_t, gen_pils, gt_pils, sks, rts, nms in loader:
                gen_t = gen_t.to(device, non_blocking=True)
                gt_t  = gt_t.to(device, non_blocking=True)
                scores = compute_batch_scores(gen_t, gt_t, gen_pils, gt_pils, bundle, device)

                B = len(nms)
                for i in range(B):
                    row = {
                        "sketch": sks[i],
                        "sketch_name": SKETCH_NAMES.get(sks[i], "?"),
                        "ratio": rts[i],
                        "image_name": nms[i],
                    }
                    for m in args.metrics:
                        row[m] = scores[m][i] if m in scores else ""
                    img_writer.writerow(row)
                fimg.flush()
                n_done += B
                total_processed += B

            dt = time.time() - t0
            rate = n_done / max(dt, 1e-6)
            eta_total = (len(tasks) - (ti + 1)) * dt  # rough: assume same speed per task
            print(f"  → {n_done} pairs in {dt:.1f}s  ({rate:.1f}/s, "
                  f"elapsed {time.time()-start:.0f}s, est-remaining ~{eta_total:.0f}s)")
    finally:
        fimg.close()

    # build summary (skip if --no_summary, e.g. parallel writers — let an outer launcher do it)
    if not args.no_summary:
        build_summary(per_image_path, per_ratio_path, args.metrics)
    else:
        print("[summary] skipped (--no_summary)")

    print(f"\n[done] total processed this run: {total_processed}")
    print(f"[done] per_image.csv         : {per_image_path}")
    print(f"[done] per_ratio_summary.csv : {per_ratio_path}")


if __name__ == "__main__":
    main()
