#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import shutil
from glob import glob
from pathlib import Path
from contextlib import nullcontext
from collections import Counter

from natsort import natsorted

# ---- arg parse first (so we can set CUDA_VISIBLE_DEVICES before importing torch) ----
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_dir", type=str, required=True,
                   help="評価したい run ディレクトリ（例: .../det/dot/illust_dot_sketch1_inference_hint_ratio_50）")
    p.add_argument("--gt_root", type=str, required=True,
                   help="GT png のあるディレクトリ（例: .../images_test/color）")

    p.add_argument("--samples_subdir", type=str, default="test/samples_cfg_scale_5.00",
                   help="run_dir の下の相対パス")
    p.add_argument("--resize", type=int, default=256,
                   help="pixel系メトリクス用のresize（0で無効）")

    # 重要：defaultは '' にして外側の CUDA_VISIBLE_DEVICES を尊重
    p.add_argument("--gpu", type=str, default="",
                   help="CUDA_VISIBLE_DEVICES にセットする値。空文字 '' なら触らない。")

    p.add_argument("--metrics", nargs="+",
                   default=["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"],
                   help="subset of: mse psnr ssim lpips openclip dino dreamsim")

    # A_B -> idx = A*ab_base + B
    p.add_argument("--ab_base", type=int, default=0,
                   help="A_B.png を idx=A*ab_base+B に変換する ab_base。0なら自動推定。")

    p.add_argument("--out_csv", type=str, default="",
                   help="出力CSV。未指定なら <run_dir>/eval_metrics.csv")
    p.add_argument("--limit", type=int, default=0,
                   help="debug用: idx順で最初のN枚だけ評価（0で無効）")

    # dump options for verification
    p.add_argument("--dump_pairs_dir", type=str, default="",
                   help="指定すると、対応付けされた pred/gt ペア画像をここに保存（デバッグ用）")
    p.add_argument("--dump_pairs_max", type=int, default=50,
                   help="保存するペア数の上限（dump_pairs_dir指定時）")
    p.add_argument("--dump_pairs_concat", action="store_true",
                   help="pred|gt を横並び結合した画像も保存")
    p.add_argument("--dump_pairs_resized", action="store_true",
                   help="評価で使う resize 後テンソルの png も保存")

    return p.parse_args()


args = parse_args()
if args.gpu != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# ---- heavy imports after CUDA_VISIBLE_DEVICES ----
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TVF


_AB_RE = re.compile(r"^(\d+)_(\d+)\.png$", re.IGNORECASE)
_HINT_RE = re.compile(r"hint_ratio_([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_hint_ratio(run_name: str):
    m = _HINT_RE.search(run_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_ab_pair(png_name: str):
    m = _AB_RE.match(png_name)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    return a, b


def build_gt_index(gt_root: str):
    gt_root = os.path.expanduser(gt_root)
    # MUST match the inference dataloader's iteration order so that generated
    # A_B.png (idx=A*ab_base+B = dataloader position) pairs with the correct GT.
    # dataloader.py uses: sorted(f.split('/')[-1] for f in glob(...))  -> plain lexical sort of basenames.
    names = sorted(Path(f).name for f in glob(os.path.join(gt_root, "*.png")))
    paths = [os.path.join(gt_root, n) for n in names]
    return paths


def read_tensor_01(path: str, resize: int = 256):
    # CHW, uint8 -> float [0,1]
    img = torchvision.io.read_image(path)
    if img.shape[0] == 4:
        img = img[:3]  # drop alpha
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.float() / 255.0
    if resize and resize > 0:
        try:
            img = TVF.resize(img, [resize, resize], antialias=True)
        except TypeError:
            img = TVF.resize(img, [resize, resize])
    return img


def psnr_01(x, y, eps=1e-10):
    mse = torch.mean((x - y) ** 2)
    return (10.0 * torch.log10(1.0 / (mse + eps))).item()


def make_metric_bundle(metrics, device):
    metrics = set(m.lower() for m in metrics)
    bundle = {}

    # -------- MSE --------
    if "mse" in metrics:
        bundle["mse_loss"] = nn.MSELoss()

    # -------- SSIM --------
    if "ssim" in metrics:
        ssim_fn = None
        try:
            from torchmetrics.functional import structural_similarity_index_measure as ssim_fn  # type: ignore
        except Exception:
            ssim_fn = None

        if ssim_fn is not None:
            def compute_ssim(x, y):
                return float(ssim_fn(x.unsqueeze(0), y.unsqueeze(0), data_range=1.0).item())
            bundle["ssim"] = compute_ssim
        else:
            # fallback: ignite
            try:
                from ignite.engine import Engine
                from ignite.metrics import SSIM
            except Exception as e:
                raise RuntimeError(
                    "SSIM requires torchmetrics or pytorch-ignite. "
                    "Try: pip install torchmetrics pytorch-ignite"
                ) from e

            def eval_step(engine, batch):
                return batch

            engine = Engine(eval_step)
            metric = SSIM(data_range=1.0)
            metric.attach(engine, "ssim")

            def compute_ssim(x, y):
                state = engine.run([[x.unsqueeze(0), y.unsqueeze(0)]])
                return float(state.metrics["ssim"])
            bundle["ssim"] = compute_ssim

    # -------- LPIPS --------
    if "lpips" in metrics:
        import lpips  # type: ignore
        bundle["lpips_model"] = lpips.LPIPS(net="alex").to(device).eval()

    # -------- OpenCLIP --------
    if "openclip" in metrics:
        import open_clip  # type: ignore
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        bundle["openclip_model"] = model.to(device).eval()
        bundle["openclip_preprocess"] = preprocess

    # -------- DINOv2 --------
    if "dino" in metrics:
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        bundle["dino_processor"] = processor
        bundle["dino_model"] = model_dino

    # -------- DreamSim --------
    if "dreamsim" in metrics:
        from dreamsim import dreamsim  # type: ignore
        model_dreamsim, preprocess_dreamsim = dreamsim(pretrained=True)
        bundle["dreamsim_model"] = model_dreamsim.to(device).eval()
        bundle["dreamsim_preprocess"] = preprocess_dreamsim

    return bundle


@torch.no_grad()
def compute_pair_scores(pred_path, gt_path, bundle, device, resize):
    # pixel metrics use torch tensors
    pred_t = read_tensor_01(pred_path, resize=resize)
    gt_t = read_tensor_01(gt_path, resize=resize)

    out = {}

    if "mse_loss" in bundle:
        out["mse"] = float(bundle["mse_loss"](pred_t, gt_t).item())

    if "ssim" in bundle:
        out["ssim"] = float(bundle["ssim"](pred_t, gt_t))

    if "lpips_model" in bundle:
        # LPIPS expects NCHW in [-1, 1]
        x = (pred_t.unsqueeze(0).to(device) * 2.0) - 1.0
        y = (gt_t.unsqueeze(0).to(device) * 2.0) - 1.0
        out["lpips"] = float(bundle["lpips_model"](x, y).item())

    # feature metrics use PIL + each model's preprocess
    need_pil = any(k in bundle for k in ["openclip_model", "dino_model", "dreamsim_model"])
    if need_pil:
        from PIL import Image
        pred_pil = Image.open(pred_path).convert("RGB")
        gt_pil = Image.open(gt_path).convert("RGB")

    if "openclip_model" in bundle:
        preprocess = bundle["openclip_preprocess"]
        model = bundle["openclip_model"]
        amp = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
        with amp():
            x = preprocess(pred_pil).unsqueeze(0).to(device)
            y = preprocess(gt_pil).unsqueeze(0).to(device)
            fx = model.encode_image(x)
            fy = model.encode_image(y)
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)
            sim = torch.nn.functional.cosine_similarity(fx, fy, dim=-1).item()
        out["openclip"] = float((sim + 1.0) / 2.0)

    if "dino_model" in bundle:
        processor = bundle["dino_processor"]
        model_dino = bundle["dino_model"]

        in1 = processor(images=pred_pil, return_tensors="pt").to(device)
        o1 = model_dino(**in1).last_hidden_state.mean(dim=1)

        in2 = processor(images=gt_pil, return_tensors="pt").to(device)
        o2 = model_dino(**in2).last_hidden_state.mean(dim=1)

        sim = torch.nn.functional.cosine_similarity(o1, o2, dim=-1).item()
        out["dino"] = float((sim + 1.0) / 2.0)

    if "dreamsim_model" in bundle:
        model_ds = bundle["dreamsim_model"]
        pre_ds = bundle["dreamsim_preprocess"]
        x = pre_ds(pred_pil).to(device)
        y = pre_ds(gt_pil).to(device)
        out["dreamsim"] = float(model_ds(x, y).detach().cpu().item())

    return out, pred_t, gt_t


def resolve_sample_dir(run_dir: str, samples_subdir: str):
    sample_dir = os.path.join(run_dir, samples_subdir)
    if os.path.isdir(sample_dir):
        return sample_dir

    # fallback: test/samples_cfg_scale_*
    cand = glob(os.path.join(run_dir, "test", "samples_cfg_scale_*"))
    cand = [c for c in cand if os.path.isdir(c)]
    if len(cand) == 1:
        return cand[0]
    return None


def infer_ab_base_from_pairs(pairs, gt_len: int):
    """
    pairs: list[(a,b,name)]
    gt_len: len(gt_paths)
    """
    # group by a
    groups = {}
    for a, b, _ in pairs:
        groups.setdefault(a, []).append(b)

    # candidate voting:
    # - cand1 = max(b)+1 (B starts at 0)
    # - cand2 = max(b)-min(b)+1 (if B doesn't start at 0)
    vote = Counter()
    for a, bs in groups.items():
        if not bs:
            continue
        cand1 = max(bs) + 1
        cand2 = (max(bs) - min(bs) + 1)
        vote[cand1] += len(bs)
        vote[cand2] += len(bs)

    # candidates: top-voted + fallback global range
    candidates = [n for n, _ in vote.most_common(10)]
    max_b_all = max(b for _, b, _ in pairs)
    min_b_all = min(b for _, b, _ in pairs)
    candidates.append(max_b_all + 1)
    candidates.append(max_b_all - min_b_all + 1)

    # uniq preserving order, and filter non-positive
    seen = set()
    candidates = [x for x in candidates if x > 0 and not (x in seen or seen.add(x))]

    def score(N: int):
        idxs = [a * N + b for a, b, _ in pairs]
        uniq = set(idxs)
        collisions = len(idxs) - len(uniq)
        in_range = sum(0 <= i < gt_len for i in uniq)
        oob = len(uniq) - in_range
        # maximize in_range, then minimize oob, collisions
        return (in_range, -oob, -collisions)

    scored = [(N, score(N)) for N in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    print("[AB] candidates and score(in_range_unique, -oob, -collisions):")
    for N, sc in scored[:10]:
        print(f"  N={N:<6d} score={sc}  (vote_weight={vote.get(N,0)})")

    best_N = scored[0][0]
    return best_N


def main():
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"run_dir not found: {run_dir}")

    run_name = Path(run_dir).name
    hint_ratio = parse_hint_ratio(run_name)

    gt_paths = build_gt_index(args.gt_root)
    print(f"[GT] {len(gt_paths)} images (natsorted basename order)")

    sample_dir = resolve_sample_dir(run_dir, args.samples_subdir)
    if sample_dir is None:
        raise RuntimeError(f"sample dir not found under run_dir: {run_dir} (samples_subdir={args.samples_subdir})")

    print(f"[RUN] {run_name}")
    print(f"run_dir   : {run_dir}")
    print(f"sample_dir: {sample_dir}")
    print(f"hint_ratio: {hint_ratio}")

    # parse A_B files first (before loading heavy models)
    pred_files = [Path(p).name for p in glob(os.path.join(sample_dir, "*.png"))]
    pairs = []
    bad = 0
    for name in pred_files:
        ab = parse_ab_pair(name)
        if ab is None:
            bad += 1
            continue
        a, b = ab
        pairs.append((a, b, name))

    if not pairs:
        raise RuntimeError(f"no valid A_B.png in {sample_dir} (bad={bad})")

    # infer ab_base if needed
    if args.ab_base and args.ab_base > 0:
        ab_base = int(args.ab_base)
        print(f"[AB] ab_base (manual): {ab_base}")
    else:
        a_vals = [a for a, _, _ in pairs]
        b_vals = [b for _, b, _ in pairs]
        print(f"[AB] valid_files={len(pairs)} bad_name={bad}  A:[{min(a_vals)}..{max(a_vals)}]  B:[{min(b_vals)}..{max(b_vals)}]  uniqA={len(set(a_vals))} uniqB={len(set(b_vals))}")
        ab_base = infer_ab_base_from_pairs(pairs, gt_len=len(gt_paths))
        print(f"[AB] ab_base (auto): {ab_base}")

    # map: idx -> filename (keep first), store meta for debugging
    idx_to_file = {}
    idx_meta = {}
    dup = 0
    for a, b, name in pairs:
        idx = a * ab_base + b
        if idx in idx_to_file:
            dup += 1
            continue
        idx_to_file[idx] = name
        idx_meta[idx] = {"a": a, "b": b, "pred_name": name}

    idxs = sorted(idx_to_file.keys())
    if args.limit and args.limit > 0:
        idxs = idxs[: args.limit]

    if not idxs:
        raise RuntimeError(f"no idxs after mapping in {sample_dir} (bad={bad}, dup={dup})")

    # dump setup
    dump_dir = args.dump_pairs_dir.strip()
    dumped = 0
    dump_map_f = None
    dump_writer = None
    if dump_dir:
        dump_dir = os.path.abspath(os.path.expanduser(dump_dir))
        os.makedirs(dump_dir, exist_ok=True)
        map_path = os.path.join(dump_dir, "pairs_map.csv")
        dump_map_f = open(map_path, "w", newline="")
        dump_writer = csv.DictWriter(
            dump_map_f,
            fieldnames=["dump_i", "idx", "a", "b", "pred_path", "gt_path"]
        )
        dump_writer.writeheader()
        print(f"[DUMP] dir: {dump_dir}")
        print(f"[DUMP] map: {map_path}")

    # now load metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enabled = [m.lower() for m in args.metrics]
    enabled_set = set(enabled)
    bundle = make_metric_bundle(enabled, device)

    sums = {m: 0.0 for m in enabled_set}
    n = 0
    oob = 0

    for idx in idxs:
        if idx >= len(gt_paths) or idx < 0:
            oob += 1
            continue

        pred_path = os.path.join(sample_dir, idx_to_file[idx])
        gt_path = gt_paths[idx]

        scores, pred_t, gt_t = compute_pair_scores(pred_path, gt_path, bundle, device, args.resize)

        # PSNR (resize後tensorで計算)
        if "psnr" in enabled_set:
            scores["psnr"] = psnr_01(pred_t, gt_t)

        # dump matched pairs for visual inspection
        if dump_writer is not None and dumped < args.dump_pairs_max:
            meta = idx_meta.get(idx, {})
            a = meta.get("a", "")
            b = meta.get("b", "")
            prefix = f"{dumped:04d}_idx{idx:06d}_a{a}_b{b}"

            pred_dst = os.path.join(dump_dir, prefix + "_pred.png")
            gt_dst   = os.path.join(dump_dir, prefix + "_gt.png")
            shutil.copy2(pred_path, pred_dst)
            shutil.copy2(gt_path, gt_dst)

            if args.dump_pairs_resized:
                pred_u8 = (pred_t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
                gt_u8   = (gt_t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
                torchvision.io.write_png(pred_u8, os.path.join(dump_dir, prefix + "_pred_resized.png"))
                torchvision.io.write_png(gt_u8,   os.path.join(dump_dir, prefix + "_gt_resized.png"))

            if args.dump_pairs_concat:
                from PIL import Image
                pimg = Image.open(pred_path).convert("RGB")
                gimg = Image.open(gt_path).convert("RGB")
                w1, h1 = pimg.size
                w2, h2 = gimg.size
                H = max(h1, h2)
                canvas = Image.new("RGB", (w1 + w2, H), (0, 0, 0))
                canvas.paste(pimg, (0, (H - h1) // 2))
                canvas.paste(gimg, (w1, (H - h2) // 2))
                canvas.save(os.path.join(dump_dir, prefix + "_concat.png"))

            dump_writer.writerow({
                "dump_i": dumped,
                "idx": idx,
                "a": a,
                "b": b,
                "pred_path": pred_path,
                "gt_path": gt_path,
            })
            dump_map_f.flush()
            dumped += 1

        # accumulate
        for m in enabled_set:
            if m in scores:
                sums[m] += float(scores[m])
        n += 1

    # close dump map
    if dump_map_f is not None:
        dump_map_f.close()

    avgs = {m: (sums[m] / max(n, 1)) for m in enabled_set}

    print(f"\nmatched: {n}  (bad_name={bad}, dup_idx={dup}, oob_idx={oob})")
    for m in ["mse", "psnr", "ssim", "lpips", "openclip", "dino", "dreamsim"]:
        if m in enabled_set:
            print(f"{m:8s}: {avgs[m]:.6f}")

    out_path = args.out_csv.strip()
    if out_path == "":
        out_path = os.path.join(run_dir, "eval_metrics.csv")
    out_path = os.path.abspath(os.path.expanduser(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    row = {
        "run": run_name,
        "run_dir": run_dir,
        "hint_ratio": "" if hint_ratio is None else hint_ratio,
        "sample_dir": sample_dir,
        "ab_base": ab_base,
        "matched": n,
        "bad_name": bad,
        "dup_idx": dup,
        "oob_idx": oob,
        "dump_pairs_dir": dump_dir,
        "dumped_pairs": dumped if dump_dir else 0,
    }
    for m in enabled_set:
        row[m] = avgs[m]

    keys = ["run", "run_dir", "hint_ratio", "sample_dir", "ab_base",
            "matched", "bad_name", "dup_idx", "oob_idx",
            "dump_pairs_dir", "dumped_pairs"] + sorted(list(enabled_set))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(row)

    print(f"\n[CSV] wrote: {out_path}")


if __name__ == "__main__":
    main()
