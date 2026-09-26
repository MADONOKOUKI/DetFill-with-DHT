#!/home/USER/anaconda3/envs/danbooregion/bin/python
"""
DanbooRegion segmentation (Stage 1): GT illustration -> region map (random colour per
region), saved at 512px NEAREST. Stage 2 (make_scribbling, py37 env) turns these region
maps into region64/scribble hints, identical to the SLIC/Quickshift pipeline.

Runs in the `danbooregion` conda env (TF1.15 + Keras 2.2.4). The DanbooRegion UNet+SRCNN
are loaded once at import of `segment` (ai.py loads weights from the code dir's CWD).

Usage (one shard):
  cd /scratch/USER/DanbooRegion/code
  danbooregion/python D_danbooregion_seg.py --shard 0 --nshards 16 \
      --src_root /home/.../main_exp_felzenszwalb_fixdot/illust \
      --txt_dir  /home/.../BBDM_seg_retrain/configs/illust \
      --out_root /scratch/USER/seg_retrain_R2-2/danboo_regions
Resume-safe (skips an id whose region png already exists).
"""
import os
# keep each shard single-ish threaded so many shards share a node without oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU
import sys, argparse, time, traceback
import numpy as np
import cv2

DRCODE = "/scratch/USER/DanbooRegion/code"
SEG_SUBDIR = "segmentation_regions/felzenszwalb"   # GT lives here per-dir
SIZE = 512   # save region map at 512px (NEAREST) — plenty for the 64px hint downsample


def load_ids(split, txt_dir):
    splits = ["train", "valid", "test"] if split == "all" else [split]
    items = []
    for sp in splits:
        with open(os.path.join(txt_dir, f"{sp}_paper.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fn = line.split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
                dn = line.split('/')[-2]
                items.append((dn, fn))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--txt_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.chdir(DRCODE)                 # ai.py loads weights relative to CWD
    sys.path.insert(0, DRCODE)
    from segment import segment      # imports ai.py -> builds session + loads UNet/SRCNN

    items = load_ids(args.split, args.txt_dir)
    mine = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    print(f"[danbooregion shard {args.shard}/{args.nshards}] {len(mine)}/{len(items)} ids", flush=True)

    done = skip = err = 0
    t0 = time.time()
    for k, (dn, fn) in enumerate(mine):
        out_dir = os.path.join(args.out_root, dn)
        out_path = os.path.join(out_dir, f"{fn}.region.png")
        if os.path.isfile(out_path):
            skip += 1
            continue
        gt = os.path.join(args.src_root, SEG_SUBDIR, dn, f"{fn}.image.png")
        img = cv2.imread(gt)
        if img is None:
            err += 1; print(f"  MISS {gt}", flush=True); continue
        try:
            _sk, region, _flat = segment(img)
            region = cv2.resize(region, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(out_path, region)
            done += 1
        except Exception:
            err += 1; print(f"  ERR {dn}/{fn}\n{traceback.format_exc()}", flush=True); continue
        if args.limit and done >= args.limit:
            break
        if k % 100 == 0 and k:
            print(f"  ..{k}/{len(mine)} done={done} skip={skip} err={err} "
                  f"{done/(time.time()-t0+1e-9):.2f} img/s", flush=True)
    print(f"[DONE danbooregion shard {args.shard}/{args.nshards}] done={done} skip={skip} "
          f"err={err} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
