#!/usr/bin/env python3
"""
DanbooRegion Stage 2 (py37 env): DanbooRegion region map + GT -> region64 / scribble_mask64
/ scribble_col64, using the SAME make_scribbling as the SLIC/Quickshift pipeline. Output
naming matches felz/SLIC/QS so the dataloader reads it unchanged.

Processes the region maps that are LOCAL on this node (glob), so it composes with the
per-node sharding of Stage 1. Run one or more instances per node with --shard/--nshards.

Usage (per shard, on a HOST_C node):
  py37 danbooregion_hints.py --shard 0 --nshards 32 \
     --region_root /path/to/data \
     --src_root /home/.../main_exp_felzenszwalb_fixdot/illust \
     --out_root /path/to/data
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, argparse, glob, time, traceback
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_hints import make_scribbling, SEG_SUBDIR, outputs_complete, write_outputs  # reuse exact hint logic + GT layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region_root", required=True)   # local danboo region maps (<dir>/<id>.region.png)
    ap.add_argument("--src_root", required=True)       # GT root (fixdot illust)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    regions = sorted(glob.glob(os.path.join(args.region_root, "*", "*.region.png")))
    mine = [p for i, p in enumerate(regions) if i % args.nshards == args.shard]
    print(f"[danboo-hints shard {args.shard}/{args.nshards}] {len(mine)}/{len(regions)} region maps", flush=True)

    done = skip = err = 0
    t0 = time.time()
    for k, rpath in enumerate(mine):
        dn = os.path.basename(os.path.dirname(rpath))
        fn = os.path.basename(rpath).replace(".region.png", "")
        out_dir = os.path.join(args.out_root, dn)
        prefix = os.path.join(out_dir, f"{fn}.image")
        if outputs_complete(prefix):
            skip += 1; continue
        gt = os.path.join(args.src_root, SEG_SUBDIR, dn, f"{fn}.image.png")
        img = cv2.imread(gt); region = cv2.imread(rpath)
        if img is None or region is None:
            err += 1; print(f"  MISS {gt if img is None else rpath}", flush=True); continue
        try:
            region64, mask64, col64 = make_scribbling(img, region)   # DanbooRegion colours are re-encoded by the library
            write_outputs(prefix, region64, mask64, col64)
            done += 1
        except Exception:
            err += 1; print(f"  ERR {dn}/{fn}\n{traceback.format_exc()}", flush=True); continue
        if args.limit and done >= args.limit: break
        if k % 200 == 0 and k:
            print(f"  ..{k}/{len(mine)} done={done} skip={skip} err={err} "
                  f"{done/(time.time()-t0+1e-9):.2f} img/s", flush=True)
    print(f"[DONE danboo-hints shard {args.shard}/{args.nshards}] done={done} skip={skip} "
          f"err={err} elapsed={time.time()-t0:.0f}s", flush=True)
    if err:
        sys.exit(2)


if __name__ == "__main__":
    main()
