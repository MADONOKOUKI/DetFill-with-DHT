#!/usr/bin/env python3
"""Compare the metrics produced by make_examples.py with the authors' reference run in expected/metrics/.

For every experiment JSON present in both places, prints the mean metric of each cell side by side and the
absolute difference, and flags cells whose difference exceeds the tolerance below. The diffusion sampler is
seeded, so on the same GPU class the outputs are identical; across GPU generations / CPU the per-image
metrics move slightly (PSNR by a few 0.1 dB at most in our tests), which the tolerances allow.
"""
import argparse, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = {"mse": 0.004, "psnr": 0.6, "ssim": 0.02, "lpips": 0.02, "openclip": 0.01, "dino": 0.01, "dreamsim": 0.015}


def walk_means(d, path=""):
    """yield (path, mean-dict) for every dict that has a 'mean' or 'per_ratio_mean' entry."""
    if isinstance(d, dict):
        if "mean" in d and isinstance(d["mean"], dict) and "psnr" in d["mean"]:
            yield path, d["mean"]
        if "per_ratio_mean" in d:
            for r, m in d["per_ratio_mean"].items():
                yield f"{path}/ratio={r}", m
        for k, v in d.items():
            if k in ("mean", "per_ratio_mean", "per_image"):
                continue
            yield from walk_means(v, f"{path}/{k}" if path else k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--expected", default=os.path.join(HERE, "expected", "metrics"))
    a = ap.parse_args()
    n_cells = n_flag = 0
    for f in sorted(glob.glob(os.path.join(a.out, "metrics", "*.json"))):
        name = os.path.basename(f)
        ref = os.path.join(a.expected, name)
        if not os.path.exists(ref):
            print(f"[{name}] no reference in expected/ (skipped)")
            continue
        new, old = json.load(open(f)), json.load(open(ref))
        new_c = dict(walk_means(new)); old_c = dict(walk_means(old))
        print(f"\n[{name}] cells compared: {len(set(new_c) & set(old_c))}")
        for k in sorted(set(new_c) & set(old_c)):
            flags = []
            for m in ("psnr", "lpips", "dreamsim"):
                if m in new_c[k] and m in old_c[k]:
                    d = abs(new_c[k][m] - old_c[k][m])
                    flags.append(f"{m} {new_c[k][m]:.4f} vs {old_c[k][m]:.4f} (Δ{d:.4f}{' !' if d > TOL[m] else ''})")
                    n_cells += 1; n_flag += d > TOL[m]
            print("  " + k + ": " + "; ".join(flags))
    print(f"\n{n_cells} metric cells compared, {n_flag} outside tolerance ({'OK' if n_flag == 0 else 'see ! marks'})")


if __name__ == "__main__":
    main()
