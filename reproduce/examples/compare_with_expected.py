#!/usr/bin/env python3
"""Compare the metrics produced by make_examples.py with the authors' reference run in expected/metrics/.

Cells that carry per-image values are compared image by image on the ids present in both runs, and their means are
re-averaged over those common ids before being compared, so a `quick` run (4 images) is not held against the mean
of the 9-image reference. Cells that only hold a mean are compared when both runs evaluated the same ids, and
reported as "not comparable" otherwise. The diffusion sampler is seeded, so on the same GPU class the outputs are
identical; across GPU generations / CPU the per-image metrics move slightly (PSNR by a few 0.1 dB at most in our
tests), which the tolerances allow. Exit status 1 when a compared cell is outside the tolerance.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = {"mse": 0.004, "psnr": 0.6, "ssim": 0.02, "lpips": 0.02, "openclip": 0.01, "dino": 0.01, "dreamsim": 0.015}
COMPARE = ("psnr", "lpips", "dreamsim")


def _is_ratio(key):
    try:
        float(key)
        return True
    except (TypeError, ValueError):
        return False


def per_image_table(node):
    """{id: {metric: value}} from a per_image dict, whatever its nesting ({id: {...}} or {ratio: {id: {...}}})."""
    if not isinstance(node, dict) or not node:
        return {}
    first = next(iter(node.values()))
    if isinstance(first, dict) and all(isinstance(v, (int, float)) for v in first.values()):
        return {str(k): v for k, v in node.items()}                      # {id: {metric: value}}
    return {}


def walk(d, path=""):
    """yield (path, node) for every dict that has a 'mean' or a 'per_ratio_mean' entry (a metric cell)."""
    if isinstance(d, dict):
        if "mean" in d and isinstance(d["mean"], dict) and "psnr" in d["mean"]:
            yield path, d
        if "per_ratio_mean" in d and isinstance(d.get("per_image"), dict):
            for r, m in d["per_ratio_mean"].items():
                yield f"{path}/ratio={r}", {"mean": m, "per_image": (d["per_image"].get(r) if _is_ratio(next(iter(d["per_image"]), "x")) else
                                                                     {i: v.get(r) for i, v in d["per_image"].items() if isinstance(v, dict) and r in v})}
        for k, v in d.items():
            if k in ("mean", "per_ratio_mean", "per_image"):
                continue
            yield from walk(v, f"{path}/{k}" if path else k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--expected", default=os.path.join(HERE, "expected", "metrics"))
    a = ap.parse_args()
    n_cells = n_flag = n_skip = 0
    for f in sorted(glob.glob(os.path.join(a.out, "metrics", "*.json"))):
        name = os.path.basename(f)
        ref = os.path.join(a.expected, name)
        if not os.path.exists(ref):
            print(f"[{name}] no reference in expected/ (skipped)")
            continue
        new, old = json.load(open(f)), json.load(open(ref))
        same_ids = sorted(map(str, new.get("ids", []))) == sorted(map(str, old.get("ids", [])))
        new_c, old_c = dict(walk(new)), dict(walk(old))
        keys = sorted(set(new_c) & set(old_c))
        print(f"\n[{name}] ids: run {len(new.get('ids', []))}, reference {len(old.get('ids', []))}; cells in both: {len(keys)}")
        for k in keys:
            pi_new, pi_old = per_image_table(new_c[k].get("per_image")), per_image_table(old_c[k].get("per_image"))
            common = sorted(set(pi_new) & set(pi_old))
            parts = []
            if common:
                for m in COMPARE:
                    if all(m in pi_new[i] and m in pi_old[i] for i in common):
                        worst = max(abs(pi_new[i][m] - pi_old[i][m]) for i in common)
                        mn, mo = sum(pi_new[i][m] for i in common) / len(common), sum(pi_old[i][m] for i in common) / len(common)
                        flag = worst > TOL[m]
                        parts.append(f"{m} {mn:.4f} vs {mo:.4f} over {len(common)} common image(s), max per-image Δ{worst:.4f}{' !' if flag else ''}")
                        n_cells += 1; n_flag += flag
            elif same_ids and "mean" in new_c[k] and "mean" in old_c[k]:
                for m in COMPARE:
                    if m in new_c[k]["mean"] and m in old_c[k]["mean"]:
                        d = abs(new_c[k]["mean"][m] - old_c[k]["mean"][m])
                        parts.append(f"{m} {new_c[k]['mean'][m]:.4f} vs {old_c[k]['mean'][m]:.4f} (Δ{d:.4f}{' !' if d > TOL[m] else ''})")
                        n_cells += 1; n_flag += d > TOL[m]
            else:
                parts.append("not comparable: different image sets and no per-image values")
                n_skip += 1
            print("  " + k + ": " + "; ".join(parts))
    print(f"\n{n_cells} metric cells compared, {n_flag} outside tolerance, {n_skip} not comparable "
          f"({'OK' if n_flag == 0 else 'see ! marks'})")
    return 1 if n_flag else 0


if __name__ == "__main__":
    sys.exit(main())
