"""Compare the two longest-path implementations on stored 64x64 region maps.

    python compare_path_methods.py --region_dir <dir with *.image_region64.png> [--limit 20]

For every region: FilFinder path (canonical: 3x3 dilation -> FilFinder2D medial
axis/prune/longest path, intersected with the region) vs. geodesic path (this
package, on the Zhang-Suen skeleton).  Reports path-pixel counts, IoU, dot
displacement, and failure counts.  Needs numpy, scikit-image, opencv, fil_finder.
"""
import argparse, glob, json, os, sys, time, warnings
import numpy as np, cv2
from skimage.morphology import skeletonize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hintauc.longest_path import geodesic_longest_path
except ImportError:
    from longest_path import geodesic_longest_path


def filfinder_path(skeleton_s):
    from fil_finder import FilFinder2D
    import astropy.units as u
    kernel = np.ones((3, 3), np.uint8)
    sk = cv2.dilate(skeleton_s.astype(np.uint8), kernel, iterations=1) * 255
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = FilFinder2D(sk, distance=250 * u.pc, mask=sk)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=3 * u.pix, skel_thresh=3 * u.pix, prune_criteria="length")
    return fil.skeleton_longpath.astype(np.uint8)


def dot_of(path):
    idx = np.argwhere(path > 0)
    if idx.size == 0:
        return None
    return (int(idx[:, 0].mean()), int(idx[:, 1].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region_dir", required=True)
    ap.add_argument("--limit", type=int, default=20, help="number of region maps (images)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    a = ap.parse_args()
    maps = sorted(glob.glob(os.path.join(a.region_dir, "*.image_region64.png")))[: a.limit]
    n = ff_fail = geo_empty = identical = 0
    ious, len_ff, len_geo, dot_dist, dot_out_ff, dot_out_geo = [], [], [], [], 0, 0
    t_ff = t_geo = 0.0
    for p in maps:
        region = cv2.imread(p); r = region.astype(np.uint64)
        ids = r[:, :, 0] * 255 * 255 + r[:, :, 1] * 255 + r[:, :, 2]
        for lab in np.unique(ids):
            m = ids == lab
            sk = skeletonize(m)
            n += 1
            t0 = time.time()
            try:
                pf = filfinder_path(sk) * m
            except Exception:
                pf = None
            t_ff += time.time() - t0
            t0 = time.time()
            pg = geodesic_longest_path(sk).mask.astype(np.uint8)
            t_geo += time.time() - t0
            if pg.sum() == 0:
                geo_empty += 1
            if pf is None or pf.sum() == 0:
                ff_fail += 1
                continue
            inter = np.logical_and(pf > 0, pg > 0).sum(); union = np.logical_or(pf > 0, pg > 0).sum()
            ious.append(inter / union if union else 1.0)
            len_ff.append(int(pf.sum())); len_geo.append(int(pg.sum()))
            identical += int(np.array_equal(pf > 0, pg > 0))
            df, dg = dot_of(pf), dot_of(pg)
            if df and dg:
                dot_dist.append(max(abs(df[0] - dg[0]), abs(df[1] - dg[1])))
                dot_out_ff += int(not m[df]); dot_out_geo += int(not m[dg])
    ious = np.array(ious); ld = np.array(len_geo) - np.array(len_ff); dd = np.array(dot_dist)
    res = dict(images=len(maps), regions=n, filfinder_failed=ff_fail, geodesic_empty=geo_empty,
               compared=int(len(ious)), identical_paths=identical, identical_frac=identical / max(len(ious), 1),
               iou_mean=float(ious.mean()), iou_median=float(np.median(ious)), iou_p10=float(np.percentile(ious, 10)),
               path_len_diff_geo_minus_ff=dict(mean=float(ld.mean()), median=float(np.median(ld)), min=int(ld.min()), max=int(ld.max())),
               dot_chebyshev_dist=dict(mean=float(dd.mean()), frac_zero=float((dd == 0).mean()), frac_le1=float((dd <= 1).mean()), max=int(dd.max())),
               dot_outside_region_frac=dict(filfinder=dot_out_ff / max(len(dd), 1), geodesic=dot_out_geo / max(len(dd), 1)),
               ms_per_region=dict(filfinder=1000 * t_ff / n, geodesic=1000 * t_geo / n))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
