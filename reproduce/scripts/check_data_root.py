#!/usr/bin/env python3
"""Check that a data root holds every file the DetFill loader will ask for, before a long inference run starts.

    python reproduce/scripts/check_data_root.py --data_root <DATA_ROOT> --domain illust --hint scribble --types 0 1 2
    python reproduce/scripts/check_data_root.py --data_root <DATASET_PATH> --domain real --hint dot --ids_file detfill/configs/real/test.txt

Flat layout (illust, detfill/README.md layout A): segmentations/originals/<id>.image.png, sketch/<source>/<id>.png,
hint_from_regions_64_rev/<id>.image_<hint>_{col,mask}64.png (or in a 0016/ sub-directory) and
hint_from_regions_256/<id>.image_region64.png. Split layout (real): <bucket>/<id>.image.png under the data root,
sketch/<source>/<bucket>/<id>.image.png, hint_from_regions_64_rev/<bucket>/... and hint_from_regions_256/<bucket>/...
(the bucket sub-directory is optional). Prints the missing files per category and exits with status 1 when
anything is missing (``--allow_missing`` only reports), so an incomplete data root is noticed before hours of GPU
time; e.g. the released ImageNet hint maps lack one of the 3,000 listed ids (n02667379_7324).
"""
import argparse
import os
import sys

SOURCES = {0: "pysimp", 1: "XDoG", 2: "sketchkeras"}


def read_ids(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            name = parts[-1]
            for suf in (".image.png", ".image.jpg", ".png", ".jpg", ".JPEG", ".jpeg"):
                if name.endswith(suf):
                    name = name[: -len(suf)]
                    break
            items.append((parts[-2] if len(parts) > 1 else None, name))
    return items


def first_existing(*paths):
    return next((p for p in paths if os.path.isfile(p)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--domain", default="illust", choices=["illust", "real"])
    ap.add_argument("--hint", default="scribble", choices=["scribble", "dot"])
    ap.add_argument("--types", nargs="*", type=int, default=[0, 1, 2], help="line-art sources to check (0 pysimp, 1 XDoG, 2 sketchkeras)")
    ap.add_argument("--ids_file", default=None, help="split list (default: reproduce/data/splits/test.txt for illust, detfill/configs/real/test.txt for real)")
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument("--show", type=int, default=5, help="how many missing ids to print per category")
    a = ap.parse_args()
    root = a.data_root
    here = os.path.dirname(os.path.abspath(__file__))
    ids_file = a.ids_file or (os.path.join(here, "..", "data", "splits", "test.txt") if a.domain == "illust"
                              else os.path.join(here, "..", "..", "detfill", "configs", "real", "test.txt"))
    items = read_ids(ids_file)
    missing = {}

    def need(category, path_or_none, idx):
        if path_or_none is None:
            missing.setdefault(category, []).append(idx)

    for bucket, i in items:
        b = bucket or ""
        if a.domain == "illust":
            need("original", first_existing(os.path.join(root, "segmentations", "originals", f"{i}.image.png")), i)
            for t in a.types:
                need(f"sketch/{SOURCES[t]}", first_existing(os.path.join(root, "sketch", SOURCES[t], f"{i}.png")), i)
            for kind in ("col", "mask"):
                need(f"hint_{a.hint}_{kind}", first_existing(
                    os.path.join(root, "hint_from_regions_64_rev", f"{i}.image_{a.hint}_{kind}64.png"),
                    os.path.join(root, "hint_from_regions_64_rev", "0016", f"{i}.image_{a.hint}_{kind}64.png")), i)
            need("region", first_existing(os.path.join(root, "hint_from_regions_256", f"{i}.image_region64.png"),
                                          os.path.join(root, "hint_from_regions_256", "0016", f"{i}.image_region64.png")), i)
        else:
            need("original", first_existing(os.path.join(root, b, f"{i}.image.png"), os.path.join(root, f"{i}.image.png")), i)
            for t in a.types:
                need(f"sketch/{SOURCES[t]}", first_existing(os.path.join(root, "sketch", SOURCES[t], b, f"{i}.image.png"),
                                                            os.path.join(root, "sketch", SOURCES[t], f"{i}.image.png")), i)
            for kind in ("col", "mask"):
                need(f"hint_{a.hint}_{kind}", first_existing(
                    os.path.join(root, "hint_from_regions_64_rev", b, f"{i}.image_{a.hint}_{kind}64.png"),
                    os.path.join(root, "hint_from_regions_64_rev", f"{i}.image_{a.hint}_{kind}64.png")), i)
            need("region", first_existing(os.path.join(root, "hint_from_regions_256", b, f"{i}.image_region64.png"),
                                          os.path.join(root, "hint_from_regions_256", f"{i}.image_region64.png")), i)
    print(f"{len(items)} ids from {os.path.relpath(ids_file)}; data root {root} ({a.domain}, hint {a.hint}, sources {a.types})")
    if not missing:
        print("all files present")
        return 0
    for cat, ids in missing.items():
        print(f"  missing {cat}: {len(ids)} (e.g. {', '.join(ids[:a.show])})")
    if a.allow_missing:
        print("continuing (allow_missing); the loader will stop at the first missing file unless these ids are removed from the split list")
        return 0
    print("error: incomplete data root; add the files, or remove the ids from the split list (the loader reads the whole list)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
