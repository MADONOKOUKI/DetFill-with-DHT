#!/usr/bin/env python3
"""Fetch illustrations from Danbooru by post id and store them as the dataset's 512 x 512 copies.

    python reproduce/scripts/fetch_originals.py --ids 100016 1001016 --out <DATA_ROOT>/segmentations/originals
    python reproduce/scripts/fetch_originals.py --id_file reproduce/data/splits/test.txt --out <DATA_ROOT>/segmentations/originals
    python reproduce/scripts/fetch_originals.py --ids 4731016 --out originals --keep_raw originals_fullres   # also the raw file (E9)

The paper's illustrations are Danbooru posts: the Danbooru2021 ids are the post ids, and the original distribution
(https://gwern.net/danbooru2021) is no longer online. This repository does not redistribute the images except the
few that appear in the paper's figures; everything else is fetched by id with this script. For each id it asks the
Danbooru API for the post (https://danbooru.donmai.us/posts/<id>.json), downloads the file and resizes it to
512 x 512 with OpenCV's bilinear interpolation, which reproduces the dataset's ground-truth copies bit-exactly
(checked on the example ids). Posts that were deleted, or that need an account, are reported and skipped
(``--login`` / ``--api_key`` for the latter). The files are for research use under Danbooru's terms of service and
are not part of this repository.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

API = "https://danbooru.donmai.us/posts/{id}.json"
UA = "DetFill-with-DHT fetch_originals (research; https://github.com/MADONOKOUKI/DetFill-with-DHT)"


def read_ids(args):
    ids = list(args.ids or [])
    if args.id_file:
        with open(args.id_file) as f:
            for line in f:
                m = re.search(r"(\d+)(?:\.image)?(?:\.\w+)?\s*$", line.strip().split("/")[-1])
                if m:
                    ids.append(m.group(1))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out


def fetch(url, auth=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if auth:
        import base64
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", nargs="*", help="Danbooru post ids")
    ap.add_argument("--id_file", help="text file with one id (or <bucket>/<id>.image.png line) per line")
    ap.add_argument("--out", required=True, help="output directory for <id>.image.png (512 x 512)")
    ap.add_argument("--keep_raw", default=None, help="also store the downloaded file as <id>.<ext> in this directory")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--login", default=None, help="Danbooru user name (optional; some posts need an account)")
    ap.add_argument("--api_key", default=None, help="Danbooru API key (optional)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (Danbooru rate limits anonymous use)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    ids = read_ids(a)
    if not ids:
        sys.exit("no ids given (--ids or --id_file)")
    os.makedirs(a.out, exist_ok=True)
    if a.keep_raw:
        os.makedirs(a.keep_raw, exist_ok=True)
    auth = (a.login, a.api_key) if a.login and a.api_key else None
    ok, skipped = 0, []
    for n, i in enumerate(ids, 1):
        dst = os.path.join(a.out, f"{i}.image.png")
        if os.path.exists(dst) and not a.overwrite:
            ok += 1
            continue
        try:
            post = json.loads(fetch(API.format(id=i), auth))
            url = post.get("file_url")
            if not url:
                skipped.append((i, "no file_url (deleted post, or an account is needed)"))
                continue
            data = fetch(url, auth)
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                skipped.append((i, f"could not decode {url}"))
                continue
            if a.keep_raw:
                ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
                with open(os.path.join(a.keep_raw, f"{i}{ext}"), "wb") as f:
                    f.write(data)
            cv2.imwrite(dst, cv2.resize(arr, (a.size, a.size), interpolation=cv2.INTER_LINEAR))
            ok += 1
            print(f"[{n}/{len(ids)}] {i}: {arr.shape[1]}x{arr.shape[0]} -> {dst}", flush=True)
        except urllib.error.HTTPError as e:
            skipped.append((i, f"HTTP {e.code}"))
        except Exception as e:  # network errors etc.
            skipped.append((i, str(e)[:120]))
        time.sleep(a.sleep)
    print(f"done: {ok} of {len(ids)} images in {a.out}")
    for i, why in skipped:
        print(f"  skipped {i}: {why}")
    if skipped:
        sys.exit(2)


if __name__ == "__main__":
    main()
