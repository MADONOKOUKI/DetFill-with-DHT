#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


class RunningStat:
    """Welford's online algorithm for mean/std"""
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> float:
        return self.mean if self.n > 0 else float("nan")

    def get_std(self) -> float:
        if self.n < 2:
            return float("nan")
        return math.sqrt(self.M2 / (self.n - 1))


def read_all_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="MEAN",
                   help="出力行の name（例: illust_dot_det_mean）")
    p.add_argument("--in_csv", nargs="+", required=True,
                   help="calc_hint_auc_manual.py が吐いた summary CSV（複数可）")
    p.add_argument("--out_csv", type=str, required=True,
                   help="平均結果を書き出す先（上書き）")
    p.add_argument("--with_std", action="store_true",
                   help="標準偏差列も出力（列名: <col>_std）")
    p.add_argument("--only_first_row_each", action="store_true",
                   help="各入力CSVの先頭1行だけ使う（重複実行で行が増えた場合の保険）")
    args = p.parse_args()

    # ---- load all rows ----
    all_rows: List[Dict[str, str]] = []
    src_info: List[Tuple[str, int]] = []  # (path, n_rows_used)

    for pth in args.in_csv:
        pth = os.path.abspath(os.path.expanduser(pth))
        if not os.path.isfile(pth):
            raise RuntimeError(f"not found: {pth}")

        rows = read_all_rows(pth)
        if not rows:
            print(f"[WARN] empty: {pth}")
            continue

        if args.only_first_row_each:
            rows = [rows[0]]

        all_rows.extend(rows)
        src_info.append((pth, len(rows)))

    if not all_rows:
        raise RuntimeError("no rows loaded. check --in_csv")

    # ---- find HintAUC columns ----
    auc_cols = sorted({k for r in all_rows for k in r.keys() if k.startswith("HintAUC_")})
    if not auc_cols:
        raise RuntimeError("no columns starting with 'HintAUC_' found in inputs")

    # ---- compute mean/std per column ----
    stats: Dict[str, RunningStat] = {c: RunningStat() for c in auc_cols}

    used_rows = 0
    for r in all_rows:
        row_used_any = False
        for c in auc_cols:
            v = to_float(r.get(c, ""))
            if math.isfinite(v):
                stats[c].update(v)
                row_used_any = True
        if row_used_any:
            used_rows += 1

    # ---- build output row ----
    out_row: Dict[str, str] = {
        "name": args.name,
        "n_rows_used": str(used_rows),
        "n_sources": str(len(src_info)),
        "sources": " | ".join([f"{os.path.basename(p)}(rows={n})" for p, n in src_info]),
    }

    for c in auc_cols:
        m = stats[c].get_mean()
        out_row[c] = "" if not math.isfinite(m) else f"{m:.10g}"
        if args.with_std:
            s = stats[c].get_std()
            out_row[c + "_std"] = "" if not math.isfinite(s) else f"{s:.10g}"

    # ---- write CSV (overwrite) ----
    out_csv = os.path.abspath(os.path.expanduser(args.out_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    keys = ["name", "n_rows_used", "n_sources", "sources"] + auc_cols
    if args.with_std:
        keys += [c + "_std" for c in auc_cols]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow(out_row)

    # ---- print ----
    print(f"[OK] wrote: {out_csv}")
    for c in auc_cols:
        print(f"  {c:18s}: {out_row[c]}" + (f"   std={out_row.get(c+'_std','')}" if args.with_std else ""))


if __name__ == "__main__":
    main()
