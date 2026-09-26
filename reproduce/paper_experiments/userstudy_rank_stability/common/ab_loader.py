"""Shared loader for all pairwise-comparison CSVs under usertest/csv_revs.

Each CSV row records one forced-choice trial:
    (user, alpha, image_idx, method_A, method_B, chosen_side, dt_sec)

Methods appear under URLs such as
    .../imgs/{method}/{alpha}/[conv_imgs/]{idx}.png
The "proposed" method has an extra `conv_imgs` subdir; `painttorch` and
`diffusart` do not. We extract method and alpha by locating the `imgs`
segment and reading the next two components.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

CSV_DIR = Path("/home/madorin/gitlab/tog2024/main/usertest/csv_revs")
ALPHAS = [0, 1, 3, 5, 10, 25, 50, 100]  # hint ratios (%)
METHODS = ["proposed", "painttorch", "diffusart"]


def _parse_url(url: str):
    """Return (method, alpha, img_idx) from a result-image URL.

    Returns (None, None, None) if parsing fails.
    """
    parts = url.split("/")
    try:
        i = parts.index("imgs")
    except ValueError:
        return None, None, None
    method = parts[i + 1]
    alpha = int(parts[i + 2])
    idx = os.path.splitext(parts[-1])[0]
    return method, alpha, idx


def _extract_user(path: Path) -> str:
    """e.g. selection_data_main_11_chen.csv -> 'chen'."""
    m = re.match(r"selection_data_main_(\d+)_([^.]+)\.csv", path.name)
    if not m:
        return path.stem
    return m.group(2)


def load_all(csv_dir: Path = CSV_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(csv_dir.glob("*.csv")):
        user = _extract_user(p)
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            df = pd.read_csv(fh)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.iterrows():
            sel_url = r["image_file_name"]
            ns_url = r["not_selected_image_file_name"]
            side = str(r["select image"]).strip()
            sel_m, sel_a, sel_i = _parse_url(sel_url)
            ns_m, ns_a, ns_i = _parse_url(ns_url)
            if sel_m is None or ns_m is None:
                continue
            if sel_a != ns_a or sel_i != ns_i:
                # only comparisons within same (alpha, image)
                continue
            try:
                t0 = pd.to_datetime(r["Start User Study"])
                t1 = pd.to_datetime(r["clicked time"])
                dt_sec = (t1 - t0).total_seconds()
            except Exception:
                dt_sec = np.nan
            rows.append(
                dict(
                    user=user,
                    alpha=sel_a,
                    image=sel_i,
                    selected_method=sel_m,
                    other_method=ns_m,
                    selected_side=side,
                    t_sec=dt_sec,
                )
            )
    out = pd.DataFrame(rows)
    # canonical (method_A, method_B) with A < B alphabetically, winner flag
    def _canon(row):
        m1, m2 = row.selected_method, row.other_method
        if m1 < m2:
            return pd.Series([m1, m2, 1])  # A selected
        else:
            return pd.Series([m2, m1, 0])  # B selected
    out[["method_A", "method_B", "A_won"]] = out.apply(_canon, axis=1)
    out["pair"] = out["method_A"] + "_vs_" + out["method_B"]
    return out


def delta_times(csv_dir: Path = CSV_DIR) -> pd.DataFrame:
    """Per-row response time using diff() of clicked-time within each CSV.

    The `Start User Study` is constant within a CSV; true per-trial latency
    is the gap between consecutive click timestamps. First row gap is NaN.
    """
    rows = []
    for p in sorted(csv_dir.glob("*.csv")):
        user = _extract_user(p)
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            df = pd.read_csv(fh)
        df.columns = [c.strip() for c in df.columns]
        df["clicked time"] = pd.to_datetime(df["clicked time"], errors="coerce")
        df["rt"] = df["clicked time"].diff().dt.total_seconds()
        for i, r in df.iterrows():
            sel_m, sel_a, sel_i = _parse_url(r["image_file_name"])
            ns_m, ns_a, ns_i = _parse_url(r["not_selected_image_file_name"])
            if sel_m is None or ns_m is None:
                continue
            rows.append(dict(user=user, trial=int(i), alpha=sel_a, image=sel_i,
                             selected_method=sel_m, other_method=ns_m,
                             rt_sec=r["rt"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_all()
    print("rows:", len(df))
    print("users:", df.user.nunique(), sorted(df.user.unique()))
    print("alphas:", sorted(df.alpha.unique()))
    print("methods:", sorted(set(df.selected_method) | set(df.other_method)))
    print(df.groupby(["pair", "alpha"]).size().unstack(fill_value=0))
