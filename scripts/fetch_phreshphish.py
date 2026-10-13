#!/usr/bin/env python3
"""
fetch_phreshphish.py — the dated columns of PhreshPhish (Dalton et al., 2025; CC BY 4.0), without its HTML.

PhreshPhish carries a collection date on every row of both classes, which the PhishVN corpus does
not, so a stream built from it has a true order to compare a constructed one against. Only
sha256, url, label and date are read: parquet stores columns separately, so the HTML column (most
of the ~20 GB) is never downloaded. The revision is pinned.

Writes data/external/phreshphish/phreshphish_dated.csv (both splits, with a `split` column) and
phreshphish_manifest.json (revision, per-file row counts, the output's SHA-256).

RUN:  python scripts/fetch_phreshphish.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

REPO = "datasets/phreshphish/phreshphish"
REVISION = "eabec4b7a66324b79cc8a0ad856d1731dc26fe1a"   # v1.0.1, 2026-02-09
COLUMNS = ["sha256", "url", "label", "date"]
OUT = os.path.join(ROOT, "data", "external", "phreshphish")


def main():
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{REPO}@{REVISION}/data/*.parquet"))
    if not files:
        raise SystemExit("no parquet files at the pinned revision")
    os.makedirs(OUT, exist_ok=True)
    parts, counts = [], {}
    for f in files:
        with fs.open(f, "rb", block_size=1 << 20) as fh:
            t = pq.read_table(fh, columns=COLUMNS).to_pandas()
        name = os.path.basename(f)
        t["split"] = name.split("-")[0]
        counts[name] = len(t)
        parts.append(t)
        print(f"[i] {name}: {len(t):,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    path = os.path.join(OUT, "phreshphish_dated.csv")
    df.to_csv(path, index=False)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump({"repo": REPO, "revision": REVISION, "columns": COLUMNS, "files": counts,
               "rows": len(df), "sha256": sha}, open(os.path.join(OUT, "phreshphish_manifest.json"), "w"),
              indent=2)
    print(f"[+] {path}: {len(df):,} rows, sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
