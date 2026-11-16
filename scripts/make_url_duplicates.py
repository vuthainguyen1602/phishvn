#!/usr/bin/env python3
"""
make_url_duplicates.py — list the PhishVN rows that record the same URL twice, so a user can keep one
row per URL in a single step.

WHY. The released record table carries 16,855 URLs twice once the scheme and a leading "www." are set
aside: almost every undated community-feed phishing row is the "www."-prefixed copy of a dated row.
The official group-aware split keeps both copies in the same part (it groups by registrable domain),
so it does not leak; a user who draws their own row-level random split does, and the URL benchmark
study measures how much. The release keeps every row, so that its counts stay those of the published data
article, and ships this map beside it.

RULE. Rows are grouped by the URL with its scheme and a leading "www." removed, lower-cased. In each
group with more than one row the row kept is the dated one where there is one, then the one without
"www.", then the first in file order: the same rule as the one-row-per-URL arm of the URL benchmark
study (run_p2_dedup_audit.py), and the script checks that it keeps the same
rows when that arm's output is on disk.

Output columns: url_key, keep_id, drop_id, keep_reason, split, label. One line per dropped row.

RUN: python scripts/make_url_duplicates.py --data <dir holding dataset_url.csv> --out <csv>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BENCH_DEDUP = os.path.join(ROOT, "data", "processed", "p2", "dedup_audit", "dataset_url_dedup.csv")


def url_key(urls: pd.Series) -> pd.Series:
    return (urls.astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True))


def build(u: pd.DataFrame) -> pd.DataFrame:
    s = u["collected_at"].astype(str).str.strip().str.slice(0, 10)
    dated = (pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
             .fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))).notna()
    host = u["url_norm"].astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
    k = pd.DataFrame({"i": np.arange(len(u)), "key": url_key(u["url_norm"]), "dated": dated.to_numpy(),
                      "www": host.str.startswith("www.").to_numpy()})
    k = k.sort_values(["key", "dated", "www", "i"], ascending=[True, False, True, True])
    first = k.drop_duplicates("key").set_index("key")
    rest = k[k.duplicated("key")]
    rows = []
    for _, r in rest.iterrows():
        kept = first.loc[r["key"]]
        if kept["dated"] and not r["dated"]:
            reason = "dated"
        elif kept["www"] != r["www"]:
            reason = "no_www"
        else:
            reason = "first_in_file"
        ku, du = u.iloc[int(kept["i"])], u.iloc[int(r["i"])]
        if ku["split"] != du["split"] or ku["label"] != du["label"]:
            raise SystemExit(f"duplicate pair crosses a split or a label: {ku['id']} / {du['id']}")
        rows.append({"url_key": r["key"], "keep_id": ku["id"], "drop_id": du["id"], "keep_reason": reason,
                     "split": ku["split"], "label": ku["label"]})
    return pd.DataFrame(rows).sort_values(["url_key", "drop_id"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory holding dataset_url.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    u = pd.read_csv(os.path.join(a.data, "dataset_url.csv"), low_memory=False)
    d = build(u)
    kept = set(u["id"]) - set(d["drop_id"])
    if os.path.exists(BENCH_DEDUP):
        p2 = set(pd.read_csv(BENCH_DEDUP, usecols=["id"], low_memory=False)["id"])
        if p2 != kept:
            raise SystemExit(f"kept rows differ from the benchmark's one-row-per-URL arm ({len(kept)} vs {len(p2)})")
        print(f"[ok] kept rows identical to the benchmark's one-row-per-URL arm ({len(p2):,})")
    d.to_csv(a.out, index=False)
    print(f"[+] {a.out}: {len(d):,} duplicate rows; {len(kept):,} rows remain with one row per URL; "
          f"reasons {d.keep_reason.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
