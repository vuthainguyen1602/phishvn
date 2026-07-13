#!/usr/bin/env python3
"""
fetch_tranco.py — Download HARD benign domains from the Tranco top-list.

Why: our other benign source (tinnhiemmang "trusted websites") is a *curated whitelist* of
verified, well-established .vn organisations — an easy negative. A detector that only ever sees
"clearly trusted vs clearly malicious" reports optimistic numbers that collapse in deployment,
where the hard cases are ordinary, un-curated legitimate sites. Tranco is a research-grade,
manipulation-resistant ranking of real-world traffic, so sampling benign URLs from it (especially
the mid/long tail, which looks lexically messier) is the standard way to add realistic negatives
and measure external validity.

Output: data/raw/tranco/benign.csv  (columns: domain, rank, scraped_at) — picked up by
normalize_merge.load_tranco() and merged as benign, source="tranco".

RUN:
  # full list, stratified sample across rank bands (recommended for hard negatives)
  python fetch_tranco.py --n 6000
  # keep only Vietnamese domains
  python fetch_tranco.py --n 4000 --vn-only
  # avoid overlap with the whitelist benign you already have
  python fetch_tranco.py --n 6000 --exclude data/raw/tinnhiem_web/benign.csv
  # if the download is blocked, point at a Tranco CSV you downloaded manually
  python fetch_tranco.py --from-file top-1m.csv --n 6000

Tranco: Le Pochat et al., NDSS 2019, https://tranco-list.eu — please cite the list ID you used.
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import io
import os
import random
import sys
import zipfile

DEFAULT_URL = "https://tranco-list.eu/top-1m.csv.zip"


def _read_csv_rows(fp):
    """Yield (rank, domain) from a Tranco CSV (rank,domain) with or without header."""
    for row in csv.reader(fp):
        if not row:
            continue
        if len(row) >= 2 and row[0].strip().isdigit():
            yield int(row[0]), row[1].strip().lower()
        elif len(row) == 1 and row[0].strip():           # domain-only file
            yield None, row[0].strip().lower()


def load_ranking(url: str, from_file: str) -> list[tuple[int | None, str]]:
    if from_file:
        with open(from_file, newline="", encoding="utf-8", errors="ignore") as f:
            return list(_read_csv_rows(f))
    try:
        import requests
    except Exception:
        sys.exit("requests not installed. pip install requests, or use --from-file.")
    print(f"[i] downloading {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if url.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(r.content))
        name = z.namelist()[0]
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
            return list(_read_csv_rows(text))
    return list(_read_csv_rows(io.StringIO(r.text)))


def stratified_sample(rows, n, seed=42):
    """Sample across rank bands so we get easy (head) AND hard (long-tail) benign, not just the
    top few hundred mega-sites. Bands: 1-1k, 1k-10k, 10k-100k, 100k-1M; weighted toward the tail."""
    rng = random.Random(seed)
    if rows and rows[0][0] is None:                       # no ranks -> plain random sample
        doms = [d for _, d in rows]
        rng.shuffle(doms)
        return doms[:n]
    bands = [(1, 1_000, 0.10), (1_000, 10_000, 0.20),
             (10_000, 100_000, 0.30), (100_000, 10**9, 0.40)]
    picked = []
    for lo, hi, w in bands:
        pool = [d for rk, d in rows if rk is not None and lo <= rk < hi]
        rng.shuffle(pool)
        picked += pool[:int(n * w)]
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--from-file", default="", help="Local Tranco CSV instead of downloading.")
    ap.add_argument("--n", type=int, default=6000, help="How many benign domains to keep.")
    ap.add_argument("--vn-only", action="store_true", help="Keep only .vn domains.")
    ap.add_argument("--exclude", default="", help="CSV of domains to exclude (e.g. your whitelist).")
    ap.add_argument("--out", default="data/raw/tranco/benign.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_ranking(args.url, args.from_file)
    print(f"[i] {len(rows)} ranked domains loaded")
    if args.vn_only:
        rows = [(rk, d) for rk, d in rows if d.endswith(".vn")]
        print(f"[i] {len(rows)} after --vn-only")

    exclude = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude, newline="", encoding="utf-8", errors="ignore") as f:
            for r in csv.DictReader(f):
                dom = (r.get("domain") or r.get("url") or "").strip().lower()
                if dom:
                    exclude.add(dom.split("//")[-1].strip("/"))
        rows = [(rk, d) for rk, d in rows if d not in exclude]
        print(f"[i] {len(rows)} after excluding {len(exclude)} whitelist domains")

    doms = stratified_sample(rows, args.n, seed=args.seed)
    now = _dt.date.today().isoformat()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rank_of = {d: rk for rk, d in rows}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "rank", "scraped_at"])
        for d in doms:
            w.writerow([d, rank_of.get(d, ""), now])
    print(f"[+] wrote {len(doms)} benign domains -> {args.out}")


if __name__ == "__main__":
    main()
