#!/usr/bin/env python3
"""
fetch_chongluadao.py — Import the ChongLuaDao (chongluadao.vn) community phishing blacklist.

ChongLuaDao is Vietnam's largest community anti-scam project. Its public API
(api.chongluadao.vn/v2/blacklist) went offline (DNS no longer resolves; the official browser
extension points at the same dead host), so the freshest machine-readable data is the last
snapshot kept by the daily-scrape mirror:

  https://github.com/elliotwutingfeng/ChongLuaDao-Phishing-Blocklist   (urls.txt, MIT-licensed
  scraper; blacklist content belongs to the ChongLuaDao project — cite chongluadao.vn)

The snapshot is a bare domain list with NO per-entry first-seen dates, so downstream these rows
are an UNDATED, community-reported stratum: tier 'bronze', random (not temporal) split
assignment. The mirror's git history could later be replayed to reconstruct per-domain
first-seen dates (one commit per day, 2022..2024) — left as a future enhancement.

Output: data/raw/chongluadao/blacklist.csv  (domain, snapshot_date, scraped_at)
        -> picked up by normalize_merge.load_chongluadao(), label=phishing, tier=bronze.

RUN:
  python scripts/collect/fetch_chongluadao.py
  # or from a local copy of urls.txt:
  python scripts/collect/fetch_chongluadao.py --from-file urls.txt --snapshot-date 2024-05-16
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import os
import re

RAW_URL = ("https://raw.githubusercontent.com/elliotwutingfeng/"
           "ChongLuaDao-Phishing-Blocklist/main/urls.txt")
COMMITS_API = ("https://api.github.com/repos/elliotwutingfeng/"
               "ChongLuaDao-Phishing-Blocklist/commits?path=urls.txt&per_page=1")
HEADERS = {"User-Agent": "research (contact: nvthai@utc2.edu.vn)"}
HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default="", help="Local urls.txt instead of downloading.")
    ap.add_argument("--snapshot-date", default="",
                    help="Snapshot date (yyyy-mm-dd); auto-detected from the mirror's last "
                         "commit when downloading.")
    ap.add_argument("--out", default="data/raw/chongluadao/blacklist.csv")
    args = ap.parse_args()

    snapshot = args.snapshot_date
    if args.from_file:
        text = open(args.from_file, encoding="utf-8", errors="ignore").read()
    else:
        import requests
        print(f"[i] downloading {RAW_URL} ...")
        r = requests.get(RAW_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
        text = r.text
        if not snapshot:
            try:
                c = requests.get(COMMITS_API, headers=HEADERS, timeout=30).json()
                snapshot = c[0]["commit"]["committer"]["date"][:10]
            except Exception:
                snapshot = ""

    doms, seen = [], set()
    for ln in text.splitlines():
        d = ln.strip().lower()
        if not d or d.startswith("#"):
            continue
        d = re.sub(r"^https?://", "", d).split("/")[0]
        if HOST_RE.match(d) and d not in seen:
            seen.add(d)
            doms.append(d)

    now = _dt.date.today().isoformat()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "snapshot_date", "scraped_at"])
        for d in doms:
            w.writerow([d, snapshot, now])
    print(f"[+] wrote {len(doms)} domains -> {args.out} (snapshot {snapshot or 'unknown'})")


if __name__ == "__main__":
    main()
