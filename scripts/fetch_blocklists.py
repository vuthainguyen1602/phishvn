#!/usr/bin/env python3
"""
fetch_blocklists.py — a dated snapshot of the two public phishing blocklists, once a day.

WHY. The subdomain-evasion audit could not evaluate its blocklist question (RQ3) because the
only reference on disk was a 300-URL OpenPhish file from 14 July 2026, before the first capture
it would have been compared with, and fetching today's feed answers a different historical
question. A blocklist snapshot is evidence only for observations it could have preceded, so the
snapshots have to accumulate alongside the revisits, day by day, with their hashes recorded.

WHAT IT STORES. data/raw/blocklists/<feed>/<YYYY-MM-DD>.<ext>.gz for each feed, and one line per
fetch in data/raw/blocklists/manifest.csv (date, feed, url, http status, bytes, sha256, rows).
A failed fetch is a manifest row with the status and no file; it is never a silent gap.

FEEDS. OpenPhish community feed (feed.txt, one URL per line, refreshed every 12 h upstream) and
PhishTank's online-valid CSV (no API key, which PhishTank rate-limits; one fetch a day is well
inside it). Both are used as published; nothing is filtered here.

RUN (cron, daily):  python3 scripts/fetch_blocklists.py
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, os, sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

OUTDIR = os.path.join(ROOT, "data", "raw", "blocklists")
MANIFEST = os.path.join(OUTDIR, "manifest.csv")
FEEDS = {
    "openphish": ("https://openphish.com/feed.txt", "txt"),
    "phishtank": ("http://data.phishtank.com/data/online-valid.csv", "csv"),
}
UA = "PhishVN-research-collector/1.0 (+https://github.com/vuthainguyen1602/phishvn)"
FIELDS = ["fetched_at_utc", "date", "feed", "url", "status", "bytes", "sha256", "rows", "file"]


def fetch(url: str) -> tuple[int, bytes]:
    import requests
    r = requests.get(url, headers={"User-Agent": UA}, timeout=120, allow_redirects=True)
    return r.status_code, r.content


def main() -> int:
    global OUTDIR, MANIFEST
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeds", nargs="+", default=list(FEEDS), choices=list(FEEDS))
    ap.add_argument("--outdir", default=OUTDIR, help="override the output directory (tests only)")
    a = ap.parse_args()
    OUTDIR, MANIFEST = a.outdir, os.path.join(a.outdir, "manifest.csv")
    now = datetime.now(timezone.utc)
    day = now.date().isoformat()
    os.makedirs(OUTDIR, exist_ok=True)
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for feed in a.feeds:
            url, ext = FEEDS[feed]
            row = {"fetched_at_utc": now.isoformat(timespec="seconds"), "date": day, "feed": feed,
                   "url": url, "status": "", "bytes": 0, "sha256": "", "rows": 0, "file": ""}
            try:
                status, data = fetch(url)
                row["status"] = status
                if status == 200 and data:
                    d = os.path.join(OUTDIR, feed)
                    os.makedirs(d, exist_ok=True)
                    path = os.path.join(d, f"{day}.{ext}.gz")
                    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
                        g.write(data)
                    row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                               rows=data.count(b"\n"), file=os.path.relpath(path, ROOT))
            except Exception as e:  # noqa: BLE001
                row["status"] = f"error:{type(e).__name__}"
            w.writerow(row)
            mf.flush()
            print(f"[{feed}] status={row['status']} bytes={row['bytes']} rows={row['rows']} "
                  f"{row['file'] or '(not stored)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
