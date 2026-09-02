#!/usr/bin/env python3
"""Reconstruct the examined-pages ledger for scans decoded before the ledger existed.

qr_scan.py used to record only the scan id of everything it looked at. That is enough to avoid
re-decoding a scan, but not enough to state a prevalence: URLScan rescans the same page many times,
so the count of scans examined is 2-3x the count of pages examined, and dividing unique pages by
scans understates the rate by that factor. The page URL for every scan is already in the search
response the crawler walks, so this walks the same query again and writes the pairs out. Search
quota only; no retrieve call is made.
"""
import csv, os, sys, time
import requests

API = "https://urlscan.io/api/v1/search/"
PAGE = 200
QUERY = "task.tags:quishing OR (task.tags:phishing AND page.domain:vn)"
SEEN = "data/raw/.qr_scan_seen"
LEDGER = "data/raw/.qr_scan_examined.csv"


def main():
    key = os.environ.get("URLSCAN_API_KEY", "")
    if not key:
        sys.exit("[!] URLSCAN_API_KEY not set (source scripts/.env and export it)")
    if not os.path.isfile(SEEN):
        sys.exit(f"[!] no {SEEN}; nothing to backfill")
    with open(SEEN, encoding="utf-8") as f:
        seen = {ln.strip() for ln in f if ln.strip()}

    have = {}
    if os.path.isfile(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            have = {r["urlscan_uuid"]: r["page_url"] for r in csv.DictReader(f)}

    found, cursor, calls = dict(have), None, 0
    while True:
        params = {"q": QUERY, "size": PAGE}
        if cursor:
            params["search_after"] = cursor
        r = requests.get(API, params=params, headers={"API-Key": key}, timeout=45)
        calls += 1
        if r.status_code != 200:
            print(f"[!] search returned {r.status_code} after {calls} call(s); keeping what we have")
            break
        batch = r.json().get("results", [])
        if not batch:
            break
        for res in batch:
            uuid = res.get("task", {}).get("uuid")
            url = res.get("page", {}).get("url") or ""
            if uuid and uuid in seen and uuid not in found:
                found[uuid] = url
        sort = batch[-1].get("sort")
        if not sort:
            break
        cursor = ",".join(str(x) for x in sort)
        print(f"  ...{calls} call(s), {len(found)}/{len(seen)} resolved")
        time.sleep(2)

    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["urlscan_uuid", "page_url"])
        for uuid in sorted(found):
            w.writerow([uuid, found[uuid]])

    pages = len({u for u in found.values() if u})
    missing = len(seen) - len(found)
    print(f"\n  scans examined     {len(seen):>6,}")
    print(f"  resolved to a page {len(found):>6,}   ({missing:,} unresolved)")
    print(f"  unique pages       {pages:>6,}")
    if len(found):
        print(f"  rescan factor      {len(found)/max(pages,1):>6.2f}x")


if __name__ == "__main__":
    main()
