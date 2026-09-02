#!/usr/bin/env python3
"""
qr_prevalence.py — the prevalence figure, counted the one way that is correct.

WHY A SCRIPT FOR ONE DIVISION. Because the division has a trap in it and the trap has already been
walked into, twice, by me: urlscan_qrs.csv holds one row per SCAN, and URLScan rescans the same page
on different days. The three rows this corpus has held for a week are one page, one payload, one
finding -- so reading len(csv) as a numerator reports 0.26% where the truth is 0.09%, an overstatement
of nearly three, in the direction that flatters the study.

Prevalence is: unique pages carrying a decodable QR, over pages examined. A page examined is a
screenshot that was decoded (the in-the-wild arm) plus a domain that was submitted and returned an
image (the self-feeding arm); a domain that was dead when urlscan visited was never examined and
does not belong in the denominator either.

RUN
    python3 scripts/qr_prevalence.py                      # from synced data
    python3 scripts/qr_prevalence.py --host bvdung@...    # straight off a collector
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from urllib.parse import urlparse

QRS = os.path.join("data", "raw", "urlscan_qrs.csv")
SHOTS = os.path.join("data", "raw", "urlscan_screenshots")
LEDGER = os.path.join("data", "raw", "qr_submit_ledger.csv")
EXAMINED = os.path.join("data", "raw", ".qr_scan_examined.csv")


def _remote(host: str, cmd: str) -> str:
    try:
        return subprocess.run(["ssh", "-o", "ConnectTimeout=8", host, cmd],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="", help="read from a collector instead of local data/raw")
    a = ap.parse_args()

    if a.host:
        n_shots = int(_remote(a.host, "ls ~/PhishVN/%s | wc -l" % SHOTS) or 0)
        rows = _remote(a.host, "cat ~/PhishVN/%s" % QRS).splitlines()
        rd = list(csv.DictReader(rows)) if rows else []
    else:
        n_shots = len(os.listdir(SHOTS)) if os.path.isdir(SHOTS) else 0
        rd = list(csv.DictReader(open(QRS, newline="", encoding="utf-8"))) if os.path.isfile(QRS) else []

    # The submitted arm: only rows that came back with an image were examined. Counted as distinct
    # domains, because the submitter sends https://<domain> once -- one row is one page.
    sub_domains = set()
    if os.path.isfile(LEDGER):
        with open(LEDGER, newline="", encoding="utf-8") as f:
            sub_domains = {(r.get("domain") or "").strip().lower()
                           for r in csv.DictReader(f) if (r.get("shot_file") or "").strip()}
    n_sub = len(sub_domains)

    # Both sides of the ratio must be pages. A screenshot is a scan, and URLScan rescans the same
    # page: the first sweep decoded 896 scans of 398 distinct URLs, so counting screenshots in the
    # denominator while counting unique pages in the numerator understated the rate 2.25x. The
    # ledger carries the page URL per scan; without it we fall back to scans and say so.
    n_pages_seen, denom_is_pages, crawled_hosts = 0, False, set()
    if os.path.isfile(EXAMINED):
        with open(EXAMINED, newline="", encoding="utf-8") as f:
            urls = {(r.get("page_url") or "").strip()
                    for r in csv.DictReader(f) if (r.get("page_url") or "").strip()}
        n_pages_seen = len(urls)
        crawled_hosts = {(urlparse(u).hostname or "").lower() for u in urls} - {""}
        denom_is_pages = n_pages_seen > 0

    # The two arms can reach the same site: one searches what others submitted, the other submits
    # what this repository detected. Adding the counts would double-count the overlap.
    overlap = len(crawled_hosts & sub_domains) if denom_is_pages else 0
    examined = (n_pages_seen if denom_is_pages else n_shots) + n_sub - overlap
    scans = len(rd)
    pages = len({r.get("source_page", "") for r in rd if r.get("source_page")})
    payloads = len({r.get("qr_decoded_url", "") for r in rd if r.get("qr_decoded_url")})
    kinds: dict = {}
    for r in rd:
        kinds[r.get("payload_kind", "?")] = kinds.get(r.get("payload_kind", "?"), 0) + 1
    # Benign by construction: a messenger link, or a code pointing back at the site that served it.
    # Counted in PAGES, like the numerator above. The first version of this counted rows here and
    # pages there, then subtracted one from the other -- the very confusion this script exists to
    # stop, committed inside it. A page counts as benign only when every payload found on it is.
    by_page: dict = {}
    for r in rd:
        pg = r.get("source_page", "")
        ok = (r.get("payload_kind") in ("messenger",)) or r.get("same_site") == "1"
        by_page[pg] = by_page.get(pg, True) and ok
    benign = sum(1 for pg, ok in by_page.items() if pg and ok)

    if not examined:
        print("[!] nothing examined yet", file=sys.stderr)
        return 1
    if denom_is_pages:
        print(f"  pages examined          {examined:>8,}   ({n_pages_seen:,} crawled, from {n_shots:,} scans"
              f" + {n_sub:,} submitted - {overlap:,} on both)")
    else:
        print(f"  scans examined          {examined:>8,}   ({n_shots:,} crawled + {n_sub:,} submitted)")
        print("  [!] no examined-ledger; denominator is scans, not pages -- rate is understated")
    print(f"  scans that saw a QR     {scans:>8,}   {100*scans/examined:6.2f}%  <- NOT the prevalence")
    print(f"  unique pages with a QR  {pages:>8,}   {100*pages/examined:6.2f}%  <- the prevalence")
    print(f"  unique payloads         {payloads:>8,}")
    print(f"  of those, benign        {benign:>8,}   (messenger link, or same site as the page)")
    print(f"  quishing found          {max(0, pages - benign):>8,}")
    if scans and pages and scans != pages:
        print(f"\n  Note: {scans} rows describe {pages} page(s). Reporting rows as prevalence would "
              f"overstate it by {scans/pages:.1f}x.")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"    payload_kind {k:<14} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
