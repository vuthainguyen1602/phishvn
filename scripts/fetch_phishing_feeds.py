#!/usr/bin/env python3
"""
fetch_phishing_feeds.py — Aggregate several public phishing feeds, then filter to Vietnamese-
targeting domains. Solves the low-VN-yield problem of any single feed by taking the UNION of many
global feeds and keeping the VN slice (.vn or a Vietnamese unaccented token in the name).

Sources (all public; each is optional and failures are tolerated):
  * OpenPhish community feed            (recent, live)
  * URLhaus online URLs                 (abuse.ch; phishing + malware — tagged by source)
  * Phishing.Database ACTIVE domains    (mitchellkrogza; large, freshness-filtered to ACTIVE)
  * ChongLuaDao current denylist        (rotating sample, VN community project)

Output: data/interim/vn_phishing_candidates.csv  (domain, sources, fetched_at) — VN-targeting only.
Feed these to a liveness check + urlscan capture (crawl-at-detection), then build_content_manifest.
Because Phishing.Database is pre-filtered to ACTIVE, its VN slice is far more likely to be live than
an aged snapshot — mitigating the survivorship bias documented in the P1b limitations.

RUN:  python scripts/collect/fetch_phishing_feeds.py
      python scripts/collect/fetch_phishing_feeds.py --sources openphish phishdb   # subset
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import os
import re
import sys

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from vn_filter import host_of, is_vn_target  # noqa: E402

H = {"User-Agent": "research (contact: nvthai@utc2.edu.vn)"}
OUT = os.path.join("data", "interim", "vn_phishing_candidates.csv")
DENYLIST_URL = "https://chongluadao.vn/database/denylist"
# Attributes may sit between the class value and the closing > -- see watch_chongluadao.py.
CLD_ENTRY_RE = re.compile(r'class="_urlText_[^"]*"[^>]*>\s*(https?://[^<\s]+)', re.I)


def _get(url, timeout=60):
    r = requests.get(url, headers=H, timeout=timeout)
    r.raise_for_status()
    return r


def src_openphish():
    return [host_of(u) for u in _get("https://openphish.com/feed.txt").text.splitlines() if u.strip()]


def src_urlhaus():
    out = []
    for line in _get("https://urlhaus.abuse.ch/downloads/csv_online/").text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            parts = next(csv.reader([line]))
        except Exception:
            continue
        if len(parts) > 2:
            out.append(host_of(parts[2]))
    return out


def src_phishdb():
    # host_of, like every other source here. The list is nominally bare domains but today's file
    # carries 99 entries with a scheme, path or trailing dot, and is_vn_target tests
    # d.endswith(".vn") against the raw string -- so a compromised Vietnamese WordPress site
    # arriving as http://anhhungphat.vn/wp-content/... was rejected, which is the exact population
    # this feed is here for.
    url = ("https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/"
           "phishing-domains-ACTIVE.txt")
    return [host_of(ln) for ln in _get(url).text.splitlines()
            if ln.strip() and not ln.startswith("#")]


def src_chongluadao():
    text = _get(DENYLIST_URL, timeout=30).text
    hosts = [host_of(m) for m in CLD_ENTRY_RE.findall(text)]
    if not hosts:
        # See watch_chongluadao.fetch_denylist_webpage: an empty parse is a markup change, and
        # main() only logs a source that RAISES, so a silent [] would drop this feed unnoticed.
        raise RuntimeError(f"denylist page returned {len(text)} bytes but CLD_ENTRY_RE matched none")
    return hosts


SOURCES = {"openphish": src_openphish, "urlhaus": src_urlhaus,
           "phishdb": src_phishdb, "chongluadao": src_chongluadao}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    now = _dt.datetime.now().replace(microsecond=0).isoformat()

    dom_src = {}      # domain -> set of sources that listed it
    for name in args.sources:
        try:
            hosts = SOURCES[name]()
        except Exception as e:
            print(f"[!] {name}: {type(e).__name__} {str(e)[:80]}", file=sys.stderr); continue
        vn = 0
        for h in hosts:
            if h and "." in h and is_vn_target(h):
                dom_src.setdefault(h, set()).add(name); vn += 1
        print(f"[i] {name}: {len(hosts)} entries, {vn} Vietnamese-targeting")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["domain", "sources", "fetched_at"])
        for d in sorted(dom_src):
            w.writerow([d, "|".join(sorted(dom_src[d])), now])
    print(f"[+] {len(dom_src)} unique Vietnamese-targeting phishing domains -> {args.out}")


if __name__ == "__main__":
    main()
