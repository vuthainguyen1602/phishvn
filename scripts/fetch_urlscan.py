#!/usr/bin/env python3
"""
fetch_urlscan.py — Get landing page content via urlscan.io (safe, without visiting malicious pages).

urlscan.io has already scanned many URLs and stores the DOM/HTML + screenshot + title. We query by
domain, take the most recent result, and download the DOM => fill html_captured/title/num_forms for the P1 schema.

INSTALL:  pip install requests
API KEY (recommended, raises the rate limit):  export URLSCAN_API_KEY=xxxx   (register free at urlscan.io)

RUN:
  python fetch_urlscan.py --in data/processed/dataset_url.csv --col domain \
         --out data/interim/urlscan.csv --dom-dir data/raw/landing --limit 500
  # also download the screenshot PNGs locally:
  python fetch_urlscan.py --in data/processed/dataset_url.csv --col domain \
         --out data/interim/urlscan.csv --dom-dir data/raw/landing \
         --save-shots --shot-dir data/raw/landing_shots --limit 500
  # coverage check BEFORE a big fetch (no downloads): how many URLs have HTML+screenshot, per tier:
  python fetch_urlscan.py --in data/processed/dataset_url.csv --col domain \
         --report-only --limit 300
"""
from __future__ import annotations
import argparse, csv, os, re, time
import requests

SEARCH = "https://urlscan.io/api/v1/search/"
RESULT = "https://urlscan.io/api/v1/result/{uuid}/"
DOM = "https://urlscan.io/dom/{uuid}/"
KEY = os.environ.get("URLSCAN_API_KEY", "")
H = {"User-Agent": "research (contact: nvthai@utc2.edu.vn)"}
if KEY:
    H["API-Key"] = KEY


def get(url, **kw):
    for a in range(4):
        try:
            r = requests.get(url, headers=H, timeout=30, **kw)
            if r.status_code == 200:
                return r
            if r.status_code == 429:      # rate limit exceeded -> wait
                time.sleep(8 * (a + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(3 * (a + 1))
    return None


def search_domain(domain: str):
    r = get(SEARCH, params={"q": f"domain:{domain}", "size": 1})
    if not r:
        return None
    res = r.json().get("results", [])
    return res[0] if res else None


def fetch_dom(uuid: str) -> str:
    r = get(DOM.format(uuid=uuid))
    return r.text if r else ""


def fetch_shot(uuid: str, shot_dir: str) -> str:
    """Download the urlscan.io screenshot PNG locally; return the file path (or '')."""
    r = get(f"https://urlscan.io/screenshots/{uuid}.png")
    if not r or not r.content:
        return ""
    path = os.path.join(shot_dir, f"{uuid}.png")
    with open(path, "wb") as g:
        g.write(r.content)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV containing a domain/url column")
    ap.add_argument("--col", default="domain", help="Name of the column holding domain/url")
    ap.add_argument("--out", default="data/interim/urlscan.csv")
    ap.add_argument("--dom-dir", default="data/raw/landing", help="Where to save the .html DOM files")
    ap.add_argument("--save-shots", action="store_true", help="Also download the screenshot PNGs locally")
    ap.add_argument("--shot-dir", default="data/raw/landing_shots", help="Where to save the .png screenshots")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--delay", type=float, default=2.5, help="Seconds to pause between queries (polite)")
    ap.add_argument("--tier-col", default="tier",
                    help="Column holding the label-confidence tier (for the coverage breakdown)")
    ap.add_argument("--report-only", action="store_true",
                    help="Only measure content-coverage (1 API call/domain, no downloads); "
                         "print how many URLs have an available HTML page + screenshot, per tier")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # first-seen domain -> tier map (so coverage can be broken down by tier)
    domains, dom_tier, seen = [], {}, set()
    for r in rows:
        d = (r.get(args.col) or "").strip().lower()
        d = re.sub(r"^https?://", "", d).split("/")[0]
        if d and d not in seen:
            seen.add(d); domains.append(d)
            dom_tier[d] = (r.get(args.tier_col) or "unknown").strip() or "unknown"
    domains = domains[:args.limit]

    if args.report_only:
        return report_coverage(domains, dom_tier, args)

    os.makedirs(args.dom_dir, exist_ok=True)
    if args.save_shots:
        os.makedirs(args.shot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fields = ["domain", "tier", "scan_uuid", "scanned_url", "title", "num_forms",
              "html_len", "screenshot", "dom_file", "shot_file", "found"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, d in enumerate(domains, 1):
            hit = search_domain(d)
            row = {"domain": d, "tier": dom_tier.get(d, "unknown"), "found": 0}
            if hit:
                uuid = hit.get("_id", "")
                dom = fetch_dom(uuid)
                dom_file = ""
                if dom:
                    dom_file = os.path.join(args.dom_dir, f"{uuid}.html")
                    with open(dom_file, "w", encoding="utf-8") as g:
                        g.write(dom)
                shot_file = fetch_shot(uuid, args.shot_dir) if args.save_shots else ""
                row.update({
                    "found": 1,
                    "scan_uuid": uuid,
                    "scanned_url": hit.get("task", {}).get("url", ""),
                    "title": (hit.get("page", {}) or {}).get("title", ""),
                    "num_forms": len(re.findall(r"<form\b", dom, re.I)),
                    "html_len": len(dom),
                    "screenshot": f"https://urlscan.io/screenshots/{uuid}.png",
                    "dom_file": dom_file,
                    "shot_file": shot_file,
                })
            w.writerow(row); f.flush()
            print(f"[{i}/{len(domains)}] {d} -> {'OK' if row['found'] else 'no scan'}")
            time.sleep(args.delay)
    print(f"Done -> {args.out}. You can submit new scans for 'no scan' domains if you have an API key.")


def report_coverage(domains, dom_tier, args):
    """Measure content-coverage without downloading: for each domain, 1 search call ->
    does a urlscan snapshot exist (has HTML) and a screenshot? Break down by tier."""
    from collections import defaultdict
    tot = defaultdict(lambda: {"n": 0, "found": 0, "shot": 0})
    for i, d in enumerate(domains, 1):
        tier = dom_tier.get(d, "unknown")
        hit = search_domain(d)
        found = 1 if hit else 0
        shot = 1 if (hit and hit.get("screenshot")) else 0
        for key in (tier, "ALL"):
            tot[key]["n"] += 1; tot[key]["found"] += found; tot[key]["shot"] += shot
        print(f"[{i}/{len(domains)}] {d} ({tier}) -> {'scan' if found else 'no scan'}")
        time.sleep(args.delay)
    print("\n== Content coverage (share of URLs with an available urlscan snapshot / screenshot) ==")
    print(f"{'tier':<10}{'N':>7}{'HTML%':>9}{'shot%':>9}")
    for key in sorted(tot, key=lambda k: (k != "ALL", k)):
        s = tot[key]; n = max(s["n"], 1)
        print(f"{key:<10}{s['n']:>7}{100*s['found']/n:>8.1f}{100*s['shot']/n:>9.1f}")
    print("\nLow HTML% on a tier (e.g. taken-down 'gold' domains) => that tier's content "
          "modalities will be sparse; consider urlscan API submissions or a live crawl.")


if __name__ == "__main__":
    main()
