#!/usr/bin/env python3
"""
scrape_trusted_orgs.py — Scrape the TRUSTED ORGANIZATIONS directory (legitimate) from tinnhiemmang.vn.
Used as the BENIGN CLASS (legitimate orgs/websites) + brand -> real domain mapping for dataset P1.

Source: https://tinnhiemmang.vn/to-chuc-tin-nhiem  (7k+ organizations, ?page=1..712)
  * List page: organization name + detail link + certification date.
  * The real website/tax code lives on the DETAIL PAGE (/danh-ba-tin-nhiem/...) -> use --enrich to fetch.

INSTALL:  pip install requests beautifulsoup4
RUN:
  # list only (fast):
  python scrape_trusted_orgs.py --end-page 712 --out data/raw/tinnhiem_org/orgs.csv
  # with website/tax code (slow, visits each detail page):
  python scrape_trusted_orgs.py --end-page 712 --enrich --out data/raw/tinnhiem_org/orgs.csv --resume

NOTE: public data from NCSC/NCA. Keep a polite delay, respect robots.txt, cite the source.
Tax code/phone are organization data — still handle carefully (do not redistribute the sensitive parts).
"""
from __future__ import annotations
import argparse, csv, os, re, time
from urllib.parse import urlparse, urlencode, parse_qsl
import requests
from bs4 import BeautifulSoup

BASE = "https://tinnhiemmang.vn/to-chuc-tin-nhiem"
SITE = "https://tinnhiemmang.vn"
HEADERS = {"User-Agent": "research-crawler (contact: you@example.edu.vn)", "Accept-Language": "vi"}
CERT_RE = re.compile(r"Tín nhiệm mạng:\s*(\d{2}/\d{2}/\d{4})")
MST_RE = re.compile(r"Mã số thuế[:\s]*([0-9]{10}(?:-[0-9]{3})?)")
SKIP = ("tinnhiemmang", "facebook", "youtube", "google", "cdn-cgi", "javascript")


def fetch(url, retries=4):
    for a in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                time.sleep(6 * (a + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(3 * (a + 1))
    return None


def parse_list(html: str):
    """Each org = one /danh-ba-tin-nhiem/ link + has 'Tin nhiem mang: <date>' in the block."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/danh-ba-tin-nhiem/"]'):
        href = a.get("href", "")
        name = a.get_text(" ", strip=True)
        if not name or any(s in href.lower() for s in SKIP):
            continue
        box = a.find_parent(["li", "div"])
        text = box.get_text(" ", strip=True) if box else ""
        m = CERT_RE.search(text)
        if not m:                       # drop navigation/footer links (no date)
            continue
        if not href.startswith("http"):
            href = SITE + href
        out.append({"org_name": name, "detail_url": href, "cert_date": m.group(1)})
    # dedupe by detail_url within a single page
    uniq, seen = [], set()
    for r in out:
        if r["detail_url"] not in seen:
            seen.add(r["detail_url"]); uniq.append(r)
    return uniq


def enrich(detail_url: str):
    """Visit the detail page to get website + tax code (best-effort)."""
    html = fetch(detail_url)
    if not html:
        return {"website": "", "domain": "", "mst": ""}
    soup = BeautifulSoup(html, "html.parser")
    website = ""
    for a in soup.select('a[href^="http"]'):
        h = a.get("href", "")
        if any(s in h.lower() for s in SKIP):
            continue
        website = h; break
    mst = MST_RE.search(soup.get_text(" "))
    dom = urlparse(website).hostname or "" if website else ""
    return {"website": website, "domain": dom.lower(), "mst": mst.group(1) if mst else ""}


def load_seen(path, key):
    seen = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add(r.get(key, ""))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tinnhiem_org/orgs.csv")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page", type=int, default=712)
    ap.add_argument("--delay", type=float, default=0.8)
    ap.add_argument("--enrich", action="store_true", help="Visit detail pages to get website/tax code (slow)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--query", default="", help="Sector filter query (copy from the browser URL)")
    ap.add_argument("--tag", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tags = dict(kv.split("=", 1) for kv in args.tag if "=" in kv)
    base_params = dict(parse_qsl(args.query)) if args.query else {}

    seen = load_seen(args.out, "detail_url") if args.resume else set()
    is_new = not (args.resume and os.path.exists(args.out))
    fields = ["org_name", "detail_url", "cert_date", "label", "source"] \
        + (["website", "domain", "mst"] if args.enrich else []) \
        + list(tags.keys()) + ["page", "scraped_at"]

    from datetime import datetime, timezone
    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    with open(args.out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        total = 0
        for page in range(args.start_page, args.end_page + 1):
            params = dict(base_params); params["page"] = page
            html = fetch(f"{BASE}?{urlencode(params)}")
            if not html:
                print(f"[!] page {page} error", flush=True); time.sleep(args.delay); continue
            now = datetime.now(timezone.utc).isoformat()
            added = 0
            for r in parse_list(html):
                if r["detail_url"] in seen:
                    continue
                seen.add(r["detail_url"])
                r.update({"label": "benign", "source": "tinnhiem_org"})
                if args.enrich:
                    r.update(enrich(r["detail_url"])); time.sleep(args.delay)
                r.update(tags); r.update({"page": page, "scraped_at": now})
                w.writerow(r); added += 1
            total += added; f.flush()
            print(f"[+] page {page}/{args.end_page}: +{added} (total {total})", flush=True)
            time.sleep(args.delay)
    print(f"Done -> {args.out} ({total} organizations)")


if __name__ == "__main__":
    main()
