#!/usr/bin/env python3
"""
scrape_vn_phishing.py
Collect a list of Vietnamese fake/scam websites for research (dataset P1).

Main source: Tin Nhiem Mang (NCSC/NCA) — https://tinnhiemmang.vn/website-lua-dao. Server-rendered,
paginated ?page=1..N (~20 items/page, ~125k items); each item gives domain, detection date,
impersonated organization, status. Bonus: OpenPhish + URLhaus public feeds.

ETHICAL/LEGAL NOTE: a PUBLIC blacklist used for non-profit RESEARCH. Be POLITE — keep the delay, no
parallel threads, real User-Agent. Respect https://tinnhiemmang.vn/robots.txt and cite the
NCSC/NCA source. Do not redistribute without permission; comply with Decree 13/2023.

INSTALL: pip install requests beautifulsoup4

RUN (examples):
  # 'list' endpoint (website-lua-dao?page): server-rendered but CAPPED at ~1k unique items.
  python scrape_vn_phishing.py --end-page 50 --out tinnhiemmang.csv

  # 'filter' endpoint (filterObj AJAX): the FULL blacklist (~125k). Auto-fetches CSRF+cookies;
  # auto-stops when it runs out of new items:
  python scrape_vn_phishing.py --endpoint filter --type fake --auto-stop 10 \
      --out data/raw/tinnhiemmang/blacklist.csv --delay 0.5
  # If Cloudflare blocks the fresh session, copy Cookie + x-csrf-token from your browser
  # (DevTools > Network > filterObj > Copy as cURL) and pass them:
  python scrape_vn_phishing.py --endpoint filter --auto-stop 10 \
      --token 'YOUR_CSRF' --cookie 'XSRF-TOKEN=...; tinnhiemmang_session=...; cf_clearance=...' \
      --out data/raw/tinnhiemmang/blacklist.csv

  # Add OpenPhish + URLhaus feeds:
  python scrape_vn_phishing.py --feeds --out-feeds feeds.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

BASE = "https://tinnhiemmang.vn/website-lua-dao"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-crawler; contact: thaivn_ph@utc.edu.vn)",
    "Accept-Language": "vi,en;q=0.8",
}

HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?$", re.I)
DATE_RE = re.compile(r"Đã phát hiện ngày\s*(\d{2}/\d{2}/\d{4})")
ORG_RE = re.compile(r"Mạo danh tổ chức:\s*(.+)")
STATUSES = ("Đang chờ", "Đang xác minh", "Đang xử lý", "Đã xử lý")


def fetch(url: str, retries: int = 4, backoff: float = 2.0) -> str | None:
    """GET with retry + backoff; returns HTML or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):  # temporarily blocked -> wait longer
                time.sleep(backoff * (attempt + 1) * 3)
                continue
            return None
        except requests.RequestException:
            time.sleep(backoff * (attempt + 1))
    return None


# ---- filterObj endpoint (the FULL, paginated blacklist behind the site's AJAX filter) ----
FILTER_BASE = "https://tinnhiemmang.vn/filterObj"
MAIN_URL = "https://tinnhiemmang.vn/website-lua-dao"
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36")
CSRF_RE = re.compile(r'name="csrf-token"\s+content="([^"]+)"')


def make_session(cookie: str | None = None, token: str | None = None):
    """Session that mimics the browser: visit the main page to obtain cookies + the CSRF token.
    Pass --cookie/--token (copied from the browser) if Cloudflare blocks a fresh session."""
    s = requests.Session()
    s.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "vi,en;q=0.8"})
    if cookie:
        s.headers["Cookie"] = cookie
    tok = token
    try:
        r = s.get(MAIN_URL, timeout=30)
        if not tok:
            m = CSRF_RE.search(r.text)
            tok = m.group(1) if m else None
    except requests.RequestException:
        pass
    return s, tok


def fetch_filter(session, token, page, type_="fake", from_date="", end_date="", retries=4, backoff=2.0):
    """GET the filterObj AJAX endpoint for one page; returns response text or None.
    from_date/end_date are 'yyyy-mm-dd' (the site filters by detection date)."""
    params = {"_token": token or "", "name_obj": "", "type": type_,
              "from_date": from_date, "end_date": end_date, "page": page, "_": int(time.time() * 1000)}
    headers = {"X-CSRF-TOKEN": token or "", "X-Requested-With": "XMLHttpRequest",
               "Referer": MAIN_URL, "Accept": "*/*"}
    for attempt in range(retries):
        try:
            r = session.get(FILTER_BASE, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                time.sleep(backoff * (attempt + 1) * 3); continue
            if r.status_code in (419, 401, 403):   # token/session expired or blocked
                return f"__AUTH_ERROR__{r.status_code}"
            return None
        except requests.RequestException:
            time.sleep(backoff * (attempt + 1))
    return None


def parse_trusted(html: str) -> list[dict]:
    """Parse the TRUSTED-website listing (type=web, website-tin-nhiem): the domain sits in a
    <span> inside an <a href=.../danh-ba-tin-nhiem/...> link; there is no date/org/status."""
    soup = BeautifulSoup(html or "", "html.parser")
    items, seen = [], set()
    for a in soup.find_all("a", href=re.compile(r"danh-ba-tin-nhiem")):
        span = a.find("span")
        dom = (span.get_text() if span else a.get_text()).strip().lower()
        if HOST_RE.match(dom) and dom not in seen:
            seen.add(dom)
            items.append({"domain": dom, "detected_date": "", "impersonated_org": "",
                          "status": "", "type": guess_type(dom)})
    return items


def parse_response(text: str, type_: str = "fake") -> list[dict]:
    """filterObj returns raw HTML (rarely JSON-wrapped). Route to the right parser by type:
    type=web -> trusted/benign listing; otherwise the scam/phishing listing."""
    if not text:
        return []
    html = text
    t = text.lstrip()
    if t[:1] in "{[":
        import json
        try:
            obj = json.loads(text)

            def find_html(o):
                if isinstance(o, str):
                    return o if ("<" in o) else ""
                if isinstance(o, dict):
                    for v in o.values():
                        h = find_html(v)
                        if h:
                            return h
                if isinstance(o, list):
                    for v in o:
                        h = find_html(v)
                        if h:
                            return h
                return ""
            html = find_html(obj) or text
        except ValueError:
            pass
    return parse_trusted(html) if type_ == "web" else parse_page(html)


SOCIAL_HINTS = ("facebook.com", "fb.com", "zalo.me", "t.me", "tiktok.com/@", "instagram.com", "youtube.com/@")


def guess_type(domain: str) -> str:
    """Guess the type: social media vs website (heuristic; prefer scraping via the 'Type' filter)."""
    d = domain.lower()
    return "social" if any(h in d for h in SOCIAL_HINTS) else "website"


def parse_page(html: str) -> list[dict]:
    """
    Extract items from a single page. Parse by TEXT (robust to CSS changes):
    iterate lines, and whenever we hit 'Da phat hien ngay <date>' take:
      - domain: the nearest hostname line above
      - org:    the 'Mao danh to chuc: ...' line below
      - status: the status line below
    """
    soup = BeautifulSoup(html, "html.parser")
    lines = [ln.strip() for ln in soup.get_text("\n").split("\n")]
    lines = [ln for ln in lines if ln]  # drop empty lines

    items: list[dict] = []
    for i, ln in enumerate(lines):
        m = DATE_RE.search(ln)
        if not m:
            continue
        detected = m.group(1)

        # domain: the nearest hostname above
        domain = None
        for j in range(i - 1, max(-1, i - 6), -1):
            cand = lines[j].strip().lower()
            if HOST_RE.match(cand):
                domain = cand
                break
        if not domain:
            continue

        # org + status: scan a few lines below
        org, status = None, None
        for k in range(i + 1, min(len(lines), i + 8)):
            om = ORG_RE.search(lines[k])
            if om and not org:
                org = om.group(1).strip()
            if lines[k].strip() in STATUSES and not status:
                status = lines[k].strip()
        items.append({
            "domain": domain,
            "detected_date": detected,
            "impersonated_org": org or "",
            "status": status or "",
            "type": guess_type(domain),
        })
    return items


def load_existing(out_csv: str) -> set[str]:
    seen: set[str] = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row.get("domain", ""))
    return seen


def scrape(out_csv, start_page, end_page, delay, resume, query="", tags=None,
           endpoint="list", type_="fake", cookie=None, token=None, auto_stop=0):
    from urllib.parse import urlencode, parse_qsl
    tags = tags or {}
    base_params = dict(parse_qsl(query)) if query else {}
    seen = load_existing(out_csv) if resume else set()
    is_new_file = not (resume and os.path.exists(out_csv))
    mode = "a" if (resume and os.path.exists(out_csv)) else "w"
    fields = ["domain", "detected_date", "impersonated_org", "status", "type"] \
        + list(tags.keys()) + ["page", "scraped_at"]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    session = tok = None
    if endpoint == "filter":
        session, tok = make_session(cookie, token)
        if not tok:
            print("[!] No CSRF token found. If pages fail, pass --token and --cookie copied "
                  "from your browser (DevTools > Copy as cURL).", file=sys.stderr)

    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new_file:
            w.writeheader()

        total = 0
        empty_streak = 0
        for page in range(start_page, end_page + 1):
            if endpoint == "filter":
                text = fetch_filter(session, tok, page, type_)
                if isinstance(text, str) and text.startswith("__AUTH_ERROR__"):
                    print(f"[!] Page {page}: auth/blocked ({text[13:]}). Re-run with --cookie/--token "
                          "from your browser.", file=sys.stderr)
                    break
                rows = parse_response(text, type_) if text else None
            else:
                params = dict(base_params); params["page"] = page
                html = fetch(f"{BASE}?{urlencode(params)}")
                rows = parse_page(html) if html else None
            if rows is None:
                print(f"[!] Page {page}: fetch error, skipping", file=sys.stderr)
                time.sleep(delay); continue
            now = datetime.now(timezone.utc).isoformat()
            added = 0
            for r in rows:
                if r["domain"] in seen:
                    continue
                seen.add(r["domain"])
                r.update(tags)
                r.update({"page": page, "scraped_at": now})
                w.writerow(r)
                added += 1
            total += added
            f.flush()
            print(f"[+] Page {page}/{end_page}: +{added} (total {total})")
            # auto-stop after N consecutive pages with no NEW items
            empty_streak = empty_streak + 1 if added == 0 else 0
            if auto_stop and empty_streak >= auto_stop:
                print(f"[i] auto-stop: {auto_stop} consecutive pages with no new items; done.")
                break
            time.sleep(delay)  # be polite
    print(f"Done. Saved {total} new items to {out_csv}")


def scrape_windows(out_csv, start, end, window_days, delay, resume, type_="fake",
                   cookie=None, token=None, tags=None, auto_stop=8):
    """Scrape the FULL blacklist by iterating DATE WINDOWS (from_date/end_date), which unlocks
    historical records the plain pagination caps out on. Dates are 'yyyy-mm-dd'."""
    from datetime import date, timedelta
    tags = tags or {}
    s0 = date.fromisoformat(start)
    s1 = date.fromisoformat(end)
    seen = load_existing(out_csv) if resume else set()
    is_new_file = not (resume and os.path.exists(out_csv))
    mode = "a" if (resume and os.path.exists(out_csv)) else "w"
    fields = ["domain", "detected_date", "impersonated_org", "status", "type"] \
        + list(tags.keys()) + ["page", "scraped_at"]

    session, tok = make_session(cookie, token)
    if not tok:
        print("[!] No CSRF token; pass --token/--cookie from your browser if pages fail.", file=sys.stderr)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if is_new_file:
            w.writeheader()
        total = 0
        cur = s0
        while cur <= s1:
            w_end = min(cur + timedelta(days=window_days - 1), s1)
            frm, to = cur.isoformat(), w_end.isoformat()
            page, empty, win_added = 1, 0, 0
            while True:
                text = fetch_filter(session, tok, page, type_, frm, to)
                if isinstance(text, str) and text.startswith("__AUTH_ERROR__"):
                    print(f"[!] auth/blocked ({text[13:]}). Re-run with --cookie/--token.", file=sys.stderr)
                    print(f"Done. Saved {total} new items to {out_csv}"); return
                rows = parse_response(text, type_) if text else []
                now = datetime.now(timezone.utc).isoformat()
                added = 0
                for r in rows:
                    if r["domain"] in seen:
                        continue
                    seen.add(r["domain"]); r.update(tags)
                    r.update({"page": page, "scraped_at": now})
                    w.writerow(r); added += 1
                total += added; win_added += added
                empty = empty + 1 if added == 0 else 0
                if not rows or (auto_stop and empty >= auto_stop):
                    break
                page += 1; f.flush(); time.sleep(delay)
            if win_added:
                print(f"[window {frm}..{to}] +{win_added} (total {total})")
            cur = w_end + timedelta(days=1)
    print(f"Done. Saved {total} new items to {out_csv}")


def fetch_feeds(out_csv: str):
    """Bonus: OpenPhish community feed (recent international phishing URLs — usually still LIVE,
    so they are also the best candidates for HTML/screenshot capture). URLhaus was dropped on
    purpose: it tracks malware distribution, not phishing, so its rows must not be labelled
    phishing."""
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    # OpenPhish community feed (text, one URL per line)
    try:
        txt = requests.get("https://openphish.com/feed.txt", headers=HEADERS, timeout=30).text
        for u in txt.splitlines():
            if u.strip():
                rows.append({"url": u.strip(), "source": "openphish", "scraped_at": now})
    except requests.RequestException as e:
        print(f"[!] OpenPhish error: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "source", "scraped_at"])
        w.writeheader()
        w.writerows(rows)
    print(f"Feeds: saved {len(rows)} URLs to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Collect Vietnamese phishing data for research.")
    ap.add_argument("--out", default="tinnhiemmang.csv", help="Output CSV file (tinnhiemmang)")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page", type=int, default=6281, help="Last page number (default ~everything)")
    ap.add_argument("--delay", type=float, default=0.8, help="Seconds to pause between pages (polite)")
    ap.add_argument("--resume", action="store_true", help="Resume, skipping domains already collected")
    ap.add_argument("--query", default="",
                    help="Filter query copied from the browser URL after clicking a filter "
                         "(e.g. 'linh_vuc=...&loai=...'). The script adds page automatically.")
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                    help="Attach a fixed label column to every row (repeatable), "
                         "e.g. --tag sector='Ngan hang - Tai chinh' --tag type_filter=website")
    ap.add_argument("--endpoint", choices=["list", "filter"], default="list",
                    help="'list' = website-lua-dao?page (capped ~1k); 'filter' = filterObj AJAX "
                         "(the FULL blacklist, ~125k)")
    ap.add_argument("--type", default="fake", help="filterObj type filter (default 'fake' = scam websites)")
    ap.add_argument("--cookie", default=None, help="Raw Cookie header from your browser (if Cloudflare blocks)")
    ap.add_argument("--token", default=None, help="CSRF token from your browser (x-csrf-token)")
    ap.add_argument("--auto-stop", type=int, default=0, metavar="N",
                    help="Stop after N consecutive pages with no new items (0 = disabled)")
    ap.add_argument("--by-window", action="store_true",
                    help="Scrape the FULL blacklist by date windows (from_date/end_date) — unlocks "
                         "historical records the plain pagination caps out on. Implies --endpoint filter.")
    ap.add_argument("--from", dest="from_date", default="2020-01-01", help="Window start (yyyy-mm-dd)")
    ap.add_argument("--to", dest="to_date", default="", help="Window end (yyyy-mm-dd; default today)")
    ap.add_argument("--window-days", type=int, default=15, help="Days per window (smaller = safer vs the ~1k/query cap)")
    ap.add_argument("--feeds", action="store_true", help="Also fetch the OpenPhish community feed")
    ap.add_argument("--out-feeds", default="data/raw/openphish/feed.csv")
    args = ap.parse_args()

    tags = {}
    for kv in args.tag:
        if "=" in kv:
            k, v = kv.split("=", 1)
            tags[k.strip()] = v.strip()

    if args.feeds:
        fetch_feeds(args.out_feeds)
    elif args.by_window:
        end = args.to_date or datetime.now().date().isoformat()
        scrape_windows(args.out, args.from_date, end, args.window_days, args.delay, args.resume,
                       type_=args.type, cookie=args.cookie, token=args.token, tags=tags,
                       auto_stop=args.auto_stop or 8)
    else:
        scrape(args.out, args.start_page, args.end_page, args.delay, args.resume,
               query=args.query, tags=tags, endpoint=args.endpoint, type_=args.type,
               cookie=args.cookie, token=args.token, auto_stop=args.auto_stop)


if __name__ == "__main__":
    main()
