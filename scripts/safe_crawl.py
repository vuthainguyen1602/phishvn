#!/usr/bin/env python3
"""
safe_crawl.py — Safely crawl LIVE landing pages with Playwright (headless).

SAFETY WARNING (read carefully):
  * Phishing pages may contain malware/drive-by. ONLY run in a disposable VM/CONTAINER,
    on an isolated network, and DO NOT log into real accounts.
  * The script does NOT fill forms, does NOT download files, does NOT run interactions — it only opens
    the page, reads the HTML, and takes a screenshot.
  * Prefer urlscan.io (fetch_urlscan.py) for most URLs; only crawl directly when you need a live page.

INSTALL:
  pip install playwright
  playwright install chromium

RUN:
  python safe_crawl.py --in data/processed/dataset_url.csv --col url \
         --out data/interim/crawl.csv --shot-dir data/raw/landing_shots --limit 200
"""
from __future__ import annotations
import argparse, csv, os, re, time


def run(inp, col, out, shot_dir, limit, delay, timeout):
    from playwright.sync_api import sync_playwright  # lazy import for a clear error if not installed

    os.makedirs(shot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with open(inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    urls, seen = [], set()
    for r in rows:
        u = (r.get(col) or "").strip()
        if u and u not in seen:
            seen.add(u); urls.append(u)
    urls = urls[:limit]

    fields = ["url", "final_url", "status", "title", "num_forms", "num_inputs",
              "html_len", "shot_file", "html_file", "ok"]
    with sync_playwright() as p, open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        for i, u in enumerate(urls, 1):
            url = u if re.match(r"^https?://", u, re.I) else "http://" + u
            rec = {"url": u, "ok": 0}
            ctx = browser.new_context(accept_downloads=False, ignore_https_errors=True)
            page = ctx.new_page()
            page.on("download", lambda d: d.cancel())  # block file downloads
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                page.wait_for_timeout(1500)
                html = page.content()
                uid = re.sub(r"[^a-z0-9]+", "_", u.lower())[:60]
                shot = os.path.join(shot_dir, f"{uid}.png")
                htmlf = os.path.join(shot_dir, f"{uid}.html")
                try:
                    page.screenshot(path=shot, full_page=False)
                except Exception:
                    shot = ""
                with open(htmlf, "w", encoding="utf-8") as g:
                    g.write(html)
                rec.update({
                    "ok": 1,
                    "final_url": page.url,
                    "status": resp.status if resp else "",
                    "title": (page.title() or "")[:300],
                    "num_forms": len(page.query_selector_all("form")),
                    "num_inputs": len(page.query_selector_all("input")),
                    "html_len": len(html),
                    "shot_file": shot,
                    "html_file": htmlf,
                })
            except Exception as e:
                rec["status"] = f"error: {type(e).__name__}"
            finally:
                ctx.close()
            w.writerow(rec); f.flush()
            print(f"[{i}/{len(urls)}] {u} -> {'OK' if rec['ok'] else rec.get('status')}")
            time.sleep(delay)
        browser.close()
    print(f"Done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--col", default="url")
    ap.add_argument("--out", default="data/interim/crawl.csv")
    ap.add_argument("--shot-dir", default="data/raw/landing_shots")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=20, help="Timeout in seconds per page")
    args = ap.parse_args()
    run(args.inp, args.col, args.out, args.shot_dir, args.limit, args.delay, args.timeout)


if __name__ == "__main__":
    main()
