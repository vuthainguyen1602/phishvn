#!/usr/bin/env python3
"""
qr_scan.py — automated quishing collection via threat intelligence (design-stage probe)

This script queries the public URLScan.io API for recent malicious URLs, downloads their 
DOM screenshots, and scans the images for embedded QR codes using our qr_decode library.
If a QR code is found (e.g., a fake VietQR payment page), it extracts the URL and saves it.

Run:
  python3 scripts/qr_scan.py --limit 50 --out data/raw/urlscan_qrs.csv
  Key: QR_SCAN_API_KEY, or URLSCAN_API_KEY from scripts/.env (gitignored).
"""
import argparse
import csv
import datetime as _dt
import os
import sys
import time
import requests

# Add lib to path so we can import qr_decode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
try:
    import qr_decode
except ImportError:
    raise SystemExit("Could not import qr_decode. Make sure it's in scripts/")


# Mechanical triage of a decoded payload. NOT a verdict: this records what the payload IS, and
# leaves what it MEANS to the audit, the way the label auditor does for the infrastructure arm. The first
# three QR codes this collector found were a Zalo "contact us" code on sinvoice.vn, Viettel's
# legitimate e-invoice site -- indistinguishable in the CSV from a fraudulent VietQR until
# something records that the payload was a messenger link on the same site that served it.
MESSENGER = ("zaloapp.com", "zalo.me", "m.me", "wa.me", "t.me", "messenger.com",
             "line.me", "api.whatsapp.com")
SHORTENER = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly", "rb.gy",
             "shorturl.at", "s.net.vn", "rebrand.ly")


def triage(source_page: str, payload: str) -> dict:
    """host of the payload, whether it points back at the page that carried it, and its kind."""
    from urllib.parse import urlparse
    try:
        ph = (urlparse(payload).hostname or "").lower().lstrip("www.")
        sh = (urlparse(source_page).hostname or "").lower().lstrip("www.")
    except ValueError:
        ph = sh = ""
    # EMVCo first: a payment string is not a URL and would otherwise be filed as "non_url" and
    # left unread, which is exactly backwards -- in Vietnam the payment code IS the quishing
    # payload, and the fraud is that the account it names is not the one the victim means to pay.
    emv = {}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
        import emvco
        emv = emvco.parse(payload)
    except Exception:
        emv = {}
    if emv.get("is_emvco"):
        return {"qr_host": "", "same_site": 0,
                "payload_kind": "emvco_valid" if emv.get("crc_ok") else "emvco_badcrc",
                "emv_bin": emv.get("bin", ""), "emv_account": emv.get("account", ""),
                "emv_amount": emv.get("amount", ""), "emv_merchant": emv.get("merchant", "")}

    kind = "other"
    if any(ph == m or ph.endswith("." + m) for m in MESSENGER):
        kind = "messenger"
    elif any(ph == m or ph.endswith("." + m) for m in SHORTENER):
        kind = "shortener"
    elif payload[:5].lower() not in ("http:", "https"):
        # A QR does not have to carry a URL. VietQR payment codes are EMVCo strings, which is the
        # case this study exists for, so they must not be silently dropped as malformed.
        kind = "non_url"
    same = int(bool(ph) and bool(sh) and (ph == sh or ph.endswith("." + sh) or sh.endswith("." + ph)))
    return {"qr_host": ph, "same_site": same, "payload_kind": kind,
            "emv_bin": "", "emv_account": "", "emv_amount": "", "emv_merchant": ""}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true",
                    help="ignore the saved cursor and start from the newest scans")
    ap.add_argument("--dom", action="store_true",
                    help="also save each scan's DOM (doubles retrievals)")
    ap.add_argument("--query", default=None,
                    help="URLScan search query; default targets quishing + .vn phishing")
    ap.add_argument("--limit", type=int, default=20, help="Number of recent scans to fetch")
    ap.add_argument("--out", default="data/raw/urlscan_qrs.csv")
    args = ap.parse_args()

    # The free tier refuses verdicts.overall.malicious and page.domain wildcards (both 403), and a
    # bare text query is worse than useless: q=phishing matches pages that CONTAIN the word, so it
    # returned Microsoft's documentation about phishing -- 7 results in total, none of them a
    # phishing page. Measured against the live API 2026-08-30:
    #   q=phishing                                    7
    #   task.tags:qr                                  2
    #   task.tags:quishing                           32
    #   task.tags:phishing AND page.domain:vn     1,666
    #   the two below, combined                   1,698
    # The first arm is the direct hit; the second is where a Vietnamese quishing page would be
    # tagged as phishing without anyone tagging the QR. Override with --query to widen it.
    QUERY = "task.tags:quishing OR (task.tags:phishing AND page.domain:vn)"
    q = getattr(args, "query", None) or QUERY
    
    api_key = os.environ.get("QR_SCAN_API_KEY")
    headers = {"API-Key": api_key} if api_key else {}
    if not api_key:
        print("[!] LƯU Ý: Chưa có biến môi trường QR_SCAN_API_KEY. API của URLScan có thể trả về lỗi 403 Forbidden.")
    
    # Paginated with search_after. One call returns the NEWEST `size` scans and nothing else, so a
    # daily cron kept re-reading the same slice: the query has 1,698 matches and a run of 50 could
    # only ever see 50 of them, the same 50 the previous run saw. The API returns each result's
    # `sort` cursor for exactly this; --pages walks it. A page is capped at 100 by the API.
    # The cursor is persisted, so tomorrow's run resumes where this one stopped instead of
    # restarting at the newest scan and re-treading the same ground (--restart forces the top).
    PAGE = min(args.limit, 100)
    cursor_file = os.path.join(os.path.dirname(args.out) or ".", ".qr_scan_cursor")
    after = None
    if not args.restart and os.path.isfile(cursor_file):
        after = open(cursor_file, encoding="utf-8").read().strip() or None
        if after:
            print(f"[*] resuming after cursor {after[:40]}")

    print(f"[*] Querying URLScan for up to {args.limit} scans in pages of {PAGE}...")
    results, last_sort = [], None
    while len(results) < args.limit:
        params = {"q": q, "size": min(PAGE, args.limit - len(results))}
        if after:
            params["search_after"] = after
        try:
            resp = requests.get("https://urlscan.io/api/v1/search/",
                                params=params, headers=headers, timeout=20)
            if resp.status_code == 429:
                print("[!] 429 on search — stopping this run rather than hammering")
                break
            resp.raise_for_status()
            batch = resp.json().get("results", [])
        except Exception as e:
            if not results:
                raise SystemExit(f"[!] Failed to fetch from URLScan API: {e}")
            print(f"[!] search failed mid-walk ({e}); keeping the {len(results)} already fetched")
            break
        if not batch:
            print("[*] query exhausted — no further pages")
            after = None          # start again from the newest next time
            break
        results.extend(batch)
        srt = batch[-1].get("sort")
        if not srt:
            break
        last_sort = ",".join(str(x) for x in srt)
        after = last_sort
    try:
        with open(cursor_file, "w", encoding="utf-8") as f:
            f.write(after or "")
    except OSError:
        pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("data/raw/urlscan_screenshots", exist_ok=True)
    
    decode_func, backend_name = qr_decode.get_decoder()
    print(f"[*] Using QR decoder backend: {backend_name}")

    # Scans this corpus already holds. Without it every run re-decoded the screenshots it had
    # cached and appended the same rows again: after two runs the CSV carried each of its three
    # uuids twice. A scan is identified by its uuid, so that is the key -- one page scanned three
    # times is three observations and all three belong here, but the SAME scan is one.
    # TWO sets, because they answer different questions. The CSV holds the scans that YIELDED a QR;
    # the seen-file holds every scan already looked at, QR or not. Only the first was tracked at
    # first, which was fine for a daily run but not for an hourly one: once the query pool is swept
    # the walk restarts at the newest scan, and every pass then re-decoded the same cached PNGs
    # because a page with no QR was never recorded as examined. With both, a re-walk costs one
    # search call and stops.
    seen_file = os.path.join(os.path.dirname(args.out) or ".", ".qr_scan_seen")
    seen = set()
    if os.path.isfile(args.out):
        with open(args.out, newline="", encoding="utf-8") as f:
            seen = {r.get("urlscan_uuid", "") for r in csv.DictReader(f)}
    if os.path.isfile(seen_file):
        with open(seen_file, encoding="utf-8") as f:
            seen |= {ln.strip() for ln in f if ln.strip()}
    if seen:
        print(f"[*] {len(seen)} scan(s) already examined; they will be skipped")
    newly_examined = []
    # The seen-file answers "have I decoded this scan already"; it cannot answer "how many pages
    # have I examined", because URLScan rescans the same page repeatedly -- 840 checks in the first
    # sweep covered 328 distinct URLs. Prevalence counts unique pages on both sides of the ratio, so
    # the denominator needs the page URL, not the scan id. Kept in its own ledger to leave the
    # skip-logic that reads .qr_scan_seen untouched.
    examined_ledger = os.path.join(os.path.dirname(args.out) or ".", ".qr_scan_examined.csv")
    newly_examined_pages = []

    found_qrs = []
    skipped = 0

    for i, res in enumerate(results):
        task = res.get("task", {})
        page = res.get("page", {})
        uuid = task.get("uuid")
        url = page.get("url")
        
        if not uuid:
            continue
        if uuid in seen:
            skipped += 1
            continue
            
        screenshot_url = f"https://urlscan.io/screenshots/{uuid}.png"
        img_path = f"data/raw/urlscan_screenshots/{uuid}.png"
        
        print(f"[{i+1}/{len(results)}] Checking {url}")
        
        # Download screenshot if we don't have it.
        # WITH the API key. This used to fetch anonymously, which is not the same request:
        # watch_urlscan_brands.py sends the key on every retrieval, and an anonymous fetch is
        # throttled per IP rather than counted against the account. When URLScan does throttle it
        # the reply is simply not a 200 and the loop `continue`s, so a run that quietly retrieved
        # nothing looks exactly like a batch with no QR in it. Same quota class either way --
        # retrieve, 10,000/day -- so there is nothing to save by staying anonymous.
        if not os.path.exists(img_path):
            try:
                img_resp = requests.get(screenshot_url, headers=headers, timeout=15)
                if img_resp.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(img_resp.content)
                else:
                    # Say which code. A 429 is a throttle to back off from, a 404 is a scan with no
                    # screenshot, and before this they were indistinguishable from each other and
                    # from a page that simply carried no QR.
                    if img_resp.status_code == 429:
                        print(f"    [!] 429 rate-limited on {uuid} — backing off 30s")
                        time.sleep(30)
                    elif img_resp.status_code != 404:
                        print(f"    [!] HTTP {img_resp.status_code} fetching screenshot for {uuid}")
                    continue
            except Exception as e:
                print(f"    [!] screenshot fetch failed for {uuid}: {e}")
                continue

        # The DOM, when asked for. The quishing corpus needs the payload, but a page that carries a
        # QR is also a page, and the sibling collectors read DOM; without it this corpus can say what the QR pointed
        # at and nothing about what surrounded it. Off by default because it doubles the retrievals.
        if args.dom:
            dom_path = f"data/raw/urlscan_qr_dom/{uuid}.html"
            if not os.path.exists(dom_path):
                try:
                    dr = requests.get(f"https://urlscan.io/dom/{uuid}/", headers=headers, timeout=30)
                    if dr.status_code == 200 and dr.text:
                        os.makedirs("data/raw/urlscan_qr_dom", exist_ok=True)
                        with open(dom_path, "w", encoding="utf-8") as f:
                            f.write(dr.text)
                except Exception:
                    pass
                
        newly_examined.append(uuid)
        newly_examined_pages.append((uuid, url or "", _dt.datetime.now().isoformat(timespec="seconds")))

        # Scan for QR code
        try:
            decoded_urls = decode_func(img_path)
            for d_url in decoded_urls:
                d_url = d_url.strip()
                if d_url:
                    print(f"    [+] BINGO! Found QR Code -> {d_url}")
                    row = {
                        "source_page": url,
                        "urlscan_uuid": uuid,
                        "qr_decoded_url": d_url,
                        "screenshot_file": img_path,
                    }
                    row.update(triage(url, d_url))
                    print(f"        {row['payload_kind']}"
                          f"{', same site' if row['same_site'] else ''}")
                    found_qrs.append(row)
        except Exception as e:
            print(f"    [!] Error decoding {img_path}: {e}")
            
        time.sleep(1) # Respect API rate limits

    # Save results
    # Schema upgrade runs whether or not this batch found anything. It was inside the
    # `if found_qrs:` branch, so a run with no new QR -- the common case once the cache is warm --
    # left the old four-column header in place and the upgrade never happened.
    file_exists = os.path.isfile(args.out)
    if file_exists:
        with open(args.out, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            old_rows, old_cols = list(rd), rd.fieldnames or []
        if "payload_kind" not in old_cols:
            for r in old_rows:
                r.update(triage(r.get("source_page", ""), r.get("qr_decoded_url", "")))
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["source_page", "urlscan_uuid",
                                                  "qr_decoded_url", "screenshot_file",
                                                  "qr_host", "same_site", "payload_kind"])
                w.writeheader(); w.writerows(old_rows)
            print(f"[*] upgraded {args.out} to the triage schema ({len(old_rows)} row(s))")

    if found_qrs:
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_page", "urlscan_uuid", "qr_decoded_url",
                                                "screenshot_file", "qr_host", "same_site",
                                                "payload_kind", "emv_bin", "emv_account",
                                                "emv_amount", "emv_merchant"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(found_qrs)
        print(f"\n[+] Success! Found {len(found_qrs)} NEW QR codes. Saved to {args.out}")
    else:
        print("\n[-] No NEW QR codes in this batch. Try a larger --limit or a different --query.")
    if newly_examined:
        try:
            with open(seen_file, "a", encoding="utf-8") as f:
                f.write("\n".join(newly_examined) + "\n")
        except OSError:
            pass
    if newly_examined_pages:
        try:
            fresh = not os.path.isfile(examined_ledger)
            with open(examined_ledger, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if fresh:
                    w.writerow(["urlscan_uuid", "page_url", "examined_at"])
                w.writerows(newly_examined_pages)
        except OSError:
            pass
    if skipped:
        print(f"[*] skipped {skipped} scan(s) already examined")

if __name__ == "__main__":
    main()
