#!/usr/bin/env python3
"""
qr_submit.py — the self-feeding half of the quishing probe.

WHY THIS EXISTS. qr_scan.py searches URLScan for scans SOMEONE ELSE submitted and decodes their
screenshots. Measured 2026-08-30, that pool holds 1,698 scans and grows by 2 a week, so it is swept
in under two days and then produces nothing. Worse, its sample is "pages a stranger chose to scan",
which cannot answer the question the paper asks -- how often a VIETNAMESE PHISHING page carries a
QR -- because the population is not the one being claimed about.

This submits domains the repository found itself, so the sample is the population: urlscan renders
each in its sandbox, and the screenshot that comes back is one we commissioned. Same mechanism as
ct_capture_bridge.py, which does this for the certificate-transparency hits, and the submit/poll routines here follow
its rules deliberately: a 429 is a fact about US and must not cost the domain an attempt, while a
400 (dead DNS, blocklisted) is a fact about the domain and does.

QUOTA, measured rather than assumed. The account's unlisted-scan allowance is 1,000/day and
ct_capture_bridge uses 70 of it, so ~930 sit unused daily. The default budget here is 200, which
leaves 730 spare for the infrastructure collector. The hourly cap is 100, so a run is capped well under it.

RUN:
  python3 scripts/qr_submit.py --budget 200
  Key: QR_SCAN_API_KEY, or URLSCAN_API_KEY from scripts/.env (gitignored).
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import qr_decode  # noqa: E402

H = {"User-Agent": "PhishVN-research/1.0 (+quishing prevalence study)"}
SUBMIT = "https://urlscan.io/api/v1/scan/"
SHOT_DIR = os.path.join("data", "raw", "urlscan_screenshots")
LEDGER = os.path.join("data", "raw", "qr_submit_ledger.csv")
LEDGER_FIELDS = ["domain", "attempt", "attempted_at", "scan_uuid", "shot_file", "qr_count", "note"]

# Where the domains come from, in the order they are drawn. Phishing detections first: they are the
# population the prevalence claim is about. host_infra is the long tail and is only reached once the
# detections are exhausted.
#
# These are FEED-ASSIGNED labels, the same caveat data/docs/infra records for host_infra.csv: a row
# is here because phishdb, openphish, urlhaus or ChongLuaDao put it there, not because anyone
# verified it. Legitimate sites appear because they were compromised and served a phishing page --
# 119.vn and 1ship.vn are both phishdb rows. For a prevalence denominator that is the right
# population anyway (it is what a defender's feed says is phishing), but any claim about what the
# QR codes MEAN has to carry the caveat.
# Registry-wildcard IPs: a whole TLD answering for every name under it, so a "domain" here was
# never a live site. The registry-wildcard audit identified dotPH's on 2026-08-16 -- 797 names then, 2,596 now, every one
# resolving to 45.79.222.138 and serving a ParkLogic parking page with zero phishing content. They
# are also the NEWEST rows in host_infra, so ordering by detection date puts them at the very front
# of the queue: without this the budget would go almost entirely to names that cannot carry a QR
# because there is no page. Excluded by IP rather than by suffix, so a genuine .ph phishing site
# still qualifies. audit_p4_labels.py probes for these rather than hardcoding them; this collector
# reads the answer it already reached.
WILDCARD_IPS = {"45.79.222.138"}

SOURCES = [
    (os.path.join("data", "raw", "vn_phishing_live", "detections.csv"), "domain", "first_detected"),
    (os.path.join("data", "raw", "chongluadao_live", "detections.csv"), "domain", "first_detected"),
    (os.path.join("data", "raw", "host_infra", "host_infra.csv"), "domain", "first_detected"),
]


def submit(domain: str, key: str) -> tuple:
    """(uuid, note). uuid is '__RATE__' on 429 and None on failure; note says WHY it failed.

    A bare None told us nothing: two of the first five domains came back submit_failed and the
    ledger could not say whether urlscan refused them (400 -- dead DNS or blocklisted, which is
    ordinary for a phishing domain days after detection) or something on our side broke."""
    try:
        r = requests.post(SUBMIT, headers={**H, "API-Key": key, "Content-Type": "application/json"},
                          json={"url": "https://" + domain, "visibility": "unlisted"}, timeout=30)
        if r.status_code == 200:
            return r.json().get("uuid"), ""
        if r.status_code == 429:
            return "__RATE__", ""
        detail = ""
        try:
            detail = (r.json().get("description") or "")[:60]
        except ValueError:
            pass
        return None, f"submit_{r.status_code}" + (f":{detail}" if detail else "")
    except requests.RequestException as e:
        return None, f"submit_error:{type(e).__name__}"


def wait_result(uuid: str, key: str, tries: int = 10, delay: float = 6.0) -> bool | None:
    """True ready, False not yet, None on 429. A 429 must not cost the domain an attempt."""
    for _ in range(tries):
        try:
            r = requests.get(f"https://urlscan.io/api/v1/result/{uuid}/",
                             headers={**H, "API-Key": key}, timeout=30)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                return None
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


def load_ledger() -> dict:
    if not os.path.isfile(LEDGER):
        return {}
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return {r["domain"]: r for r in csv.DictReader(f)}


def candidates(done: dict, want: int) -> list:
    """Domains not yet attempted, NEWEST DETECTION FIRST, deduplicated across sources.

    Order matters more than it looks. The sources span 2026-06-10 to 2026-08-30, and reading them
    in file order meant working through July first: 32 of the first 61 submissions came back
    submit_400, "could not be resolved", because the domain had been taken down weeks earlier. A
    dead domain yields no screenshot, and no screenshot means no chance of seeing a QR at all --
    so the run spends its budget on the domains least able to answer the question. Newest first
    puts the budget where a page might still be up.

    Rows with no date sort last rather than being dropped: host_infra fills first_detected for
    everything, but a source that ever stops would otherwise vanish from the queue silently.
    """
    rows, seen = [], set(done)
    for path, col, datecol in SOURCES:
        if not os.path.isfile(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = (row.get(col) or "").strip().lower().lstrip("www.")
                if not d or d in seen:
                    continue
                a = (row.get("a_records") or "")
                if a and all(ip in WILDCARD_IPS for ip in a.split(";") if ip):
                    seen.add(d)          # remember it so no later source re-offers it
                    continue
                seen.add(d)
                rows.append((row.get(datecol) or "", d))
    rows.sort(key=lambda r: (r[0] == "", r[0]), reverse=True)
    return [d for _, d in rows[:want]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200,
                    help="submissions this run (unlisted/day is 1000; ct_capture_bridge uses ~70)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be submitted, submit nothing")
    a = ap.parse_args()

    key = os.environ.get("QR_SCAN_API_KEY") or os.environ.get("URLSCAN_API_KEY")
    if not key:
        print("[!] no QR_SCAN_API_KEY / URLSCAN_API_KEY — submission needs one", file=sys.stderr)
        return 1

    done = load_ledger()
    todo = candidates(done, a.budget)
    if not todo:
        print(f"[*] every domain in the sources has been attempted ({len(done)} in the ledger)")
        return 0
    print(f"[*] {len(done)} attempted before; submitting {len(todo)} this run")
    if a.dry_run:
        for d in todo[:20]:
            print("   ", d)
        print(f"    ... ({len(todo)} total)")
        return 0

    decode_func, backend = qr_decode.get_decoder()
    print(f"[*] decoder backend: {backend}")
    os.makedirs(SHOT_DIR, exist_ok=True)
    new_rows, qr_rows, rate_stop = [], [], False

    for i, dom in enumerate(todo, 1):
        uuid, why = submit(dom, key)
        if uuid == "__RATE__":
            # Stop the run. Not an attempt against this domain: the limit is ours, not its fault.
            print(f"[!] 429 on submit at {i}/{len(todo)} — stopping, {dom} keeps its budget")
            rate_stop = True
            break
        if not uuid:
            new_rows.append({"domain": dom, "attempt": 1, "attempted_at": time.strftime("%FT%T"),
                             "scan_uuid": "", "shot_file": "", "qr_count": 0, "note": why})
            continue
        ready = wait_result(uuid, key)
        if ready is None:
            print(f"[!] 429 while waiting on {dom} — stopping, it keeps its budget")
            rate_stop = True
            break
        row = {"domain": dom, "attempt": 1, "attempted_at": time.strftime("%FT%T"),
               "scan_uuid": uuid, "shot_file": "", "qr_count": 0,
               "note": "" if ready else "not_ready"}
        if ready:
            img = os.path.join(SHOT_DIR, f"{uuid}.png")
            # A scan can complete with no screenshot: the page was dead, and urlscan records the
            # attempt without an image. That is a legitimate outcome for a phishing domain days
            # after detection -- and a measurable one -- so the ledger says WHICH it was rather
            # than leaving a bare 404 that reads like a fault of ours.
            page_status = ""
            try:
                meta = requests.get(f"https://urlscan.io/api/v1/result/{uuid}/",
                                    headers={**H, "API-Key": key}, timeout=30).json()
                page_status = str((meta.get("page") or {}).get("status", ""))
            except (requests.RequestException, ValueError):
                pass
            try:
                sr = requests.get(f"https://urlscan.io/screenshots/{uuid}.png",
                                  headers={**H, "API-Key": key}, timeout=30)
                if sr.status_code == 200 and sr.content:
                    with open(img, "wb") as fh:
                        fh.write(sr.content)
                    row["shot_file"] = img
                    for d_url in (u.strip() for u in decode_func(img)):
                        if d_url:
                            row["qr_count"] += 1
                            qr_rows.append({"source_page": "https://" + dom,
                                            "urlscan_uuid": uuid, "qr_decoded_url": d_url,
                                            "screenshot_file": img})
                            print(f"    [+] QR on {dom} -> {d_url}")
                else:
                    row["note"] = ("no_screenshot"
                                   + (f"_page_{page_status}" if page_status else "")
                                   + f"_http_{sr.status_code}")
            except (requests.RequestException, OSError) as e:
                row["note"] = f"shot_error:{type(e).__name__}"
        new_rows.append(row)
        if i % 25 == 0:
            print(f"    [{i}/{len(todo)}] submitted")
        time.sleep(1.2)          # the per-minute cap is 60; stay well under it

    if new_rows:
        exists = os.path.isfile(LEDGER)
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    print(f"[+] {len(new_rows)} domain(s) recorded, {len(qr_rows)} QR found"
          + (" (stopped early on 429)" if rate_stop else ""))
    if qr_rows:
        print("[i] run qr_scan.py to fold these into urlscan_qrs.csv with its triage columns, or "
              "read them from the ledger's qr_count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
