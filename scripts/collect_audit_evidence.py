#!/usr/bin/env python3
"""
collect_audit_evidence.py — find, for each row of the label-audit sample, what evidence exists.

WHAT THIS IS FOR. The audit's cost is not judgement, it is lookup: only 12 of the 200 sampled
domains have a capture inside this corpus, so an annotator otherwise hand-searches archives for
the other 188. That part is mechanical and belongs to a script.

WHAT THIS DELIBERATELY DOES NOT DO. It writes no verdict and suggests none, not even a hint, because
a suggestion in the sheet is an anchor and the quantity being measured is whether two people agree
independently. It also never reads key.csv: the sample is blinded, and a tool that joined the source
label onto the evidence table would un-blind it for anyone who opened the file.

Sources, matching the order the codebook fixes (docs/verify/CODEBOOK.md):
  1. this corpus's own capture      (data/interim/content_manifest.csv)
  2. a public urlscan scan           (data/interim/urlscan.csv -> a result URL)
  3. a web archive near first-seen   (Wayback CDX, nearest snapshot to the detection date)
  4. the trusted-organisation registry (data/raw/tinnhiem_org)

RUN:  python scripts/audit/collect_audit_evidence.py
      python scripts/audit/collect_audit_evidence.py --limit 20     # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
SHEET = os.path.join(ROOT, "data", "docs", "verify", "annotator_A.csv")
OUT = os.path.join(ROOT, "data", "docs", "verify", "EVIDENCE.csv")
UA = "Mozilla/5.0 (research; contact nvthai@utc2.edu.vn)"


def _read(path, key=None):
    if not os.path.exists(path):
        return {} if key else []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r[key]: r for r in rows if r.get(key)} if key else rows


def wayback(domain: str, near: str | None, retries: int = 3):
    """Nearest archived snapshot to `near` (YYYYMMDD), plus how many exist at all."""
    q = ("http://web.archive.org/cdx/search/cdx?url="
         + urllib.parse.quote(domain)
         + "&output=json&fl=timestamp,statuscode&collapse=timestamp:6&limit=200")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                rows = json.loads(r.read().decode("utf-8") or "[]")
            break
        except Exception:
            if attempt == retries - 1:
                return None, 0, "lookup-failed"
            time.sleep(2 ** attempt)
    if len(rows) < 2:
        return None, 0, ""
    stamps = [x[0] for x in rows[1:] if x and x[0]]
    if not stamps:
        return None, 0, ""
    pick = min(stamps, key=lambda t: abs(int(t[:8]) - int(near))) if near else stamps[0]
    return pick, len(stamps), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    sheet = _read(SHEET)
    if not sheet:
        raise SystemExit(f"no sample at {SHEET}")
    if any((r.get("verdict") or "").strip() for r in sheet):
        raise SystemExit("The sheet already carries verdicts. Refusing to run: evidence collected "
                         "after annotation cannot inform it and can only look like it did.")

    manifest = _read(os.path.join(ROOT, "data", "interim", "content_manifest.csv"), "domain")
    scans = _read(os.path.join(ROOT, "data", "interim", "urlscan.csv"), "domain")
    registry = {r.get("domain", "").strip().lower()
                for r in _read(os.path.join(ROOT, "data", "raw", "tinnhiem_org", "orgs.csv"))
                if r.get("domain")}
    # collected_at arrives in two formats, dd/mm/yyyy and ISO, in that precedence -- the same order
    # normalize_merge uses. Parsing only one silently discards half the corpus's dates, which is
    # exactly the bug check_paper_claims.parse_event_date exists to prevent.
    def event_date(s: str):
        s = (s or "").strip()[:10]
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        return None

    dates = {}
    for r in _read(os.path.join(ROOT, "data", "processed", "dataset_url.csv")):
        u, d = r.get("url", ""), event_date(r.get("collected_at"))
        if u and d:
            dates[u] = d.strftime("%Y%m%d")

    rows = sheet[:args.limit] if args.limit else sheet
    out = []
    for i, r in enumerate(rows, 1):
        dom = r["url"]
        near = dates.get(dom)
        cap = manifest.get(dom, {})
        scan = scans.get(dom, {})
        stamp, n_snap, note = wayback(dom, near)
        out.append({
            "vid": r["vid"],
            "url": dom,
            "first_seen": near or "",
            "corpus_capture": cap.get("shot_file", "") or cap.get("dom_file", ""),
            "capture_provenance": cap.get("provenance", ""),
            "urlscan": (f"https://urlscan.io/result/{scan['scan_uuid']}/"
                        if scan.get("scan_uuid") else ""),
            "registry": "yes" if dom.lower() in registry else "",
            "wayback_snapshots": n_snap,
            "wayback_nearest": (f"https://web.archive.org/web/{stamp}/http://{dom}"
                                if stamp else ""),
            "note": note,
        })
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)
        time.sleep(args.sleep)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    have = sum(1 for r in out if r["corpus_capture"] or r["urlscan"]
               or r["wayback_nearest"] or r["registry"])
    print(f"[+] {OUT}")
    print(f"    {have}/{len(out)} rows have at least one source to look at "
          f"({100.0 * have / len(out):.0f}%)")
    for k, lab in (("corpus_capture", "corpus capture"), ("urlscan", "urlscan scan"),
                   ("wayback_nearest", "web archive"), ("registry", "registry")):
        print(f"      {lab:15s} {sum(1 for r in out if r[k]):3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
