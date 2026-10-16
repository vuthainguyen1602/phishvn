#!/usr/bin/env python3
"""
build_content_manifest.py — Assemble the P1b content manifest from all page captures, with the
language + quality filtering that de-confounds the content experiment.

For every captured HTML file: extract visible text, drop near-empty (<40 chars) and
parked/suspended/challenge/error pages, keep only Vietnamese-content pages for the *_vi manifest.
Sources are merged in priority order so LIVE captures beat RETRO ones; the retrospective
ChongLuaDao captures are excluded outright (survivorship bias). Each row carries 'js_dir' so the
JS modality sees the full script surface, not just inline code.

Outputs (with a 'provenance' column so live vs retro can be sliced):
  data/interim/content_manifest.csv     — all captured pages (both languages), for reference
  data/interim/content_manifest_vi.csv  — Vietnamese-content, non-parked pages (used by P1b)

RUN:  python scripts/build_content_manifest.py
The full source list, its priority order and the exclusions: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import csv
import glob
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from vn_filter import is_vietnamese_text, visible_text

INTERIM = os.path.join(ROOT, "data", "interim")
RAW = os.path.join(ROOT, "data", "raw")

PARKED = re.compile(
    r"(for sale|is for sale|parked|expired|suspended|account suspended|not found|404|403|"
    r"under construction|coming soon|apache|nginx|welcome to nginx|index of /|godaddy|sedo|"
    r"namecheap|has expired|forbidden|error|maintenance|plesk|cpanel|it works|um momento|"
    r"just a moment|checking your browser|cloudflare|access denied|domain default page|"
    r"this domain (may be|is) for sale)", re.I)
# Two capture classes that are not site content at all and were being scored for Vietnameseness.
# The registry wildcard: dotPH answers every unregistered .ph name with one ParkLogic page, which
# is why brand-stuffed names like vietcombank-primevideo.com.ph have captures at all (the same
# wildcard audit_capture_labels.py flags as verdict `registry_wildcard`). The browser error page: the
# scanner's own failure screen, matched on Chrome's error CODE because that survives localisation
# while the surrounding prose does not -- the pages arrive in French and Portuguese.
NOT_CONTENT = re.compile(
    r"(available to be registered|click here to register|category search|"
    r"ERR_[A-Z_]{4,}|DNS_PROBE_FINISHED|HTTP ERROR \d{3})")
# retrospective ChongLuaDao captures — excluded (see the design note / P1b limitations)
SKIP_FILES = {"urlscan_cld_search.csv", "urlscan_cld_live.csv"}





def _backup_manifest(path: str) -> None:
    """Copy an existing manifest aside, dated, so a rebuild is reversible."""
    if not os.path.exists(path):
        return
    import datetime as _dt
    import shutil as _sh
    stamp = _dt.date.fromtimestamp(os.path.getmtime(path)).isoformat()
    dst = path.replace(".csv", f".{stamp}.bak.csv")
    if not os.path.exists(dst):
        _sh.copy2(path, dst)
        print(f"[i] previous manifest kept at {dst}")


def capture_sources():
    """Ordered list of (csv_path, label, provenance). Earlier entries win domain-level dedup, so
    LIVE captures take precedence over the retrospective crawl for the same domain."""
    srcs = []
    # 1. LIVE: Jetson/collector detections (phishing, crawl-at-detection) — may be multiple files
    for p in sorted(glob.glob(os.path.join(RAW, "*_live", "detections.csv"))):
        srcs.append((p, "phishing", "live"))
    # 1b. LIVE brand-lookalike captures (dirs don't match *_live; ct_brands' captures live in
    # captures.csv — its detections.csv is cert identities with no dom_file)
    # urlscan_brands keeps BOTH: detections.csv is the identity record (one row per domain, at
    # first detection) and captures.csv is the attempt log a later run writes when it finally
    # reaches a domain the capture budget skipped. Reading only the first loses those pages.
    for rel in (("urlscan_brands", "detections.csv"), ("urlscan_brands", "captures.csv"),
                ("ct_brands", "captures.csv")):
        p = os.path.join(RAW, *rel)
        if os.path.exists(p):
            srcs.append((p, "phishing", "live"))
    # 2a. benign LIVE: certified trusted-org captures (collect_benign_orgs.py). Deliberately not
    # under a *_live dir — the glob above labels those phishing.
    p = os.path.join(RAW, "tinnhiem_benign", "detections.csv")
    if os.path.exists(p):
        srcs.append((p, "benign", "live"))
    # 2b. benign retro crawl
    p = os.path.join(INTERIM, "urlscan_benign.csv")
    if os.path.exists(p):
        srcs.append((p, "benign", "retro"))
    # 3. retro phishing (the remaining interim urlscan*.csv, minus benign + skipped)
    for p in sorted(glob.glob(os.path.join(INTERIM, "urlscan*.csv"))):
        b = os.path.basename(p)
        if b == "urlscan_benign.csv" or b in SKIP_FILES:
            continue
        srcs.append((p, "phishing", "retro"))
    return srcs


def main():
    # cross-check phishing/benign label against the URL dataset when a domain is known
    ph_lab = {}
    dpath = os.path.join(ROOT, "data", "processed", "dataset_url.csv")
    if os.path.exists(dpath):
        for _, r in pd.read_csv(dpath, low_memory=False).iterrows():
            ph_lab.setdefault(str(r["domain"]).lower(), r["label"])

    seen_dom = set()
    rows_all, rows_vi = [], []
    stats = {"total": 0, "dup": 0, "empty": 0, "parked": 0, "non_vi": 0, "kept_vi": 0}
    prov_counts = {}
    for path, default_label, provenance in capture_sources():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("found") != "1":
                continue
            domf = (r.get("dom_file") or "").strip()
            if not domf:
                continue
            if not os.path.isabs(domf):
                domf = os.path.join(ROOT, domf)
            if not os.path.exists(domf):
                continue
            dom = (r.get("domain") or "").strip().lower()
            key = dom[4:] if dom.startswith("www.") else dom  # www.x and x are the same page
            if key in seen_dom:               # domain-level dedup (live wins, processed first)
                stats["dup"] += 1; continue
            seen_dom.add(key)
            stats["total"] += 1
            try:
                html = open(domf, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            txt = visible_text(html)
            tit = r.get("title") or ""
            # benign source stays benign; phishing source cross-checked against the URL dataset.
            # A capture absent from the released corpus is DROPPED, not assumed phishing: the
            # watcher admits by vn_filter.is_vn_target, a selection an audit found unreliable
            # (0 confirmed phishing, 31% verifiably legitimate). The old assume-phishing fallback
            # put agimexpharm.com.vn and an FPT school into the content study's positive class.
            if default_label == "benign":
                lab = "benign"
            else:
                lab = ph_lab.get(dom) or ph_lab.get(key)
                if not lab:
                    stats["off_corpus"] = stats.get("off_corpus", 0) + 1
                    continue
            # external JS captured alongside the DOM: explicit js_dir column (benign log) or the
            # capture convention <landing dir>_js/<scan_uuid> (live logs); "" when nothing exists
            jsd = (r.get("js_dir") or "").strip()
            if jsd and not os.path.isabs(jsd):
                jsd = os.path.join(ROOT, jsd)
            uuid = (r.get("scan_uuid") or "").strip()
            if not jsd and uuid:
                jsd = os.path.dirname(domf) + "_js" + os.sep + uuid
            if not os.path.isdir(jsd):
                jsd = ""
            row = {"domain": dom, "label": lab, "provenance": provenance,
                   "tier": r.get("tier", ""), "dom_file": domf,
                   "shot_file": r.get("shot_file", ""), "scan_uuid": r.get("scan_uuid", ""),
                   "js_dir": jsd}
            rows_all.append(row)
            if len(txt) < 40:
                stats["empty"] += 1; continue
            if PARKED.search(tit + " " + txt[:400]) or NOT_CONTENT.search(txt[:600]):
                stats["parked"] += 1; continue
            if not is_vietnamese_text(txt):
                stats["non_vi"] += 1; continue
            stats["kept_vi"] += 1
            prov_counts[(provenance, lab)] = prov_counts.get((provenance, lab), 0) + 1
            rows_vi.append(row)

    # A rebuild moves the corpus a submitted paper was measured on (the content study's balanced subset is
    # drawn from it), and on 2026-08-22 an unattended rebuild inside a data sync left its
    # previous state unrecoverable. Keep the outgoing manifest before overwriting it.
    _backup_manifest(os.path.join(INTERIM, "content_manifest.csv"))
    _backup_manifest(os.path.join(INTERIM, "content_manifest_vi.csv"))
    pd.DataFrame(rows_all).to_csv(os.path.join(INTERIM, "content_manifest.csv"), index=False)
    vi = pd.DataFrame(rows_vi)
    vi.to_csv(os.path.join(INTERIM, "content_manifest_vi.csv"), index=False)
    print(f"captured pages (domain-deduped): {stats['total']}  (skipped {stats['dup']} duplicate domains)")
    # off_corpus was counted but never printed, so the largest single drop in the phishing arm was
    # invisible in the build log -- and it happens BEFORE the quality gate, which is why there is no
    # "Vietnamese-content phishing before the corpus cross-check" quantity to report.
    print(f"  dropped: off-corpus {stats.get('off_corpus', 0)}, empty {stats['empty']}, "
          f"parked {stats['parked']}, non-Vietnamese {stats['non_vi']}")
    print(f"  kept (Vietnamese, cleaned): {stats['kept_vi']}")
    if len(vi):
        print("  by label:", vi.label.value_counts().to_dict())
        print("  by provenance x label:", {f"{p}/{l}": n for (p, l), n in sorted(prov_counts.items())})


if __name__ == "__main__":
    main()
