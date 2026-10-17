#!/usr/bin/env python3
"""
p3_sample_flow_replay.py — what the off-corpus cut removes from the phishing arm (revision 2, R3.8).

Replays build_content_manifest.py read-only with the capture-file cutoff of the pinned manifest
(content_manifest_vi.csv, written 2026-08-22 18:07): every capture file newer than the cutoff is
ignored, domain dedup and source order are the build's own. The replay must reproduce Table 2
(4,475 captures after dedup, 329 duplicates, 1,010 off-corpus) or it stops. It then reports, per
capture source, how many phishing-arm captures the off-corpus cut removed and how many of those
would have passed the remaining gates, so the cut's effect on the composition of the phishing
class is printed rather than described.

RUN:  python scripts/p3_sample_flow_replay.py
OUT:  data/processed/p3/results/sample_flow_replay.json
      papers/P3_multimodal/sections/gen_offcorpus.tex
"""
import csv
import datetime as dt
import os
import sys
from collections import Counter

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
import build_content_manifest as b
from vn_filter import is_vietnamese_text, visible_text
from genfile import write_generated, write_results

CUTOFF = dt.datetime(2026, 8, 22, 18, 7).timestamp()
EXPECT = {"total": 4475, "dup": 329, "off": 1010}
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
# Capture source -> how it admits a domain. The brand watchers admit on a name-level rule
# (a Vietnamese brand token in the hostname); the live feeds poll public blocklists.
KIND = {"urlscan_brands": "watcher", "ct_brands": "watcher",
        "vn_phishing_live": "feed", "chongluadao_live": "feed"}


def main():
    ph = {}
    for _, r in pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"),
                            low_memory=False).iterrows():
        ph.setdefault(str(r["domain"]).lower(), r["label"])
    seen, st, res = set(), Counter(), Counter()
    for path, lab0, _prov in b.capture_sources():
        src = os.path.basename(os.path.dirname(path))
        src = "retro" if src == "interim" else src
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("found") != "1":
                continue
            f = (r.get("dom_file") or "").strip()
            if not f:
                continue
            if not os.path.isabs(f):
                f = os.path.join(ROOT, f)
            if not os.path.exists(f) or os.path.getmtime(f) > CUTOFF:
                continue
            d = (r.get("domain") or "").strip().lower()
            k = d[4:] if d.startswith("www.") else d
            if k in seen:
                st["dup"] += 1
                continue
            seen.add(k)
            st["total"] += 1
            if lab0 != "phishing":
                continue
            lab = ph.get(d) or ph.get(k)
            off = not lab
            st["off"] += off
            txt = visible_text(open(f, encoding="utf-8", errors="ignore").read())
            if len(txt) < 40:
                g = "empty"
            elif (b.PARKED.search((r.get("title") or "") + " " + txt[:400])
                  or b.NOT_CONTENT.search(txt[:600])):
                g = "parked"
            elif not is_vietnamese_text(txt):
                g = "non_vi"
            else:
                g = "kept"
            # A listed domain keeps the corpus's label: the watchers caught legitimate operators
            # the corpus lists as benign, and the build files them as benign rows.
            res[("off" if off else "in" if str(lab).lower() == "phishing" else "benign",
                 src, g)] += 1
    for key, want in EXPECT.items():
        if st[key] != want:
            raise SystemExit(f"replay does not reproduce Table 2: {key}={st[key]}, expected {want}")

    def n(cut, kind=None, gate=None):
        return sum(v for (c, s, g), v in res.items()
                   if c == cut and (kind is None or KIND.get(s, "retro") == kind)
                   and (gate is None or g == gate))
    out = {"stats": dict(st), "by_source": {"|".join(k): v for k, v in sorted(res.items())}}
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results",
                               "sample_flow_replay.json"), out)
    kept_in = {s: v for (c, s, g), v in res.items() if c == "in" and g == "kept"}
    to_benign = sum(v for (c, s, g), v in res.items() if c == "benign" and g == "kept")
    write_generated(
        os.path.join(SEC, "gen_offcorpus.tex"),
        f"Of the $1{{,}}010$ off-corpus captures, ${n('off', 'watcher')}$ come from the brand "
        f"watchers, which admit a hostname on a name-level rule, and ${n('off', 'feed')}$ from the "
        f"live blocklist feeds, which the released corpus does not draw on, so they carry no corpus "
        f"label or tier. Had they been kept, ${n('off', gate='kept')}$ of them "
        f"would have passed the remaining gates (${n('off', 'watcher', 'kept')}$ watcher, "
        f"${n('off', 'feed', 'kept')}$ feed). A further ${to_benign}$ watcher captures that pass "
        f"every gate are of domains the corpus lists as \\emph{{benign}} (legitimate operators "
        f"such as a carrier and a payment gateway), and the build files them in the benign arm "
        f"under the corpus label. The ${sum(kept_in.values())}$ phishing pages that remain are "
        f"${kept_in.get('chongluadao_live', 0)}$ captures, taken in July and August 2026, of domains "
        f"on a ChongLuaDao list that has not been updated since May 2024, and "
        f"${kept_in.get('retro', 0)}$ retrospective captures, so the cut narrows the phishing "
        f"class to domains the released corpus lists as phishing\n")
    print(st, dict(kept_in))


if __name__ == "__main__":
    main()
