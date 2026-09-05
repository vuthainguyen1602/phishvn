#!/usr/bin/env python3
"""
p3_jaccard_check.py — the calibration instrument for P3's controlled paraphrase attack.

The successor study (papers/P3_multimodal/protocols/PARAPHRASE_BAND_PROTOCOL.md) fixes attack
strength as token Jaccard inside [0.20, 0.30]; rewrites are revised until they land there, and this
is what the revision is driven by. It DELIBERATELY cannot import a model, train anything or compute
a miss rate — that firewall is the only thing separating "calibrating the instrument" from tuning
the attack against its own outcome, and check_paper_claims.py asserts that no learning library
can be named in this file at all.

RUN:
  python scripts/p3_jaccard_check.py                 # audit the current paraphrase set
  python scripts/p3_jaccard_check.py --out-of-band   # only the rows still needing revision
  python scripts/p3_jaccard_check.py --ids a,b,c     # specific sources
  python scripts/p3_jaccard_check.py --tsv FILE      # audit candidate rewrites: id<TAB>variant<TAB>text
The tokenisation and the outcome-blindness rule: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

BAND = (0.20, 0.30)
LINK = "http://sim.example.com/x"
URL_RE = re.compile(r"https?://\S+|\bsim\.example\.vn\S*", re.I)
BRAND_STOP = ("vietcom", "techcom", "vietin", "agribank", "bidv", "mbbank", "sacombank",
              "acb", "tpbank", "vpbank", "momo", "zalopay", "vnpay", "shopee", "lazada",
              "tiki", "sendo", "grab", "viettel", "vinaphone", "mobifone", "facebook",
              "zalo", "tiktok", "instagram", "ghtk", "ghn", "vnpost", "jtexpress")


def toks(s: str) -> set[str]:
    return set(URL_RE.sub(" ", str(s)).lower().split())


def jaccard(a: str, b: str) -> float:
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def guardrail_problems(text: str) -> list[str]:
    """Every rule the shipped attack set must satisfy, re-checked here so calibration cannot
    quietly trade a guardrail for a Jaccard target."""
    bad = []
    for u in URL_RE.findall(text):
        if u.rstrip(".,;") != LINK:
            bad.append(f"non-simulated URL {u!r}")
    low = text.lower()
    bad += [f"real brand token {b!r}" for b in BRAND_STOP if b in low]
    if any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text)):
        bad.append("diacritics (orthography control)")
    if len(text) < 20:
        bad.append("implausibly short")
    if LINK not in text:
        bad.append("simulated link dropped")
    return bad


def sources() -> dict[str, str]:
    import pandas as pd
    frames = [pd.read_csv(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))
              for t in ("sms", "email")]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["label"] == "phishing"]
    return {r["id"]: str(r["text"]) for _, r in d.iterrows()}


def current_rewrites() -> list[dict]:
    p = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing — run scripts/p3_paraphrase_corpus.py first.")
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def from_tsv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise SystemExit(f"malformed line (need id<TAB>variant<TAB>text): {ln[:60]!r}")
            rows.append({"src_id": parts[0].strip(), "variant": parts[1].strip(),
                         "text": parts[2].strip()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", help="audit candidate rewrites from a TSV instead of the shipped set")
    ap.add_argument("--out-of-band", action="store_true", help="print only rows needing revision")
    ap.add_argument("--ids", help="comma-separated source ids to restrict to")
    args = ap.parse_args()

    src = sources()
    rows = from_tsv(args.tsv) if args.tsv else current_rewrites()
    if args.ids:
        want = {i.strip() for i in args.ids.split(",")}
        rows = [r for r in rows if r["src_id"] in want]

    lo, hi = BAND
    n_in = n_out = n_bad = 0
    for r in rows:
        s = src.get(r["src_id"])
        if s is None:
            print(f"[!] unknown source id {r['src_id']}")
            n_bad += 1
            continue
        j = jaccard(s, r["text"])
        problems = guardrail_problems(r["text"])
        ok = lo <= j <= hi and not problems
        n_in += ok
        n_out += (not ok)
        n_bad += bool(problems)
        if ok and args.out_of_band:
            continue
        flag = "OK  " if ok else ("LOW " if j < lo else "HIGH" if j > hi else "BAD ")
        note = ("  !! " + "; ".join(problems)) if problems else ""
        print(f"{flag} J={j:.3f}  {r['src_id']}/{r['variant']}{note}")
        if not ok and not args.out_of_band:
            shared = sorted(toks(s) & toks(r["text"]))
            print(f"       shared tokens ({len(shared)}): {' '.join(shared)}")

    total = n_in + n_out
    print(f"\nband [{lo:.2f}, {hi:.2f}]: {n_in}/{total} in band, {n_out} to revise"
          + (f", {n_bad} with guardrail problems" if n_bad else ""))
    if args.tsv and n_out:
        sys.exit(1)


if __name__ == "__main__":
    main()
