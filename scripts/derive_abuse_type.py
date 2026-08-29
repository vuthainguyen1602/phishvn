#!/usr/bin/env python3
"""
derive_abuse_type.py — say what KIND of abuse each listed domain is, where that can be said.

WHY. The positive class is "listed by a Vietnamese anti-fraud feed", not "phishing". ChongLuaDao
supplies 92% of it and is a chong-lua-dao project, so gambling, betting streams, investment fraud
and adult sites sit in the class beside credential phishing. `scenario` cannot express that: it is
a BRAND-IMPERSONATION taxonomy, so everything impersonating nobody lands in `other` — 91% of
positives — reading as "unclassified phishing" rather than "not impersonation at all".

WHAT IT DELIBERATELY DOES NOT DO: guess. A type is assigned only where a conservative rule fires,
`unknown` everywhere else. Opaque names are the norm (mc622.com, tse6971.com, s567.live) and no
lexical rule can read them, so a high `unknown` share is the honest outcome. Precision over recall
throughout: a paper filtering on this column would otherwise silently train on a different
population than it says it does.

Rules, in precedence order:
  brand_impersonation  the registry brand-token machinery already fired (scenario != other) — this
                       tier inherits that inference's precision, and `rule` records which fired
  gambling             Vietnamese gambling vocabulary or a named betting brand
  betting_stream       match-streaming vocabulary
  investment           crypto/forex/investment vocabulary
  adult                adult vocabulary
  unknown              no rule fired — the great majority

OUTPUT: data/processed/abuse_type.csv (url, abuse_type, rule) — joinable on `url`, deliberately a
side file rather than a new column in dataset_url.csv, so the released schema and every check
pinned to it stay unchanged.

RUN:  python scripts/derive_abuse_type.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
SRC = os.path.join(ROOT, "data", "processed", "dataset_url.csv")
OUT = os.path.join(ROOT, "data", "processed", "abuse_type.csv")

# Vietnamese gambling vocabulary and named betting brands. Deliberately no bare English "bet",
# "club" or "vip": those appear in legitimate names often enough to cost precision, and the .vip
# TLD alone is not evidence of anything.
# Long, unambiguous tokens may match anywhere in the name.
GAMBLING = re.compile(
    r"taixiu|tai-xiu|xocdia|xoc-dia|nohu|no-hu|danhbai|danh-bai|baucua|bau-cua|"
    r"sunwin|rikvip|go88|b52club|b52-club|hitclub|hit-club|may88|jun88|f8bet|"
    r"188bet|fun88|nhatvip|vin777|789club|789-club|casino|baccarat|xosomienbac", re.I)
# Short brand tokens need a boundary: "m88" matched inside clmm888, which is not a betting brand.
# A digit-run either side is what distinguishes the brand from a coincidence in a random string.
GAMBLING_SHORT = re.compile(
    r"(?:^|[^a-z0-9])(?:iwin|luck8|w88|12bet|fb88|vn88|m88|bk8|xoso)(?:[^a-z0-9]|$)", re.I)
BETTING_STREAM = re.compile(
    r"bong88|keonhacai|keo-nha-cai|kqbd|xoilac|xoi-lac|90phut|socolive|thapcam|"
    r"tructiepbongda|truc-tiep-bong-da|bongdaso|banthang", re.I)
INVESTMENT = re.compile(
    r"forex|binance|ethereum|bitcoin|crypto|(?:^|[-.])coin|xingtou|dautu|dau-tu|"
    r"sanguonvon|iqoption|olymptrade", re.I)
ADULT = re.compile(r"(?:^|[-.])sex|xxx|porn|(?:^|[-.])jav(?:[-.]|\d)|phimsex|truyensex", re.I)

RULES = [("gambling", GAMBLING), ("gambling", GAMBLING_SHORT),
         ("betting_stream", BETTING_STREAM),
         ("investment", INVESTMENT), ("adult", ADULT)]


def classify(url: str, scenario: str):
    if (scenario or "").strip() and scenario.strip().lower() != "other":
        return "brand_impersonation", f"scenario={scenario.strip().lower()}"
    for name, pat in RULES:
        m = pat.search(url or "")
        if m:
            return name, f"token={m.group(0).strip('-.').lower()}"
    return "unknown", ""


def main() -> int:
    if not os.path.exists(SRC):
        raise SystemExit(f"missing {SRC} — run `make data` first")
    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out, counts = [], Counter()
    for r in rows:
        if (r.get("label") or "").strip().lower() != "phishing":
            continue                       # the benign arm is not abuse and gets no type
        t, why = classify(r.get("url", ""), r.get("scenario", ""))
        out.append({"url": r["url"], "abuse_type": t, "rule": why})
        counts[t] += 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url", "abuse_type", "rule"])
        w.writeheader()
        w.writerows(out)

    n = len(out)
    print(f"[+] {OUT}  ({n:,} listed domains)")
    for t, c in counts.most_common():
        print(f"      {t:20s} {c:6,}  ({100.0 * c / n:5.1f}%)")
    known = n - counts["unknown"]
    print(f"    typed by a rule: {known:,} ({100.0 * known / n:.1f}%) — the rest is `unknown` by "
          f"design, not by omission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
