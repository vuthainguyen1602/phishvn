#!/usr/bin/env python3
"""
align_compphish.py — Featurize URLs into the CompPhish 26-column schema so VN data can be
trained/tested jointly (cross-dataset) with CompPhish (Goenka 2026, 10.17632/fmbs4kp9wz.2).

Input : a CSV with a URL column (+ optional label). Works on data/processed/dataset_url.csv
        (uses url_norm/url and maps phishing->1, benign->0) or ANY external URL list, so that
        third-party corpora can be re-featurised into the same schema for a multi-corpus
        cross-dataset study. Column matching is case-insensitive.
Output: CSV with exactly: <25 CompPhish features>, label

RUN:
  # our VN dataset -> CompPhish format
  python align_compphish.py --in data/processed/dataset_url.csv --out data/processed/vn_compphish.csv
  # an EXTERNAL corpus -> same schema. CHECK THE POLARITY FIRST: to_label() below reads a
  # numeric 1 as PHISHING, and not every corpus agrees. PhiUSIIL ships 1 = LEGITIMATE
  # (134,850 ones against 100,945 zeros, matching its published class counts), so feeding its
  # raw file straight in --- as an earlier version of this very usage line told you to ---
  # inverts the labels silently and produces a corpus that looks fine and scores below chance
  # on transfer. Convert to the string labels first (data/external/phiusiil_urllabel.csv is the
  # converted copy this project actually uses, spot-checked by hand) and pass those:
  python align_compphish.py --in data/external/phiusiil_urllabel.csv --url-col url \
         --label-col label --out data/processed/phiusiil_compphish.csv
  # Grambeddings / PhishTank / Mendeley URL sets work the same way (point --url-col/--label-col),
  # and each needs the same polarity check before it is trusted.
  # then concatenate the *_compphish.csv files (identical columns) and train/test across corpora.
"""
from __future__ import annotations
import argparse, csv, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from compphish_features import extract, COLUMNS


def to_label(v):
    s = str(v).strip().lower()
    if s in ("1", "phishing", "phish", "spam", "smishing", "malicious"):
        return 1
    if s in ("0", "benign", "legit", "legitimate", "ham"):
        return 0
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url-col", default="", help="URL column (auto: url_norm/url/dom/domain)")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Empty input.")
    cols = list(rows[0].keys())
    lower = {c.lower(): c for c in cols}                       # case-insensitive lookup
    if args.url_col:
        ucol = lower.get(args.url_col.lower(), args.url_col)
    else:
        ucol = next((lower[c] for c in ("url_norm", "url", "dom", "domain") if c in lower), None)
    if not ucol or ucol not in cols:
        raise SystemExit(f"No URL column found; have {cols}. Pass --url-col.")
    lcol = lower.get(args.label_col.lower(), args.label_col)
    scol = lower.get("split")   # carry over the group-aware split if the input has one
    tcol = lower.get("tier")    # carry the label-confidence tier so benchmarks can slice on it

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = COLUMNS + ["label"] + (["split"] if scol else []) + (["tier"] if tcol else [])
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            u = (r.get(ucol) or "").strip()
            if not u:
                continue
            feat = extract(u)
            feat["label"] = to_label(r.get(lcol, ""))
            if scol:
                feat["split"] = r.get(scol, "")
            if tcol:
                feat["tier"] = r.get(tcol, "")
            w.writerow(feat); n += 1
    print(f"Wrote {n} rows -> {args.out} (CompPhish schema)")


if __name__ == "__main__":
    main()
