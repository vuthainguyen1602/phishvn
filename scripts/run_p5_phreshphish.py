#!/usr/bin/env python3
"""
run_p5_phreshphish.py — the construction test on a corpus whose every row is dated.

PhishVN dates 41% of its rows, so its stream can only be built, never checked against an arrival
order. PhreshPhish (Dalton et al., 2025) dates every row of both classes with its collection day,
which gives a true order to compare a constructed stream against. Design, fixed before any run:

  true      : every row at its own date, sorted, 20 equal-count windows (as on PhishVN).
  constructed: the dates of a set of rows are hidden and those rows are placed by the same hash
             spreading the PhishVN stream uses (SHA-1 of salt + id across the dated span). Four
             masks, each chosen by a hash of the row's sha256 so it is fixed:
               random25 / random59 / random75 — that share of all rows (59% is PhishVN's undated
                                               share);
               benign   — every benign row, the case where only one class carries dates.
  For each stream and policy: AUTC on F1 and on ROC-AUC (the over-time policies of
  retrain_drift.run, Random Forest, seed 0), and the staleness gap (better updating policy minus
  static). The four constructions are re-placed under ten salts, the released placement ("")
  included, so a constructed gap is read against how far placement alone moves it.

The corpus is sampled to PhishVN's size (53,116 rows, the rows with the smallest hash of their
sha256, so the draw is fixed and class-blind), which also gives the PhishVN window size of about
2,650 rows; the full corpus at 20 windows x 41 streams of refits is not tractable here.
Features are the 21 lexical features PhishVN uses, computed from each URL by the same extractor.
Amendment, 2026-10-13, after the first run stopped in feature extraction and before any result:
URLs that urllib cannot parse (a bracketed host that is not an IP address) are dropped from the
corpus before the sample is drawn, and their count is reported.
Outcome, 2026-10-13: the true-order stream cannot be run. In date order the first window is all
phishing (windows range from 11% to 100% phishing; benign was collected in campaigns, almost none
from November 2024 to April 2025), so the window-1 model sees one class. The date order records
the collection schedule, not arrival. The design is not amended after seeing this.
One row per sha256 as released (URLs are not deduplicated; the duplicate share is reported).

Reads data/external/phreshphish/phreshphish_dated.csv (fetch_phreshphish.py). Writes
data/processed/p5/phreshphish/: pp_stream.csv (features + date + id), pp_constructions.csv (one
row per construction x salt x policy), pp_true_windows.csv (per-window F1 of the true stream) and
pp_corpus.json (rows, classes, span, duplicate URLs, mask sizes).

RUN:  python scripts/run_p5_phreshphish.py [--no-salts]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from compphish_features import extract
from make_p5_assets import F1_DROP, PERIOD, PSI_TAU, WINDOWS
from retrain_drift import load, run

SRC = os.path.join(ROOT, "data", "external", "phreshphish", "phreshphish_dated.csv")
OUT = os.path.join(ROOT, "data", "processed", "p5", "phreshphish")
STREAM = os.path.join(OUT, "pp_stream.csv")
MASKS = {"random25": 0.25, "random59": 0.59, "random75": 0.75, "benign": None}
SAMPLE = 53116                                     # PhishVN's row count
SALTS = [""] + [f"s{k}:" for k in range(1, 10)]   # "" is the first placement, as on PhishVN
FEATURES = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
            "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
            "letter_ratio", "digit_ratio", "spec_ratio", "slash_cnt", "entropy", "path_len",
            "query_len"]


def unit(key: str) -> float:
    """A fixed number in [0, 1) from a string: the mask draw, independent of the placement hash."""
    return int(hashlib.sha1(("mask:" + key).encode()).hexdigest(), 16) % 10000 / 10000


def parses(url: str) -> bool:
    from urllib.parse import urlparse
    import re
    try:
        urlparse(url if re.match(r"^[a-z]+://", url, re.I) else "http://" + url)
        return True
    except ValueError:
        return False


def build_stream() -> dict:
    full = pd.read_csv(SRC, low_memory=False)
    ok = full["url"].astype(str).map(parses)
    unparsable = int((~ok).sum())
    full = full[ok].reset_index(drop=True)
    draw = full["sha256"].astype(str).map(lambda k: hashlib.sha1(("sample:" + k).encode()).hexdigest())
    src = full.loc[draw.sort_values(kind="stable").index[:SAMPLE]].sort_index().reset_index(drop=True)
    feats = pd.DataFrame([extract(u) for u in src["url"].astype(str)])[FEATURES]
    feats["label"] = src["label"].map({"phish": "phishing", "benign": "benign"}).values
    feats["collected_at"] = pd.to_datetime(src["date"]).dt.strftime("%Y-%m-%d").values
    feats["id"] = src["sha256"].values
    os.makedirs(OUT, exist_ok=True)
    feats.to_csv(STREAM, index=False)
    t = pd.to_datetime(feats["collected_at"])
    bare = (src["url"].astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True))
    return {"unparsable_dropped": unparsable, "corpus_rows": len(full), "corpus_phishing": int((full.label == "phish").sum()),
            "rows": len(feats), "phishing": int((feats.label == "phishing").sum()),
            "benign": int((feats.label == "benign").sum()),
            "first_date": f"{t.min():%Y-%m-%d}", "last_date": f"{t.max():%Y-%m-%d}",
            "distinct_dates": int(t.nunique()), "duplicate_urls": int(len(bare) - bare.nunique())}


def masked(name: str) -> pd.DataFrame:
    """The stream file with the dates of the masked rows blanked."""
    s = pd.read_csv(STREAM, low_memory=False)
    share = MASKS[name]
    hide = (s.label == "benign") if share is None else s["id"].astype(str).map(unit) < share
    s.loc[hide, "collected_at"] = ""
    return s


def one(name: str, df: pd.DataFrame, feats: list[str], salt: str):
    tr = {}
    strat, counts, _ = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP, include_static_arch=False,
                           trace=tr)
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
    pri = [float(df.y.iloc[i].mean()) for i in idx]
    return strat, [{"construction": name, "salt": salt, "rows": len(df), "policy": k,
             "autc": float(np.mean(v)), "end_f1": float(v[-1]), "retrains": counts[k],
             "auc_autc": float(np.nanmean(tr[k]["auc"])), "w1_prior": pri[0],
             "prior_min": min(pri), "prior_max": max(pri)} for k, v in strat.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-salts", action="store_true", help="the first placement only")
    a = ap.parse_args()
    corpus = build_stream()
    print(json.dumps(corpus), flush=True)

    rows = []
    df, feats = load(STREAM, "collected_at", spread_undated=False)
    assert df["_t"].notna().all() and len(feats) == len(FEATURES)
    strat, r = one("true", df, feats, "")
    rows += r
    win = pd.DataFrame(strat)
    win.insert(0, "window", range(2, 2 + len(win)))
    win.to_csv(os.path.join(OUT, "pp_true_windows.csv"), index=False)
    print("[true] done", flush=True)

    corpus["masked_rows"] = {}
    for salt in (SALTS[:1] if a.no_salts else SALTS):
        for name in MASKS:
            path = os.path.join(OUT, f"_masked_{name}.csv")
            if salt == "":
                m = masked(name)
                corpus["masked_rows"][name] = int((m["collected_at"].fillna("") == "").sum())
                m.to_csv(path, index=False)
            d, f = load(path, "collected_at", spread_undated=True, salt=salt)
            rows += one(name, d, f, salt)[1]
            print(f"[{name} salt={salt!r}] done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "pp_constructions.csv"), index=False)
    for name in MASKS:
        os.remove(os.path.join(OUT, f"_masked_{name}.csv"))
    json.dump(corpus, open(os.path.join(OUT, "pp_corpus.json"), "w"), indent=2, sort_keys=True)
    out = pd.DataFrame(rows)
    g = out.groupby(["construction", "salt"]).apply(
        lambda x: pd.Series({"gap_f1": x.set_index("policy").autc.drop("static").max() - x.set_index("policy").autc["static"],
                             "gap_auc": x.set_index("policy").auc_autc.drop("static").max() - x.set_index("policy").auc_autc["static"]}))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
