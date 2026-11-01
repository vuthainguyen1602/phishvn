#!/usr/bin/env python3
"""
run_p2_phreshphish_temporal.py — the protocol contrast on a corpus that dates both classes
(revision 1, 2026-10-31).

WHY. PhishVN dates its phishing rows but almost none of its benign ones, so its forward-in-time
protocol orders the phishing class only and splits the benign class at random. Whether that
one-sided design understates the cost of time cannot be measured on PhishVN. PhreshPhish (Dalton
et al., 2025; CC BY 4.0) dates every row of both classes with its collection day; the drift study already
draws a fixed, class-blind 53,116-row sample of it with the same 21 lexical features
(data/processed/p5/phreshphish/pp_stream.csv, run_p5_phreshphish.py).

DESIGN. On those rows, three ways to split 70/30, each with the registrable-domain guard of
run_p2_temporal_strict.py (a test row whose registrable domain occurs in training is dropped):
  two_class_temporal   every row ordered by date; train = earliest 70%, test = the rest
  phishing_temporal    PhishVN's design: phishing ordered by date (earliest 70% train), benign
                       split at random at the same rate
  random_guarded       a random 70/30 split, the guard applied, the benign test rows subsampled so
                       the test prior equals two_class_temporal's
Seven families, five seeds. The two temporal phishing splits are fixed; seeds move the benign draw
(phishing_temporal), the split (random_guarded) and the model.

One caution the reading must carry, measured on 2026-10-13 for the drift study: PhreshPhish's benign rows were
collected in campaigns (almost none from November 2024 to April 2025), so its date order records
the collection schedule, and the two-class temporal test window is not a deployment stream either.

Writes data/processed/p2/revision/phreshphish_temporal.csv.

RUN: OPENBLAS_NUM_THREADS=1 python scripts/run_p2_phreshphish_temporal.py
"""
from __future__ import annotations

import argparse
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
from train_url_baseline import COMPPHISH
from run_p2_benchmark import FAMILIES, run_one
from run_p2_temporal_strict import match_prior
from audit_p2_harmonisation import host_of
from psl import registered_domain

STREAM = "data/processed/p5/phreshphish/pp_stream.csv"
URLS = "data/external/phreshphish/phreshphish_dated.csv"
OUT = "data/processed/p2/revision/phreshphish_temporal.csv"
CUT = 0.70


def load() -> pd.DataFrame:
    s = pd.read_csv(STREAM)
    u = pd.read_csv(URLS, usecols=["sha256", "url"]).drop_duplicates("sha256")
    s = s.merge(u, left_on="id", right_on="sha256", how="left")
    s["y"] = (s.label.astype(str).str.lower() == "phishing").astype(int)
    s["date"] = pd.to_datetime(s.collected_at)
    s["rdom"] = s.url.map(host_of).map(registered_domain)
    # stable order inside a day: the row id, so the split never depends on file order
    return s.sort_values(["date", "id"]).reset_index(drop=True)


def guard(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    return te[~te.rdom.isin(set(tr.rdom))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    k = int(len(df) * CUT)
    tt_tr = df.iloc[:k]
    tt_te = guard(tt_tr, df.iloc[k:])
    prior = float(tt_te.y.mean())
    ph = df[df.y == 1]
    be = df[df.y == 0]
    kp = int(len(ph) * CUT)
    print(f"rows {len(df)}, features {len(feats)}; two-class cut at {df.date.iloc[k].date()}: "
          f"train {len(tt_tr)} (prior {tt_tr.y.mean():.3f}), test {len(tt_te)} after the guard "
          f"(prior {prior:.3f}); phishing cut at {ph.date.iloc[kp].date()}", flush=True)

    rows = []
    for s in range(args.seeds):
        rng = np.random.RandomState(s)
        bm = rng.rand(len(be)) < CUT
        p_tr = pd.concat([ph.iloc[:kp], be[bm]])
        p_te = guard(p_tr, pd.concat([ph.iloc[kp:], be[~bm]]))
        rm = rng.rand(len(df)) < CUT
        r_tr = df[rm]
        r_te = match_prior(guard(r_tr, df[~rm]), prior, rng)
        for proto, (tr, te) in {"two_class_temporal": (tt_tr, tt_te),
                                "phishing_temporal": (p_tr, p_te),
                                "random_guarded": (r_tr, r_te)}.items():
            for name in args.families:
                met = run_one(name, tr[feats].to_numpy(float), tr.y.to_numpy(int),
                              te[feats].to_numpy(float), te.y.to_numpy(int), s)
                met.update(protocol=proto, n_test=len(te), test_prior=float(te.y.mean()))
                rows.append(met)
                print(f"  {name:<13} {proto:<19} seed={s} F1={met['F1']:.3f} "
                      f"ROC-AUC={met['ROC-AUC']:.3f} prior={te.y.mean():.3f}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["protocol", "family"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
          .mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
