#!/usr/bin/env python3
"""
run_p2_temporal_strict.py — P2's honest temporal protocol, built only from rows whose detection
date is real.

WHY THIS EXISTS. normalize_merge's `split` column is only PARTLY temporal: sources without a
date are split randomly, so run_p2_benchmark's "temporal" protocol tracked "random" almost
exactly — that similarity is a property of the split, not evidence of deployment robustness.
Here the phishing class is split STRICTLY by `collected_at` (train = the older 70% of dated
phishing, test = the strictly newer 30%; undated phishing is dropped), which is the class that
actually drifts — campaigns, brands and hosting churn. The benign class is barely dated in this
corpus (~2k of 33k rows), so it is split randomly per seed at the same 70/30 rate; that choice
is stated in the paper, and is the standard protocol in the phishing-drift literature (benign
web infrastructure is comparatively stationary).

LEAKAGE GUARD. A registrable domain seen in the train window is removed from the test window
(kept in train): re-detections of the same campaign domain would otherwise let the model score
"memorised domain", not "generalises forward in time".

The comparison that matters is strict-temporal vs random ON THE SAME ROW SET — same phishing
subset, same benign pool — so protocol is the only variable. Output rows carry protocol
"random_same_rows" and "temporal_strict"; families and metrics mirror run_p2_benchmark.

RUN:
  python scripts/train/run_p2_temporal_strict.py                 # 7 families x 2 protocols x seeds
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label  # noqa: E402
from run_p2_benchmark import (DETERMINISTIC, FAMILIES, pr_curve_row, run_one,  # noqa: E402
                              write_curves)
from psl import registered_domain  # noqa: E402  (same PSL logic the collector folds by)

URL_CSV = "data/processed/dataset_url.csv"
ALIGNED = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2_temporal_strict.csv"
CURVES = "data/processed/p2_pr_curves_strict.csv"


def parse_date(s):
    """dd/mm/yyyy first, then ISO — the same order normalize_merge uses; one format alone
    silently drops half the corpus (the trap check_paper_claims documents)."""
    s = (s if isinstance(s, str) else "").strip()[:10]
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def load():
    meta = pd.read_csv(URL_CSV, usecols=["url_norm", "label", "collected_at"], dtype=str)
    meta["date"] = meta["collected_at"].map(parse_date)
    df = add_label(pd.read_csv(ALIGNED))
    df = df.merge(meta.rename(columns={"url_norm": "url"})[["url", "date"]],
                  on="url", how="left")
    df = df.dropna(subset=[c for c in COMPPHISH if c in df.columns])
    df["rdom"] = df["dom"].astype(str).str.lower().map(registered_domain)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.70,
                    help="rolling-origin robustness: train fraction for the phishing time cut "
                         "AND the benign mask rate (0.70 = the canonical split)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--curves", default=CURVES)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * args.cut)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    seen = set(ph_tr.rdom)
    leaked = ph_te.rdom.isin(seen)
    ph_te = ph_te[~leaked]
    print(f"phishing dated: {len(ph)}  train<= {ph_tr.date.max().date()}  "
          f"test>= {ph_te.date.min().date()}  ({int(leaked.sum())} test rows dropped as "
          f"same-registrable-domain re-detections)")
    print(f"benign pool: {len(be)} (random 70/30 per seed)")

    rows, curves = [], []
    for name in args.families:
        for proto in ("temporal_strict", "random_same_rows"):
            # even "deterministic" families need all seeds here: the benign 70/30 mask is drawn
            # per seed, so no family is deterministic under either protocol
            for s in range(args.seeds):
                rng = np.random.RandomState(s)
                bmask = rng.rand(len(be)) < args.cut
                if proto == "temporal_strict":
                    tr = pd.concat([ph_tr, be[bmask]])
                    te = pd.concat([ph_te, be[~bmask]])
                else:
                    pool = pd.concat([ph_tr, ph_te])
                    pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                    tr = pd.concat([pool[pmask], be[bmask]])
                    te = pd.concat([pool[~pmask], be[~bmask]])
                met, y_t, sc = run_one(name, tr[feats].to_numpy(float),
                                       tr["y"].to_numpy(int), te[feats].to_numpy(float),
                                       te["y"].to_numpy(int), s, return_scores=True)
                met["protocol"] = proto
                rows.append(met)
                curves.append({"family": name, "protocol": proto, "seed": s,
                               "precision": pr_curve_row(y_t, sc)})
                print(f"  {name:<13} {proto:<17} seed={s} F1={met['F1']:.3f} "
                      f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


if __name__ == "__main__":
    main()
