#!/usr/bin/env python3
"""
run_p6_attribution_drift.py — P6 experiment 3: is attribution drift a leading indicator of
performance drift?

Experiments 1-2 float one operational proposal without testing it: monitor the drifting URL-shape
features' attribution magnitudes as an early warning. One detector is trained ONCE on the earliest
dated phishing and applied forward over equal-count windows, measuring recall_w and |SHAP|_w per
window. Descriptive only — a handful of windows, a rank correlation, no powered lead/lag test, and
the generated prose says so.

OUTPUT:
  data/processed/p6/p6_attribution_drift.csv   one row per window: dates, n, recall, per-feature |SHAP|
RUN:  python scripts/run_p6_attribution_drift.py
The design and why the windows are phishing-only: kept in the development repository, not shipped in this mirror
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
from run_p2_temporal_strict import load
from run_p2_benchmark import make_any_model
from train_url_baseline import COMPPHISH

OUT = "data/processed/p6/p6_attribution_drift.csv"
# the features experiment 1 named as the movers (URL-shape counts) plus the dominant prior, so the
# figure/table track the ones the monitoring proposal is actually about
KEY = ["dot_cnt", "dash_cnt", "tld_len"]


def shap_profile(model, X, feats, sample, rng):
    import shap
    sub = X if len(X) <= sample else X[rng.choice(len(X), size=sample, replace=False)]
    sv = shap.TreeExplainer(model).shap_values(sub)
    if isinstance(sv, list):
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(axis=0), index=feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--train-frac", type=float, default=0.30,
                    help="earliest fraction of dated phishing used to train the deployed model")
    ap.add_argument("--windows", type=int, default=6, help="forward equal-count time windows")
    ap.add_argument("--sample", type=int, default=2000, help="max rows per SHAP profile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0]
    rng = np.random.RandomState(args.seed)

    # deployed model: earliest train-frac of dated phishing + the (stationary) benign class
    cut = int(len(ph) * args.train_frac)
    ph_tr, ph_fwd = ph.iloc[:cut], ph.iloc[cut:]
    Xtr = pd.concat([ph_tr[feats], be[feats]]).to_numpy(float)
    ytr = np.r_[np.ones(len(ph_tr), int), np.zeros(len(be), int)]
    model = make_any_model(args.family, args.seed)
    model.fit(Xtr, ytr)
    print(f"[i] deployed {args.family}: train on {len(ph_tr)} phishing "
          f"(<= {ph_tr.date.max().date()}) + {len(be)} benign; "
          f"forward on {len(ph_fwd)} phishing in {args.windows} windows")

    # equal-count forward windows
    ph_fwd = ph_fwd.reset_index(drop=True)
    bins = np.array_split(np.arange(len(ph_fwd)), args.windows)
    rows = []
    for w, idx in enumerate(bins):
        win = ph_fwd.iloc[idx]
        X = win[feats].to_numpy(float)
        recall = float((model.predict(X) == 1).mean())          # phishing-only -> recall = catch rate
        prof = shap_profile(model, X, feats, args.sample, rng)
        row = {"window": w, "n": len(win),
               "date_start": win.date.min().date().isoformat(),
               "date_end": win.date.max().date().isoformat(),
               "date_mid": win.date.iloc[len(win) // 2].date().isoformat(),
               "recall": round(recall, 4)}
        for f in feats:
            row[f"shap_{f}"] = round(float(prof[f]), 5)
        rows.append(row)
        print(f"  w{w} {row['date_start']}..{row['date_end']} n={len(win)} "
              f"recall={recall:.3f} " + " ".join(f"{k}={prof[k]:.3f}" for k in KEY))

    out = pd.DataFrame(rows)

    # descriptive co-variation: Spearman rho between recall and each key feature's |SHAP| over
    # windows. Reported as direction + magnitude, NOT as a significance test (too few windows).
    from scipy.stats import spearmanr
    corr = {}
    for f in KEY:
        r, _ = spearmanr(out["recall"], out[f"shap_{f}"])
        corr[f] = round(float(r), 3)
    out.attrs["recall_shap_spearman"] = corr
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    # stash the correlation as a trailing comment row-free sidecar the asset generator reads
    with open(OUT.replace(".csv", "_corr.csv"), "w", encoding="utf-8") as f:
        f.write("feature,recall_shap_spearman\n")
        for k, v in corr.items():
            f.write(f"{k},{v}\n")
    print(f"[+] {OUT}")
    print(f"[i] Spearman rho(recall, |SHAP|) over windows: {corr}")
    print(f"[i] recall drift: {out.recall.iloc[0]:.3f} -> {out.recall.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
