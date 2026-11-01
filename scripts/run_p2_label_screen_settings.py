#!/usr/bin/env python3
"""
run_p2_label_screen_settings.py — does the label-noise screen's reading depend on its settings?
(revision 1, 2026-11-01)

WHY. The confident-learning screen (audit_label_noise.py) flags bronze phishing rows at about the
rate of the human-verified gold tier, and the paper reads that as "the screen does not single out
bronze", not as a label-error estimate. A reviewer asked whether that reading holds under other
settings of the screen. This reruns it under each combination of:
  scorer      calibrated HistGB (the canonical) or calibrated logistic regression;
  threshold   per-class mean confidence (the canonical), per-class median, or the 0.5 argmax;
  folds       5 (the canonical) or 10.
For each setting: the flag rate of phishing rows by tier and the bronze-minus-gold excess, mean
over three seeds. Nothing else of the audit changes.

Writes data/processed/p2/revision/label_screen_settings.csv.

RUN: python scripts/run_p2_label_screen_settings.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label

CORPUS = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2/revision/label_screen_settings.csv"
SEEDS = 3


def scorer(kind: str, seed: int):
    if kind == "HistGB":
        base = HistGradientBoostingClassifier(random_state=seed)
    else:
        base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def oof(X, y, kind, seed, folds):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        p[te] = scorer(kind, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def flags(y, p_phish, rule):
    p_benign = 1 - p_phish
    if rule == "argmax":
        t_phish = t_benign = 0.5
    else:
        agg = np.mean if rule == "mean" else np.median
        t_phish, t_benign = agg(p_phish[y == 1]), agg(p_benign[y == 0])
    return np.where(y == 1, p_benign >= t_benign, p_phish >= t_phish)


def main() -> int:
    df = add_label(pd.read_csv(CORPUS, low_memory=False))
    for c in COMPPHISH:  # as audit_label_noise.py reads the corpus
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X, y = df[list(COMPPHISH)].to_numpy(float), df.y.to_numpy(int)
    tier = df["tier"].fillna("?").to_numpy()
    rows = []
    for kind, folds in itertools.product(("HistGB", "LogReg"), (5, 10)):
        for s in range(SEEDS):
            p = oof(X, y, kind, s, folds)
            for rule in ("mean", "median", "argmax"):
                fl = flags(y, p, rule)
                r = {"scorer": kind, "folds": folds, "rule": rule, "seed": s}
                for t in ("gold", "silver", "bronze"):
                    m = (y == 1) & (tier == t)
                    r[f"phish_{t}"] = float(fl[m].mean()) if m.any() else np.nan
                r["benign"] = float(fl[y == 0].mean())
                rows.append(r)
            print(f"  {kind:<7} folds={folds:<2} seed={s} done", flush=True)
    out = pd.DataFrame(rows)
    out["excess_bronze_gold"] = out.phish_bronze - out.phish_gold
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print((out.groupby(["scorer", "folds", "rule"])[["phish_gold", "phish_silver", "phish_bronze",
                                                      "excess_bronze_gold"]].mean() * 100).round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
