#!/usr/bin/env python3
"""
run_p2_ngram.py — a second string representation beside the character-CNN: character n-gram
TF-IDF read by logistic regression (revision 1, 2026-11-01).

WHY. The character-CNN beats the seven feature-based families, and its gain could come from the
input (the raw string instead of 21 hand-built features) or from the model's capacity (a
convolutional network instead of a shallow learner). The two cannot be separated with one string
model. This arm keeps the input and drops the capacity: the same raw string, encoded as character
1- to 5-grams, read by a linear model. If it lands near the CNN, the input carries the gain; if it
lands near the feature-based families, the capacity does.

DESIGN. The CNN arm's rows and protocols exactly (run_p2_charcnn.py: phishing-temporal and the
unguarded random control on the dated rows, the canonical per-seed benign mask), five seeds. The
vocabulary and IDF weights are fitted on the training window of each seed only, as the CNN's
vocabulary is. Logistic regression with default regularisation (C = 1), up to 2,000 iterations.
Nothing is tuned.

Writes data/processed/p2/revision/ngram.csv in the canonical schema (family "CharNgramLR").

RUN: python scripts/run_p2_ngram.py --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import _metrics
from run_p2_temporal_strict import benign_mask, load, split_phishing

OUT = "data/processed/p2/revision/ngram.csv"


def fit_predict(tr_urls, ytr, te_urls, seed):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=True, sublinear_tf=True,
                          min_df=2)
    Xtr = vec.fit_transform(tr_urls)
    m = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, ytr)
    return m.predict_proba(vec.transform(te_urls))[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    df = load()
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    rows = []
    for proto in ("temporal_strict", "random_same_rows"):
        for s in range(a.seeds):
            rng = np.random.RandomState(s)
            bmask = benign_mask(be, rng, 0.70)
            if proto == "temporal_strict":
                tr, te = pd.concat([ph_tr, be[bmask]]), pd.concat([ph_te, be[~bmask]])
            else:
                pool = pd.concat([ph_tr, ph_te])
                pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                tr = pd.concat([pool[pmask], be[bmask]])
                te = pd.concat([pool[~pmask], be[~bmask]])
            t0 = time.time()
            score = fit_predict(tr.url.astype(str).tolist(), tr.y.to_numpy(int),
                                te.url.astype(str).tolist(), s)
            met = _metrics(te.y.to_numpy(int), score)
            met.update(family="CharNgramLR", seed=s, protocol=proto,
                       fit_seconds=round(time.time() - t0, 2))
            rows.append(met)
            print(f"  CharNgramLR {proto:<17} seed={s} F1={met['F1']:.3f} "
                  f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).groupby("protocol")[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]].mean().round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
