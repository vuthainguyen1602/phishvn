#!/usr/bin/env python3
"""
run_p2_ngram_xdata.py — the character n-gram model in the four-corpus transfer matrix
(revision 1, 2026-11-03).

WHY. The character-CNN softens the transfer collapse without repairing it, which leaves open
whether that is a property of the corpus pairs or of the one string model tried. The n-gram model
(run_p2_ngram.py: character 1- to 5-gram TF-IDF read by logistic regression) is a second string
representation with almost no capacity, and on the dated rows it matches the CNN. Running it in
the same matrix says whether the CNN's transfer cells are what reading the string gives, or what
the network gives.

SAME DESIGN AS THE CNN MATRIX (run_p2_charcnn_xdata.py). Diagonal cells use the
registrable-domain-grouped 70/30 split of run_cross_dataset.in_dataset_split; off-diagonal cells
fit on the whole source corpus and score the whole target, one fit per (source, seed). The URL
scheme is stripped everywhere, as there, because only PhishVN's aligned file carries one. The
vocabulary and IDF weights are fitted on the source training rows only. Nothing is tuned. The fit
is deterministic for a given training set, so seeds vary only the diagonal split.

RUN: python scripts/run_p2_ngram_xdata.py --seeds 5
Writes data/processed/p2/revision/ngram_xdata.csv (one row per cell and seed, every metric) and
data/processed/p2/cross_dataset_{F1,ROC-AUC}_CharNgramLR.csv in the matrix layout.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import _metrics
from run_cross_dataset import load_corpus, in_dataset_split
from run_p2_charcnn_xdata import CORPORA, strip_scheme
from run_p2_ngram import fit_predict

OUT = "data/processed/p2/revision/ngram_xdata.csv"
MATRIX_PREFIX = "data/processed/p2/cross_dataset_"
FAMILY = "CharNgramLR"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--matrix-prefix", default=MATRIX_PREFIX,
                    help="prefix for the two matrix CSVs; empty string = do not write them")
    a = ap.parse_args()
    os.chdir(ROOT)

    corpora = {}
    for name, path in CORPORA:
        df = load_corpus(path)
        df["url_s"] = df["url"].map(strip_scheme)
        corpora[name] = df
        print(f"[i] {name}: {len(df)} rows (phishing={int(df.y.sum())})", flush=True)
    names = [n for n, _ in CORPORA]

    rows = []
    for src in names:
        S = corpora[src]
        # off-diagonal: the fit does not depend on the seed beyond the solver's, so it is made once
        # per source and its scores are repeated for every seed's row
        targets = [t for t in names if t != src]
        t0 = time.time()
        off_scores = fit_predict(S.url_s.tolist(), S.y.to_numpy(int),
                                 pd.concat([corpora[t].url_s for t in targets]).tolist(), 0)
        off_fit = round(time.time() - t0, 1)
        for s in range(a.seeds):
            tr, te = in_dataset_split(S, s)
            t0 = time.time()
            sc = fit_predict(tr.url_s.tolist(), tr.y.to_numpy(int), te.url_s.tolist(), s)
            m = _metrics(te.y.to_numpy(int), sc)
            m.update(family=FAMILY, train=src, test=src, seed=s, n_train=len(tr), n_test=len(te),
                     fit_seconds=round(time.time() - t0, 1))
            rows.append(m)
            print(f"  {src:<12}-> {src:<12} seed={s} F1={m['F1']:.3f} ROC={m['ROC-AUC']:.3f}",
                  flush=True)
            off = 0
            for t in targets:
                T = corpora[t]
                sc = off_scores[off:off + len(T)]
                off += len(T)
                m = _metrics(T.y.to_numpy(int), sc)
                m.update(family=FAMILY, train=src, test=t, seed=s, n_train=len(S),
                         n_test=len(T), fit_seconds=off_fit if (s == 0 and t == targets[0]) else 0.0)
                rows.append(m)
                if s == 0:
                    print(f"  {src:<12}-> {t:<12} F1={m['F1']:.3f} ROC={m['ROC-AUC']:.3f}",
                          flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[+] {len(out)} rows -> {a.out}")
    if a.matrix_prefix:
        for metric in ("F1", "ROC-AUC"):
            mat = (out.pivot_table(index="train", columns="test", values=metric, aggfunc="mean")
                      .reindex(index=names, columns=names).round(3))
            mat.index.name = None
            path = f"{a.matrix_prefix}{metric}_{FAMILY}.csv"
            mat.to_csv(path)
            v = mat.to_numpy(float)
            d, o = np.diag(v).mean(), v[~np.eye(len(v), dtype=bool)].mean()
            print(f"[+] {path}  diag={d:.3f} off={o:.3f} gap={d - o:.3f} "
                  f"below-chance={int((v[~np.eye(len(v), dtype=bool)] < 0.5).sum())}")
            print(mat.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
