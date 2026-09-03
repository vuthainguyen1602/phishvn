#!/usr/bin/env python3
r"""
p3_xdata_bootstrap.py — uncertainty for the cross-dataset headline quantities (Random Forest arm).

The transfer section reported a diagonal mean, an off-diagonal mean, their gap and a count of
below-chance cells as bare point estimates. Resamples TEST ROWS, not seeds: the seeds vary only the
learner's own randomness and their spread is not an estimate of how far the numbers would move on
another sample of URLs. Percentile intervals over B replicates, for these four corpora only — four
is far too few to resample corpora, and the paper says so where the numbers are printed.

The file name keeps "p3" because p2_xdata_bootstrap.py imports its machinery; the sentence it emits
now lands in papers/P2_url_benchmark/sections/gen_xdata_ci_rf.tex.

RUN:  python scripts/p3_xdata_bootstrap.py            # B=500, seeds=5
      python scripts/p3_xdata_bootstrap.py --boot 100 # quick check
The 2026-08-19 venue move and the resampling argument in full: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated
from run_cross_dataset import load_corpus, in_dataset_split
from train_url_baseline import COMPPHISH, make_model

PROC = os.path.join(ROOT, "data", "processed")
SEC = os.path.join(ROOT, "papers", "P2_url_benchmark", "sections")
# The same four files and the same order as the matrix in data/processed/p2/cross_dataset_F1.csv.
CORPORA = {
    "PhishVN": "vn_compphish.csv",
    "PhiUSIIL": "external/phiusiil_compphish.csv",
    "ISCXURL2016": "external/iscx_compphish.csv",
    "PhishStorm": "external/phishstorm_compphish.csv",
}
SEEDS = 5


def _fit(model: str, seed: int):
    """The matrix's own factory. RandomForest (P3's matrix) comes from the shared baseline
    module; P2's F1 matrix is CatBoost, which only run_p2_benchmark knows how to build."""
    if model == "RandomForest":
        return make_model(model, seed, {})
    from run_p2_benchmark import make_any_model
    return make_any_model(model, seed)


def collect_scores(seeds: int = SEEDS, model: str = "RandomForest"):
    """(y_true, score) for every (cell, seed), reproducing run_cross_dataset.cell_score.

    `model` defaults to the Random Forest P3's matrix is built from; P2 passes "CatBoost",
    whose F1 matrix is the one that paper prints."""
    data = {n: load_corpus(os.path.join(PROC, f)) for n, f in CORPORA.items()}
    names = list(data)
    held = {}
    for i in names:
        for j in names:
            same = i == j
            per_seed = []
            t0 = time.time()
            for s in range(seeds):
                if same:
                    tr, te = in_dataset_split(data[i], s)
                else:
                    tr, te = data[i], data[j]
                m = _fit(model, s).fit(tr[COMPPHISH], tr["y"])
                sc = m.predict_proba(te[COMPPHISH])[:, 1]
                per_seed.append((te["y"].to_numpy().astype(int), sc))
            held[(i, j)] = per_seed
            f1 = np.mean([f1_score(y, sc >= 0.5, zero_division=0) for y, sc in per_seed])
            roc = np.mean([roc_auc_score(y, sc) for y, sc in per_seed])
            print(f"[i] {i:12s} -> {j:12s} F1={f1:.3f} ROC-AUC={roc:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return names, held


def _cell_metrics(y, sc, idx):
    yb, sb = y[idx], sc[idx]
    if len(np.unique(yb)) < 2:
        return np.nan, np.nan
    return (f1_score(yb, sb >= 0.5, zero_division=0), roc_auc_score(yb, sb))


def bootstrap(names, held, boot: int, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    keys = list(held)
    out = {k: {"F1": [], "ROC-AUC": []} for k in keys}
    for b in range(boot):
        for k in keys:
            f1s, rocs = [], []
            for y, sc in held[k]:
                idx = rng.integers(0, len(y), len(y))
                f1, roc = _cell_metrics(y, sc, idx)
                f1s.append(f1); rocs.append(roc)
            out[k]["F1"].append(np.nanmean(f1s))
            out[k]["ROC-AUC"].append(np.nanmean(rocs))
        if (b + 1) % 50 == 0:
            print(f"    ... {b+1}/{boot} replicates", flush=True)
    return out


def summarise(names, held, reps, boot):
    """Point estimates from the un-resampled scores; intervals from the replicates."""
    diag = [(n, n) for n in names]
    off = [k for k in reps if k[0] != k[1]]
    rows = []
    point = {}
    for metric in ("F1", "ROC-AUC"):
        obs = {}
        for k, per_seed in held.items():
            fn = (lambda y, sc: f1_score(y, sc >= 0.5, zero_division=0)) if metric == "F1" \
                else (lambda y, sc: roc_auc_score(y, sc))
            obs[k] = float(np.mean([fn(y, sc) for y, sc in per_seed]))
        d = float(np.mean([obs[k] for k in diag]))
        o = float(np.mean([obs[k] for k in off]))
        point[metric] = {"diag": d, "off": o, "gap": d - o, "cells": obs}
        dv = np.array([np.mean([reps[k][metric][b] for k in diag]) for b in range(boot)])
        ov = np.array([np.mean([reps[k][metric][b] for k in off]) for b in range(boot)])
        for label, series, pt in (("diagonal", dv, d), ("off-diagonal", ov, o),
                                  ("gap", dv - ov, d - o)):
            lo, hi = np.percentile(series, [2.5, 97.5])
            rows.append({"metric": metric, "quantity": label, "point": round(pt, 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "boot": boot})
        # Per-cell interval for every off-diagonal cell, so "N cells are below chance" can be
        # stated as "N cells whose whole interval is below chance" rather than as a bare count.
        for k in off:
            s = np.array(reps[k][metric])
            lo, hi = np.percentile(s, [2.5, 97.5])
            rows.append({"metric": metric, "quantity": f"cell {k[0]}->{k[1]}",
                         "point": round(obs[k], 4), "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4), "boot": boot})
    return pd.DataFrame(rows), point


def emit(df, point, boot):
    def ci(metric, quantity):
        r = df[(df.metric == metric) & (df.quantity == quantity)].iloc[0]
        return r["point"], r["ci_lo"], r["ci_hi"]

    f1_gap, f1_lo, f1_hi = ci("F1", "gap")
    roc_gap, roc_lo, roc_hi = ci("ROC-AUC", "gap")
    f1_off, f1_off_lo, f1_off_hi = ci("F1", "off-diagonal")
    roc_off, roc_off_lo, roc_off_hi = ci("ROC-AUC", "off-diagonal")
    cells = df[(df.metric == "ROC-AUC") & (df.quantity.str.startswith("cell "))]
    below = cells[cells.ci_hi < 0.5]
    point_below = cells[cells.point < 0.5]
    sent = (
        f"Both headline quantities carry a bootstrap interval over test rows "
        f"(${boot}$ resamples; the five seeds vary only the learner's own randomness and are not "
        f"a sampling estimate, so they are not used as one). The F1 gap is "
        f"${f1_gap:.3f}$ (95\\% CI ${f1_lo:.3f}$--${f1_hi:.3f}$) and the ROC-AUC gap "
        f"${roc_gap:.3f}$ (${roc_lo:.3f}$--${roc_hi:.3f}$); the off-diagonal means are "
        f"${f1_off:.3f}$ (${f1_off_lo:.3f}$--${f1_off_hi:.3f}$) and ${roc_off:.3f}$ "
        f"(${roc_off_lo:.3f}$--${roc_off_hi:.3f}$). Of the ${len(point_below)}$ transfer cells "
        f"whose ROC-AUC point estimate is below chance, ${len(below)}$ have an entire interval "
        f"below $0.5$. The resampling is over rows, not corpora: with four corpora these "
        f"intervals describe how far these numbers would move on another draw of URLs from "
        f"\\emph{{these}} sources, and carry no claim about corpora in general."
    )
    write_generated(os.path.join(SEC, "gen_xdata_ci_rf.tex"), sent + "\n",
                    f"(F1 gap {f1_gap:.3f} [{f1_lo:.3f},{f1_hi:.3f}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default=os.path.join(PROC, "p3", "p3_xdata_bootstrap.csv"))
    args = ap.parse_args()

    names, held = collect_scores(args.seeds)
    reps = bootstrap(names, held, args.boot)
    df, point = summarise(names, held, reps, args.boot)
    df.to_csv(args.out, index=False)
    print(f"[+] {args.out}")
    for metric in ("F1", "ROC-AUC"):
        p = point[metric]
        print(f"    {metric}: diagonal {p['diag']:.3f}, off-diagonal {p['off']:.3f}, "
              f"gap {p['gap']:.3f}")
    emit(df, point, args.boot)


if __name__ == "__main__":
    main()
