#!/usr/bin/env python3
"""
run_p2_fcts.py — E5: is a stack's meta-learner better trained on forward-chained OOF?

Stack[CB+LR] beats CatBoost under phishing-temporal and loses under random, yet its meta-learner is
fitted on RANDOM-fold OOF probabilities — in-distribution predictions, not the regime the model
meets at test. Three variants differ ONLY in which folds produced that matrix: random_fold,
forward_chained (TimeSeriesSplit in time order) and shuffled_order (the same blocked structure in a
random order), the last being the control that makes the comparison mean anything. Exploratory by
pre-specification (PREREG_refresh_window.md). The meta-learner's coefficients are recorded per run.

RUN:
  python scripts/run_p2_fcts.py                       # 5 seeds, canonical cut
  python scripts/run_p2_fcts.py --seeds 10 --bases CatBoost+LogReg
Why the control, and how the training set is time-ordered: kept in the development repository, not shipped in this mirror
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
from train_url_baseline import COMPPHISH, _metrics
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load, split_phishing
from run_p2_stacking_baseline import ABBREV
from paired_eval import corrected_paired_t

OUT = "data/processed/p2/p2_fcts.csv"
VARIANTS = ("random_fold", "forward_chained", "shuffled_order")


def blocked_folds(order: np.ndarray, n_splits: int = 5):
    """TimeSeriesSplit over `order` (row indices in chain order), mapped back to row indices.
    The first block is train-only, so the OOF matrix covers the last n_splits/(n_splits+1)."""
    from sklearn.model_selection import TimeSeriesSplit
    for tr_pos, oo_pos in TimeSeriesSplit(n_splits=n_splits).split(order):
        yield order[tr_pos], order[oo_pos]


def oof_matrix(bases, X, y, folds):
    """P[i, j] = base j's P(phishing) for row i, from a model that never saw row i."""
    P = np.full((len(y), len(bases)), np.nan)
    for tr_idx, oo_idx in folds:
        for j, name in enumerate(bases):
            m = make_any_model(name, 0)
            m.fit(X[tr_idx], y[tr_idx])
            P[oo_idx, j] = m.predict_proba(X[oo_idx])[:, 1]
    return P, ~np.isnan(P).any(axis=1)


def chain_order(tr: pd.DataFrame, rng: np.random.RandomState) -> np.ndarray:
    """Row positions of `tr` in time order: dated rows by date, undated spread uniformly."""
    n = len(tr)
    pos = np.empty(n, dtype=float)
    dated = tr["date"].notna().to_numpy()
    if dated.any():
        r = tr.loc[dated, "date"].rank(method="first").to_numpy()
        pos[dated] = (r - 0.5) / dated.sum()
    pos[~dated] = rng.rand((~dated).sum())
    return np.argsort(pos, kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bases", default="CatBoost+LogReg",
                    help="base learners joined by + (the crossover stack by default)")
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    bases = args.bases.split("+")
    for b in bases:
        if b not in ABBREV:
            raise SystemExit(f"unknown family in --bases: {b}")
    tag = "Stack[" + "+".join(ABBREV[b] for b in bases) + "]"

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, None)
    print(f"phishing dated: {len(ph)}  train {len(ph_tr)}  test {len(ph_te)} "
          f"({n_leaked} guard-dropped)  benign pool: {len(be)}  features: {len(feats)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    rows = []
    for proto in ("temporal_strict", "random_same_rows"):
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
            tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)

            # Prediction-time bases: fitted once on all of train and shared by the variants, so
            # the only thing that varies is the matrix the meta-learner was fitted on.
            Pte = np.column_stack([make_any_model(b, s).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                                   for b in bases])
            order = chain_order(tr, np.random.RandomState(1000 + s))

            for variant in VARIANTS:
                t0 = time.time()
                if variant == "random_fold":
                    folds = list(StratifiedKFold(args.splits, shuffle=True,
                                                 random_state=s).split(Xtr, ytr))
                elif variant == "forward_chained":
                    folds = list(blocked_folds(order, args.splits))
                else:
                    folds = list(blocked_folds(rng.permutation(len(ytr)), args.splits))
                P, cov = oof_matrix(bases, Xtr, ytr, folds)
                meta = LogisticRegression(max_iter=1000).fit(P[cov], ytr[cov])
                met = _metrics(yte, meta.predict_proba(Pte)[:, 1])
                met.update({"family": tag, "variant": variant, "protocol": proto, "seed": s,
                            "meta_rows": int(cov.sum()),
                            "fit_seconds": round(time.time() - t0, 2)})
                met.update({f"coef_{ABBREV[b]}": round(float(c), 3)
                            for b, c in zip(bases, meta.coef_[0])})
                rows.append(met)
                print(f"  {variant:<16} {proto:<17} seed={s} PR-AUC={met['PR-AUC']:.4f} "
                      f"F1={met['F1']:.3f} meta_n={met['meta_rows']} "
                      f"coef={np.round(meta.coef_[0], 2)} ({met['fit_seconds']}s)")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["protocol", "variant"])[["PR-AUC", "F1", "FPR@R0.90"]]
             .mean().round(4).to_string())

    t = out[out.protocol == "temporal_strict"].pivot(index="seed", columns="variant",
                                                     values="PR-AUC")
    vs_shuf = corrected_paired_t(t.forward_chained - t.shuffled_order)
    vs_rand = corrected_paired_t(t.forward_chained - t.random_fold)
    print(f"\ntemporal: forward_chained - shuffled_order  {vs_shuf['mean']:+.4f} "
          f"({vs_shuf['wins']}/{vs_shuf['k']} seeds, p={vs_shuf['p']:.3f})")
    print(f"temporal: forward_chained - random_fold     {vs_rand['mean']:+.4f} "
          f"({vs_rand['wins']}/{vs_rand['k']} seeds, p={vs_rand['p']:.3f})")
    beats = vs_shuf["mean"] > 0 and vs_shuf["p"] <= 0.05 and vs_rand["mean"] > 0
    print("E5: " + ("FORWARD CHAINING HELPS — the order carries signal, follow it up"
                    if beats else
                    "NULL — forward-chained OOF is not better than the shuffled-order control; "
                    "the arrow of time inside the training window buys the meta-learner nothing"))


if __name__ == "__main__":
    main()
