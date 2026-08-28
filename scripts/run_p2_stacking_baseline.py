#!/usr/bin/env python3
"""
run_p2_stacking_baseline.py — SOTA-style stacking + hybrid feature selection, evaluated under
P2's honest temporal protocol.

WHY THIS EXISTS. The recent ensemble literature (representative: Connolly & Atlam 2025, JNCA —
stacking RF + XGBoost under a meta-learner, with hybrid feature selection) reports ~99.5%
accuracy under random splits on public corpora. P2's claim is that the binding constraint on
this corpus is temporal shift, not in-distribution fit — so the interesting question is whether
that recipe closes the temporal gap or merely polishes the random-split number. This script
reproduces the recipe's two ingredients on the CompPhish-21 schema:

  * Stacking   — RandomForest + XGBoost base learners (the same configs the P2 benchmark uses
                 for the single-family rows, so the comparison isolates the ensemble), 5-fold
                 out-of-fold probabilities, LogisticRegression meta-learner.
  * Stacking+HFS — the same ensemble on a hybrid-selected feature subset: mutual-information
                 ranking (filter 1) with a pairwise-correlation redundancy prune (filter 2,
                 |Pearson r| > 0.90 drops the lower-MI feature), fitted on TRAIN ONLY, top-k kept.

Split, rows, metrics and output format all mirror run_p2_temporal_strict (same load(), same
per-seed benign 70/30 mask, same leakage guard), so rows from both scripts can sit in one table;
`family` is "Stacking" / "Stacking+HFS".

Base learners are configurable (--bases), so the same harness answers the follow-up question
"does a DIVERSE stack close the gap where the tree-only one didn't?" — e.g. CatBoost+LogReg
pairs the strongest booster with the family whose random-vs-temporal gap is near zero.

RUN:
  python scripts/train/run_p2_stacking_baseline.py --hfs      # the JNCA recipe: RF+XGB, +HFS ablation
  python scripts/train/run_p2_stacking_baseline.py \
      --bases XGBoost+LightGBM CatBoost+HistGB CatBoost+LogReg CatBoost+MLP \
              CatBoost+HistGB+LogReg+MLP \
      --out data/processed/p2/p2_stacking_combos.csv \
      --curves data/processed/p2/p2_pr_curves_stacking_combos.csv   # base-learner sweep
  python scripts/train/run_p2_stacking_baseline.py --seeds 20 --bases CatBoost+LogReg \
      --test-after 2025-02-18 --out data/processed/p2/p2_refresh_cblr_k20.csv --curves ""
      # PREREG_refresh_window.md Test 1: the window is run_p2_temporal_strict.split_phishing's
      # --test-after mode (train = all dated phishing on or before the date, test = strictly
      # later, same guard); pair with the CatBoost file from run_p2_temporal_strict --test-after
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
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, _metrics  # noqa: E402
from run_p2_benchmark import make_any_model, pr_curve_row, write_curves  # noqa: E402
from run_p2_temporal_strict import load, split_phishing  # noqa: E402

OUT = "data/processed/p2/p2_stacking_baseline.csv"
CURVES = "data/processed/p2/p2_pr_curves_stacking.csv"


def hybrid_select(Xtr: np.ndarray, ytr: np.ndarray, names: list[str], k: int,
                  seed: int) -> list[int]:
    """Filter-filter hybrid on TRAIN only: MI ranks relevance, correlation prunes redundancy.
    Returns column indices in original order."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(Xtr, ytr, random_state=seed)
    order = np.argsort(mi)[::-1]
    with np.errstate(invalid="ignore"):  # zero-variance columns yield NaN correlations
        corr = np.nan_to_num(np.corrcoef(Xtr, rowvar=False), nan=0.0)
    kept: list[int] = []
    for i in order:
        if any(abs(corr[i, j]) > 0.90 for j in kept):
            continue
        kept.append(int(i))
        if len(kept) == k:
            break
    return sorted(kept)


# short labels for the `family` column, so combo rows stay readable in tables
ABBREV = {"RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CB",
          "HistGB": "HGB", "LogReg": "LR", "MLP": "MLP"}


def make_stacker(bases: list[str], seed: int):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    return StackingClassifier(
        estimators=[(ABBREV[b], make_any_model(b, seed)) for b in bases],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=5, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=12, help="features kept by the hybrid selector")
    ap.add_argument("--bases", nargs="+", default=["RandomForest+XGBoost"],
                    help="base-learner combos, e.g. CatBoost+LogReg (family names joined by +)")
    ap.add_argument("--hfs", action="store_true",
                    help="also run each combo on the hybrid-selected feature subset")
    ap.add_argument("--cut", type=float, default=0.70,
                    help="rolling-origin robustness: phishing time-cut fraction and benign "
                         "mask rate (0.70 = canonical, mirrors run_p2_temporal_strict)")
    ap.add_argument("--test-after", default=None, metavar="YYYY-MM-DD",
                    help="refresh window (PREREG_refresh_window.md): test = dated phishing "
                         "strictly after this date, train = all dated phishing on or before it; "
                         "overrides the fractional --cut for the phishing class only")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--curves", default=CURVES)
    args = ap.parse_args()
    combos = [c.split("+") for c in args.bases]
    for combo in combos:
        for b in combo:
            if b not in ABBREV:
                raise SystemExit(f"unknown family in --bases: {b}")

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, args.test_after)
    if ph_te.empty:
        raise SystemExit(f"no dated phishing after {args.test_after}: the refresh window has "
                         f"not arrived (corpus max date {ph.date.max().date()})")
    print(f"phishing dated: {len(ph)}  train<= {ph_tr.date.max().date()} ({len(ph_tr)})  "
          f"test>= {ph_te.date.min().date()} ({len(ph_te)}; {n_leaked} guard-dropped)  "
          f"benign pool: {len(be)}  features: {len(feats)}")

    rows, curves = [], []
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
            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)

            sel = hybrid_select(Xtr, ytr, feats, args.k, s) if args.hfs else []
            for combo in combos:
                tag = "Stack[" + "+".join(ABBREV[b] for b in combo) + "]"
                variants = [(tag, list(range(len(feats))))]
                if args.hfs:
                    variants.append((tag + "+HFS", sel))
                for name, cols in variants:
                    t0 = time.time()
                    m = make_stacker(combo, s)
                    m.fit(Xtr[:, cols], ytr)
                    score = m.predict_proba(Xte[:, cols])[:, 1]
                    met = _metrics(yte, score)
                    met.update({"family": name, "seed": s, "protocol": proto,
                                "fit_seconds": round(time.time() - t0, 2),
                                "n_features": len(cols),
                                "features": "" if len(cols) == len(feats) else
                                            "|".join(feats[i] for i in cols)})
                    rows.append(met)
                    curves.append({"family": name, "protocol": proto, "seed": s,
                                   "precision": pr_curve_row(yte, score)})
                    print(f"  {name:<22} {proto:<17} seed={s} F1={met['F1']:.3f} "
                          f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f} "
                          f"({met['fit_seconds']}s, {len(cols)} feats)")
            if args.hfs:
                print(f"    HFS kept: {', '.join(feats[i] for i in sel)}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


if __name__ == "__main__":
    main()
