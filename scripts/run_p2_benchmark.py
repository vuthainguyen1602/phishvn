#!/usr/bin/env python3
"""
run_p2_benchmark.py — The P2 benchmark: which classical ML family fits Vietnamese phishing URLs?

Seven model families (LogReg, RandomForest, HistGB, MLP from train_url_baseline, plus XGBoost,
LightGBM, CatBoost) on the CompPhish 21-feature schema, under BOTH evaluation protocols:

  * random   — stratified 70/30, repeated over --seeds (the optimistic, most-published protocol)
  * temporal — normalize_merge's group-aware time split baked into vn_compphish.csv's `split`
               column: train strictly earlier, test strictly later (the deployment protocol)

Reporting both is the point of the paper: a family that wins under random and drops under
temporal is memorising time-local artefacts, and single-protocol benchmarks cannot see that.
Non-deterministic families are averaged over --seeds under BOTH protocols (under temporal the
split is fixed but the model seed still varies).

Metrics per run: F1, PR-AUC, ROC-AUC, FPR at 0.90 recall (imbalanced data — accuracy is not reported).
Output: one row per (family, protocol, seed) appended to data/processed/p2/p2_benchmark.csv;
downstream tables aggregate mean ± std. Cross-dataset transfer of the winning family reuses
run_cross_dataset.py; HPO of the winner reuses hpo_gwo.py — neither is duplicated here.

RUN:
  python scripts/run_p2_benchmark.py                    # full: 7 families x 2 protocols x seeds
  python scripts/run_p2_benchmark.py --families XGBoost LightGBM --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, _metrics, add_label, make_model

CORPUS = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2/p2_benchmark.csv"
CURVES = "data/processed/p2/p2_pr_curves.csv"

FAMILIES = ["LogReg", "RandomForest", "HistGB", "MLP", "XGBoost", "LightGBM", "CatBoost"]
DETERMINISTIC = {"LogReg"}
# The three families the shared baseline module trains with class_weight="balanced"; the four
# others are unweighted. Since 2026-10-09 P2 fits every family the same way, unweighted: mixing the
# two produced the whole 5.3-point random-split margin the paper once led with (the spread was
# 0.053 mixed and 0.005 uniform). P2_UNWEIGHTED=0 restores the mixed configuration as a legacy arm,
# and refuse_canonical() stops that arm from writing over a canonical CSV.
WEIGHTED = {"LogReg", "RandomForest", "HistGB"}
UNWEIGHTED = os.environ.get("P2_UNWEIGHTED", "1") == "1"


def weighting_params(name, unweighted=None):
    u = UNWEIGHTED if unweighted is None else unweighted
    return {"class_weight": None} if (u and name in WEIGHTED) else None


def refuse_canonical(out, canonical, *extra):
    if not UNWEIGHTED:
        for path in (canonical,) + extra:
            if path and out and os.path.abspath(out) == os.path.abspath(path):
                raise SystemExit(f"P2_UNWEIGHTED=0 (the legacy mixed-weighting arm) must not write "
                                 f"the canonical {path}; pass --out")


def make_any_model(name: str, seed: int, params: dict | None = None):
    """The three boosters are P2 additions; the rest defer to train_url_baseline.make_model so
    P2's numbers for those families stay comparable with P1b's published baselines. `params`
    overrides the booster defaults (the GWO HPO path); None keeps the defaults the benchmark
    reports."""
    p = params or {}
    if name == "Stack[CB+LR]":
        # The crossover ensemble (see run_p2_stacking_baseline): exposed here so companion
        # studies (P6's group-threshold run) can score it through the same factory.
        from catboost import CatBoostClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        return StackingClassifier(
            estimators=[("CB", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                                  random_seed=seed, verbose=False)),
                        ("LR", LogisticRegression(max_iter=1000, class_weight="balanced"))],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba", cv=5, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                                "eval_metric": "logloss", "random_state": seed, "n_jobs": -1,
                                **p})
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.1,
                                 "random_state": seed, "n_jobs": -1, "verbosity": -1, **p})
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{"iterations": 300, "depth": 6, "learning_rate": 0.1,
                                     "random_seed": seed, "verbose": False, **p})
    return make_model(name, seed, params)


# The recall axis every PR curve in P2 is interpolated onto. Fixed and shared so curves from
# different scripts (this one and run_p2_temporal_strict) can be drawn on one pair of axes.
PR_GRID = np.linspace(0.02, 1.0, 200)


def pr_curve_row(y, score):
    """Precision at each point of PR_GRID, by vertical averaging at fixed recall.

    Persisting the CURVE rather than the per-sample scores keeps the artefact small (200 floats
    per run instead of ~16k) and keeps it honest: the figure is then drawn from the same fitted
    model as the table row beside it, not from a re-run that would resample the split."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, score)
    return np.interp(PR_GRID, rec[::-1], prec[::-1])


def write_curves(curves, path):
    """Seed-mean precision per (family, protocol), long-format so a figure can read one row set.

    Averaged over seeds for the same reason the tables report means: a single-seed curve beside a
    five-seed table invites reconciling two numbers that were never the same quantity."""
    if not curves or not path:
        return
    import collections
    acc = collections.defaultdict(list)
    for c in curves:
        acc[(c["family"], c["protocol"])].append(c["precision"])
    rows = []
    for (fam, proto), mats in acc.items():
        mean = np.mean(mats, axis=0)
        rows += [{"family": fam, "protocol": proto, "n_seeds": len(mats),
                  "recall": round(float(r), 4), "precision": round(float(pv), 6)}
                 for r, pv in zip(PR_GRID, mean)]
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        # a partial run (--families) refreshes only its own curves and keeps everyone else's
        old = pd.read_csv(path)
        keep = ~old.set_index(["family", "protocol"]).index.isin(list(acc))
        new = pd.concat([old[keep], new], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)
    print(f"[+] {len(acc)} curve(s) -> {path}")


def run_one(name, Xtr, ytr, Xte, yte, seed, return_scores=False, unweighted=None):
    t0 = time.time()
    params = weighting_params(name, unweighted)
    m = make_any_model(name, seed, params)
    m.fit(Xtr, ytr)
    fit_s = time.time() - t0
    score = m.predict_proba(Xte)[:, 1]
    met = _metrics(yte, score)
    met.update({"family": name, "seed": seed, "fit_seconds": round(fit_s, 2)})
    return (met, np.asarray(yte), score) if return_scores else met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=CORPUS)
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--curves", default=CURVES)
    ap.add_argument("--unweighted", action="store_true",
                    help="kept for old command lines; unweighted is the default (P2_UNWEIGHTED=0 "
                         "selects the legacy mixed-weighting arm)")
    args = ap.parse_args()
    args.unweighted = args.unweighted or UNWEIGHTED
    refuse_canonical(args.out, OUT)
    refuse_canonical(args.curves, CURVES)

    df = add_label(pd.read_csv(args.inp))
    feats = [c for c in COMPPHISH if c in df.columns]
    if len(feats) != len(COMPPHISH):
        raise SystemExit(f"corpus lacks CompPhish schema ({len(feats)}/{len(COMPPHISH)} features)")
    df = df.dropna(subset=feats)

    protocols = {}
    X, y = df[feats].to_numpy(float), df["y"].to_numpy(int)
    protocols["random"] = lambda s: train_test_split(X, y, test_size=0.30, stratify=y,
                                                     random_state=s)
    if "split" in df.columns:
        tr, te = df[df.split == "train"], df[df.split == "test"]
        Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
        protocols["temporal"] = lambda s: (Xtr, Xte, ytr, yte)
        print(f"temporal split: train={len(tr)} test={len(te)}")
    else:
        print("[!] no `split` column — temporal protocol skipped")
    print(f"corpus: {len(df)} rows, {int(y.sum())} phishing / {len(y) - int(y.sum())} benign")

    rows, curves = [], []
    for name in args.families:
        for proto, splitter in protocols.items():
            seeds = [0] if (name in DETERMINISTIC and proto == "temporal") else range(args.seeds)
            for s in seeds:
                Xtr, Xte, ytr, yte = splitter(s)
                met, y_t, sc = run_one(name, Xtr, ytr, Xte, yte, s, return_scores=True,
                                       unweighted=args.unweighted)
                met["protocol"] = proto
                rows.append(met)
                curves.append({"family": name, "protocol": proto, "seed": s,
                               "precision": pr_curve_row(y_t, sc)})
                print(f"  {name:<13} {proto:<9} seed={s} "
                      f"F1={met['F1']:.3f} PR-AUC={met['PR-AUC']:.3f} "
                      f"FPR@R0.90={met['FPR@R0.90']:.3f} ({met['fit_seconds']}s)")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    agg = (out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
              .agg(["mean", "std"]).round(4))
    print(agg.to_string())


if __name__ == "__main__":
    main()
