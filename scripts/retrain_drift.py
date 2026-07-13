#!/usr/bin/env python3
"""
retrain_drift.py — Concept-drift-aware continual retraining experiment (P6).

Streams a time-ordered dataset in windows and compares retraining strategies:
  static | periodic(k) | drift-triggered (PSI / performance drop).
Reports F1 per window + retrain count (labelling-budget proxy).

Input: a CSV with numeric feature columns + label (+ a time column, default collected_at).
INSTALL: pip install pandas scikit-learn
RUN:
  python retrain_drift.py --in data/processed/dataset_url.csv --windows 10 --period 3 --psi 0.2
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

DROP = {"label", "y", "is_llm", "split", "page", "id"}


def psi(expected, actual, bins=10):
    """Population Stability Index between two 1-D distributions."""
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1) + 1e-6
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def load(path, time_col):
    df = pd.read_csv(path)
    df["y"] = (df["label"].astype(str).str.lower().isin(["phishing", "1", "spam", "smishing"])).astype(int) \
        if df["label"].dtype == object else pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if time_col in df:
        df["_t"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
        df = df.sort_values("_t", na_position="first")
    feats = [c for c in df.columns if c not in DROP and c != time_col and c != "_t"
             and pd.api.types.is_numeric_dtype(df[c])]
    return df.reset_index(drop=True), feats


def fit(df, feats):
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0)
    m.fit(df[feats], df.y)
    return m


def run(df, feats, windows, period, psi_tau, f1_drop):
    idx = np.array_split(np.arange(len(df)), windows)
    chunks = [df.iloc[i] for i in idx if len(i)]
    strategies = {"static": [], "periodic": [], "drift": []}
    counts = {"static": 1, "periodic": 1, "drift": 1}

    base = chunks[0]
    models = {k: fit(base, feats) for k in strategies}
    acc = base.copy()  # accumulated data for retraining
    ref_feat = base[feats[0]].values if feats else np.zeros(1)
    last_f1 = {k: 1.0 for k in strategies}

    for w in range(1, len(chunks)):
        cur = chunks[w]
        acc = pd.concat([acc, cur])
        for strat in strategies:
            m = models[strat]
            pred = m.predict(cur[feats])
            f1 = f1_score(cur.y, pred, zero_division=0)
            strategies[strat].append(round(f1, 3))
            # decide retraining for NEXT window
            retrain = False
            if strat == "periodic" and w % period == 0:
                retrain = True
            elif strat == "drift":
                p = psi(ref_feat, cur[feats[0]].values) if feats else 0
                if p > psi_tau or (last_f1[strat] - f1) > f1_drop:
                    retrain = True
            if retrain:
                models[strat] = fit(acc, feats); counts[strat] += 1
                if strat == "drift":
                    ref_feat = cur[feats[0]].values
            last_f1[strat] = f1
    return strategies, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--time-col", default="collected_at")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--period", type=int, default=3, help="Retrain every k windows (periodic)")
    ap.add_argument("--psi", type=float, default=0.2, help="PSI drift threshold")
    ap.add_argument("--f1-drop", type=float, default=0.1, help="F1 drop that triggers retrain")
    args = ap.parse_args()

    df, feats = load(args.inp, args.time_col)
    if df["y"].nunique() < 2 or not feats:
        raise SystemExit("Need both classes and numeric feature columns.")
    print(f"n={len(df)} features={len(feats)} windows={args.windows}")
    strat, counts = run(df, feats, args.windows, args.period, args.psi, args.f1_drop)
    print("F1 per window (from window 2):")
    for k, v in strat.items():
        mean = np.mean(v) if v else float("nan")
        print(f"  {k:9} retrains={counts[k]:2d}  meanF1={mean:.3f}  {v}")
    print("Lower retrains + higher meanF1 = better cost/accuracy trade-off.")


if __name__ == "__main__":
    main()
