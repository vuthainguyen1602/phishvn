#!/usr/bin/env python3
"""
retrain_drift.py — Concept-drift-aware continual retraining experiment (P5).

Streams a time-ordered dataset in windows and compares retraining strategies:
  static | periodic(k) | drift-triggered (PSI / performance drop)
plus two never-retrained ARCHITECTURE baselines (added 2026-08-17): static CatBoost and static
Stack[CatBoost+LogReg]. The cost of not retraining may differ by architecture, so these rows price
whether a different static model buys back part of what retraining buys, with zero new labels.
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

# rank is benign-only (Tranco) -> NaN on phishing, a hard label leak; is_llm is constant here;
# is_https is excluded from modelling as a collection artifact (kept out here for consistency).
DROP = {"label", "y", "is_llm", "split", "page", "id", "rank", "tier", "censored", "is_https"}


def psi(expected, actual, bins=10):
    """Population Stability Index between two 1-D distributions."""
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1) + 1e-6
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def load(path, time_col, spread_undated=False, seed=42):
    df = pd.read_csv(path, low_memory=False)
    df["y"] = (df["label"].astype(str).str.lower().isin(["phishing", "1", "spam", "smishing"])).astype(int) \
        if df["label"].dtype == object else pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if time_col in df:
        # explicit formats (dd/mm/yyyy then ISO), matching normalize_merge
        s = df[time_col].astype(str).str.strip().str.slice(0, 10)
        df["_t"] = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce").fillna(
            pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
        if spread_undated and df["_t"].notna().any():
            # Undated rows (whitelist/top-list benign) carry no drift signal, so treat them as a
            # STATIONARY background: spread them uniformly across the dated span instead of dumping
            # them all at t0. Deterministic (hash of a stable key) so the stream is reproducible.
            lo, hi = df["_t"].min(), df["_t"].max()
            span = (hi - lo).days or 1
            miss = df["_t"].isna()
            key = df.loc[miss, "id"].astype(str) if "id" in df else df.loc[miss].index.astype(str)
            frac = key.map(lambda k: int(__import__("hashlib").sha1(k.encode()).hexdigest(), 16) % 10000 / 10000)
            df.loc[miss, "_t"] = lo + pd.to_timedelta((frac * span).round().astype(int), unit="D")
        df = df.sort_values("_t", na_position="first")
    feats = [c for c in df.columns if c not in DROP and c != time_col and c != "_t"
             and pd.api.types.is_numeric_dtype(df[c])]
    # RandomForest rejects NaN; fill any residual gaps with the column median (0 if all-NaN)
    for c in feats:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)
    return df.reset_index(drop=True), feats


def fit(df, feats, kind="rf", seed=0):
    """kind: rf (the stream policies' model), cb (CatBoost, 300 iterations, depth 6),
    cblr (Stack[CatBoost+LogReg], five-fold, logistic-regression meta-learner)."""
    if kind == "cb":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                               random_seed=seed, verbose=False)
    elif kind == "cblr":
        from catboost import CatBoostClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        m = StackingClassifier(
            estimators=[("cb", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                                  random_seed=seed, verbose=False)),
                        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba", cv=5, n_jobs=-1)
    else:
        m = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1,
                                   random_state=seed)
    m.fit(df[feats], df.y)
    return m


# never-retrained architecture baselines: fitted once on the first window, frozen thereafter
STATIC_ARCH = {"static_cb": "cb", "static_cblr": "cblr"}


def run(df, feats, windows, period, psi_tau, f1_drop, include_static_arch=True, seed=0):
    idx = np.array_split(np.arange(len(df)), windows)
    chunks = [df.iloc[i] for i in idx if len(i)]
    strategies = {"static": [], "periodic": [], "drift": []}
    if include_static_arch:
        strategies.update({k: [] for k in STATIC_ARCH})
    counts = {k: 0 for k in strategies}  # true retrains (initial fit excluded)

    def mean_psi(ref_df, cur_df):
        """Average PSI across ALL features — a single feature is too noisy a drift signal."""
        vals = [psi(ref_df[c].values, cur_df[c].values) for c in feats]
        return float(np.mean(vals)) if vals else 0.0

    base = chunks[0]
    models = {k: fit(base, feats, STATIC_ARCH.get(k, "rf"), seed=seed) for k in strategies}
    acc = base.copy()  # accumulated data for retraining
    ref = {k: base for k in strategies}  # reference window per strategy (for PSI)
    last_f1 = {k: 1.0 for k in strategies}
    psi_trace = []

    for w in range(1, len(chunks)):
        cur = chunks[w]
        acc = pd.concat([acc, cur])
        psi_trace.append(round(mean_psi(ref["drift"], cur), 3))
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
                if mean_psi(ref[strat], cur) > psi_tau or (last_f1[strat] - f1) > f1_drop:
                    retrain = True
            if retrain:
                models[strat] = fit(acc, feats, seed=seed); counts[strat] += 1
                ref[strat] = cur
            last_f1[strat] = f1
    return strategies, counts, psi_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--time-col", default="collected_at")
    ap.add_argument("--seed", type=int, default=0, help="Model seed shared by all policies and refits")
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--period", type=int, default=3, help="Retrain every k windows (periodic)")
    ap.add_argument("--psi", type=float, default=0.2, help="PSI drift threshold")
    ap.add_argument("--f1-drop", type=float, default=0.1, help="F1 drop that triggers retrain")
    ap.add_argument("--spread-undated", action="store_true",
                    help="Spread undated rows uniformly across the dated span (stationary "
                         "background) instead of dumping them at t0 — needed when only one class "
                         "carries timestamps (e.g. phishing dated, benign a current snapshot).")
    args = ap.parse_args()

    df, feats = load(args.inp, args.time_col, spread_undated=args.spread_undated)
    if df["y"].nunique() < 2 or not feats:
        raise SystemExit("Need both classes and numeric feature columns.")
    span = f"{df['_t'].min():%Y-%m} .. {df['_t'].max():%Y-%m}" if "_t" in df else "n/a"
    print(f"n={len(df)} features={len(feats)} windows={args.windows} span={span}")
    strat, counts, psi_trace = run(df, feats, args.windows, args.period, args.psi, args.f1_drop, seed=args.seed)
    print("F1 per window (from window 2):")
    for k, v in strat.items():
        mean = np.mean(v) if v else float("nan")
        print(f"  {k:9} retrains={counts[k]:2d}  meanF1={mean:.3f}  {v}")
    print(f"  PSI/window (drift signal): {psi_trace}")
    print("Lower retrains + higher meanF1 = better cost/accuracy trade-off.")


if __name__ == "__main__":
    main()
