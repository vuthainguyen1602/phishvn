#!/usr/bin/env python3
"""
train_url_baseline.py — Baseline phishing detection from URL features (runs out of the box).

Auto-detects the feature schema:
  * CompPhish schema (Goenka 2026) if its 22 numeric columns are present, or
  * our basic schema (from normalize_merge) otherwise.

Supports CROSS-DATASET evaluation via --test-in (train on --in, test on a different file), e.g.
train on CompPhish and test on PhishVN (and vice versa) — the P1b generalization experiment.

Metrics: F1/PR-AUC/ROC-AUC + FPR at a fixed recall (phishing is class-imbalanced, so not accuracy).

INSTALL:  pip install pandas scikit-learn joblib
RUN:
  python train_url_baseline.py --in data/processed/dataset_url.csv --out models/url_rf.joblib
  # cross-dataset (train CompPhish, test PhishVN):
  python train_url_baseline.py --in compphish.csv --test-in data/processed/vn_compphish.csv
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, roc_curve)
import joblib

# Model families available to the baselines and the cross-dataset study. Keeping several families
# lets us test whether the cross-dataset generalisation gap holds regardless of model (evidence that
# a gap is a data artefact, not a model quirk).
MODEL_NAMES = ["LogReg", "RandomForest", "HistGB", "MLP"]
DETERMINISTIC = {"LogReg"}  # no seed effect; reported once

# Small, sane search spaces for optional hyperparameter tuning (--tune). Tuning is done ONLY on the
# training corpus's own validation data — never on a target test corpus — to avoid leakage.
PARAM_GRID = {
    "LogReg": {"C": [0.1, 1.0, 10.0]},
    "RandomForest": {"n_estimators": [200, 400], "max_depth": [None, 20, 40],
                     "min_samples_leaf": [1, 2, 5]},
    "HistGB": {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31, 63],
               "max_depth": [None, 10]},
    "MLP": {"mlpclassifier__hidden_layer_sizes": [(64,), (128, 64)],
            "mlpclassifier__alpha": [1e-4, 1e-3]},
}

# our basic schema (from normalize_merge)
BASIC = ["url_len", "num_dots", "num_subdomains", "has_ip", "has_at", "suspicious_tld",
         "domain_len", "num_hyphens", "num_digits"]
# CompPhish numeric schema (21 features).
# NOTE: is_https is intentionally EXCLUDED — in our collection benign was stored with https
# and phishing with http, making the scheme a spurious perfect predictor (F1=1.0 artifact).
# The column still exists in the CSV for CompPhish schema compatibility; we just don't train on it.
COMPPHISH = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
             "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
             "letter_ratio", "digit_ratio", "spec_ratio", "slash_cnt", "entropy",
             "path_len", "query_len"]


def add_basic_feats(df):
    u = df.get("url", pd.Series([""] * len(df))).fillna("").astype(str)
    df["domain_len"] = df.get("domain", pd.Series([""] * len(df))).fillna("").astype(str).str.len()
    df["num_hyphens"] = u.str.count("-")
    df["num_digits"] = u.str.count(r"\d")
    return df


def add_label(df):
    lab = df["label"]
    if lab.dtype != object:
        df["y"] = pd.to_numeric(lab, errors="coerce").fillna(0).astype(int)
    else:
        pos = {"phishing", "phish", "spam", "smishing", "malicious", "1"}
        df["y"] = lab.astype(str).str.lower().isin(pos).astype(int)
    return df


def load(path):
    df = pd.read_csv(path)
    df = add_label(df)
    feats = COMPPHISH if all(c in df.columns for c in COMPPHISH) else BASIC
    if feats is BASIC:
        df = add_basic_feats(df)
    for c in feats:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df, feats


def fpr_at_recall(y, score, target_recall=0.90):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def report(name, y, pred, score):
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    print(f"  [{name}] F1={f1_score(y,pred):.3f} P={precision_score(y,pred,zero_division=0):.3f} "
          f"R={recall_score(y,pred,zero_division=0):.3f} ROC-AUC={auc:.3f} "
          f"PR-AUC={average_precision_score(y,score):.3f} FPR@R0.90={fpr_at_recall(y,score):.3f}")


def make_model(name, seed, params=None):
    params = params or {}
    if name == "LogReg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", **params)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=seed, **params)
    if name == "HistGB":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
    if name == "MLP":  # scaled inputs matter for a neural net on tabular features
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                           random_state=seed, **params))
    raise ValueError(f"unknown model: {name}")


def tune_params(name, Xtr, ytr, seed):
    """Light RandomizedSearch on the training data (scoring=PR-AUC). Returns best params dict."""
    from sklearn.model_selection import RandomizedSearchCV
    grid = PARAM_GRID.get(name, {})
    if not grid:
        return {}
    n_comb = 1
    for v in grid.values():
        n_comb *= len(v)
    search = RandomizedSearchCV(make_model(name, seed), grid, n_iter=min(8, n_comb),
                                scoring="average_precision", cv=3, random_state=seed, n_jobs=-1)
    search.fit(Xtr, ytr)
    return search.best_params_


def _metrics(y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    y = np.asarray(y)
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    return {"F1": f1_score(y, pred, zero_division=0),
            "PR-AUC": average_precision_score(y, score),
            "ROC-AUC": auc,
            "FPR@R0.90": fpr_at_recall(y, score)}


def bootstrap_ci(y, score, B=1000, seed=0, thr=0.5):
    """Percentile bootstrap 95% CI for each metric by resampling the TEST set with replacement.
    Captures test-set sampling variability (complements the seed spread, which captures model
    variability). Together they give the ±/CI reporting reviewers expect for a Q1 venue."""
    y = np.asarray(y); score = np.asarray(score)
    n = len(y)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        for k, v in _metrics(y[idx], score[idx], thr).items():
            acc[k].append(v)
    out = {}
    for k, a in acc.items():
        a = [x for x in a if x == x]  # drop NaN
        out[k] = (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"),) * 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/dataset_url.csv")
    ap.add_argument("--test-in", default="", help="Separate test file (cross-dataset).")
    ap.add_argument("--out", default="models/url_rf.joblib")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Seeds for stochastic models; report mean±std (default 5).")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Bootstrap resamples for the best model's 95%% test CI (0 disables).")
    ap.add_argument("--models", default="LogReg,RandomForest",
                    help=f"Comma-separated model families. Available: {','.join(MODEL_NAMES)}.")
    ap.add_argument("--tune", action="store_true",
                    help="Hyperparameter search on the training data (PR-AUC, no test leakage).")
    ap.add_argument("--tune-method", default="random", choices=["random", "gwo"],
                    help="random = RandomizedSearch; gwo = Grey Wolf Optimizer (benchmark fairly).")
    args = ap.parse_args()

    df, feats = load(args.inp)
    print(f"schema={'CompPhish' if feats is COMPPHISH else 'basic'} ({len(feats)} features)")
    if df["y"].nunique() < 2:
        raise SystemExit("Need BOTH classes (phishing + benign) in the training file.")

    if args.test_in:  # cross-dataset: train on all --in, test on all --test-in
        te, _ = load(args.test_in)
        for c in feats:
            if c not in te:
                te[c] = 0
        tr = va = df
        print(f"CROSS-DATASET  train={len(tr)} ({os.path.basename(args.inp)})  "
              f"test={len(te)} ({os.path.basename(args.test_in)})")
    elif "split" in df and df["split"].notna().any():
        tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test"]
        if len(te) == 0:
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=0); va = te
    else:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=0); va = te

    Xtr, ytr = tr[feats], tr.y
    print(f"train={len(tr)} test={len(te)} | phishing ratio test={te.y.mean():.2f}")

    try:
        import mlflow
        mlflow.set_experiment("phishvn-url")
    except Exception:
        mlflow = None

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    yte = te.y.to_numpy()
    best, best_ap, best_name, best_score = None, -1, "", None
    for name in model_list:
        params = {}
        if args.tune:
            if args.tune_method == "gwo":
                from hpo_gwo import gwo_search
                params, cvf = gwo_search(name, Xtr, ytr, 0)
                print(f"  [{name}] GWO tuned (CV PR-AUC={cvf:.3f}): {params}")
            else:
                params = tune_params(name, Xtr, ytr, 0)
                print(f"  [{name}] tuned: {params}")
        # deterministic models (e.g. LogReg) are reported once; others vary with the seed.
        seeds = range(1) if name in DETERMINISTIC else range(args.seeds)
        runs = {k: [] for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")}
        last_model = last_score = None
        for s in seeds:
            m = make_model(name, s, params).fit(Xtr, ytr)
            score = m.predict_proba(te[feats])[:, 1]
            for k, v in _metrics(yte, score).items():
                runs[k].append(v)
            last_model, last_score = m, score
        tag = "deterministic" if name in DETERMINISTIC else f"{len(list(seeds))} seeds"
        cells = " ".join(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" for k, v in runs.items())
        print(f"  [{name}] ({tag}) {cells}")
        ap_ = float(np.mean(runs["PR-AUC"]))
        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "schema": "compphish" if feats is COMPPHISH else "basic",
                                   "n_train": len(tr), "cross_dataset": bool(args.test_in),
                                   "tuned": bool(args.tune), "seeds": len(list(seeds))})
                for k, v in runs.items():
                    mlflow.log_metric(k.replace("@", "_at_").replace("-", "_").replace(".", ""),
                                      float(np.mean(v)))
                    mlflow.log_metric("std_" + k.split("@")[0].replace("-", "_"), float(np.std(v)))
        if ap_ > best_ap:
            best, best_ap, best_name, best_score = last_model, ap_, name, last_score

    if args.bootstrap and best_score is not None:
        ci = bootstrap_ci(yte, best_score, B=args.bootstrap)
        print(f"  [best={best_name}] bootstrap 95% CI (B={args.bootstrap}): " +
              "  ".join(f"{k}=[{lo:.3f},{hi:.3f}]" for k, (lo, hi) in ci.items()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"model": best, "features": feats}, args.out)
    print(f"Saved best model ({best_name}) -> {args.out}")


if __name__ == "__main__":
    main()
