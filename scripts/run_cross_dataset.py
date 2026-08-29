#!/usr/bin/env python3
"""
run_cross_dataset.py — The P1b flagship experiment: a train x test generalisation matrix.

Given several corpora already re-featurised into the CompPhish 21-feature schema (via
align_compphish.py), it trains a detector on each corpus and tests on every corpus. Cell (i, j)
= score of a model trained on corpus i and evaluated on corpus j. The DIAGONAL is in-distribution;
OFF-DIAGONAL cells measure transfer. A large (diagonal - off-diagonal) gap means models overfit
corpus-specific artefacts rather than learning transferable phishing signal.

Feature schema is the modelling subset of CompPhish (is_https excluded — a known collection
artefact). Random Forest is averaged over --seeds for stability.

RUN:
  python run_cross_dataset.py \
      --corpora PhishVN=data/processed/vn_compphish.csv \
                CompPhish=compphish_compphish.csv \
                PhiUSIIL=phiusiil_compphish.csv \
                Grambeddings=grambeddings_compphish.csv \
      --metric F1 --seeds 5 --out data/processed/p2/cross_dataset_F1.csv
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import (COMPPHISH, MODEL_NAMES, DETERMINISTIC,
                                add_label, make_model, tune_params, _metrics)


def load_corpus(path):
    df = pd.read_csv(path)
    df = add_label(df)
    for c in COMPPHISH:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["y"].nunique() < 2:
        raise SystemExit(f"{path}: needs both classes (phishing + benign).")
    return df


LEGACY_SPLIT = False  # set by --legacy-split, to reproduce the pre-2026-08-19 matrices


def in_dataset_split(df, seed):
    """Split a corpus for its own DIAGONAL cell, group-aware by registrable domain.

    WHY GROUP-AWARE, AND WHY THIS WAS A BUG. align_compphish reduces every corpus to
    registrable-domain granularity, and the 21 features are computed from the domain string, so
    two rows sharing a domain are an IDENTICAL feature vector with an IDENTICAL label. The old
    fallback here was a plain row-level stratified split, applied to every corpus except PhishVN
    -- the only one shipping its own group-aware `split` column. Measured (scripts/
    audit_xdata_leakage.py): 82.0% of ISCXURL2016's rows share a domain and 92.0% of a random
    test fold was already present in training; PhishStorm 48.1%/81.0%; PhiUSIIL 23.6%/66.5%.
    ISCXURL2016's in-distribution F1 falls 0.965 -> 0.758 once domains cannot span the split.

    The leak is DIAGONAL-ONLY -- off-diagonal cells train and test on different corpora and cannot
    leak this way -- so it inflated the diagonal, and therefore the generalisation gap, and only in
    the corpora the study did not own. P2's own temporal protocol already carries a
    "registrable-domain leakage guard"; this brings the cross-dataset diagonal to the same
    standard rather than inventing a new one.
    """
    if "split" in df and df["split"].astype(str).isin(["train", "test"]).any():
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        if len(te) and te["y"].nunique() > 1:
            return tr, te
    if LEGACY_SPLIT or "dom" not in df:
        return train_test_split(df, test_size=0.3, stratify=df["y"], random_state=seed)
    gi, gj = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(df, groups=df["dom"]))
    tr, te = df.iloc[gi], df.iloc[gj]
    if te["y"].nunique() < 2:  # degenerate draw: fall back rather than crash the cell
        return train_test_split(df, test_size=0.3, stratify=df["y"], random_state=seed)
    return tr, te


def _psd_power(C, p):
    """C^p for a symmetric PSD matrix via eigendecomposition (eigenvalues clipped at 0)."""
    w, V = np.linalg.eigh(C)
    return (V * np.clip(w, 0, None) ** p) @ V.T


class CoralAligner:
    """CORAL (Sun, Feng & Saenko 2016): align the source features' second-order statistics to the
    target's, using UNLABELED target features only — the standard 'is there a model-side fix'
    baseline for the transfer collapse. Each domain is first standardized by its own statistics
    (unsupervised), then the source is whitened with its covariance and re-colored with the
    target's (lambda*I regularization as in the paper). The model trains on aligned source
    features and is evaluated on target features standardized by target statistics."""

    def __init__(self, Xs, Xt, lam=1.0):
        Xs = np.asarray(Xs, float)
        Xt = np.asarray(Xt, float)
        self.mu_s, self.sd_s = Xs.mean(0), Xs.std(0) + 1e-9
        self.mu_t, self.sd_t = Xt.mean(0), Xt.std(0) + 1e-9
        Zs = (Xs - self.mu_s) / self.sd_s
        Zt = (Xt - self.mu_t) / self.sd_t
        d = Zs.shape[1]
        Cs = np.cov(Zs, rowvar=False) + lam * np.eye(d)
        Ct = np.cov(Zt, rowvar=False) + lam * np.eye(d)
        self.A = _psd_power(Cs, -0.5) @ _psd_power(Ct, 0.5)

    def source(self, X):
        return ((np.asarray(X, float) - self.mu_s) / self.sd_s) @ self.A

    def target(self, X):
        return (np.asarray(X, float) - self.mu_t) / self.sd_t


def cell_score(train_df, test_df, same, metric, seeds, model="RandomForest", tune=False,
               feats=COMPPHISH, adapt="none"):
    n_seed = 1 if model in DETERMINISTIC else seeds
    params = {}
    if tune:  # tune ONLY on the training corpus (never on the target test corpus)
        params = tune_params(model, train_df[feats], train_df["y"], 0)
    vals = []
    for s in range(n_seed):
        if same:
            tr, te = in_dataset_split(train_df, s)
        else:
            tr, te = train_df, test_df
        if model in MODEL_NAMES:
            m = make_model(model, s, params)
        else:  # P2's booster additions; default hyperparameters, no tune_params support
            from run_p2_benchmark import make_any_model
            m = make_any_model(model, s)
        if adapt == "coral" and not same:  # diagonal cells stay unadapted by construction
            al = CoralAligner(tr[feats].to_numpy(), te[feats].to_numpy())
            m = m.fit(al.source(tr[feats]), tr["y"])
            score = m.predict_proba(al.target(te[feats]))[:, 1]
        else:
            m = m.fit(tr[feats], tr["y"])
            score = m.predict_proba(te[feats])[:, 1]
        vals.append(_metrics(te["y"].to_numpy(), score)[metric])
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="name=path.csv entries, each already in the CompPhish schema.")
    ap.add_argument("--metric", default="F1", choices=["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--model", default="RandomForest",
                    choices=MODEL_NAMES + ["XGBoost", "LightGBM", "CatBoost"],
                    help="Model family for the matrix (run once per family for a gap-across-models "
                         "table). The booster families come from run_p2_benchmark and ignore --tune.")
    ap.add_argument("--tune", action="store_true", help="Tune on each train corpus (no test leakage).")
    ap.add_argument("--drop", default="",
                    help="Comma-separated features to EXCLUDE (SHAP-guided artefact-pruning "
                         "intervention: does removing corpus artefacts close the transfer gap?).")
    ap.add_argument("--adapt", default="none", choices=["none", "coral"],
                    help="Unsupervised domain adaptation for off-diagonal cells (coral = align "
                         "source second-order statistics to the unlabeled target).")
    ap.add_argument("--legacy-split", action="store_true",
                    help="Reproduce the pre-2026-08-19 leaky row-level diagonal split.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global LEGACY_SPLIT
    LEGACY_SPLIT = args.legacy_split

    drop = {f.strip() for f in args.drop.split(",") if f.strip()}
    unknown = drop - set(COMPPHISH)
    if unknown:
        raise SystemExit(f"--drop names not in the CompPhish schema: {sorted(unknown)}")
    feats = [c for c in COMPPHISH if c not in drop]
    if drop:
        print(f"[i] artefact-pruning: dropping {sorted(drop)} -> {len(feats)} features")

    corpora = {}
    for entry in args.corpora:
        if "=" not in entry:
            raise SystemExit(f"Bad --corpora entry (need name=path): {entry}")
        name, path = entry.split("=", 1)
        corpora[name] = load_corpus(path)
        print(f"[i] {name}: {len(corpora[name])} rows "
              f"(phishing={int(corpora[name]['y'].sum())})")
    names = list(corpora)

    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    print(f"\nComputing {len(names)}x{len(names)} matrix "
          f"(metric={args.metric}, seeds={args.seeds}) ...")
    for i in names:
        for j in names:
            mean, std = cell_score(corpora[i], corpora[j], i == j, args.metric, args.seeds,
                                   model=args.model, tune=args.tune, feats=feats,
                                   adapt=args.adapt)
            mat.loc[i, j] = round(mean, 3)
            print(f"  train={i:14s} test={j:14s} {args.metric}={mean:.3f}±{std:.3f}"
                  f"{'  (in-dist)' if i == j else ''}")

    diag = np.array([mat.loc[n, n] for n in names], dtype=float)
    off = mat.values[~np.eye(len(names), dtype=bool)].astype(float)
    print("\n=== " + args.metric + " matrix (rows=train, cols=test) ===")
    print(mat.to_string())
    print(f"\nmean diagonal (in-distribution) = {diag.mean():.3f}")
    print(f"mean off-diagonal (transfer)    = {np.nanmean(off):.3f}")
    print(f"generalisation gap              = {diag.mean() - np.nanmean(off):.3f}  "
          f"(smaller = features transfer better)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        mat.to_csv(args.out)
        print(f"[+] saved matrix -> {args.out}")


if __name__ == "__main__":
    main()
