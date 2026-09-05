#!/usr/bin/env python3
"""
run_p2_residual_lambda.py — E4: residual-backbone lambda sensitivity (exploratory; decides whether
PREREG_refresh_window.md's Test 2 keeps its BH slot).

If the LogReg backbone carries the forward-stable signal, CatBoost should model only what it leaves
unexplained: fitted with the LR logit as its `baseline`, its trees are a residual model and the
combined score is sigmoid(logit_LR + lambda*f_resid). The CONTROL is the lambda-mix of two
independently fitted models — same endpoints, no residual structure. An interior optimum the mix
curve does not share earns Test 2 its slot; a monotone curve forfeits it and Test 1 runs with m=1.

Split, seeds and benign masks are run_p2_stacking_baseline's canonical setting; each model is
fitted ONCE per seed/protocol and lambda is swept at scoring time.

RUN:
  python scripts/run_p2_residual_lambda.py            # 5 seeds, lambda in {0,0.1,...,1}
The two variants written out, and what decides the slot: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import os
import sys

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

OUT = "data/processed/p2/p2_residual_lambda.csv"
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_scores(Xtr, ytr, Xte, seed):
    """One LR fit, one residual CatBoost fit (baseline = LR logit), one independent CatBoost
    fit. Returns the test-set pieces the lambda sweep needs."""
    from catboost import Pool
    lr = make_any_model("LogReg", seed).fit(Xtr, ytr)
    z_tr, z_te = lr.decision_function(Xtr), lr.decision_function(Xte)
    cb_res = make_any_model("CatBoost", seed)
    cb_res.fit(Pool(Xtr, ytr, baseline=z_tr))
    f_te = cb_res.predict(Xte, prediction_type="RawFormulaVal")   # trees only, no baseline
    # sanity: trees + baseline must be what CatBoost itself would score with the offset
    chk = cb_res.predict(Pool(Xte, baseline=z_te), prediction_type="RawFormulaVal")
    assert np.allclose(chk, z_te + f_te, atol=1e-6), "baseline decomposition mismatch"
    p_cb = make_any_model("CatBoost", seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return z_te, f_te, sigmoid(z_te), p_cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut)
    print(f"phishing dated: {len(ph)}  train {len(ph_tr)}  test {len(ph_te)} "
          f"({n_leaked} guard-dropped)  benign pool: {len(be)}")

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
            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
            z, f, p_lr, p_cb = fit_scores(Xtr, ytr, Xte, s)
            for lam in LAMBDAS:
                for variant, score in (("residual", sigmoid(z + lam * f)),
                                       ("mix", (1 - lam) * p_lr + lam * p_cb)):
                    m = _metrics(yte, score)
                    rows.append({"seed": s, "protocol": proto, "lambda": lam,
                                 "variant": variant, "pr_auc": m["PR-AUC"], "f1": m["F1"],
                                 "fpr90": m["FPR@R0.90"]})
            r = [x["pr_auc"] for x in rows if x["seed"] == s and x["protocol"] == proto
                 and x["variant"] == "residual"]
            print(f"  {proto:<17} seed={s} residual PR-AUC at lambda=0/0.5/1: "
                  f"{r[0]:.4f}/{r[5]:.4f}/{r[10]:.4f}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} rows -> {args.out}")

    # ---- verdict (temporal_strict is the arm that matters) --------------------------------
    for proto in ("temporal_strict", "random_same_rows"):
        sub = out[out.protocol == proto]
        tab = sub.pivot_table(index="lambda", columns="variant", values="pr_auc",
                              aggfunc=["mean", "std"])
        print(f"\n{proto}: mean (sd) PR-AUC per lambda")
        print(pd.DataFrame({"residual": tab["mean"]["residual"].round(4),
                            "residual_sd": tab["std"]["residual"].round(4),
                            "mix": tab["mean"]["mix"].round(4),
                            "mix_sd": tab["std"]["mix"].round(4)}).to_string())
    t = out[out.protocol == "temporal_strict"]
    res = t[t.variant == "residual"].groupby("lambda")["pr_auc"]
    mix = t[t.variant == "mix"].groupby("lambda")["pr_auc"].mean()
    rm, rs = res.mean(), res.std()
    lam_star = float(rm.idxmax())
    interior = 0.0 < lam_star < 1.0
    beats_mix = interior and rm[lam_star] > mix[lam_star]
    margin = interior and rm[lam_star] - max(rm[0.0], rm[1.0]) > rs[lam_star]
    kept = interior and beats_mix and margin
    print(f"\nE4 lambda* = {lam_star}  residual mean PR-AUC {rm[lam_star]:.4f} "
          f"(sd {rs[lam_star]:.4f}); mix at lambda* {mix[lam_star]:.4f}; endpoints "
          f"residual lambda=0 {rm[0.0]:.4f}, lambda=1 {rm[1.0]:.4f}")
    print(f"  interior optimum: {interior}  beats mix at lambda*: {beats_mix}  "
          f"exceeds both endpoints by > 1 seed-sd: {margin}")
    print("E4: SLOT KEPT" if kept else "E4: SLOT FORFEITED")


if __name__ == "__main__":
    main()
