#!/usr/bin/env python3
"""
run_p2_row_influence.py — what single training rows are worth to the logistic regression tested
forward in time: influence functions, row by row (revision 1, 2026-11-11).

WHY. run_p2_group_value.py values the undated phishing rows as a group, by training with and
without them. The data-valuation literature the article cites values single rows. For the
logistic regression of the benchmark that value has a closed form: the influence function of
Koh and Liang (2017) needs the Hessian of the training objective, which for 21 features and an
intercept is a 22 x 22 matrix, so it is computed exactly, without the approximations the method
needs for deep models.

THE FIT. An influence function is a derivative at the optimum of the training objective. The
benchmark fits its logistic regression with scikit-learn's default solver and tolerance, which
stops close to the optimum but not at it, and at that point refitting noise is larger than the
effect of removing a few hundred rows (the first version of this script found summed values and
measured changes uncorrelated, r = 0.2). The same objective (L2 penalty, C = 1, no class weights,
the 21 features as the benchmark feeds them) is therefore solved to convergence here with Newton's
method. The test F1 of the two fits agrees to the third decimal, and both are written out.

WHAT IS COMPUTED. For the "+undated_all" training set of run_p2_group_value.py (the dated
training window, the undated phishing rows that share no registrable domain with the test window,
and the seed's benign training rows) and the canonical dated test window:

  value_i = g_test' H^-1 grad_i

  grad_i  gradient of row i's log-loss at the fitted parameters
  H       Hessian of the training objective (sum of row losses + ||w||^2 / 2C, intercept free)
  g_test  gradient of the mean test log-loss

value_i is the first-order change in mean test log-loss if row i were removed and the model
refitted: positive for a row whose removal would raise the test loss (a helpful row), negative for
a row whose removal would lower it (a harmful row). The test labels enter g_test, so this is a
diagnostic of the data, not a selection rule a deployment could apply.

CHECKS, because a first-order estimate at a numerically fitted optimum can be wrong:
  linearity   for random subsets of training rows, the summed values against the change in test
              loss measured by removing the subset and refitting (Pearson r over the subsets);
  single      the three lowest and three highest valued rows, each removed alone;
  removal     refit without the 0.5%, 1% and 5% of training rows with the lowest value, each
              against the same number of rows drawn at random, and without every undated row. A
              first-order value is local: it is expected to hold for a few rows and to fail for
              a large group chosen by the value itself, and the arms show where it stops holding.

Writes data/processed/p2/revision/row_influence.csv (seed x group), row_influence_removal.csv
(seed x arm), row_influence_check.csv (seed x subset) and row_influence_single.csv (six rows per
seed, without their hostnames). No row-level file is written: the rows carry hostnames.

RUN: OPENBLAS_NUM_THREADS=1 python scripts/run_p2_row_influence.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH
from sklearn.linear_model import LogisticRegression
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import benign_mask, split_phishing
from run_p2_scores import load_with_meta

REV = "data/processed/p2/revision"
N_SUBSETS, SUBSET_SIZE, SHARES = 20, 500, (0.005, 0.01, 0.05)


def design(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))])


def fit(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """The benchmark's logistic-regression objective, solved to convergence."""
    ref = make_any_model("LogReg", seed)
    assert ref.penalty in ("l2", "deprecated") and ref.class_weight is None and ref.fit_intercept
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=ref.C, solver="newton-cholesky", tol=1e-12,
                                  max_iter=2000).fit(X, y)


def fit_benchmark(X: np.ndarray, y: np.ndarray, seed: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_any_model("LogReg", seed).fit(X, y)


def theta(model) -> np.ndarray:
    return np.concatenate([model.coef_.ravel(), model.intercept_])


def log_loss_mean(model, X: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def influence(model, Xtr, ytr, Xte, yte) -> tuple[np.ndarray, float]:
    """value_i for every training row, and the norm of the objective's gradient at the fit."""
    th, C = theta(model), float(model.C)
    A, B = design(Xtr), design(Xte)
    p_tr = 1.0 / (1.0 + np.exp(-(A @ th)))
    p_te = 1.0 / (1.0 + np.exp(-(B @ th)))
    reg = np.ones(len(th)) / C
    reg[-1] = 0.0                                     # the intercept is not penalised
    H = (A * (p_tr * (1 - p_tr))[:, None]).T @ A + np.diag(reg)
    g_test = ((p_te - yte)[:, None] * B).mean(axis=0)
    s = np.linalg.solve(H, g_test)                    # H is symmetric: s' grad_i = g' H^-1 grad_i
    values = (p_tr - ytr) * (A @ s)
    grad_obj = ((p_tr - ytr)[:, None] * A).sum(axis=0) + reg * th
    return values, float(np.linalg.norm(grad_obj) / len(ytr))


def scores(model, Xte, yte) -> dict:
    p = model.predict_proba(Xte)[:, 1]
    return {"logloss": log_loss_mean(model, Xte, yte), "F1": f1_score(yte, p >= 0.5),
            "PR-AUC": average_precision_score(yte, p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default=REV)
    args = ap.parse_args()
    df = load_with_meta()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    und = df[(df.y == 1) & df.date.isna()]
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    und = und[~und.rdom.isin(set(ph_te.rdom))]
    groups, removal, check, single = [], [], [], []
    for s in range(args.seeds):
        bm = benign_mask(be, np.random.RandomState(s), 0.70)
        parts = [("dated phishing", ph_tr),
                 ("undated, ChongLuaDao", und[und.source == "chongluadao"]),
                 ("undated, OpenPhish", und[und.source == "openphish"]),
                 ("benign", be[bm])]
        tr = pd.concat([p for _, p in parts])
        grp = np.concatenate([[name] * len(p) for name, p in parts])
        te = pd.concat([ph_te, be[~bm]])
        Xtr, ytr = tr[feats].to_numpy(float), tr.y.to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te.y.to_numpy(int)
        base = fit(Xtr, ytr, s)
        v, gnorm = influence(base, Xtr, ytr, Xte, yte)
        b = scores(base, Xte, yte)
        f1_benchmark = scores(fit_benchmark(Xtr, ytr, s), Xte, yte)["F1"]
        print(f"seed {s}: train {len(tr)}, test {len(te)}, test log-loss {b['logloss']:.4f}, "
              f"F1 {b['F1']:.4f} (benchmark solver {f1_benchmark:.4f}), |grad|/n {gnorm:.1e}",
              flush=True)

        def measured(drop) -> dict:
            keep = np.setdiff1d(np.arange(len(tr)), drop)
            return scores(fit(Xtr[keep], ytr[keep], s), Xte, yte)

        for name in dict.fromkeys(grp):
            m = grp == name
            groups.append({"seed": s, "group": name, "n": int(m.sum()),
                           "mean_value": v[m].mean(), "median_value": float(np.median(v[m])),
                           "sum_value": v[m].sum(), "share_harmful": float((v[m] < 0).mean()),
                           "share_abs": float(np.abs(v[m]).sum() / np.abs(v).sum()),
                           "test_logloss": b["logloss"], "F1": b["F1"],
                           "F1_benchmark_solver": f1_benchmark, "grad_norm": gnorm})
            print(f"  {name:<22} n={int(m.sum()):>6} sum={v[m].sum():+.4f} "
                  f"harmful={(v[m] < 0).mean():.3f}", flush=True)
        # ---- linearity: summed values against measured change for random subsets
        rng = np.random.RandomState(1000 + s)
        for k in range(N_SUBSETS):
            drop = rng.choice(len(tr), SUBSET_SIZE, replace=False)
            check.append({"seed": s, "subset": k, "predicted": v[drop].sum(),
                          "actual": measured(drop)["logloss"] - b["logloss"]})
        c = pd.DataFrame([r for r in check if r["seed"] == s])
        print(f"  linearity over {N_SUBSETS} subsets of {SUBSET_SIZE}: "
              f"r={np.corrcoef(c.predicted, c.actual)[0, 1]:.3f}", flush=True)
        # ---- single rows at the two ends of the ranking
        order = np.argsort(v)
        for i in list(order[:3]) + list(order[-3:]):
            single.append({"seed": s, "group": grp[i], "predicted": v[i],
                           "actual": measured([i])["logloss"] - b["logloss"]})
        # ---- removal: the rows the values rank lowest, against the same number at random
        arms = {}
        for share in SHARES:
            k = int(round(share * len(tr)))
            arms[f"lowest {share:.1%}"] = order[:k]
            arms[f"random {share:.1%}"] = rng.choice(len(tr), k, replace=False)
        arms["all undated"] = np.flatnonzero(np.char.startswith(grp.astype(str), "undated"))
        removal.append({"seed": s, "arm": "none", "n_removed": 0, "predicted": 0.0, **b})
        for arm, drop in arms.items():
            m = measured(drop)
            removal.append({"seed": s, "arm": arm, "n_removed": len(drop),
                            "predicted": v[drop].sum(), **m})
            print(f"  remove {arm:<12} n={len(drop):>5} log-loss {m['logloss'] - b['logloss']:+.4f} "
                  f"(predicted {v[drop].sum():+.4f}) F1 {m['F1'] - b['F1']:+.4f}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(groups).to_csv(os.path.join(args.outdir, "row_influence.csv"), index=False)
    pd.DataFrame(removal).to_csv(os.path.join(args.outdir, "row_influence_removal.csv"), index=False)
    pd.DataFrame(check).to_csv(os.path.join(args.outdir, "row_influence_check.csv"), index=False)
    pd.DataFrame(single).to_csv(os.path.join(args.outdir, "row_influence_single.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
