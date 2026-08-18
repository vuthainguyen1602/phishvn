#!/usr/bin/env python3
"""
hpo_gwo.py — Grey Wolf Optimizer (GWO) for hyperparameter search, plus a random-search baseline on
the SAME evaluation budget so any claimed gain is defensible (not an unfair comparison).

Honest framing: metaheuristic HPO is only worth reporting when it is benchmarked fairly against a
strong baseline under an equal number of model evaluations. This module makes that comparison easy:
`gwo_search` and `random_search` both take a `budget` (number of fitness evaluations) and maximise
3-fold CV PR-AUC on the TRAINING data only (no test leakage). Reference: Mirjalili, Mirjalili &
Lewis, "Grey Wolf Optimizer", Advances in Engineering Software, 2014.

Used by train_url_baseline.py via `--tune --tune-method gwo`.
"""
from __future__ import annotations
import os
import sys

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import make_model  # noqa: E402

# Continuous / large search spaces (metaheuristics only matter when the space is big enough that
# exhaustive grid search is impractical). Each entry: (name, low, high, kind).
SPACE = {
    "LogReg": [("C", -3, 3, "logfloat")],
    "RandomForest": [("n_estimators", 100, 600, "int"), ("max_depth", 3, 50, "int"),
                     ("min_samples_leaf", 1, 10, "int"), ("max_features", 0.1, 1.0, "float")],
    "HistGB": [("learning_rate", -2, -0.5, "logfloat"), ("max_leaf_nodes", 15, 255, "int"),
               ("max_depth", 3, 30, "int"), ("l2_regularization", 0.0, 10.0, "float"),
               ("max_iter", 100, 500, "int")],
    "MLP": [("mlpclassifier__alpha", -5, -2, "logfloat"),
            ("mlpclassifier__learning_rate_init", -4, -2, "logfloat")],
    # P2 booster additions — constructed via run_p2_benchmark.make_any_model (see _factory)
    "CatBoost": [("iterations", 100, 600, "int"), ("depth", 4, 10, "int"),
                 ("learning_rate", -2, -0.5, "logfloat"), ("l2_leaf_reg", 1.0, 10.0, "float")],
    "XGBoost": [("n_estimators", 100, 600, "int"), ("max_depth", 3, 10, "int"),
                ("learning_rate", -2, -0.5, "logfloat"), ("subsample", 0.5, 1.0, "float"),
                ("colsample_bytree", 0.5, 1.0, "float")],
}


def _factory(model, seed, params):
    """make_model knows the four P1b families; the boosters live in run_p2_benchmark."""
    if model in ("CatBoost", "XGBoost", "LightGBM"):
        from run_p2_benchmark import make_any_model
        return make_any_model(model, seed, params)
    return make_model(model, seed, params)


def _decode(pos, space):
    """Map a position in [0,1]^d to a params dict."""
    params = {}
    for x, (name, lo, hi, kind) in zip(pos, space):
        v = lo + x * (hi - lo)
        if kind == "int":
            params[name] = int(round(v))
        elif kind == "logfloat":
            params[name] = float(10.0 ** v)
        else:
            params[name] = float(v)
    return params


def _fitness(model, params, X, y, seed):
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return float(cross_val_score(_factory(model, seed, params), X, y,
                                     scoring="average_precision", cv=cv, n_jobs=-1).mean())
    except Exception:
        return -1.0  # invalid config -> worst score


def gwo_budget(n_agents, iters):
    """True number of fitness evaluations gwo_search(n_agents, iters) performs."""
    return n_agents * (iters + 1)


def random_search(model, X, y, seed=0, budget=48):
    """Baseline: uniform random search over the same space and budget as GWO."""
    space = SPACE.get(model, [])
    if not space:
        return {}, float("nan")
    rng = np.random.default_rng(seed)
    best_p, best_f = {}, -1.0
    for _ in range(budget):
        p = _decode(rng.random(len(space)), space)
        f = _fitness(model, p, X, y, seed)
        if f > best_f:
            best_p, best_f = p, f
    return best_p, best_f


def gwo_search(model, X, y, seed=0, n_agents=6, iters=8):
    """Grey Wolf Optimizer maximising CV PR-AUC.

    Budget = n_agents * (iters + 1) evaluations: the initial population is scored once,
    then every wolf is re-scored each iteration. Use `gwo_budget` to grant a baseline
    the same number of fitness evaluations.
    """
    space = SPACE.get(model, [])
    if not space:
        return {}, float("nan")
    d = len(space)
    rng = np.random.default_rng(seed)
    wolves = rng.random((n_agents, d))
    fit = np.array([_fitness(model, _decode(w, space), X, y, seed) for w in wolves])
    # alpha/beta/delta = three best (we maximise, so take the largest)
    order = np.argsort(-fit)
    alpha, beta, delta = wolves[order[0]].copy(), wolves[order[1]].copy(), wolves[order[2]].copy()
    a_fit = fit[order[0]]
    for t in range(iters):
        a = 2 - 2 * t / max(1, iters - 1)  # linearly 2 -> 0
        for i in range(n_agents):
            for lead in (alpha, beta, delta):
                r1, r2 = rng.random(d), rng.random(d)
                A = 2 * a * r1 - a
                C = 2 * r2
                Dlead = np.abs(C * lead - wolves[i])
                wolves[i] = np.clip(0.5 * (wolves[i] + (lead - A * Dlead)), 0, 1)
        fit = np.array([_fitness(model, _decode(w, space), X, y, seed) for w in wolves])
        order = np.argsort(-fit)
        if fit[order[0]] > a_fit:
            alpha, a_fit = wolves[order[0]].copy(), fit[order[0]]
        beta, delta = wolves[order[1]].copy(), wolves[order[2]].copy()
    return _decode(alpha, space), float(a_fit)


if __name__ == "__main__":  # quick self-test
    import argparse, pandas as pd
    from train_url_baseline import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/vn_compphish.csv")
    ap.add_argument("--model", default="HistGB")
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    df, feats = load(a.inp)
    X, y = df[feats], df["y"]
    budget = gwo_budget(a.agents, a.iters)
    gp, gf = gwo_search(a.model, X, y, n_agents=a.agents, iters=a.iters)
    rp, rf = random_search(a.model, X, y, budget=budget)
    print(f"GWO    best CV PR-AUC={gf:.4f}  params={gp}")
    print(f"Random best CV PR-AUC={rf:.4f}  params={rp}  (same budget={budget})")
