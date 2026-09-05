#!/usr/bin/env python3
"""
run_p2_hpo.py — P2 experiment (d): does tuning change the ranking, or only inflate numbers?

GWO vs equal-budget uniform random search, tuned on the STRICT-TEMPORAL train window only
(3-fold CV PR-AUC — the tuner never sees the future), then evaluated once on the phishing-temporal
test window. Families: the phishing-temporal top tier (CatBoost, HistGB) plus LogReg (tuned C) —
if tuning cannot lift the complex families clearly past tuned-LogReg under the honest protocol,
the paper's "protocol beats hyperparameters" claim holds.

Data machinery is run_p2_temporal_strict's, verbatim. Output: data/processed/p2/p2_hpo.csv.
Use multiple search seeds before making an optimizer claim: one stochastic search run is only a diagnostic.

RUN:  python scripts/run_p2_hpo.py
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
from hpo_gwo import _factory, gwo_budget, gwo_search, random_search
from run_p2_temporal_strict import load

OUT = "data/processed/p2/p2_hpo.csv"
FAMILIES = ["CatBoost", "HistGB", "LogReg"]
N_AGENTS, ITERS = 6, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--seeds", type=int, default=3, help="model seeds for the test-side eval")
    ap.add_argument("--search-seeds", type=int, default=1,
                    help="independent GWO/random-search initialisations per family")
    ap.add_argument("--search-seed-start", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    rng = np.random.RandomState(0)
    bmask = rng.rand(len(be)) < 0.70
    tr = pd.concat([ph_tr, be[bmask]])
    te = pd.concat([ph_te, be[~bmask]])
    Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
    print(f"phishing-temporal: train={len(tr)} test={len(te)}")

    def eval_test(name, params):
        vals = []
        for s in range(args.seeds):
            m = _factory(name, s, params).fit(Xtr, ytr)
            vals.append(_metrics(yte, m.predict_proba(Xte)[:, 1]))
        return {k: (float(np.mean([v[k] for v in vals])),
                    float(np.std([v[k] for v in vals]))) for k in vals[0]}

    budget = gwo_budget(N_AGENTS, ITERS)
    rows = []

    def checkpoint():
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    for name in args.families:
        res = eval_test(name, {})
        print(f"[default {name}] " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items()))
        rows.append({"family": name, "config": "default", "cv_prauc": np.nan, "params": "{}",
                     **{k: m for k, (m, s) in res.items()},
                     **{k + "_std": s for k, (m, s) in res.items()}})
        checkpoint()

        for search_seed in range(args.search_seed_start,
                                 args.search_seed_start + args.search_seeds):
            t0 = time.time()
            gp, gf = gwo_search(name, Xtr, ytr, seed=search_seed,
                                n_agents=N_AGENTS, iters=ITERS)
            rp, rf = random_search(name, Xtr, ytr, seed=search_seed, budget=budget)
            adopted = gf >= rf  # pre-specified, equal-budget selection rule
            best_params, src = (gp, "gwo") if adopted else (rp, "random-search")
            res = eval_test(name, best_params)
            print(f"[tuned  {name} seed={search_seed}] CV: gwo={gf:.4f} random={rf:.4f} -> "
                  f"adopted {src} ({time.time()-t0:.0f}s, budget={budget})")
            print("               " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items()))
            rows.append({"family": name, "config": f"tuned({src})", "search_seed": search_seed,
                         "cv_prauc": max(gf, rf), "cv_prauc_gwo": gf,
                         "cv_prauc_random": rf, "budget": budget,
                         "params": str(best_params),
                         **{k: m for k, (m, s) in res.items()},
                         **{k + "_std": s for k, (m, s) in res.items()}})
            checkpoint()

    print(f"[+] -> {args.out}")


if __name__ == "__main__":
    main()
