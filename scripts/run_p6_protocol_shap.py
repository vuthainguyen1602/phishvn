#!/usr/bin/env python3
"""
run_p6_protocol_shap.py — P6's opening experiment: WHAT the model relies on, per protocol.

P2 measured that the family ranking collapses between a random split and a phishing-temporal one;
this asks WHY in feature terms, by comparing global SHAP attributions of the same family trained
under both. A feature that carries the random-split model but loses rank in the temporal one is a
time-local artefact. The benign mask is drawn independently per (protocol, seed) run, and the
between-protocol rank correlation is calibrated against a within-protocol null.

Outputs:
  data/processed/p6/p6_protocol_shap.csv       per-feature mean |SHAP| + ranks (first family,
                                            seed-averaged) — the paper's contrast table
  data/processed/p6/p6_protocol_shap_runs.csv  long format: family, protocol, seed, feature, shap
  data/processed/p6/p6_protocol_shap_rho.csv   rho summary: between-protocol vs within-protocol

RUN:
  python scripts/run_p6_protocol_shap.py                    # CatBoost+XGBoost, 5 seeds
  python scripts/run_p6_protocol_shap.py --families CatBoost --seeds 1   # v1-equivalent
Why v1's shared mask was tautological, and what the null buys: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import itertools
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
from train_url_baseline import COMPPHISH
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load

OUT = "data/processed/p6/p6_protocol_shap.csv"
OUT_RUNS = "data/processed/p6/p6_protocol_shap_runs.csv"
OUT_RHO = "data/processed/p6/p6_protocol_shap_rho.csv"


def shap_importance(model, X, feats, sample, seed):
    import shap
    rng = np.random.RandomState(seed)
    sub = X[rng.choice(len(X), size=min(sample, len(X)), replace=False)]
    sv = shap.TreeExplainer(model).shap_values(sub)
    if isinstance(sv, list):  # binary API variants
        sv = sv[1]
    return pd.Series(np.abs(sv).mean(axis=0), index=feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["CatBoost", "XGBoost"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    pool = pd.concat([ph_tr, ph_te])

    def make_split(proto, rng):
        """Benign mask (and, for the control, the phishing mask) drawn from THIS run's rng —
        no sharing across protocols."""
        bmask = rng.rand(len(be)) < 0.70
        if proto == "temporal_strict":
            return pd.concat([ph_tr, be[bmask]]), pd.concat([ph_te, be[~bmask]])
        pmask = rng.rand(len(pool)) < (len(ph_tr) / len(pool))
        return (pd.concat([pool[pmask], be[bmask]]),
                pd.concat([pool[~pmask], be[~bmask]]))

    runs = []   # (family, proto, seed) -> Series
    for fi, fam in enumerate(args.families):
        for pi, proto in enumerate(("random_same_rows", "temporal_strict")):
            for s in range(args.seeds):
                rng = np.random.RandomState(100_000 * (fi + 1) + 10_000 * (pi + 1) + s)
                tr, te = make_split(proto, rng)
                m = make_any_model(fam, s)
                m.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))
                imp = shap_importance(m, te[feats].to_numpy(float), feats, args.sample, s)
                runs.append({"family": fam, "protocol": proto, "seed": s, "imp": imp})
                print(f"[i] {fam:<9} {proto:<17} seed={s} trained on {len(tr)}, "
                      f"SHAP over {min(args.sample, len(te))} test rows", flush=True)

    # long-format dump
    long = pd.concat(
        [r["imp"].rename("shap").rename_axis("feature").reset_index()
           .assign(family=r["family"], protocol=r["protocol"], seed=r["seed"])
         for r in runs], ignore_index=True)
    long.to_csv(OUT_RUNS, index=False)

    # per-feature contrast table for the FIRST family, seed-averaged
    fam0 = args.families[0]

    def mean_imp(fam, proto):
        sel = [r["imp"] for r in runs if r["family"] == fam and r["protocol"] == proto]
        return pd.concat(sel, axis=1).mean(axis=1)

    tab = pd.DataFrame({
        "shap_random": mean_imp(fam0, "random_same_rows"),
        "shap_temporal": mean_imp(fam0, "temporal_strict"),
    })
    tab["rank_random"] = tab.shap_random.rank(ascending=False).astype(int)
    tab["rank_temporal"] = tab.shap_temporal.rank(ascending=False).astype(int)
    tab["rank_delta"] = tab.rank_random - tab.rank_temporal
    tab = tab.sort_values("shap_temporal", ascending=False).round(5)
    tab.to_csv(args.out, index_label="feature")

    # rho summary: between-protocol (same-seed pairs) vs within-protocol (re-seed pairs)
    from scipy.stats import spearmanr

    def rho(a, b):
        return spearmanr(a, b).statistic

    rows = []
    for fam in args.families:
        get = lambda proto, s: next(r["imp"] for r in runs
                                    if r["family"] == fam and r["protocol"] == proto
                                    and r["seed"] == s)
        between = [rho(get("random_same_rows", s), get("temporal_strict", s))
                   for s in range(args.seeds)]
        within = {proto: [rho(get(proto, i), get(proto, j))
                          for i, j in itertools.combinations(range(args.seeds), 2)]
                  for proto in ("random_same_rows", "temporal_strict")}
        rows.append({
            "family": fam,
            "rho_between_mean": np.mean(between), "rho_between_sd": np.std(between, ddof=1),
            "rho_within_random_mean": np.mean(within["random_same_rows"]),
            "rho_within_random_sd": np.std(within["random_same_rows"], ddof=1),
            "rho_within_temporal_mean": np.mean(within["temporal_strict"]),
            "rho_within_temporal_sd": np.std(within["temporal_strict"], ddof=1),
            "n_seeds": args.seeds,
        })
        print(f"\n{fam}: rho between-protocol {np.mean(between):.3f}±{np.std(between, ddof=1):.3f}"
              f" | within-random {np.mean(within['random_same_rows']):.3f}"
              f" | within-temporal {np.mean(within['temporal_strict']):.3f}")
    pd.DataFrame(rows).round(4).to_csv(OUT_RHO, index=False)

    print(f"\n{tab.to_string()}")
    print(f"[+] -> {args.out}, {OUT_RUNS}, {OUT_RHO}")


if __name__ == "__main__":
    main()
