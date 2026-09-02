#!/usr/bin/env python3
"""
run_p6_case_studies.py — P6 experiment 4: the .vn blind spot in named cases.

Experiment 2 gives the aggregate (90.4% of .vn-registered phishing missed, traced to a benign-ward
tld_len prior); this grounds it in concrete rows — specific missed .vn domains with the model's
score and top signed SHAP contributions, plus a caught non-.vn contrast showing the same feature
pushing the right way. The examples are historical, taken-down indicators from public national
blocklists.

RUN:  python scripts/run_p6_case_studies.py
Selection rule, disclosure policy and outputs: kept in the development repository, not shipped in this mirror
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
from run_p2_temporal_strict import load
from run_p2_benchmark import make_any_model
from train_url_baseline import COMPPHISH

OUT = "data/processed/p6/p6_case_studies.csv"
N_MISSED = 4      # missed .vn phishing examples
N_CAUGHT = 2      # caught contrasts (one .vn if any is caught, one non-.vn)


def top_signed(sv_row, feats, k=3):
    """The k features with the largest |SHAP| for this row, as 'feature:+0.83' strings, signed
    (positive = pushed the score phishing-ward, negative = benign-ward)."""
    order = np.argsort(-np.abs(sv_row))[:k]
    return "; ".join(f"{feats[i]}:{sv_row[i]:+.2f}" for i in order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    rng = np.random.RandomState(args.seed)
    bmask = rng.rand(len(be)) < 0.70
    tr = pd.concat([ph_tr, be[bmask]])
    te = pd.concat([ph_te, be[~bmask]]).reset_index(drop=True)

    m = make_any_model(args.family, args.seed)
    m.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))

    import shap
    sv = shap.TreeExplainer(m).shap_values(te[feats].to_numpy(float))
    if isinstance(sv, list):
        sv = sv[1]
    te = te.reset_index(drop=True)
    te["pred"] = m.predict_proba(te[feats].to_numpy(float))[:, 1]
    te["tld_str"] = te["tld"].astype(str).str.lower()
    te["is_vn"] = te.tld_str.str.endswith("vn")
    i_tld = feats.index("tld_len")
    te["shap_tld_len"] = sv[:, i_tld]

    def pick(mask, sort_col, ascending, n):
        sub = te[mask].sort_values(sort_col, ascending=ascending)
        sub = sub.drop_duplicates("rdom")
        return sub.head(n)

    # missed .vn phishing: most benign-ward tld_len contribution among the false negatives
    missed = pick((te.y == 1) & te.is_vn & (te.pred < 0.5), "shap_tld_len", True, N_MISSED)
    # caught contrasts: a caught non-.vn phishing (tld_len should push phishing-ward), and a caught
    # .vn phishing if the model got any right (shows the prior is beatable when other signal is loud)
    caught_other = pick((te.y == 1) & ~te.is_vn & (te.pred >= 0.5), "pred", False, 1)
    caught_vn = pick((te.y == 1) & te.is_vn & (te.pred >= 0.5), "pred", False, 1)

    rows = []
    for kind, sub in [("missed .vn", missed), ("caught non-.vn", caught_other),
                      ("caught .vn", caught_vn)]:
        for pos, r in sub.iterrows():
            rows.append({
                "kind": kind,
                "domain": str(r.get("rdom") or r["url"]),
                "score": round(float(r["pred"]), 3),
                "shap_tld_len": round(float(r["shap_tld_len"]), 3),
                "top_signed_shap": top_signed(sv[pos], feats),
            })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    n_vn_miss = int(((te.y == 1) & te.is_vn & (te.pred < 0.5)).sum())
    n_vn = int(((te.y == 1) & te.is_vn).sum())
    print(f"[i] .vn phishing in test: {n_vn}, missed {n_vn_miss} (FNR {n_vn_miss / n_vn:.3f})")
    print(out.to_string(index=False))
    print(f"[+] {OUT}")


if __name__ == "__main__":
    main()
