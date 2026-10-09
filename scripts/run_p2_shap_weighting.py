#!/usr/bin/env python3
"""
run_p2_shap_weighting.py — TreeSHAP attribution of the PhishVN in-distribution Random Forest, with
and without class weighting, on the same seeds and the same explained rows.

tab_shap() in make_p2_bench_assets.py fits the forest through train_url_baseline.make_model, which
sets class_weight="balanced". The 2026-10-09 uniform-weighting arm (P2_UNWEIGHTED=1 in the other
P2 runners) refits that forest without it; this asks whether the attribution the paper reads off
the forest (tld_len first, top-3 share 55.9%) depends on the weighting. It writes a CSV only and
never touches the paper's table or figure.

RUN:
  python scripts/run_p2_shap_weighting.py
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
from run_cross_dataset import in_dataset_split, load_corpus
from train_url_baseline import COMPPHISH, make_model

CORPUS = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2/p2_shap_weighting.csv"


def attribution(df, seeds, n_sample, weighting):
    import shap
    params = None if weighting == "balanced" else {"class_weight": None}
    imps, signs = [], []
    for s in range(seeds):
        tr, te = in_dataset_split(df, s)
        m = make_model("RandomForest", s, params).fit(tr[COMPPHISH], tr["y"])
        # the same stratified rows tab_shap explains: the draw depends on the seed only
        sub = te.groupby("y", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n_sample // 2), random_state=s))
        sv = shap.TreeExplainer(m).shap_values(sub[COMPPHISH])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imps.append(np.abs(sv).mean(axis=0))
        signs.append([np.corrcoef(sub[c], sv[:, i])[0, 1] if sub[c].nunique() > 1 else np.nan
                      for i, c in enumerate(COMPPHISH)])
    imp = np.mean(imps, axis=0)
    return pd.DataFrame({"feature": COMPPHISH, "weighting": weighting,
                         "mean_abs_shap": imp, "sd_abs_shap": np.std(imps, axis=0),
                         "share": imp / imp.sum(),
                         "direction": np.nanmean(np.array(signs, dtype=float), axis=0)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    df = load_corpus(CORPUS)
    out = pd.concat([attribution(df, a.seeds, a.n_sample, w) for w in ("balanced", "none")])
    out["rank"] = out.groupby("weighting")["share"].rank(ascending=False).astype(int)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    for w, g in out.groupby("weighting"):
        g = g.sort_values("share", ascending=False)
        top = ", ".join(f"{r.feature} {100 * r.share:.1f}%" for r in g.head(5).itertuples())
        print(f"[{w}] top-3 share {100 * g.share.head(3).sum():.1f}%  |  {top}")
    print(f"[+] {a.out}")


if __name__ == "__main__":
    main()
