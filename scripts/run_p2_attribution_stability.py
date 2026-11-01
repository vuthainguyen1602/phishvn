#!/usr/bin/env python3
"""
run_p2_attribution_stability.py — is the TreeSHAP ranking a property of the forest, of the seed,
or of the corpus? (revision 1, 2026-10-31)

WHY. Section 6 reads TreeSHAP on the PhishVN random forest and finds the attribution concentrated
in three features (tld_len, subdom_cnt, dot_cnt). That is one model family on one corpus. This
repeats the attribution for the five tree families, on each of the four transfer corpora, over
three seeds, so the paper can say whether the same features lead everywhere or whether the
ranking moves with the model or the data. It is still association: a feature the model leans on,
not a feature shown to cause the transfer failure (run_p2_corpus_probe.py gives the
corpus-signature side of that question).

DESIGN. Each corpus's diagonal cell as run_cross_dataset.py builds it (in_dataset_split, grouped
by registrable domain); each family fitted as the benchmark fits it (unweighted defaults); mean |SHAP| on a
class-stratified 600-row sample of the test split. Writes one row per corpus x family x seed x
feature to data/processed/p2/revision/attribution_stability.csv, and prints per corpus the top
three features of each family and the mean Spearman correlation of the importance vectors across
seeds (same family) and across families (same seed).

RUN: OPENBLAS_NUM_THREADS=1 python scripts/run_p2_attribution_stability.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH
from run_p2_benchmark import make_any_model, weighting_params
from run_cross_dataset import in_dataset_split, load_corpus
from audit_p2_harmonisation import ALIGNED

OUT = "data/processed/p2/revision/attribution_stability.csv"
TREE_FAMILIES = ("RandomForest", "HistGB", "XGBoost", "LightGBM", "CatBoost")
SEEDS, N_SAMPLE = 3, 600  # 600 explained rows per cell: exact TreeSHAP on the 300-tree forest is minutes per thousand rows


def shap_importance(model, X: pd.DataFrame) -> np.ndarray:
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
    return np.abs(np.asarray(sv)).mean(axis=0)


def main() -> int:
    feats = list(COMPPHISH)
    rows = []
    for corpus, path in ALIGNED.items():
        df = load_corpus(path)
        for s in range(SEEDS):
            tr, te = in_dataset_split(df, s)
            sub = te.groupby("y", group_keys=False).apply(
                lambda g: g.sample(min(len(g), N_SAMPLE // 2), random_state=s))
            for fam in TREE_FAMILIES:
                m = make_any_model(fam, s, weighting_params(fam))
                m.fit(tr[feats].to_numpy(float), tr.y.to_numpy(int))
                imp = shap_importance(m, sub[feats].astype(float))
                rows += [{"corpus": corpus, "family": fam, "seed": s, "feature": f,
                          "mean_abs_shap": float(v)} for f, v in zip(feats, imp)]
                top = [feats[i] for i in np.argsort(-imp)[:3]]
                print(f"  {corpus:<12} {fam:<13} seed={s} top3={top}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    for corpus, g in out.groupby("corpus"):
        piv = g.pivot_table(index=["family", "seed"], columns="feature", values="mean_abs_shap")
        seed_rho = [spearmanr(piv.loc[(f, a)], piv.loc[(f, b)]).statistic
                    for f in TREE_FAMILIES for a, b in itertools.combinations(range(SEEDS), 2)]
        fam_rho = [spearmanr(piv.loc[(a, s)], piv.loc[(b, s)]).statistic
                   for s in range(SEEDS) for a, b in itertools.combinations(TREE_FAMILIES, 2)]
        share = g.groupby("family").apply(
            lambda h: h.groupby("feature").mean_abs_shap.mean().nlargest(3).sum()
            / h.groupby("feature").mean_abs_shap.mean().sum())
        tops = g.groupby(["family", "feature"]).mean_abs_shap.mean().groupby(level=0).nlargest(3)
        print(f"\n{corpus}: rho across seeds {np.nanmean(seed_rho):.3f}, across families "
              f"{np.nanmean(fam_rho):.3f}; top-3 share {share.round(3).to_dict()}")
        print(tops.reset_index(level=0, drop=True).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
