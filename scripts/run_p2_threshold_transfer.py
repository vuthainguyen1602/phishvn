#!/usr/bin/env python3
"""
run_p2_threshold_transfer.py — does a threshold chosen on one corpus survive a change of corpus?

WHY. A selected threshold has to survive temporal AND cross-dataset shift to be an operating point. run_p2_val_threshold.py answers the temporal half (tau* barely moves forward in time); the
transfer matrices report F1 at 0.5 and ROC-AUC, so nothing measured what a threshold fixed on the
source corpus does on a target corpus.

DESIGN. Four corpora in the shared 21-feature schema, three families (logistic regression, random
forest, CatBoost), five seeds, fitted unweighted like every family of the benchmark. For a source corpus i and seed
s: i is split by registrable domain (run_cross_dataset.in_dataset_split, the transfer matrices'
own split); a 10% domain-grouped validation slice is carved from the training side; the model is
fitted on the rest and tau_src is the F1-maximising threshold on that slice (the rule of
run_p2_val_threshold.best_threshold). For every target j (j = i is the in-corpus reference) the
target is split the same way and the source model scores j's test side at three thresholds:

  tau_src    the source's validation threshold, carried over unchanged (what a deployed model does);
  tau_tgt    the F1-maximising threshold on a 10% labelled slice of j's training side, scored by
             the same source model (what re-selecting the threshold with a few target labels does);
  0.5        the default.

All three readings use the same model and the same target test rows; they differ in threshold only.
Reported per cell: both thresholds, and FPR, recall and F1 at each.

RUN:
  P2_UNWEIGHTED=1 python scripts/run_p2_threshold_transfer.py --seeds 5
OUT: data/processed/p2/revision/threshold_transfer.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, MODEL_NAMES, make_model
from run_cross_dataset import in_dataset_split, load_corpus
from run_p2_val_threshold import best_threshold

CORPORA = {"PhishVN": "data/processed/vn_compphish.csv",
           "PhiUSIIL": "data/processed/external/phiusiil_compphish.csv",
           "ISCX": "data/processed/external/iscx_compphish.csv",
           "PhishStorm": "data/processed/external/phishstorm_compphish.csv"}
FAMILIES = ["LogReg", "RandomForest", "CatBoost"]
OUT = "data/processed/p2/revision/threshold_transfer.csv"
VAL = 0.10


def model_for(name: str, seed: int):
    if name in MODEL_NAMES:
        params = {"class_weight": None} if name in {"LogReg", "RandomForest", "HistGB"} else {}
        return make_model(name, seed, params)
    from run_p2_benchmark import make_any_model
    return make_any_model(name, seed)


def carve(df: pd.DataFrame, seed: int, frac: float = VAL):
    """(rest, slice): a domain-grouped slice of `frac`, retried until it holds both classes."""
    groups = df["dom"] if "dom" in df else np.arange(len(df))
    for k in range(20):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed + 1000 * k)
                    .split(df, groups=groups))
        if df.iloc[b]["y"].nunique() == 2 and df.iloc[a]["y"].nunique() == 2:
            return df.iloc[a], df.iloc[b]
    raise RuntimeError("no two-class validation slice")


def reading(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    p = (s >= tau).astype(int)
    neg, pos = y == 0, y == 1
    return {"FPR": float(p[neg].mean()) if neg.any() else np.nan,
            "recall": float(p[pos].mean()) if pos.any() else np.nan,
            "F1": float(f1_score(y, p, zero_division=0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if os.environ.get("P2_UNWEIGHTED") != "1":
        raise SystemExit("the benchmark fits every family unweighted: set P2_UNWEIGHTED=1")
    os.chdir(ROOT)
    corpora = {n: load_corpus(p) for n, p in CORPORA.items()}
    feats = list(COMPPHISH)
    splits = {(n, s): in_dataset_split(df, s) for n, df in corpora.items() for s in range(a.seeds)}
    rows = []
    for fam in a.families:
        for src in corpora:
            for s in range(a.seeds):
                tr, _ = splits[(src, s)]
                fit, val = carve(tr, s)
                m = model_for(fam, s).fit(fit[feats], fit["y"])
                tau_src = best_threshold(val["y"].to_numpy(), m.predict_proba(val[feats])[:, 1])
                for tgt in corpora:
                    ttr, tte = splits[(tgt, s)]
                    _, tval = carve(ttr, s)
                    tau_tgt = best_threshold(tval["y"].to_numpy(),
                                             m.predict_proba(tval[feats])[:, 1])
                    y = tte["y"].to_numpy()
                    sc = m.predict_proba(tte[feats])[:, 1]
                    row = {"family": fam, "source": src, "target": tgt, "seed": s,
                           "tau_src": tau_src, "tau_tgt": tau_tgt}
                    for tag, tau in (("src", tau_src), ("tgt", tau_tgt), ("half", 0.5)):
                        row.update({f"{k}_{tag}": v for k, v in reading(y, sc, tau).items()})
                    rows.append(row)
                    print(f"{fam:<12} {src:>10} -> {tgt:<10} s={s} tau_src={tau_src:.3f} "
                          f"tau_tgt={tau_tgt:.3f} FPR@src={row['FPR_src']:.3f} "
                          f"recall@src={row['recall_src']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[+] {len(out)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
