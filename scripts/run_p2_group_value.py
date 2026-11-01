#!/usr/bin/env python3
"""
run_p2_group_value.py — what the undated phishing rows are worth to a model tested forward in time
(revision 1, 2026-10-31).

WHY. The composition step removes the 17,071 undated phishing rows (16,775 from ChongLuaDao and
296 from OpenPhish, all bronze). Section 6.3 asks whether they are mislabelled; the complementary
data-centric question is what they contribute to performance on the dated test population.
Influence functions or Data Shapley value single rows; at the level the paper argues, the unit is
the group, and a group's leave-in value is measurable directly: train on the phishing-temporal
training window with and without the group, test on the same dated test window.

DESIGN. Phishing-temporal split of run_p2_temporal_strict.py (seed's benign mask, 0.70 cut). Arms:
  dated_only      the canonical training window
  + undated_all   plus every undated phishing row
  + undated_cld   plus the ChongLuaDao undated rows only
  + undated_op    plus the OpenPhish undated rows only
The test window is the canonical guarded one in every arm. An undated row whose registrable
domain occurs in the test window is not added (most undated rows are re-listings of dated hosts,
and adding them would put test domains in training), so the arms differ in training rows alone and
the count of undated rows each arm actually adds is reported. A first version shrank the test
window instead, dropping any test row whose domain an undated row shares; that left 484 of the
test window's phishing rows, which is itself the measure of how much the undated rows duplicate
the dated ones. Undated rows carry no date, so some may postdate the test window; the arm measures what
adding them does, not whether a deployment could have had them. LogReg, RandomForest and
CatBoost, five seeds; F1 at 0.5, PR-AUC, ROC-AUC.

Writes data/processed/p2/revision/group_value.csv.

RUN: OPENBLAS_NUM_THREADS=1 python scripts/run_p2_group_value.py
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
from train_url_baseline import COMPPHISH
from run_p2_benchmark import run_one
from run_p2_temporal_strict import benign_mask, split_phishing
from run_p2_scores import load_with_meta

OUT = "data/processed/p2/revision/group_value.csv"
FAMILIES = ("LogReg", "RandomForest", "CatBoost")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    df = load_with_meta()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    und = df[(df.y == 1) & df.date.isna()]
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    groups = {"dated_only": und.iloc[:0], "+undated_all": und,
              "+undated_cld": und[und.source == "chongluadao"],
              "+undated_op": und[und.source == "openphish"]}
    shared = und.rdom.isin(set(ph_te.rdom))
    print(f"undated rows sharing a registrable domain with the test window: {int(shared.sum())} "
          f"of {len(und)}; test phishing rows sharing one with an undated row: "
          f"{int(ph_te.rdom.isin(set(und.rdom)).sum())} of {len(ph_te)}", flush=True)
    groups = {k: v[~v.rdom.isin(set(ph_te.rdom))] for k, v in groups.items()}
    print(f"dated train {len(ph_tr)}, undated {len(und)} "
          f"({', '.join(f'{k} {len(v)}' for k, v in groups.items())}), test phishing {len(ph_te)}",
          flush=True)
    rows = []
    for s in range(args.seeds):
        bm = benign_mask(be, np.random.RandomState(s), 0.70)
        te = pd.concat([ph_te, be[~bm]])
        for arm, extra in groups.items():
            tr = pd.concat([ph_tr, extra, be[bm]])
            for fam in FAMILIES:
                met = run_one(fam, tr[feats].to_numpy(float), tr.y.to_numpy(int),
                              te[feats].to_numpy(float), te.y.to_numpy(int), s)
                met.update(arm=arm, n_added=len(extra), n_train=len(tr), n_test=len(te))
                rows.append(met)
                print(f"  {fam:<13} {arm:<13} seed={s} F1={met['F1']:.4f} "
                      f"PR-AUC={met['PR-AUC']:.4f} ROC-AUC={met['ROC-AUC']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["family", "arm"])[["F1", "PR-AUC", "ROC-AUC"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
