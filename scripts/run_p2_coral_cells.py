#!/usr/bin/env python3
"""
run_p2_coral_cells.py — every off-diagonal transfer cell, unadapted and CORAL-adapted, with the
metrics that show whether a cell degenerated (revision 1, 2026-10-31).

WHY. The paper argues that CORAL's apparent F1 gain is the adapted model drifting toward
predicting everything positive, and supports it with a count of degenerate cells
(gen_coral_degeneracy). The matrices it reads hold one metric each. This prints the cells
themselves: precision, recall, F1, the predicted-positive rate at the 0.5 threshold, ROC-AUC, and
the target's all-positive F1, for the random forest and logistic regression, base and adapted.

Same cells as run_cross_dataset.py --adapt coral: train on the whole source corpus, score the
whole target corpus, CORAL with lambda = 1 aligning the source to the unlabelled target, the
canonical seeds (logistic regression is deterministic and runs once). The corpus files are the
four HOW_TO_RUN names.

Writes data/processed/p2/revision/coral_cells.csv (one row per model x config x cell x seed).

RUN: python scripts/run_p2_coral_cells.py --seeds 5
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, DETERMINISTIC, make_model
from run_cross_dataset import CoralAligner, load_corpus
from audit_p2_harmonisation import ALIGNED

OUT = "data/processed/p2/revision/coral_cells.csv"
MODELS = ("RandomForest", "LogReg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    corp = {k: load_corpus(v) for k, v in ALIGNED.items()}
    feats = list(COMPPHISH)
    rows = []
    for model in MODELS:
        for src, tgt in itertools.permutations(corp, 2):
            tr, te = corp[src], corp[tgt]
            y = te.y.to_numpy(int)
            allpos = f1_score(y, np.ones_like(y))
            for cfg in ("base", "coral"):
                for s in range(1 if model in DETERMINISTIC else args.seeds):
                    # unweighted, as every benchmark family is fitted since 2026-10-09
                    m = make_model(model, s, {"class_weight": None})
                    if cfg == "coral":
                        al = CoralAligner(tr[feats].to_numpy(), te[feats].to_numpy())
                        m.fit(al.source(tr[feats]), tr.y)
                        sc = m.predict_proba(al.target(te[feats]))[:, 1]
                    else:
                        m.fit(tr[feats], tr.y)
                        sc = m.predict_proba(te[feats])[:, 1]
                    pred = (sc >= 0.5).astype(int)
                    rows.append({"model": model, "config": cfg, "source": src, "target": tgt,
                                 "seed": s, "precision": precision_score(y, pred, zero_division=0),
                                 "recall": recall_score(y, pred), "F1": f1_score(y, pred),
                                 "pred_pos_rate": float(pred.mean()),
                                 "ROC-AUC": roc_auc_score(y, sc), "allpos_F1": allpos})
            r = pd.DataFrame(rows)
            r = r[(r.model == model) & (r.source == src) & (r.target == tgt)]
            print(r.groupby("config")[["precision", "recall", "F1", "pred_pos_rate", "ROC-AUC"]]
                  .mean().round(3).assign(cell=f"{model} {src}->{tgt}").to_string(header=False),
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[+] {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
