#!/usr/bin/env python3
r"""p2_xdata_bootstrap.py — Bootstrap confidence intervals for P2 cross-dataset matrices.

Resamples test rows across seeds to compute 95% CIs for CatBoost F1 and Random Forest ROC-AUC.

Usage:
    python scripts/p2_xdata_bootstrap.py [--boot 500] [--seeds 5]
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated
from p3_xdata_bootstrap import bootstrap, collect_scores, summarise

PROC = os.path.join(ROOT, "data", "processed")
SEC = os.path.join(ROOT, "papers", "P2_url_benchmark", "sections")
OUT = os.path.join(PROC, "p2", "p2_xdata_bootstrap.csv")

# (model, metric) pairs: the model that produced each matrix P2 prints.
ARMS = [("CatBoost", "F1"), ("RandomForest", "ROC-AUC")]


def emit(df: pd.DataFrame, boot: int) -> None:
    def row(model, metric, quantity):
        r = df[(df.model == model) & (df.metric == metric)
               & (df.quantity == quantity)].iloc[0]
        return float(r["point"]), float(r["ci_lo"]), float(r["ci_hi"])

    f1_gap = row("CatBoost", "F1", "gap")
    f1_off = row("CatBoost", "F1", "off-diagonal")

    sent = (

        f"On the same row bootstrap (${boot}$ resamples) CatBoost's F1 generalisation gap is "
        f"${f1_gap[0]:.3f}$ (95\\% CI ${f1_gap[1]:.3f}$--${f1_gap[2]:.3f}$) around an "
        f"off-diagonal mean of ${f1_off[0]:.3f}$ (${f1_off[1]:.3f}$--${f1_off[2]:.3f}$)."
    )
    write_generated(os.path.join(SEC, "gen_xdata_ci.tex"), sent + "\n",
                    f"(F1 gap {f1_gap[0]:.3f} [{f1_gap[1]:.3f},{f1_gap[2]:.3f}])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    frames = []
    for model, metric in ARMS:
        print(f"\n=== {model} ({metric} matrix) ===", flush=True)
        names, held = collect_scores(args.seeds, model=model)
        reps = bootstrap(names, held, args.boot)
        df, point = summarise(names, held, reps, args.boot)
        df.insert(0, "model", model)
        frames.append(df[df.metric == metric])
        p = point[metric]
        print(f"    {model}/{metric}: diagonal {p['diag']:.3f}, off-diagonal {p['off']:.3f}, "
              f"gap {p['gap']:.3f}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[+] {args.out}")
    emit(out, args.boot)


if __name__ == "__main__":
    main()
