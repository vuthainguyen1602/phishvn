#!/usr/bin/env python3
"""
run_p5_robustness.py — the three checks the 2026-10-09 review asked of the constructed stream.

1. Window composition. Per window: rows, the share of hash-spread (undated) rows, the class prior,
   and the static model's precision, recall and F1 beside the periodic and drift-triggered F1. The
   twenty-point staleness gap is read against where the spread rows sit.
2. Dated-only stream. The same three policies on the dated rows alone, cut into the same number of
   equal-count windows, so no row's position comes from the hash. If the gap is a property of the
   spreading scheme, it should shrink here.
3. Budget-frontier seeds. The dense labelling-budget grid repeated over model seeds 0-4 (the seed
   sets the forest and the random label draws together), so the frontier's single-seed gaps get a
   spread.

Writes CSVs only (data/processed/p5/p5_window_composition.csv, p5_dated_only.csv,
p5_budget_seeds.csv); the paper's tables are not touched.

RUN:  python scripts/run_p5_robustness.py [--seeds 5] [--only window,dated,budget]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from make_p5_assets import DRIFT_CSV, F1_DROP, PERIOD, PSI_TAU, WINDOWS
from make_p5_budget_figure import DENSE
from make_p5_assets import BUDGETS
from retrain_drift import fit, load, run
from drift_detectors import run_budget

OUT = os.path.join(ROOT, "data", "processed", "p5")


def dated_mask(df):
    """Rows whose own event date parsed: the same two formats load() reads, nothing spread."""
    s = df["collected_at"].astype(str).str.strip().str.slice(0, 10)
    t = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce").fillna(
        pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
    return t.notna().to_numpy()


def window_composition():
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    dated = dated_mask(df)
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
    strat, _, _ = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP, include_static_arch=False)
    static = fit(df.iloc[idx[0]], feats)
    rows = []
    for w, i in enumerate(idx):
        cur = df.iloc[i]
        r = {"window": w + 1, "rows": len(cur), "spread_share": float(1 - dated[i].mean()),
             "prior": float(cur.y.mean())}
        if w > 0:
            pred = static.predict(cur[feats])
            r.update({"static_precision": precision_score(cur.y, pred, zero_division=0),
                      "static_recall": recall_score(cur.y, pred, zero_division=0),
                      "static_f1": f1_score(cur.y, pred, zero_division=0),
                      "periodic_f1": strat["periodic"][w - 1], "drift_f1": strat["drift"][w - 1]})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "p5_window_composition.csv"), index=False)
    print(out.round(3).to_string(index=False))


def dated_only():
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=False)
    df = df[df["_t"].notna()].reset_index(drop=True)
    strat, counts, _ = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP, include_static_arch=False)
    rows = [{"stream": "dated_only", "rows": len(df), "prior": float(df.y.mean()), "policy": k,
             "autc": float(np.mean(v)), "end_f1": float(v[-1]), "retrains": counts[k]}
            for k, v in strat.items()]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "p5_dated_only.csv"), index=False)
    print(out.round(3).to_string(index=False))
    win = pd.DataFrame({k: v for k, v in strat.items()})
    win.insert(0, "window", range(2, 2 + len(win)))
    win.to_csv(os.path.join(OUT, "p5_dated_only_windows.csv"), index=False)


def budget_seeds(seeds):
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    grid = sorted(set(DENSE) | set(BUDGETS))
    rows = []
    for s in range(seeds):
        res = run_budget(df, feats, WINDOWS, grid, period=PERIOD, psi_tau=PSI_TAU,
                         f1_drop=F1_DROP, seed=s)
        for B, d in res.items():
            for k, (autc, labels) in d.items():
                rows.append({"seed": s, "budget": B, "policy": k, "autc": autc, "labels": labels})
        print(f"[seed {s}] done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "p5_budget_seeds.csv"), index=False)
    out = pd.DataFrame(rows)
    w = out.pivot_table(index=["seed", "budget"], columns="policy", values="autc")
    gaps = pd.DataFrame({"drift_minus_periodic": w["drift"] - w["periodic"],
                         "active_minus_periodic": w["active"] - w["periodic"]})
    print(gaps.groupby("budget").agg(["mean", "std"]).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="window,dated,budget")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    parts = set(a.only.split(","))
    if "window" in parts:
        window_composition()
    if "dated" in parts:
        dated_only()
    if "budget" in parts:
        budget_seeds(a.seeds)


if __name__ == "__main__":
    main()
