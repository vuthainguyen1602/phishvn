#!/usr/bin/env python3
"""
run_p5_dedup.py — the P5 stream built three ways, and what each construction does to the gap.

  released : the released corpus, undated rows hash-spread (the stream of make_p5_assets.py)
  nowww    : the same rows with a leading "www." removed from every URL and features recomputed
  dedup    : one row per URL (P2's run_p2_dedup_audit.py --corpus-only), undated rows hash-spread

The constructed stream spreads undated rows across the dated span by hash. Most undated phishing
rows are the "www."-prefixed copy of a dated row (see run_p5_robustness.py --only format and
P2's run_p2_dedup_audit.py), so the static model is fitted on one URL format and scored on the
other. This runner rebuilds the stream from P2's one-row-per-URL corpus
(data/processed/p2/dedup_audit/, written by run_p2_dedup_audit.py --corpus-only) and reruns the
over-time policies, the window composition and the label-free budget frontier on it.

A fourth construction, the dated rows alone, is run in the sweep for reference, and the three
placed constructions are re-drawn under nine further hash salts.

Writes data/processed/p5/dedup/: p5_construction_sweep.csv (four constructions; F1, ROC-AUC,
balanced-accuracy and unseen-vector AUTC per policy), p5_construction_salts.csv, p5_psi_traces.csv,
p5_dedup_overtime.csv (per-window F1 per policy), p5_dedup_summary.csv (AUTC, end F1, retrains),
p5_dedup_windows.csv and p5_dedup_budget.csv.

RUN:  python scripts/run_p5_dedup.py [--skip-budget] [--no-salts]
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
from make_p5_assets import F1_DROP, PERIOD, PSI_TAU, WINDOWS, BUDGETS
from make_p5_budget_figure import DENSE
from retrain_drift import fit, load, run
from drift_detectors import run_budget

DD = os.path.join(ROOT, "data", "processed", "p2", "dedup_audit")
OUT = os.path.join(ROOT, "data", "processed", "p5", "dedup")
STREAM = os.path.join(OUT, "drift_compphish_dedup.csv")


def build_stream():
    u = pd.read_csv(os.path.join(DD, "dataset_url_dedup.csv"), low_memory=False)
    c = pd.read_csv(os.path.join(DD, "vn_compphish_dedup.csv"), low_memory=False)
    if len(u) != len(c):
        raise SystemExit("dedup corpus files are not row-aligned; rerun run_p2_dedup_audit.py --corpus-only")
    c = c.copy()
    for col in ("collected_at", "id", "tier"):
        c[col] = u[col].values
    os.makedirs(OUT, exist_ok=True)
    c.to_csv(STREAM, index=False)


NOWWW = os.path.join(OUT, "drift_compphish_nowww.csv")


def build_nowww():
    """Strip a leading www. from every URL and recompute the CompPhish features with the same
    aligner make_p5_assets.py's input comes from; rows, dates and ids are unchanged."""
    import subprocess, tempfile
    u = pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"), low_memory=False)
    strip = lambda s: s.astype(str).str.replace(r"^((?:[A-Za-z][A-Za-z0-9+.-]*://)?)www\.", r"\1",
                                                regex=True, case=False)
    for col in ("url", "url_norm"):
        if col in u:
            u[col] = strip(u[col])
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "url.csv"), os.path.join(td, "cp.csv")
        u.to_csv(src, index=False)
        aligner = next(p for p in (os.path.join(ROOT, "scripts", "core", "dataset", "align_compphish.py"),
                                   os.path.join(os.path.dirname(_HERE), "align_compphish.py"),
                                   os.path.join(_HERE, "align_compphish.py")) if os.path.exists(p))
        subprocess.run([sys.executable, aligner, "--in", src, "--out", dst], check=True,
                       stdout=subprocess.DEVNULL)
        c = pd.read_csv(dst, low_memory=False)
    for col in ("collected_at", "id", "tier"):
        c[col] = u[col].values
    os.makedirs(OUT, exist_ok=True)
    c.to_csv(NOWWW, index=False)


SALTS = [""] + [f"s{k}:" for k in range(1, 10)]  # "" is the released placement


def _one(name, path, salt):
    df, feats = load(path, "collected_at", spread_undated=(name != "dated"), salt=salt)
    if name == "dated":
        df = df[df["_t"].notna()].reset_index(drop=True)
    tr = {}
    strat, counts, psi_trace = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP,
                                   include_static_arch=False, trace=tr)
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
    pri = [float(df.y.iloc[i].mean()) for i in idx]
    dated = dated_mask(df)
    rows = []
    for k, v in strat.items():
        t = tr[k]
        rows.append({"construction": name, "salt": salt, "rows": len(df), "policy": k,
                     "autc": float(np.mean(v)), "end_f1": float(v[-1]), "retrains": counts[k],
                     "auc_autc": float(np.nanmean(t["auc"])), "bacc_autc": float(np.nanmean(t["bacc"])),
                     "novel_autc": float(np.nanmean(t["novel_f1"])),
                     "novel_pos_share": float(np.nanmean(t["novel_share"])),
                     "w1_prior": pri[0], "w1_spread": float(1 - dated[idx[0]].mean()),
                     "prior_min": min(pri), "prior_max": max(pri)})
    return rows, psi_trace


def sweep(salts=True):
    """The constructions side by side under the released placement (salt ""), then the three
    placed constructions re-drawn under nine further salts. Per policy: AUTC on F1, ROC-AUC and
    balanced accuracy, AUTC on rows whose feature vector window 1 never held, and window priors."""
    from make_p5_assets import DRIFT_CSV
    cons = (("released", DRIFT_CSV), ("nowww", NOWWW), ("dedup", STREAM), ("dated", DRIFT_CSV))
    rows, traces = [], []
    for name, path in cons:
        r, psi_trace = _one(name, path, "")
        rows += r
        traces.append(pd.DataFrame({"construction": name, "window": range(2, 2 + len(psi_trace)),
                                    "psi_vs_reference": psi_trace}))
        print(f"[sweep] {name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "p5_construction_sweep.csv"), index=False)
    pd.concat(traces).to_csv(os.path.join(OUT, "p5_psi_traces.csv"), index=False)
    print(out.round(3).to_string(index=False))
    if not salts:
        return
    srows = [r for r in rows if r["construction"] != "dated"]
    for salt in SALTS[1:]:
        for name, path in cons[:3]:
            srows += _one(name, path, salt)[0]
        print(f"[salt {salt}] done", flush=True)
        pd.DataFrame(srows).to_csv(os.path.join(OUT, "p5_construction_salts.csv"), index=False)


def dated_mask(df):
    s = df["collected_at"].astype(str).str.strip().str.slice(0, 10)
    t = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce").fillna(
        pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
    return t.notna().to_numpy()


def overtime():
    df, feats = load(STREAM, "collected_at", spread_undated=True)
    strat, counts, psi_trace = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP)
    win = pd.DataFrame(strat)
    win.insert(0, "window", range(2, 2 + len(win)))
    win["psi"] = psi_trace
    win.to_csv(os.path.join(OUT, "p5_dedup_overtime.csv"), index=False)
    summ = pd.DataFrame([{"policy": k, "autc": float(np.mean(v)), "end_f1": float(v[-1]),
                          "retrains": counts[k]} for k, v in strat.items()])
    summ.insert(0, "rows", len(df))
    summ.to_csv(os.path.join(OUT, "p5_dedup_summary.csv"), index=False)
    print(summ.round(3).to_string(index=False))
    dated = dated_mask(df)
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
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
                      "static_f1": f1_score(cur.y, pred, zero_division=0)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "p5_dedup_windows.csv"), index=False)


def budget(seeds):
    df, feats = load(STREAM, "collected_at", spread_undated=True)
    grid = sorted(set(DENSE) | set(BUDGETS))
    rows = []
    for s in range(seeds):
        for B, d in run_budget(df, feats, WINDOWS, grid, period=PERIOD, psi_tau=PSI_TAU,
                               f1_drop=F1_DROP, seed=s).items():
            for k, (autc, labels) in d.items():
                rows.append({"seed": s, "budget": B, "policy": k, "autc": autc, "labels": labels})
        print(f"[seed {s}] done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "p5_dedup_budget.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-budget", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-salts", action="store_true", help="skip the nine re-drawn placements")
    a = ap.parse_args()
    build_stream()
    build_nowww()
    overtime()
    sweep(salts=not a.no_salts)
    if not a.skip_budget:
        budget(a.seeds)


if __name__ == "__main__":
    main()
