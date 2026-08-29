#!/usr/bin/env python3
"""
run_gwo_temporal.py — the two P1b results the methodology promises but Section 6 lacked:

1. TEMPORAL-SPLIT URL baseline: train on the temporally earlier window of the full PhishVN URL
   corpus (vn_compphish.csv, group-aware time-based split from normalize_merge.py) and test on
   the strictly later window. This is the protocol Section 4.3 describes; the split is temporal
   for dated sources and random for undated ones (documented in normalize_merge.temporal_split).

2. GWO vs EQUAL-BUDGET RANDOM SEARCH (Algorithm 2's protocol): tune RandomForest and HistGB with
   the Grey Wolf Optimizer on the TRAINING split only (3-fold CV PR-AUC), grant uniform random
   search the same number of fitness evaluations, and evaluate the GWO configuration on the
   temporal test set. Per the stated protocol, the tuned configuration is only *adopted* when it
   beats the equal-budget baseline in CV.

Outputs: data/processed/p3/temporal_gwo.csv (raw numbers) and
papers/P3_multimodal/sections/tab_temporal_gwo.tex (the table Section 6 inputs).
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, _metrics, add_label, make_model
from hpo_gwo import gwo_budget, gwo_search, random_search

INP = os.path.join(ROOT, "data/processed/vn_compphish.csv")
OUT_CSV = os.path.join(ROOT, "data/processed/p3/temporal_gwo.csv")
# The table left P3 on 2026-08-21 (the subsection now cites P2 for the protocol and keeps only
# the CSV's headline numbers), so the rendered table goes beside the CSV, not into the paper.
OUT_TEX = os.path.join(ROOT, "data/processed/p3/temporal_gwo_table.tex")

BASELINE_MODELS = ["LogReg", "RandomForest", "HistGB", "MLP"]
TUNED_MODELS = ["RandomForest", "HistGB"]  # rich-enough spaces for a metaheuristic to matter
SEEDS = 5
DETERMINISTIC = {"LogReg"}
N_AGENTS, ITERS = 6, 8  # budget = 6 * (8+1) = 54 fitness evaluations per method


def eval_on_test(name, params, Xtr, ytr, Xte, yte):
    """Fit (multi-seed where stochastic) and report mean/std of each metric on the test window."""
    seeds = range(1) if name in DETERMINISTIC else range(SEEDS)
    runs = {k: [] for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")}
    for s in seeds:
        m = make_model(name, s, params).fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        for k, v in _metrics(yte, score).items():
            runs[k].append(v)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in runs.items()}


def main():
    df = pd.read_csv(INP, low_memory=False)
    df = add_label(df)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    Xtr, ytr = tr[COMPPHISH], tr.y
    Xte, yte = te[COMPPHISH], te.y
    print(f"temporal split: train={len(tr)} (older window)  test={len(te)} (newer window)  "
          f"phishing ratio test={yte.mean():.2f}")

    rows = []

    # --- 1. default-hyperparameter baselines on the temporal split ---
    for name in BASELINE_MODELS:
        t0 = time.time()
        res = eval_on_test(name, {}, Xtr, ytr, Xte, yte)
        cells = " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items())
        print(f"  [default {name}] {cells}  ({time.time()-t0:.0f}s)")
        rows.append({"model": name, "config": "default", "cv_prauc": np.nan,
                     **{k: m for k, (m, s) in res.items()},
                     **{k + "_std": s for k, (m, s) in res.items()}})

    # --- 2. GWO vs equal-budget random search (tuned on the training window only) ---
    budget = gwo_budget(N_AGENTS, ITERS)
    for name in TUNED_MODELS:
        t0 = time.time()
        gp, gf = gwo_search(name, Xtr, ytr, seed=0, n_agents=N_AGENTS, iters=ITERS)
        print(f"  [GWO {name}] CV PR-AUC={gf:.4f} params={gp} ({time.time()-t0:.0f}s)")
        t0 = time.time()
        rp, rf = random_search(name, Xtr, ytr, seed=0, budget=budget)
        print(f"  [random {name}] CV PR-AUC={rf:.4f} params={rp} ({time.time()-t0:.0f}s)")
        res = eval_on_test(name, gp, Xtr, ytr, Xte, yte)
        cells = " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items())
        print(f"  [GWO-tuned {name} on temporal test] {cells}")
        rows.append({"model": name, "config": "gwo", "cv_prauc": gf, "cv_prauc_random": rf,
                     "budget": budget, "params": str(gp),
                     **{k: m for k, (m, s) in res.items()},
                     **{k + "_std": s for k, (m, s) in res.items()}})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[+] {OUT_CSV}")
    write_table(out, len(tr), len(te))


def fmt(r, k):
    return f"{r[k]:.3f}\\,$\\pm$\\,{r[k + '_std']:.3f}"


def write_table(out, n_tr, n_te):
    label = {"LogReg": "Logistic regression", "RandomForest": "Random Forest",
             "HistGB": "HistGB", "MLP": "MLP"}
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{URL/infrastructure channel under the corpus's group-aware temporal split"
        f" ({n_tr:,} older training rows, {n_te:,} strictly newer test rows);"
        f" mean\\,$\\pm$\\,std over {SEEDS} seeds. GWO-tuned rows use Algorithm~\\ref{{alg:gwo}}'s"
        f" configuration, tuned on the training window only.}}",
        "\\label{tab:temporal_gwo}",
        "\\begin{tabular}{l c c c c}", "\\toprule",
        "Model & F1 & PR-AUC & ROC-AUC & FPR@90\\%rec. \\\\", "\\midrule",
    ]
    for _, r in out[out.config == "default"].iterrows():
        lines.append(f"{label[r['model']]} & " + " & ".join(
            fmt(r, k) for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")) + " \\\\")
    tuned = out[out.config == "gwo"]
    if len(tuned):
        lines.append("\\midrule")
        for _, r in tuned.iterrows():
            lines.append(f"{label[r['model']]} (GWO-tuned) & " + " & ".join(
                fmt(r, k) for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\end{table}",
    ]
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] {OUT_TEX}")


if __name__ == "__main__":
    main()
