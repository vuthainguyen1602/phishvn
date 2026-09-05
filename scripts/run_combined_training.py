#!/usr/bin/env python3
"""
run_combined_training.py — P2: does POOLING corpora rescue transfer?

The label read P3 until 2026-09-05. It was stale: the transfer material moved to the URL
benchmark on 2026-08-19, and OUT_TEX below has written into that paper's sections/ ever
since. The file keeps its p3_multimodal/ home so no import path moves.

For each held-out corpus j, train on the UNION of the other three and test on all of j — the
strongest "just add more corpora" baseline the cross-dataset matrix admits. Run once on the full
21-feature CompPhish schema and once with the three SHAP-named artefact features pruned, so pooling
and pruning can be compared and stacked against the per-cell matrix numbers. Reference columns are
read from the matrix rather than recomputed.

Outputs: data/processed/p2/combined_training.csv
The two interventions and how they stack: kept in the development repository, not shipped in this mirror
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
from train_url_baseline import COMPPHISH, _metrics, make_model
from run_cross_dataset import load_corpus

PROC = os.path.join(ROOT, "data/processed")
OUT_CSV = os.path.join(PROC, "p2", "combined_training.csv")
OUT_TEX = os.path.join(ROOT, "papers/P2_url_benchmark/sections/tab_combined.tex")

CORPORA = {
    "PhishVN": "vn_compphish.csv",
    "PhiUSIIL": "external/phiusiil_compphish.csv",
    "ISCXURL2016": "external/iscx_compphish.csv",
    "PhishStorm": "external/phishstorm_compphish.csv",
}
DROP = {"tld_len", "subdom_cnt", "dot_cnt"}
PRUNED = [c for c in COMPPHISH if c not in DROP]
SEEDS = 5


def main():
    data = {n: load_corpus(os.path.join(PROC, f)) for n, f in CORPORA.items()}
    names = list(data)
    f1_mat = pd.read_csv(os.path.join(PROC, "p2", "cross_dataset_F1.csv"), index_col=0)
    roc_mat = pd.read_csv(os.path.join(PROC, "p2", "cross_dataset_ROC-AUC.csv"), index_col=0)

    rows = []
    for held in names:
        union = pd.concat([data[n] for n in names if n != held], ignore_index=True)
        te = data[held]
        print(f"held-out={held}: union train={len(union)} rows, test={len(te)}")
        for tag, feats in (("full", COMPPHISH), ("pruned", PRUNED)):
            t0 = time.time()
            f1s, rocs = [], []
            for s in range(SEEDS):
                m = make_model("RandomForest", s, {}).fit(union[feats], union["y"])
                score = m.predict_proba(te[feats])[:, 1]
                met = _metrics(te["y"].to_numpy(), score)
                f1s.append(met["F1"])
                rocs.append(met["ROC-AUC"])
            singles_f1 = [f1_mat.loc[i, held] for i in names if i != held]
            rows.append({
                "held_out": held, "config": tag,
                "F1": float(np.mean(f1s)), "F1_std": float(np.std(f1s)),
                "ROC-AUC": float(np.mean(rocs)), "ROC-AUC_std": float(np.std(rocs)),
                "best_single_F1": float(max(singles_f1)),
                "best_single_ROC": float(max(roc_mat.loc[i, held] for i in names if i != held)),
                "indist_F1": float(f1_mat.loc[held, held]),
            })
            print(f"  [{tag:6s}] F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} "
                  f"ROC-AUC={np.mean(rocs):.3f}±{np.std(rocs):.3f} "
                  f"(best single {max(singles_f1):.3f}, in-dist {f1_mat.loc[held, held]:.3f}) "
                  f"({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[+] {OUT_CSV}")
    write_table(out)


def write_table(out):
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Pooled training, F1 (random forest, five seeds; $\\pm$ = seed spread):"
        " trained on the union of the other three corpora; ``Best single'' = strongest"
        " single-source cell of Table~\\ref{tab:xdatasetrf}.}",
        "\\label{tab:combined}",
        "\\begin{tabular}{l c c c c}", "\\toprule",
        "Held-out corpus & Best single & Union (21) & Union (18, pruned) & In-dist.\\ (ref.) \\\\",
        "\\midrule",
    ]
    for held in out.held_out.unique():
        full = out[(out.held_out == held) & (out.config == "full")].iloc[0]
        pr = out[(out.held_out == held) & (out.config == "pruned")].iloc[0]
        lines.append(
            f"{held} & {full['best_single_F1']:.3f} & "
            f"{full['F1']:.3f}\\,$\\pm$\\,{full['F1_std']:.3f} & "
            f"{pr['F1']:.3f}\\,$\\pm$\\,{pr['F1_std']:.3f} & {full['indist_F1']:.3f} \\\\")
    m = {c: out[out.config == c] for c in ("full", "pruned")}
    # The "In-dist. (ref.)" column is the 21-feature diagonal for BOTH union columns; the pruned
    # diagonal is read from run_cross_dataset's pruned matrix so the footnote cannot go stale.
    pm = pd.read_csv(os.path.join(PROC, "p2", "cross_dataset_F1_pruned.csv"), index_col=0)
    pruned_diag = float(np.mean(np.diag(pm.values)))
    lines += [
        "\\midrule",
        f"Mean & {out[out.config=='full'].best_single_F1.mean():.3f} & "
        f"{m['full'].F1.mean():.3f} & {m['pruned'].F1.mean():.3f} & "
        f"{m['full'].indist_F1.mean():.3f} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\footnotesize In-distribution reference: 21-feature diagonal;"
        f" the pruned diagonal is {pruned_diag:.3f} (Table~\\ref{{tab:pruned}}).}} \\\\",
        "\\bottomrule", "\\end{tabular}",
        "\\end{table}",
    ]
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] {OUT_TEX}")


if __name__ == "__main__":
    main()
