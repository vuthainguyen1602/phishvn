#!/usr/bin/env python3
"""
audit_xdata_leakage.py — measure how much duplicate-domain leakage inflates the cross-dataset
DIAGONAL, and therefore the generalisation gap the URL benchmark builds on.

align_compphish.py reduces every corpus to registrable-domain granularity and nothing deduplicates
afterwards, while run_cross_dataset.in_dataset_split falls back to a row-level stratified split for
every corpus except PhishVN. The 21 features are computed from the domain string, so a domain on
both sides puts an IDENTICAL feature vector with an IDENTICAL label in train and test.

RUN: python scripts/audit_xdata_leakage.py [--seeds 3]
The measurement and what it implies for the gap: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label, make_model, _metrics

CORPORA = {
    "PhishVN":     "data/processed/vn_compphish.csv",
    "PhiUSIIL":    "data/processed/external/phiusiil_compphish.csv",
    "ISCXURL2016": "data/processed/external/iscx_compphish.csv",
    "PhishStorm":  "data/processed/external/phishstorm_compphish.csv",
}
FEATS = [c for c in COMPPHISH]


def load(path):
    df = pd.read_csv(os.path.join(ROOT, path))
    df = add_label(df)
    for c in FEATS:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def score(tr, te, seed, metric):
    m = make_model("RandomForest", seed=seed)
    m.fit(tr[FEATS], tr["y"])
    p = m.predict_proba(te[FEATS])[:, 1]
    return _metrics(te["y"].values, p)[metric]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    rows = []
    for name, path in CORPORA.items():
        df = load(path)
        gcol = "dom" if "dom" in df else "url"
        n, uniq = len(df), df[gcol].nunique()
        # how much of a random test fold is already in train, feature-identically?
        tr, te = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=0)
        seen = set(map(tuple, tr[FEATS].values))
        overlap = np.mean([tuple(v) in seen for v in te[FEATS].values])

        out = {"corpus": name, "rows": n, "unique_dom": uniq,
               "dup_rows_pct": 100 * (1 - uniq / n), "test_in_train_pct": 100 * overlap}

        ded = df.drop_duplicates(subset=FEATS + ["y"])
        for metric in ("F1", "ROC-AUC"):
            asrun, dedup, group = [], [], []
            for s in range(a.seeds):
                t1, e1 = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=s)
                asrun.append(score(t1, e1, s, metric))
                t2, e2 = train_test_split(ded, test_size=0.3, stratify=ded["y"], random_state=s)
                dedup.append(score(t2, e2, s, metric))
                gi, gj = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                                random_state=s).split(df, groups=df[gcol]))
                group.append(score(df.iloc[gi], df.iloc[gj], s, metric))
            out[f"{metric}_asrun"] = np.mean(asrun)
            out[f"{metric}_dedup"] = np.mean(dedup)
            out[f"{metric}_group"] = np.mean(group)
        rows.append(out)
        print(f"[+] {name}")

    r = pd.DataFrame(rows).set_index("corpus")
    pd.set_option("display.width", 200)
    print("\n=== duplication and leak exposure ===")
    print(r[["rows", "unique_dom", "dup_rows_pct", "test_in_train_pct"]].round(1))
    for metric in ("F1", "ROC-AUC"):
        cols = [f"{metric}_asrun", f"{metric}_dedup", f"{metric}_group"]
        print(f"\n=== in-distribution {metric} (the DIAGONAL) ===")
        print(r[cols].round(3))
        print(f"  diagonal mean: as-run {r[cols[0]].mean():.3f} | "
              f"dedup {r[cols[1]].mean():.3f} | group-aware {r[cols[2]].mean():.3f}")
    out = os.path.join(ROOT, "data/reports/xdata_leakage_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r.to_csv(out)
    print(f"\n[+] {out}")


if __name__ == "__main__":
    main()
