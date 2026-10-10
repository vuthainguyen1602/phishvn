#!/usr/bin/env python3
"""
run_p2_nowww_xdata.py — the transfer matrix with every host in canonical form.

WHY. The transfer arm featurises hosts, and a leading "www." carries opposite labels in different
corpora: on PhishVN and ISCX-URL2016 it is almost only on phishing rows, on PhiUSIIL it is on every
benign row and on PhishStorm on two thirds of them. subdom_cnt, dot_cnt and the length features all
count it, so a model can learn "www. means phishing" on one corpus and be scored on a corpus where
it means the opposite. This arm removes the prefix from every host in all four corpora, recomputes
the CompPhish features, and reruns the matrices, so the share of the below-chance transfer that
the prefix explains is measured rather than argued. PhishVN enters as one row per host (P2's
dedup_audit corpus), since stripping the prefix from the released corpus would turn its 16,855
twin rows into exact duplicates.

Writes data/processed/p2/nowww/: {phishvn,phiusiil,iscx,phishstorm}_nowww.csv, www_share.csv
(share of rows with the prefix, per corpus and class, before stripping) and the matrices
cross_dataset_{F1,ROC-AUC,MCC}.csv (random forest) and cross_dataset_F1_CatBoost.csv.

RUN:  P2_UNWEIGHTED=1 python scripts/run_p2_nowww_xdata.py [--skip-matrices]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from compphish_features import extract

PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(PROC, "p2", "nowww")
SOURCES = {"phishvn": os.path.join(PROC, "p2", "dedup_audit", "vn_compphish_dedup.csv"),
           "phiusiil": os.path.join(PROC, "external", "phiusiil_compphish.csv"),
           "iscx": os.path.join(PROC, "external", "iscx_compphish.csv"),
           "phishstorm": os.path.join(PROC, "external", "phishstorm_compphish.csv")}
NAMES = {"phishvn": "PhishVN", "phiusiil": "PhiUSIIL", "iscx": "ISCXURL2016",
         "phishstorm": "PhishStorm"}
WWW = re.compile(r"^((?:[A-Za-z][A-Za-z0-9+.-]*://)?)www\.", re.I)


def is_phish(v) -> bool:
    return str(v).strip().lower() in {"1", "phishing", "phish", "malicious", "bad"}


def build():
    os.makedirs(OUT, exist_ok=True)
    share = []
    for key, path in SOURCES.items():
        df = pd.read_csv(path, low_memory=False)
        url = df["url"].astype(str)
        has = url.map(lambda u: bool(WWW.match(u)))
        y = df["label"].map(is_phish)
        for cls, m in (("phishing", y), ("benign", ~y)):
            share.append({"corpus": NAMES[key], "class": cls, "rows": int(m.sum()),
                          "www_share": float(has[m].mean())})
        feats = pd.DataFrame([extract(WWW.sub(r"\1", u)) for u in url])
        for col in df.columns:
            if col not in feats.columns:
                feats[col] = df[col].values
        feats[df.columns].to_csv(os.path.join(OUT, f"{key}_nowww.csv"), index=False)
        print(f"[i] {NAMES[key]}: {len(df)} rows, {int(has.sum())} prefixes removed", flush=True)
    rel = pd.read_csv(os.path.join(PROC, "vn_compphish.csv"), usecols=["url", "label"])
    has, y = rel["url"].astype(str).map(lambda u: bool(WWW.match(u))), rel["label"].map(is_phish)
    for cls, m in (("phishing", y), ("benign", ~y)):
        share.append({"corpus": "PhishVN (released)", "class": cls, "rows": int(m.sum()),
                      "www_share": float(has[m].mean())})
    pd.DataFrame(share).to_csv(os.path.join(OUT, "www_share.csv"), index=False)
    print(pd.DataFrame(share).round(3).to_string(index=False))


def matrices():
    x = os.path.join(os.path.dirname(_HERE), "p3_multimodal", "run_cross_dataset.py")
    if not os.path.exists(x):  # flat mirror layout
        x = os.path.join(_HERE, "run_cross_dataset.py")
    corpora = [f"{NAMES[k]}={os.path.join(OUT, k + '_nowww.csv')}" for k in SOURCES]
    env = dict(os.environ, P2_UNWEIGHTED="1", OPENBLAS_NUM_THREADS="1")
    for metric, model, suf in (("F1", "RandomForest", ""), ("ROC-AUC", "RandomForest", ""),
                               ("MCC", "RandomForest", ""), ("F1", "CatBoost", "_CatBoost")):
        subprocess.run([sys.executable, "-u", x, "--corpora", *corpora, "--seeds", "5",
                        "--metric", metric, "--model", model,
                        "--out", os.path.join(OUT, f"cross_dataset_{metric}{suf}.csv")],
                       cwd=ROOT, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-matrices", action="store_true")
    a = ap.parse_args()
    build()
    if not a.skip_matrices:
        matrices()


if __name__ == "__main__":
    main()
