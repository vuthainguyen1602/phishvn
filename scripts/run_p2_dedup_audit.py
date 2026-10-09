#!/usr/bin/env python3
"""
run_p2_dedup_audit.py — the duplicated-host sensitivity arm of P2.

The released corpus records 16,855 hosts twice: almost every undated community-feed phishing row
is the "www."-prefixed copy of a dated row for the same host. This arm keeps one row per host
(the host with any leading "www." removed; the dated row is kept when there is one, then the row
without "www."), leaving 36,261 rows, and reruns the P2 arms whose rows the duplicates touch.

The feature rows are a subset of data/processed/vn_compphish.csv (row-aligned with
dataset_url.csv), so nothing is re-featurised. The standard runners are run unchanged inside a
scratch tree whose data/processed/{dataset_url,vn_compphish}.csv are the deduplicated files and
whose data/processed/p2 is a copy, so no canonical CSV is ever written; the outputs are copied
to data/processed/p2/dedup_audit/, which gen_dedup_sensitivity in make_p2_bench_assets.py reads.

RUN:  python scripts/run_p2_dedup_audit.py [--tree /tmp/p2dedup]
      (about two hours: seven families x two protocols x five seeds, plus the guarded control,
       three transfer matrices, the duplicate-leakage audit and TreeSHAP)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(PROC, "p2", "dedup_audit")
COPIED = ["p2_benchmark", "p2_temporal_strict", "p2_temporal_strict_guarded", "cross_dataset_F1",
          "cross_dataset_ROC-AUC", "cross_dataset_F1_CatBoost", "p2_dup_leakage",
          "p2_shap_weighting"]


def bare_host(urls: pd.Series) -> pd.Series:
    return (urls.astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True))


def dedup() -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv(os.path.join(PROC, "dataset_url.csv"), low_memory=False)
    c = pd.read_csv(os.path.join(PROC, "vn_compphish.csv"), low_memory=False)
    if len(u) != len(c):
        raise SystemExit("dataset_url.csv and vn_compphish.csv are not row-aligned")
    s = u["collected_at"].astype(str).str.strip().str.slice(0, 10)
    dated = (pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
             .fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))).notna()
    host = u["url_norm"].astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
    k = pd.DataFrame({"i": np.arange(len(u)), "bare": bare_host(u["url_norm"]),
                      "dated": dated, "www": host.str.startswith("www.")})
    k = k.sort_values(["bare", "dated", "www", "i"], ascending=[True, False, True, True])
    keep = np.sort(k.drop_duplicates("bare")["i"].to_numpy())
    return u.iloc[keep].reset_index(drop=True), c.iloc[keep].reset_index(drop=True)


def build_tree(tree: str, ud: pd.DataFrame, cd: pd.DataFrame) -> None:
    os.makedirs(os.path.join(tree, "data", "processed"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "requirements.txt"), tree)
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(tree, "scripts"), dirs_exist_ok=True)
    for d in ("raw", "interim", "external", "docs"):
        src, dst = os.path.join(ROOT, "data", d), os.path.join(tree, "data", d)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    for name in os.listdir(PROC):
        src, dst = os.path.join(PROC, name), os.path.join(tree, "data", "processed", name)
        if name in ("dataset_url.csv", "vn_compphish.csv") or os.path.exists(dst):
            continue
        if name == "p2":  # a real copy: the runners write here
            os.makedirs(dst)
            for f in os.listdir(src):
                if f.endswith(".csv"):
                    shutil.copy(os.path.join(src, f), dst)
        else:
            os.symlink(src, dst)
    ud.to_csv(os.path.join(tree, "data", "processed", "dataset_url.csv"), index=False)
    cd.to_csv(os.path.join(tree, "data", "processed", "vn_compphish.csv"), index=False)
    for p in ("P2_url_benchmark",):
        os.makedirs(os.path.join(tree, "papers", p, "figures"), exist_ok=True)
        shutil.copytree(os.path.join(ROOT, "papers", p, "sections"),
                        os.path.join(tree, "papers", p, "sections"), dirs_exist_ok=True)


def run_arms(tree: str) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", P2_UNWEIGHTED="1")
    xc = ["PhishVN=data/processed/vn_compphish.csv",
          "PhiUSIIL=data/processed/external/phiusiil_compphish.csv",
          "ISCXURL2016=data/processed/external/iscx_compphish.csv",
          "PhishStorm=data/processed/external/phishstorm_compphish.csv"]
    b = "scripts/"
    x = "scripts/run_cross_dataset.py"
    arms = [[b + "run_p2_benchmark.py", "--seeds", "5"],
            [b + "run_p2_temporal_strict.py", "--seeds", "5"],
            [b + "run_p2_temporal_strict.py", "--seeds", "5", "--guard-control",
             "--out", "data/processed/p2/p2_temporal_strict_guarded.csv", "--curves", ""],
            [x, "--corpora", *xc, "--seeds", "5", "--metric", "F1", "--model", "RandomForest",
             "--out", "data/processed/p2/cross_dataset_F1.csv"],
            [x, "--corpora", *xc, "--seeds", "5", "--metric", "ROC-AUC", "--model", "RandomForest",
             "--out", "data/processed/p2/cross_dataset_ROC-AUC.csv"],
            [x, "--corpora", *xc, "--seeds", "5", "--metric", "F1", "--model", "CatBoost",
             "--out", "data/processed/p2/cross_dataset_F1_CatBoost.csv"],
            [b + "p2_dup_leakage.py"],
            [b + "run_p2_shap_weighting.py"]]
    for a in arms:
        print("[run]", " ".join(a), flush=True)
        subprocess.run([sys.executable, "-u", *a], cwd=tree, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="", help="scratch directory (default: a new temp dir)")
    ap.add_argument("--corpus-only", action="store_true",
                    help="write the deduplicated corpus to dedup_audit/ and stop")
    a = ap.parse_args()
    ud, cd = dedup()
    os.makedirs(OUT, exist_ok=True)
    ud.to_csv(os.path.join(OUT, "dataset_url_dedup.csv"), index=False)
    cd.to_csv(os.path.join(OUT, "vn_compphish_dedup.csv"), index=False)
    print(f"[i] kept {len(ud):,} of the corpus's rows, one per host")
    if a.corpus_only:
        return
    tree = a.tree or tempfile.mkdtemp(prefix="p2dedup_")
    build_tree(tree, ud, cd)
    run_arms(tree)
    for f in COPIED:
        shutil.copy(os.path.join(tree, "data", "processed", "p2", f + ".csv"),
                    os.path.join(OUT, f + ".csv"))
    print(f"[+] {len(COPIED)} outputs -> {OUT}")


if __name__ == "__main__":
    main()
