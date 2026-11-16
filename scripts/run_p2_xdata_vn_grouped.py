#!/usr/bin/env python3
"""
run_p2_xdata_vn_grouped.py — the random-forest transfer matrix with PhishVN's diagonal drawn like
the other three corpora's.

WHY. In the four-corpus transfer matrix the diagonal cell of PhiUSIIL, ISCX-URL2016 and PhishStorm
is a 70/30 split grouped by registrable domain, while PhishVN's uses the train/test split the
corpus ships with (also domain-disjoint). This arm drops PhishVN's split column so that
run_cross_dataset draws the same grouped 70/30 split for it, and reruns the F1 and ROC-AUC
matrices. Only the PhishVN diagonal can change: off-diagonal cells train on the whole source.

RUN: P2_UNWEIGHTED=1 OPENBLAS_NUM_THREADS=1 python scripts/run_p2_xdata_vn_grouped.py
OUT: data/processed/p2/revision/xdata_vn_grouped_{F1,ROC-AUC}.csv
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(PROC, "p2", "revision")


def main() -> None:
    runner = os.path.join(os.path.dirname(_HERE), "p3_multimodal", "run_cross_dataset.py")
    if not os.path.exists(runner):  # flat mirror layout
        runner = os.path.join(_HERE, "run_cross_dataset.py")
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        vn = os.path.join(tmp, "vn_compphish_nosplit.csv")
        pd.read_csv(os.path.join(PROC, "vn_compphish.csv"), low_memory=False) \
            .drop(columns=["split"]).to_csv(vn, index=False)
        corpora = [f"PhishVN={vn}",
                   f"PhiUSIIL={os.path.join(PROC, 'external', 'phiusiil_compphish.csv')}",
                   f"ISCXURL2016={os.path.join(PROC, 'external', 'iscx_compphish.csv')}",
                   f"PhishStorm={os.path.join(PROC, 'external', 'phishstorm_compphish.csv')}"]
        env = dict(os.environ, P2_UNWEIGHTED="1", OPENBLAS_NUM_THREADS="1")
        for metric in ("F1", "ROC-AUC"):
            subprocess.run([sys.executable, "-u", runner, "--corpora", *corpora, "--seeds", "5",
                            "--metric", metric, "--model", "RandomForest",
                            "--out", os.path.join(OUT, f"xdata_vn_grouped_{metric}.csv")],
                           cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
