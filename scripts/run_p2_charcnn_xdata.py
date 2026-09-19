#!/usr/bin/env python3
"""
run_p2_charcnn_xdata.py — the character-CNN in the four-corpus transfer matrix.

WHY. Section 6.7 closes with a scoped claim: what fails to transfer is "this feature space", the
21-feature CompPhish schema, and a model that reads the URL string might place the boundary
somewhere the hand-built space cannot. The manuscript left that as a comment marked
[NEED EXPERIMENT]. The eighth arm already exists (run_p2_charcnn.py) and the matrix already
exists (run_cross_dataset.py); this runs the one in the other, so the sentence can be a
measurement either way.

SAME DESIGN AS THE TABULAR MATRIX. Diagonal cells use the same registrable-domain-grouped
70/30 split run_cross_dataset.in_dataset_split draws (PhishVN's shipped split column, a grouped
draw for the other three); off-diagonal cells train on the whole source corpus and score the
whole target. One thing differs by construction and is stated: a network fitted once on a source
corpus is the same network whichever target it is scored on, so a (source, seed) pair is fitted
once and scored three times rather than refitted per cell. The tabular runner refits per cell
because that is how its loop is written; for deterministic-per-seed fits the result is the same.

THE STRING THE NETWORK SEES. PhishVN's aligned file carries a scheme ("http://...") and the
three external files do not, so the scheme is stripped everywhere before encoding. Left in, a
model trained on PhishVN would key on seven characters no external corpus contains. Nothing else
is normalised: all four corpora were already reduced to registrable-domain granularity by
align_compphish, so the strings are hosts, as in the main arm.

DEVICE. Forty fits at 12 epochs is a GPU job; the benchmark's canonical arm was run on the CPU. The
device is recorded per row and the two runs are never compared cell to cell.

RUN:
  python scripts/run_p2_charcnn_xdata.py --seeds 5 --device mps
  python scripts/run_p2_charcnn_xdata.py --seeds 1 --epochs 2 \\
      --out /tmp/x.csv --matrix-prefix ""                                  # wiring check
Writes data/processed/p2/p2_charcnn_xdata.csv (one row per cell and seed, every metric) and
data/processed/p2/cross_dataset_{F1,ROC-AUC}_CharCNN.csv in the matrix layout every
intervention table reads.
"""
from __future__ import annotations
import argparse, os, re, sys, time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import _metrics
from run_cross_dataset import load_corpus, in_dataset_split
from run_p2_charcnn import fit_predict

CORPORA = [("PhishVN", "data/processed/vn_compphish.csv"),
           ("PhiUSIIL", "data/processed/external/phiusiil_compphish.csv"),
           ("ISCXURL2016", "data/processed/external/iscx_compphish.csv"),
           ("PhishStorm", "data/processed/external/phishstorm_compphish.csv")]
OUT = "data/processed/p2/p2_charcnn_xdata.csv"
MATRIX_PREFIX = "data/processed/p2/cross_dataset_"
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


def strip_scheme(u) -> str:
    return _SCHEME.sub("", str(u))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--matrix-prefix", default=MATRIX_PREFIX,
                    help="prefix for the two matrix CSVs; empty string = do not write them")
    a = ap.parse_args()
    os.chdir(ROOT)

    corpora = {}
    for name, path in CORPORA:
        df = load_corpus(path)
        df["url_s"] = df["url"].map(strip_scheme)
        corpora[name] = df
        print(f"[i] {name}: {len(df)} rows (phishing={int(df.y.sum())}), "
              f"{df.url_s.str.len().median():.0f} chars median")
    names = [n for n, _ in CORPORA]

    rows = []
    for src in names:
        S = corpora[src]
        for s in range(a.seeds):
            # diagonal: the grouped in-corpus split, same draw as the tabular matrix
            tr, te = in_dataset_split(S, s)
            t0 = time.time()
            sc = fit_predict(tr.url_s.tolist(), tr.y.to_numpy(int), te.url_s.tolist(),
                             s, a.epochs, device=a.device)
            m = _metrics(te.y.to_numpy(int), sc)
            m.update(train=src, test=src, seed=s, n_train=len(tr), n_test=len(te),
                     fit_seconds=round(time.time() - t0, 1), device=a.device)
            rows.append(m)
            print(f"  {src:<12}-> {src:<12} seed={s} F1={m['F1']:.3f} ROC={m['ROC-AUC']:.3f} "
                  f"({m['fit_seconds']}s, in-dist)", flush=True)
            # off-diagonal: one fit on the whole source, scored on each whole target. The
            # network is trained once and only scored three times, so the fit time is charged
            # to the first target and zero to the others; the CSV says so.
            targets = [t for t in names if t != src]
            t0 = time.time()
            net_scores = fit_predict(S.url_s.tolist(), S.y.to_numpy(int),
                                     pd.concat([corpora[t].url_s for t in targets]).tolist(),
                                     s, a.epochs, device=a.device)
            fit_s = round(time.time() - t0, 1)
            off = 0
            for t in targets:
                T = corpora[t]
                sc = net_scores[off:off + len(T)]
                off += len(T)
                m = _metrics(T.y.to_numpy(int), sc)
                m.update(train=src, test=t, seed=s, n_train=len(S), n_test=len(T),
                         fit_seconds=fit_s if t == targets[0] else 0.0, device=a.device)
                rows.append(m)
                print(f"  {src:<12}-> {t:<12} seed={s} F1={m['F1']:.3f} "
                      f"ROC={m['ROC-AUC']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)  # checkpoint after every seed

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[+] {len(out)} rows -> {a.out}")
    if a.matrix_prefix:
        for metric in ("F1", "ROC-AUC"):
            mat = (out.pivot_table(index="train", columns="test", values=metric, aggfunc="mean")
                      .reindex(index=names, columns=names).round(3))
            mat.index.name = None
            path = f"{a.matrix_prefix}{metric}_CharCNN.csv"
            mat.to_csv(path)
            v = mat.to_numpy(float)
            d, o = np.diag(v).mean(), v[~np.eye(len(v), dtype=bool)].mean()
            print(f"[+] {path}  diag={d:.3f} off={o:.3f} gap={d - o:.3f}")
            print(mat.to_string())


if __name__ == "__main__":
    main()
