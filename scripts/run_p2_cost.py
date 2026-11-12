#!/usr/bin/env python3
"""
run_p2_cost.py — what each model costs to train, to store and to run (revision 1, 2026-10-31).

WHY. The paper motivates lightweight URL screening and compares seven tabular families with a
character-CNN and a stacked ensemble, but reports no cost. This measures it on one machine, one
thread for every model, on the phishing-temporal training window of seed 0 (the same rows as
Table tab:strict).

Per model: fit seconds; serialised size (pickle bytes for the tabular models and the stack, 4
bytes per parameter for the CNN); inference latency per URL in a batch of 1,000 and for one URL
at a time (median of 200 single calls), from the 21 features for the tabular models and from the
encoded string for the CNN. Feature extraction from a raw URL (compphish_features.extract) is
timed separately, since every tabular model pays it and the CNN does not. The CNN's fit time is
read from the canonical CPU run (data/processed/p2/p2_charcnn.csv, fit_seconds) rather than
re-trained; inference does not depend on the weights, so a freshly initialised network of the
same shape is timed.

TIMING UNDER BACKGROUND LOAD. The workstation is rarely idle. Other processes can only make a
timed call slower, never faster, so the whole measurement is repeated ROUNDS times and the table
reports, per model and quantity, the minimum over rounds: the run least disturbed by the rest of
the machine. Every round is kept, with the machine's 1-minute load average when it started, in
cost_rounds.csv (and cost_cnn_batch_rounds.csv for the CNN's batch-size curve).

Writes data/processed/p2/revision/cost.csv and cost_cnn_batch.csv (the minima) beside the rounds.

RUN: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_p2_cost.py
"""
from __future__ import annotations

import os
import pickle
import platform
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
from train_url_baseline import COMPPHISH
from run_p2_benchmark import FAMILIES, make_any_model, weighting_params
from run_p2_temporal_strict import benign_mask, load, split_phishing
from compphish_features import extract

REV = "data/processed/p2/revision"
OUT = os.path.join(REV, "cost.csv")
OUT_ROUNDS = os.path.join(REV, "cost_rounds.csv")
OUT_CNN_BATCH = os.path.join(REV, "cost_cnn_batch.csv")
OUT_CNN_BATCH_ROUNDS = os.path.join(REV, "cost_cnn_batch_rounds.csv")
CNN_BATCH_SIZES = (1, 10, 32, 100, 1000)
SINGLE_CALLS = 200
WARMUP_CALLS = 3
BATCH_REPEATS = 5
ROUNDS = 5
TIMED = ["fit_s", "ms_per_url_batch", "ms_single"]


def load1() -> float:
    """The 1-minute load average, recorded beside each round."""
    try:
        return float(os.getloadavg()[0])
    except OSError:
        return float("nan")


def latency(predict, X) -> tuple[float, float]:
    """(ms per URL in a batch of 1,000, median ms for one URL).

    Both are timed after warm-up calls: the first call of a model pays one-off costs (lazy
    allocation, thread-pool start-up, kernel selection) that are not per-URL cost. The batch figure
    is the median of BATCH_REPEATS timed batches."""
    xb = X[:1000]
    for _ in range(WARMUP_CALLS):
        predict(xb)
        predict(X[:1])
    reps = []
    for _ in range(BATCH_REPEATS):
        t0 = time.perf_counter()
        predict(xb)
        reps.append((time.perf_counter() - t0) * 1e3 / len(xb))
    batch = float(np.median(reps))
    one = []
    for i in range(SINGLE_CALLS):
        t0 = time.perf_counter()
        predict(X[i:i + 1])
        one.append((time.perf_counter() - t0) * 1e3)
    return batch, float(np.median(one))


def one_round(tr, te, feats, cnn_fit: float) -> tuple[list[dict], list[dict]]:
    """Every model timed once; returns (model rows, CNN batch-size curve)."""
    Xtr, ytr, Xte = tr[feats].to_numpy(float), tr.y.to_numpy(int), te[feats].to_numpy(float)
    rows = []

    urls = te.url.astype(str).tolist()[:1000]
    for u in urls[:50]:
        extract(u)  # warm-up
    t0 = time.perf_counter()
    for u in urls:
        extract(u)
    rows.append({"model": "feature extraction (21 features)", "fit_s": np.nan, "size_kb": np.nan,
                 "ms_per_url_batch": (time.perf_counter() - t0) * 1e3 / len(urls),
                 "ms_single": np.nan})

    for name in FAMILIES:
        m = make_any_model(name, 0, weighting_params(name))
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        fit = time.perf_counter() - t0
        b, o = latency(lambda x: m.predict_proba(x), Xte)
        rows.append({"model": name, "fit_s": fit, "size_kb": len(pickle.dumps(m)) / 1024,
                     "ms_per_url_batch": b, "ms_single": o})
        print(rows[-1], flush=True)

    from run_p2_stacking_baseline import make_stacker
    st = make_stacker(["CatBoost", "LogReg"], 0)
    t0 = time.perf_counter()
    st.fit(Xtr, ytr)
    fit = time.perf_counter() - t0
    b, o = latency(lambda x: st.predict_proba(x), Xte)
    rows.append({"model": "Stack[CatBoost+LogReg]", "fit_s": fit,
                 "size_kb": len(pickle.dumps(st)) / 1024, "ms_per_url_batch": b, "ms_single": o})
    print(rows[-1], flush=True)

    import torch
    from run_p2_charcnn import build_vocab, encode, make_net
    vocab = build_vocab(tr.url.astype(str))
    net = make_net(len(vocab), 0).eval()
    Xc = torch.from_numpy(encode(te.url.astype(str).tolist()[:1000 + SINGLE_CALLS], vocab))
    curve = []
    with torch.no_grad():
        b, o = latency(lambda x: torch.sigmoid(net(x)), Xc)
        # On one thread the CNN's per-URL time depends on the batch size (its activations stop
        # fitting in cache between 10 and 32 URLs), so record the curve the table's two columns
        # sit on.
        for size in CNN_BATCH_SIZES:
            xb = Xc[:size]
            for _ in range(WARMUP_CALLS):
                net(xb)
            reps = []
            for _ in range(BATCH_REPEATS):
                t0 = time.perf_counter()
                net(xb)
                reps.append((time.perf_counter() - t0) * 1e3 / size)
            curve.append({"batch": size, "ms_per_url": float(np.median(reps))})
    rows.append({"model": "CharCNN", "fit_s": cnn_fit,
                 "size_kb": sum(p.numel() for p in net.parameters()) * 4 / 1024,
                 "ms_per_url_batch": b, "ms_single": o})
    print(rows[-1], flush=True)
    return rows, curve


def main() -> int:
    import torch
    torch.set_num_threads(1)
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    bm = benign_mask(be, np.random.RandomState(0), 0.70)
    tr, te = pd.concat([ph_tr, be[bm]]), pd.concat([ph_te, be[~bm]])
    cnn = pd.read_csv("data/processed/p2/p2_charcnn.csv")
    cnn_fit = float(cnn.loc[cnn.protocol == "temporal_strict", "fit_seconds"].mean()) \
        if "fit_seconds" in cnn else np.nan

    all_rows, all_curve = [], []
    for r in range(ROUNDS):
        load_at_start = load1()
        print(f"--- round {r + 1}/{ROUNDS}, 1-min load {load_at_start:.2f}", flush=True)
        rows, curve = one_round(tr, te, feats, cnn_fit)
        for x in rows:
            x.update(round=r + 1, load1=load_at_start)
        for x in curve:
            x.update(round=r + 1, load1=load_at_start)
        all_rows += rows
        all_curve += curve

    os.makedirs(REV, exist_ok=True)
    rounds = pd.DataFrame(all_rows)
    rounds.to_csv(OUT_ROUNDS, index=False)
    curve_rounds = pd.DataFrame(all_curve)
    curve_rounds.to_csv(OUT_CNN_BATCH_ROUNDS, index=False)

    order = list(dict.fromkeys(rounds.model))
    out = (rounds.groupby("model", sort=False)
           .agg({**{c: "min" for c in TIMED}, "size_kb": "first"})
           .reindex(order).reset_index())
    out = out[["model", "fit_s", "size_kb", "ms_per_url_batch", "ms_single"]]
    out["machine"] = f"{platform.machine()} {platform.processor() or ''} {platform.system()}".strip()
    out["torch_threads"] = torch.get_num_threads()
    out["rounds"] = ROUNDS
    out.to_csv(OUT, index=False)
    curve_rounds.groupby("batch", as_index=False).ms_per_url.min().to_csv(OUT_CNN_BATCH, index=False)
    print(out.round(4).to_string(index=False))
    print("load at round start:", rounds.groupby("round").load1.first().round(2).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
