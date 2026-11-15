#!/usr/bin/env python3
"""
run_p2_disjoint_windows.py — the protocol contrast on forward test windows that do not overlap.

WHY. The four forward cut-offs of the article (run_p2_temporal_strict.py --cut 0.5/0.6/0.7/0.8)
are nested: every test window runs to the corpus' last detection date, so a later window lies
inside the earlier ones and agreement across cuts is agreement on shared rows. This arm cuts the
dated phishing rows at the same four fractions and scores each block between two cuts on its own:

    block [0.5, 0.6)   trained on the oldest 50% of dated phishing rows
    block [0.6, 0.7)   trained on the oldest 60%
    block [0.7, 0.8)   trained on the oldest 70%
    block [0.8, 1.0]   trained on the oldest 80%  (the article's 0.80 cut, unchanged)

The four test blocks share no phishing row. Each model is trained on every dated phishing row
older than its block, so the training sets are nested and the blocks are different periods of one
corpus, not independent corpora. Everything else is the guard- and prior-matched contrast of
run_p2_temporal_strict.py: the registrable-domain guard on the test block, a random control that
re-splits the same training and test rows with the same guard and has its benign test rows thinned
to the temporal arm's prior, the benign pool split row by row at the block's training fraction,
five seeds. The last block reproduces the 0.80 cut of the article row for row, which this script
checks when that log is present.

Three of the blocks hold about a tenth of the dated phishing rows each, so their test prior is
far below the article's. Each temporal fit is therefore scored a second time with its benign test
rows thinned to the canonical test prior of the same seed (protocol temporal_canonprior), so that
a comparison of two families at tau = 0.5 across blocks is not a comparison across priors.

This arm was added after the nested cut-offs had been run and read. It is exploratory.

RUN: OPENBLAS_NUM_THREADS=1 python scripts/run_p2_disjoint_windows.py
OUT: data/processed/p2/revision/disjoint_windows.csv  (one row per block, family, protocol, seed)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, _metrics
from run_p2_benchmark import FAMILIES, run_one
from run_p2_temporal_strict import benign_mask, load, match_prior, split_phishing

OUT = "data/processed/p2/revision/disjoint_windows.csv"
NESTED_080 = "data/processed/p2/revision/windows_cut0.8.csv"
BLOCKS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
CANON_CUT = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.chdir(ROOT)

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    X = lambda d: d[feats].to_numpy(float)
    Y = lambda d: d["y"].to_numpy(int)

    # the canonical test prior of each seed (cut 0.70), the target of temporal_canonprior
    canon_te = split_phishing(ph, CANON_CUT)[1]
    canon_prior = {}
    for s in range(args.seeds):
        nb = int((~benign_mask(be, np.random.RandomState(s), CANON_CUT)).sum())
        canon_prior[s] = len(canon_te) / (len(canon_te) + nb)

    rows = []
    for a, b in BLOCKS:
        k0, k1 = int(len(ph) * a), int(len(ph) * b) if b < 1.0 else len(ph)
        ph_tr, blk = ph.iloc[:k0], ph.iloc[k0:k1]
        leaked = blk.rdom.isin(set(ph_tr.rdom))
        ph_te = blk[~leaked]
        info = dict(block=f"{a:.1f}-{b:.1f}", train_frac=a, n_train_phish=len(ph_tr),
                    n_test_phish=len(ph_te), n_guard_dropped=int(leaked.sum()),
                    train_last=str(ph_tr.date.max().date()),
                    test_first=str(ph_te.date.min().date()), test_last=str(ph_te.date.max().date()))
        print(f"block [{a:.1f}, {b:.1f}]: train {len(ph_tr)} (to {info['train_last']}), test "
              f"{len(ph_te)} ({info['test_first']} to {info['test_last']}; {info['n_guard_dropped']}"
              f" dropped by the guard)", flush=True)
        for name in args.families:
            for s in range(args.seeds):
                # the same draws, in the same order, as run_p2_temporal_strict --guard-control
                # --match-prior at cut a, with the block in place of the open-ended test window
                rng = np.random.RandomState(s)
                bm = benign_mask(be, rng, a)
                tr, te = pd.concat([ph_tr, be[bm]]), pd.concat([ph_te, be[~bm]])
                pool = pd.concat([ph_tr, ph_te])
                pm = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                p_tr, p_te = pool[pm], pool[~pm]
                p_te = p_te[~p_te.rdom.isin(set(p_tr.rdom))]
                prior = len(ph_te) / (len(ph_te) + int((~bm).sum()))
                c_tr = pd.concat([p_tr, be[bm]])
                c_te = match_prior(pd.concat([p_te, be[~bm]]), prior, rng)
                met, y_t, sc = run_one(name, X(tr), Y(tr), X(te), Y(te), s, return_scores=True)
                rows.append(dict(met, protocol="temporal", test_prior=float(Y(te).mean()), **info))
                # the same fit, scored with its benign test rows thinned to the canonical prior; a
                # separate generator, so the draws above stay those of the nested runs
                n_ph, n_be = len(ph_te), int((~bm).sum())
                k = int(round(n_ph * (1 - canon_prior[s]) / canon_prior[s]))
                pick = np.arange(n_be) if k >= n_be else \
                    np.random.RandomState(10_000 + s).choice(n_be, k, replace=False)
                keep = np.r_[np.ones(n_ph, bool), np.isin(np.arange(n_be), pick)]
                m2 = _metrics(y_t[keep], sc[keep])
                m2.update(family=name, seed=s, fit_seconds=met["fit_seconds"])
                rows.append(dict(m2, protocol="temporal_canonprior",
                                 test_prior=float(y_t[keep].mean()), **info))
                mc = run_one(name, X(c_tr), Y(c_tr), X(c_te), Y(c_te), s)
                rows.append(dict(mc, protocol="random_guarded_pm",
                                 test_prior=float(Y(c_te).mean()), **info))
            print(f"  {name:<13} done", flush=True)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} rows -> {args.out}")
    g = out.groupby(["block", "family", "protocol"]).F1.mean().unstack()
    print((g["temporal"] - g["random_guarded_pm"]).unstack(0).round(4).to_string())

    # the last block is the article's 0.80 cut: the two logs must agree row for row
    if os.path.exists(NESTED_080) and set(args.families) == set(FAMILIES):
        n = pd.read_csv(NESTED_080)
        n["protocol"] = n.protocol.map({"temporal_strict": "temporal",
                                        "random_same_rows_guarded": "random_guarded_pm"})
        m = out[out.block == "0.8-1.0"].merge(n, on=["family", "protocol", "seed"],
                                              suffixes=("", "_nested"))
        worst = max(float((m[c] - m[c + "_nested"]).abs().max()) for c in ("F1", "ROC-AUC"))
        print(f"[check] block 0.8-1.0 against the nested 0.80 cut: {len(m)} rows, "
              f"largest difference {worst:.6f}")
        if len(m) != 2 * len(FAMILIES) * args.seeds or worst > 2e-3:
            raise SystemExit("the last block does not reproduce the nested 0.80 cut")


if __name__ == "__main__":
    main()
