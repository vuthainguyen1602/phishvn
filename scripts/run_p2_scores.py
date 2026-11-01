#!/usr/bin/env python3
"""
run_p2_scores.py — per-row scores of the primary protocol contrast, for the analyses that need
more than a summary metric (revision 1, 2026-10-31).

WHY. Several questions about the protocol contrast cannot be answered from the per-run metrics the
other runners keep: calibration (Brier, ECE), precision at deployment prevalences, recall under a
strict false-positive budget, false positives per 10,000 benign URLs at a threshold chosen without
the test window, uncertainty resampled by registrable domain rather than by seed, and the effect
of plausible label errors in the test window. Each of these is a function of (label, score) on
the test rows, so this runner stores those once and analyse_p2_scores.py reads them.

DESIGN. The primary contrast: phishing-temporal against the random control that carries the same
registrable-domain guard and the same test prior (run_p2_temporal_strict.py --guard-control
--match-prior), on the same dated rows, seven families, the canonical seeds and the canonical
0.70 cut. Every model is fitted on the training window minus a validation slice carved as in
run_p2_val_threshold.py (the most recent 10% of training phishing plus a random 10% of training
benign under the temporal protocol, a stratified 10% under the random one), so a threshold read
off the validation slice is one a practitioner could have fixed before the test window. The
fitted model is therefore NOT the full-window model of Table tab:strict; its metrics differ from
that table's in the fit slice alone.

Writes data/processed/p2/revision/p2_scores.csv.gz: one row per (family, protocol, seed, part,
test or validation row) with y, score, registrable domain, tier and source; and
p2_scores_runs.csv: one row per run with the validation threshold and row counts.

RUN:
  OPENBLAS_NUM_THREADS=1 python scripts/run_p2_scores.py --seeds 5
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
from train_url_baseline import COMPPHISH
from run_p2_benchmark import FAMILIES, make_any_model, weighting_params
from run_p2_temporal_strict import URL_CSV, benign_mask, load, match_prior, split_phishing
from run_p2_val_threshold import best_threshold, carve

OUTDIR = "data/processed/p2/revision"
PROTOCOLS = ("temporal_strict", "random_same_rows_guarded")


def splits(ph_tr, ph_te, be, seed, cut=0.70):
    """(train, test) for both protocols of one seed, consuming the rng exactly as
    run_p2_temporal_strict.py --guard-control --match-prior does, so the rows match its CSV."""
    out = {}
    for proto in PROTOCOLS:
        rng = np.random.RandomState(seed)
        bmask = benign_mask(be, rng, cut)
        if proto == "temporal_strict":
            out[proto] = (pd.concat([ph_tr, be[bmask]]), pd.concat([ph_te, be[~bmask]]))
            continue
        pool = pd.concat([ph_tr, ph_te])
        pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
        p_tr, p_te = pool[pmask], pool[~pmask]
        p_te = p_te[~p_te.rdom.isin(set(p_tr.rdom))]
        te = pd.concat([p_te, be[~bmask]])
        te = match_prior(te, len(ph_te) / (len(ph_te) + int((~bmask).sum())), rng)
        out[proto] = (pd.concat([p_tr, be[bmask]]), te)
    return out


def load_with_meta() -> pd.DataFrame:
    df = load()
    # tier is already in the aligned corpus; source is not
    meta = pd.read_csv(URL_CSV, usecols=["url_norm", "source"], dtype=str)
    meta = meta.drop_duplicates("url_norm").rename(columns={"url_norm": "url"})
    return df.merge(meta, on="url", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_with_meta()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)

    keep = ["y", "rdom", "tier", "source"]
    parts, runs = [], []
    for s in range(args.seeds):
        for proto, (tr, te) in splits(ph_tr, ph_te, be, s).items():
            fit, val = carve(tr, proto, s)
            for name in args.families:
                m = make_any_model(name, s, weighting_params(name))
                m.fit(fit[feats].to_numpy(float), fit.y.to_numpy(int))
                sv = m.predict_proba(val[feats].to_numpy(float))[:, 1]
                st = m.predict_proba(te[feats].to_numpy(float))[:, 1]
                tau = best_threshold(val.y.to_numpy(int), sv)
                for part, frame, sc in (("val", val, sv), ("test", te, st)):
                    p = frame[keep].copy()
                    p["score"] = sc.astype(np.float32)
                    p["family"], p["protocol"], p["seed"], p["part"] = name, proto, s, part
                    parts.append(p)
                runs.append({"family": name, "protocol": proto, "seed": s, "tau_val": tau,
                             "n_fit": len(fit), "n_val": len(val), "n_test": len(te),
                             "test_prior": float(te.y.mean())})
                print(f"  {name:<13} {proto:<25} seed={s} tau*={tau:.3f} n_test={len(te)}",
                      flush=True)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(os.path.join(args.outdir, "p2_scores.csv.gz"), index=False, float_format="%.6g")
    pd.DataFrame(runs).to_csv(os.path.join(args.outdir, "p2_scores_runs.csv"), index=False)
    print(f"[+] {len(out)} scored rows, {len(runs)} runs -> {args.outdir}")


if __name__ == "__main__":
    main()
