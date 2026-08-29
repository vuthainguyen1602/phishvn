#!/usr/bin/env python3
"""
run_p2_temporal_strict.py — P2's honest temporal protocol, built only from rows whose detection
date is real.

WHY. normalize_merge's `split` column is only PARTLY temporal — undated sources are split
randomly, so run_p2_benchmark's "temporal" protocol tracked "random" almost exactly, a property
of the split rather than evidence of robustness. Here the phishing class is split STRICTLY by
`collected_at` (train = older 70% of dated phishing, test = strictly newer 30%, undated dropped);
that is the class that drifts. The benign class is barely dated (~2k of 33k rows) and is split
randomly per seed at the same rate — stated in the paper, and standard in the drift literature.

LEAKAGE GUARD: a registrable domain seen in train is removed from test (kept in train), else the
model scores "memorised domain" rather than "generalises forward".

The comparison that matters is strict-temporal vs random ON THE SAME ROW SET, so protocol is the
only variable; rows carry protocol "random_same_rows" and "temporal_strict".

ONE ASYMMETRY, AND THE ARM THAT CLOSES IT (round-2 review, M7). The temporal arm applies the
domain guard and the random control does not, so ~7% of the control's phishing test window shares
a domain with its own training window (scripts/p2_dup_leakage.py), inflating the control
and Delta_proto. --guard-control adds "random_same_rows_guarded", applying the identical guard
after the random re-split, into its own CSV so the canonical two-arm table is undisturbed.

REFRESH WINDOW (PREREG_refresh_window.md, Test 1). --test-after YYYY-MM-DD replaces the
fractional cut with a calendar one: test = dated phishing strictly later, train = ALL dated
phishing on or before, same guard. Set to the corpus maximum the test window is exactly what the
next refresh brings in; set to the canonical cut (2022-11-15) it reproduces the 0.70 split up to
the boundary tie (smoke test). Benign masks and model configs unchanged.

RUN:
  python scripts/run_p2_temporal_strict.py                 # 7 families x 2 protocols x seeds
  python scripts/run_p2_temporal_strict.py --guard-control \\
      --out data/processed/p2/p2_temporal_strict_guarded.csv --curves ""   # the M7 control arm
  python scripts/run_p2_temporal_strict.py --families CatBoost --seeds 20 \\
      --test-after 2025-02-18 --out data/processed/p2/p2_refresh_cb_k20.csv --curves ""  # Test 1
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label
from run_p2_benchmark import (DETERMINISTIC, FAMILIES, pr_curve_row, run_one,
                              write_curves)
from psl import registered_domain

URL_CSV = "data/processed/dataset_url.csv"
ALIGNED = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2/p2_temporal_strict.csv"
CURVES = "data/processed/p2/p2_pr_curves_strict.csv"


def parse_date(s):
    """dd/mm/yyyy first, then ISO — the same order normalize_merge uses; one format alone
    silently drops half the corpus (the trap check_paper_claims documents)."""
    s = (s if isinstance(s, str) else "").strip()[:10]
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def load():
    meta = pd.read_csv(URL_CSV, usecols=["url_norm", "label", "collected_at"], dtype=str)
    meta["date"] = meta["collected_at"].map(parse_date)
    df = add_label(pd.read_csv(ALIGNED))
    df = df.merge(meta.rename(columns={"url_norm": "url"})[["url", "date"]],
                  on="url", how="left")
    df = df.dropna(subset=[c for c in COMPPHISH if c in df.columns])
    df["rdom"] = df["dom"].astype(str).str.lower().map(registered_domain)
    return df


def split_phishing(ph: pd.DataFrame, cut: float, test_after: str | None = None):
    """The phishing time split plus the registrable-domain guard, shared by every P2 runner.

    `ph` is the dated phishing set sorted by date. Fractional `cut` is the canonical rolling-
    origin split (train = the first `cut` of rows). `test_after` (YYYY-MM-DD) is the registered
    refresh window: train = every row dated ON OR BEFORE the date, test = rows STRICTLY later.
    In both modes a test row whose registrable domain occurs anywhere in train is dropped.
    Returns (ph_tr, ph_te, n_leaked)."""
    if test_after:
        d = pd.Timestamp(test_after)
        ph_tr, ph_te = ph[ph.date <= d], ph[ph.date > d]
    else:
        k = int(len(ph) * cut)
        ph_tr, ph_te = ph.iloc[:k], ph.iloc[k:]
    leaked = ph_te.rdom.isin(set(ph_tr.rdom))
    return ph_tr, ph_te[~leaked], int(leaked.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.70,
                    help="rolling-origin robustness: train fraction for the phishing time cut "
                         "AND the benign mask rate (0.70 = the canonical split)")
    ap.add_argument("--test-after", default=None, metavar="YYYY-MM-DD",
                    help="refresh window (PREREG_refresh_window.md): test = dated phishing "
                         "strictly after this date, train = all dated phishing on or before it; "
                         "overrides the fractional --cut for the phishing class only")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--curves", default=CURVES)
    ap.add_argument("--guard-control", action="store_true",
                    help="run the M7 arm: strict-temporal against a random control that carries "
                         "the SAME registrable-domain guard, so protocol is genuinely the only "
                         "variable. Writes protocols temporal_strict + random_same_rows_guarded.")
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, args.test_after)
    if ph_te.empty:
        raise SystemExit(f"no dated phishing after {args.test_after}: the refresh window has "
                         f"not arrived (corpus max date {ph.date.max().date()})")
    print(f"phishing dated: {len(ph)}  train<= {ph_tr.date.max().date()} ({len(ph_tr)})  "
          f"test>= {ph_te.date.min().date()} ({len(ph_te)}; {n_leaked} test rows dropped as "
          f"same-registrable-domain re-detections)")
    print(f"benign pool: {len(be)} (random 70/30 per seed)")

    protos = ("temporal_strict", "random_same_rows_guarded") if args.guard_control \
        else ("temporal_strict", "random_same_rows")
    rows, curves = [], []
    for name in args.families:
        for proto in protos:
            # even "deterministic" families need all seeds here: the benign 70/30 mask is drawn
            # per seed, so no family is deterministic under either protocol
            for s in range(args.seeds):
                rng = np.random.RandomState(s)
                bmask = rng.rand(len(be)) < args.cut
                if proto == "temporal_strict":
                    tr = pd.concat([ph_tr, be[bmask]])
                    te = pd.concat([ph_te, be[~bmask]])
                else:
                    pool = pd.concat([ph_tr, ph_te])
                    pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                    p_tr, p_te = pool[pmask], pool[~pmask]
                    if proto == "random_same_rows_guarded":
                        # The temporal arm's guard, applied to the random control: drop from TEST
                        # any row whose registrable domain occurs in TRAIN. Same direction, same
                        # granularity, same PSL logic -- so the only remaining difference between
                        # the two arms is how rows were assigned.
                        p_te = p_te[~p_te.rdom.isin(set(p_tr.rdom))]
                    tr = pd.concat([p_tr, be[bmask]])
                    te = pd.concat([p_te, be[~bmask]])
                met, y_t, sc = run_one(name, tr[feats].to_numpy(float),
                                       tr["y"].to_numpy(int), te[feats].to_numpy(float),
                                       te["y"].to_numpy(int), s, return_scores=True)
                met["protocol"] = proto
                rows.append(met)
                curves.append({"family": name, "protocol": proto, "seed": s,
                               "precision": pr_curve_row(y_t, sc)})
                print(f"  {name:<13} {proto:<17} seed={s} F1={met['F1']:.3f} "
                      f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


if __name__ == "__main__":
    main()
