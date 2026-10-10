#!/usr/bin/env python3
"""
run_p2_temporal_twosided.py — the temporal protocol with BOTH classes cut by the calendar.

WHY. The phishing-temporal protocol of run_p2_temporal_strict cuts the phishing class by
detection date and splits the benign class at random, because the benign class is "essentially
undated". Essentially is not entirely: 2,026 benign rows (all gold tier) carry a collection
date, 1,116 of them on or before the phishing cut. Limitation (i) promised a two-sided split
once a dated benign stream exists; this runs the two-sided split on the rows that already have
dates, so the promise is a measurement with a stated sample size rather than a deferral.

DESIGN. Rows = dated phishing (19,635) + dated benign (2,026). One calendar cut for both classes,
the canonical phishing cut (the date at which the oldest 70% of dated phishing ends, 2022-11-15):
train = rows on or before it, test = rows strictly after, and the registrable-domain guard of
Algorithm 1 applied to the test side of BOTH classes. The control is the same rows re-split at
random per class at the same rate, with the same guard applied after the re-split, so the two
arms differ in protocol alone. Because both classes are date-fixed, the temporal arm varies only
by model seed; the control redraws its split per seed.

WHAT IT CAN AND CANNOT SAY. The test window is 87% phishing (5,9k phishing against ~0,9k benign),
so F1 at tau = 0.5 sits on a different floor from every other design in the paper and is printed
for completeness only; the protocol step is read on ROC-AUC, PR-AUC and FPR at 0.90 recall,
which the class prior does not move in the same way. And the dated benign rows are gold-tier
registry entries, not a sample of the benign class, so this is a sensitivity arm for the
one-sided protocol, not a replacement for it.

RUN:
  OPENBLAS_NUM_THREADS=1 python scripts/run_p2_temporal_twosided.py --seeds 5
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import COMPPHISH
from run_p2_benchmark import FAMILIES, refuse_canonical, run_one
from run_p2_temporal_strict import load

OUT = "data/processed/p2/p2_temporal_twosided.csv"


def guard(te: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Drop test rows whose registrable domain occurs anywhere in train (both classes)."""
    return te[~te.rdom.isin(set(tr.rdom))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.70,
                    help="phishing train fraction that defines the calendar cut date")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    refuse_canonical(a.out, OUT)
    os.chdir(ROOT)

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    dated = df[df.date.notna()].sort_values("date").reset_index(drop=True)
    ph = dated[dated.y == 1]
    be = dated[dated.y == 0]
    cut_date = ph.date.iloc[int(len(ph) * a.cut) - 1]      # the canonical phishing cut
    tr_all = dated[dated.date <= cut_date]
    te_all = guard(dated[dated.date > cut_date], tr_all)
    n_dropped = int((dated.date > cut_date).sum()) - len(te_all)
    print(f"cut {cut_date.date()}: train {len(tr_all)} (phish {int(tr_all.y.sum())}, benign "
          f"{int((tr_all.y == 0).sum())}), test {len(te_all)} (phish {int(te_all.y.sum())}, benign "
          f"{int((te_all.y == 0).sum())}); {n_dropped} test rows dropped by the domain guard")
    rate = {1: (tr_all.y == 1).sum() / len(ph), 0: (tr_all.y == 0).sum() / len(be)}

    rows = []
    for name in a.families:
        for proto in ("temporal_both", "random_both_guarded"):
            for s in range(a.seeds):
                if proto == "temporal_both":
                    tr, te = tr_all, te_all
                else:
                    rng = np.random.RandomState(s)
                    mask = np.array([rng.rand() < rate[int(y)] for y in dated.y])
                    tr, te = dated[mask], guard(dated[~mask], dated[mask])
                met, _, _ = run_one(name, tr[feats].to_numpy(float), tr.y.to_numpy(int),
                                    te[feats].to_numpy(float), te.y.to_numpy(int), s,
                                    return_scores=True)
                met.update(protocol=proto, n_train=len(tr), n_test=len(te),
                           n_test_benign=int((te.y == 0).sum()), cut_date=str(cut_date.date()))
                rows.append(met)
                print(f"  {name:<13} {proto:<20} seed={s} F1={met['F1']:.3f} "
                      f"ROC={met['ROC-AUC']:.3f} PR={met['PR-AUC']:.3f} "
                      f"FPR@0.90={met['FPR@R0.90']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


if __name__ == "__main__":
    main()
