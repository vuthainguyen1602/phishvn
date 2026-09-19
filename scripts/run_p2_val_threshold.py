#!/usr/bin/env python3
"""
run_p2_val_threshold.py — the decomposition at a threshold chosen on validation data.

WHY. Table tab:maxf1 reads each family's best F1 off its TEST precision--recall curve. That is
an oracle: it says how much of the thresholded spread is threshold placement, and it says
nothing about what a practitioner who cannot see the test window would obtain. The manuscript
says so and calls the validation-selected threshold "needed". This supplies it.

DESIGN. The three same-corpus designs of the decomposition, seven families, the canonical seeds.
For each (design, family, seed) the training window is split 90/10 into a fit slice and a
validation slice; the model is fitted on the 90%, tau* is the threshold that maximises F1 on
the validation slice, and the test window is scored once at tau*. The same fitted model is also
scored at tau = 0.5 and at its oracle test-curve maximum, so the three readings differ in the
threshold alone and never in the model.

The validation slice respects the design. Under phishing-temporal it is the most recent 10% of
the phishing training window by detection date plus a random 10% of the benign training rows,
so nothing later than the fit slice's last date informs tau*; under the two random designs it is
a stratified random 10%. A row in the test window is never used for anything.

RUN:
  python scripts/run_p2_val_threshold.py --seeds 5
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import COMPPHISH, _metrics
from run_p2_benchmark import FAMILIES, make_any_model
from run_p2_temporal_strict import load, split_phishing

OUT = "data/processed/p2/p2_val_threshold.csv"
DESIGNS = ("random_full", "random_same_rows", "temporal_strict")
VAL_FRAC = 0.10


def best_threshold(y, score) -> float:
    """The threshold maximising F1 on (y, score): the midpoint convention is not needed because
    the score is applied with >= exactly as _metrics does."""
    p, r, thr = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)[:-1]
    return float(thr[int(np.nanargmax(f))])


def max_f1(y, score) -> float:
    p, r, _ = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.nanmax(f))


def carve(tr: pd.DataFrame, design: str, seed: int):
    """(fit, val) from the training window, respecting the design."""
    rng = np.random.RandomState(10_000 + seed)
    if design == "temporal_strict":
        ph = tr[tr.y == 1].sort_values("date")
        be = tr[tr.y == 0]
        k = int(len(ph) * (1 - VAL_FRAC))
        bm = rng.rand(len(be)) < VAL_FRAC
        fit = pd.concat([ph.iloc[:k], be[~bm]])
        val = pd.concat([ph.iloc[k:], be[bm]])
        return fit, val
    fit, val = train_test_split(tr, test_size=VAL_FRAC, stratify=tr.y, random_state=seed)
    return fit, val


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.chdir(ROOT)

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, a.cut)

    rows = []
    for design in DESIGNS:
        for s in range(a.seeds):
            if design == "random_full":
                tr, te = train_test_split(df, test_size=0.30, stratify=df.y, random_state=s)
            else:
                rng = np.random.RandomState(s)          # the canonical per-seed benign mask
                bmask = rng.rand(len(be)) < a.cut
                if design == "temporal_strict":
                    tr = pd.concat([ph_tr, be[bmask]])
                    te = pd.concat([ph_te, be[~bmask]])
                else:
                    pool = pd.concat([ph_tr, ph_te])
                    pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                    tr = pd.concat([pool[pmask], be[bmask]])
                    te = pd.concat([pool[~pmask], be[~bmask]])
            fit, val = carve(tr, design, s)
            for name in a.families:
                m = make_any_model(name, s)
                m.fit(fit[feats].to_numpy(float), fit.y.to_numpy(int))
                sv = m.predict_proba(val[feats].to_numpy(float))[:, 1]
                tau = best_threshold(val.y.to_numpy(int), sv)
                st = m.predict_proba(te[feats].to_numpy(float))[:, 1]
                yte = te.y.to_numpy(int)
                met = _metrics(yte, st)                      # F1 here is at tau = 0.5
                met.update(family=name, seed=s, protocol=design, tau=round(tau, 4),
                           F1_val=f1_score(yte, (st >= tau).astype(int)),
                           F1_oracle=max_f1(yte, st),
                           n_fit=len(fit), n_val=len(val), n_test=len(te),
                           val_last_date=str(val[val.y == 1].date.max().date())
                           if design == "temporal_strict" else "")
                rows.append(met)
                print(f"  {design:<17} seed={s} {name:<13} tau*={tau:.3f} "
                      f"F1@0.5={met['F1']:.3f} F1@tau*={met['F1_val']:.3f} "
                      f"oracle={met['F1_oracle']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["protocol", "family"])[["F1", "F1_val", "F1_oracle"]].mean().round(4)
             .to_string())


if __name__ == "__main__":
    main()
