#!/usr/bin/env python3
"""
run_p2_prior_control.py — the composition step with the class prior held fixed.

WHY. Table tab:decomp's composition step changes two things at once: WHICH phishing rows are
eligible (the undated mass drops out) and the class prior (the benign pool stays at 16,410 while
phishing falls from 36,706 to 19,635, so a 69%-phishing test set becomes 55%). The manuscript
said a prior-controlled resampling experiment "would be needed" to separate them. This is that
experiment, as a 2 x 2: {full-corpus phishing, dated phishing only} x {prior 0.69, prior 0.55},
every arm a random stratified 70/30 split so the protocol never varies.

  full_natural   all 36,706 phishing + all 16,410 benign            (the paper's random design)
  full_matched   a random 19,635 of the 36,706 phishing + all benign (prior matched to dated)
  dated_natural  the 19,635 dated phishing + all benign             (the paper's dated-random)
  dated_matched  the 19,635 dated phishing + a random 8,778 benign  (prior matched to full)

Reading the square: full_matched - full_natural and dated_matched - dated_natural are the PRIOR
effect at fixed composition; dated_natural - full_matched and dated_matched - full_natural are
the COMPOSITION effect at fixed prior. Subsamples are redrawn per seed. Every row also records
the oracle max-F1 on its own test curve, so the fixed-threshold and chosen-threshold readings
of Section 6.2 can both be repeated on it.

The full arms carry no domain guard, deliberately: they reproduce the design under study.

RUN:
  python scripts/run_p2_prior_control.py --seeds 5
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import COMPPHISH
from run_p2_benchmark import FAMILIES, run_one
from run_p2_temporal_strict import load

OUT = "data/processed/p2/p2_prior_control.csv"
ARMS = ("full_natural", "full_matched", "dated_natural", "dated_matched")


def max_f1(y, score) -> float:
    p, r, _ = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.nanmax(f))


def draw(arm: str, ph_all: pd.DataFrame, ph_dated: pd.DataFrame, be: pd.DataFrame,
         rng: np.random.RandomState) -> pd.DataFrame:
    """The row set an arm evaluates, before the 70/30 split."""
    n_dated, n_be = len(ph_dated), len(be)
    prior_full = len(ph_all) / (len(ph_all) + n_be)
    if arm == "full_natural":
        return pd.concat([ph_all, be])
    if arm == "full_matched":
        idx = rng.choice(len(ph_all), n_dated, replace=False)
        return pd.concat([ph_all.iloc[np.sort(idx)], be])
    if arm == "dated_natural":
        return pd.concat([ph_dated, be])
    if arm == "dated_matched":
        n_keep = int(round(n_dated * (1 - prior_full) / prior_full))
        idx = rng.choice(n_be, n_keep, replace=False)
        return pd.concat([ph_dated, be.iloc[np.sort(idx)]])
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.chdir(ROOT)

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph_all = df[df.y == 1].reset_index(drop=True)
    ph_dated = ph_all[ph_all.date.notna()].reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    print(f"phishing {len(ph_all)} (dated {len(ph_dated)}), benign {len(be)}; "
          f"prior full {len(ph_all)/(len(ph_all)+len(be)):.3f}, "
          f"dated {len(ph_dated)/(len(ph_dated)+len(be)):.3f}")

    rows = []
    for arm in ARMS:
        for s in range(a.seeds):
            pool = draw(arm, ph_all, ph_dated, be, np.random.RandomState(1000 + s))
            X, y = pool[feats].to_numpy(float), pool["y"].to_numpy(int)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                                  random_state=s)
            for name in a.families:
                met, y_t, sc = run_one(name, Xtr, ytr, Xte, yte, s, return_scores=True)
                met.update(arm=arm, prior=round(float(y.mean()), 4), n_rows=len(pool),
                           n_phish=int(y.sum()), n_benign=int(len(y) - y.sum()),
                           n_test=len(yte), maxF1=max_f1(y_t, sc))
                rows.append(met)
                print(f"  {arm:<14} seed={s} {name:<13} F1={met['F1']:.3f} "
                      f"maxF1={met['maxF1']:.3f} PR-AUC={met['PR-AUC']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["arm", "family"])[["F1", "maxF1", "PR-AUC"]].mean().round(4)
             .unstack("arm").to_string())


if __name__ == "__main__":
    main()
