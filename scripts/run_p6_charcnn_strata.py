#!/usr/bin/env python3
"""
run_p6_charcnn_strata.py — does the suffix-length blind spot survive a change of representation?

WHY. P6's finding is a mechanism as much as a rate: the detector misses two-character-suffix
phishing because `tld_len` carries a "short suffix implies benign" prior, and pinning that one
feature moves most of the missed rows across the threshold. `tld_len` is a column of the
hand-built 21-feature schema. A character-CNN reading the raw URL string has no such column, so
the obvious question is whether it inherits the blind spot or not, and the answer changes what P6
is entitled to claim:

  * it inherits it -> the blind spot is a property of the TASK on this corpus, and P6 is stronger
    than it currently claims, because a representation with no suffix feature reproduces it;
  * it does not -> the blind spot belongs to the hand-built schema, and P6 must scope its claim
    the way the companion benchmark scoped its own on 2026-08-29.

Either answer is worth having. Neither is assumed here.

THE STRATA ARE P6'S, IMPORTED. `strata_of` and `vn_group` come from run_p6_suffix_blindspot and
run_p6_group_threshold rather than being redefined, so "two-character suffix" means exactly what
the published table means. The split, seeds and benign mask are the phishing-temporal ones every
other arm uses, so the row sets are identical and the comparison is paired by stratum.

The decision threshold is 0.5, the deployed point P6 reports its rates at. A model can only be
said to share a blind spot at the operating point where the blind spot was measured.

RUN:  python scripts/run_p6_charcnn_strata.py --seeds 3
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(os.path.dirname(_HERE))

from train_url_baseline import COMPPHISH
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load, split_phishing
from run_p6_suffix_blindspot import strata_of, STRATA
from run_p2_charcnn import fit_predict


def rates(te: pd.DataFrame, score: np.ndarray, tag: str, seed: int) -> list[dict]:
    """Per-stratum false-negative rate at the deployed threshold, plus the sizes behind it."""
    st = strata_of(te)
    y = te["y"].to_numpy(int)
    pred = (score >= 0.5).astype(int)
    out = []
    for name in STRATA:
        m = (st == name).to_numpy()
        ph = m & (y == 1)
        if ph.sum() == 0:
            continue
        out.append({"model": tag, "seed": seed, "stratum": name,
                    "n_phish": int(ph.sum()),
                    "FNR": float((pred[ph] == 0).mean()),
                    "n_benign": int((m & (y == 0)).sum())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--reference", default="CatBoost")
    ap.add_argument("--out", default="data/processed/p6/p6_charcnn_strata.csv")
    a = ap.parse_args()
    os.chdir(ROOT)

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, a.cut)

    rows = []
    for s in range(a.seeds):
        rng = np.random.RandomState(s)
        bmask = rng.rand(len(be)) < a.cut
        tr = pd.concat([ph_tr, be[bmask]])
        te = pd.concat([ph_te, be[~bmask]])

        t0 = time.time()
        sc = fit_predict(tr["url"].tolist(), tr["y"].to_numpy(int), te["url"].tolist(), s, a.epochs)
        rows += rates(te, sc, "CharCNN", s)
        print(f"  CharCNN  seed={s} ({time.time()-t0:.0f}s)")

        m = make_any_model(a.reference, s)
        m.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))
        rows += rates(te, m.predict_proba(te[feats].to_numpy(float))[:, 1], a.reference, s)
        print(f"  {a.reference:<8} seed={s}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print("\n  per-stratum FNR at tau=0.5, mean over seeds:")
    piv = out.pivot_table(index="stratum", columns="model", values="FNR", aggfunc="mean")
    n = out.groupby("stratum")["n_phish"].mean().astype(int)
    for st in STRATA:
        if st in piv.index:
            print(f"    {st:<13} n={n[st]:>5}  " +
                  "  ".join(f"{c} {piv.loc[st, c]:.3f}" for c in piv.columns))
    print(f"[+] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
