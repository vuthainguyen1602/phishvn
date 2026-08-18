#!/usr/bin/env python3
"""
run_p2_drift_forecastability.py — Can the temporal gap be anticipated from inside the training
window? Three diagnostics, one shared split infrastructure.

WHY THIS EXISTS. The pseudo-future family of drift-aware methods (dynamic expert selection,
decay-extrapolated weighting, gated backoff, temporally consistent stacking OOF) all assume the
drift visible INSIDE the train window resembles the drift ACROSS the train->test boundary. The
gatekeeper run of 2026-08-17 found no such resemblance; this script hardens that into a
paper-grade result and asks the two follow-up questions the finding raises.

  E1 — internal decay curves. Fit each family on the oldest fraction of the phishing train
       window, evaluate on 4 successive forward windows (registrable-domain guard mirrors the
       outer protocol; fixed benign eval set so only phishing drifts). Two origins (0.50, 0.60
       of ph_tr) and two metrics (PR-AUC, ROC-AUC) so the conclusion doesn't hang on one
       windowing or one prevalence-sensitive metric. The question: does the internal decay
       ranking reproduce the canonical random-minus-temporal gap ranking? (Spearman, per
       origin x metric.)
  E2 — shift-localization matrix. A phishing-only discriminator (HistGB, 5-fold OOF AUC) for
       every pair of time blocks: 4 blocks inside ph_tr (first detection per registrable
       domain only) plus ph_te. If shift accumulates smoothly, pair AUC grows with time
       distance; if the boundary is a regime change, internal pairs stay near 0.5 while
       (W4, test) jumps despite the shortest time distance.
  E3 — instance-level novelty diagnostic. Does train-only novelty (mean scaled kNN distance,
       k=10) predict WHERE CatBoost fails on the test window? AUC(u; wrong vs correct), all
       rows and phishing-only, CatBoost and LogReg. Pre-set decision rule from the review:
       AUC >= 0.65 keeps the gated-backoff idea alive as future work; below buries it.

RUN:
  python scripts/train/run_p2_drift_forecastability.py            # all three, 5 seeds
  python scripts/train/run_p2_drift_forecastability.py --only e2  # one diagnostic
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
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH  # noqa: E402
from run_p2_benchmark import FAMILIES, make_any_model  # noqa: E402
from run_p2_temporal_strict import load  # noqa: E402

OUT_DECAY = "data/processed/p2_forecastability_decay.csv"
OUT_MATRIX = "data/processed/p2_forecastability_shiftmatrix.csv"
OUT_NOVELTY = "data/processed/p2_forecastability_novelty.csv"
N_WIN = 4


def split_canonical(df, cut=0.70):
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    k = int(len(ph) * cut)
    ph_tr, ph_te = ph.iloc[:k], ph.iloc[k:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    return feats, ph_tr.reset_index(drop=True), ph_te.reset_index(drop=True), be


def guarded_windows(ph_rest, seen):
    """Contiguous time windows with the outer protocol's registrable-domain guard applied
    cumulatively: a window keeps only first-seen domains."""
    wins, seen = [], set(seen)
    for i in range(N_WIN):
        lo, hi = int(len(ph_rest) * i / N_WIN), int(len(ph_rest) * (i + 1) / N_WIN)
        w = ph_rest.iloc[lo:hi]
        wins.append(w[~w.rdom.isin(seen)])
        seen |= set(w.rdom)
    return wins


def e1_decay(feats, ph_tr, be, seeds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for origin in (0.50, 0.60):
        half = int(len(ph_tr) * origin)
        ph_fit, ph_rest = ph_tr.iloc[:half], ph_tr.iloc[half:].reset_index(drop=True)
        wins = guarded_windows(ph_rest, set(ph_fit.rdom))
        t0 = ph_fit.date.max()
        horiz = [float(w.date.map(lambda d: (d - t0).days).median()) for w in wins]
        print(f"[E1] origin={origin} fit={len(ph_fit)} windows={[len(w) for w in wins]} "
              f"median horizon={[round(h) for h in horiz]}d")
        for s in seeds:
            rng = np.random.RandomState(s)
            bmask = rng.rand(len(be)) < 0.70
            be_tr, be_ev = be[bmask], be[~bmask]
            Xtr = pd.concat([ph_fit, be_tr])[feats].to_numpy(float)
            ytr = np.r_[np.ones(len(ph_fit)), np.zeros(len(be_tr))].astype(int)
            for fam in FAMILIES:
                m = make_any_model(fam, s)
                m.fit(Xtr, ytr)
                sc_be = m.predict_proba(be_ev[feats].to_numpy(float))[:, 1]
                for k, w in enumerate(wins):
                    sc_ph = m.predict_proba(w[feats].to_numpy(float))[:, 1]
                    y = np.r_[np.ones(len(w)), np.zeros(len(be_ev))]
                    sc = np.r_[sc_ph, sc_be]
                    rows.append({"origin": origin, "family": fam, "seed": s, "win": k,
                                 "horizon_d": horiz[k], "n_ph": len(w),
                                 "PR-AUC": average_precision_score(y, sc),
                                 "ROC-AUC": roc_auc_score(y, sc)})
            print(f"[E1] origin={origin} seed={s} done")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DECAY, index=False)

    # verdict: internal decay ranking vs the canonical gap ranking, per origin x metric
    from scipy.stats import spearmanr
    canon = pd.read_csv("data/processed/p2_temporal_strict.csv")
    g = canon.groupby(["family", "protocol"])["PR-AUC"].mean().unstack()
    gap = (g["random_same_rows"] - g["temporal_strict"])
    for origin in (0.50, 0.60):
        for metric in ("PR-AUC", "ROC-AUC"):
            slopes = {}
            sub = out[out.origin == origin]
            for fam in FAMILIES:
                gg = sub[sub.family == fam].groupby("win")[["horizon_d", metric]].mean()
                slopes[fam] = np.polyfit(gg["horizon_d"], gg[metric], 1)[0]
            s_ = pd.Series(slopes)
            rho, p = spearmanr(-s_[gap.index], gap)  # larger decay should mean larger gap
            print(f"[E1] origin={origin} {metric}: Spearman(internal decay vs canonical gap) "
                  f"rho={rho:+.3f} p={p:.3f}")
    print(f"[+] {len(out)} rows -> {OUT_DECAY}")


def e2_matrix(feats, ph_tr, ph_te):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    first = ph_tr.drop_duplicates("rdom", keep="first").reset_index(drop=True)
    blocks, names = [], []
    for i in range(N_WIN):
        lo, hi = int(len(first) * i / N_WIN), int(len(first) * (i + 1) / N_WIN)
        blocks.append(first.iloc[lo:hi])
        names.append(f"W{i + 1}")
    blocks.append(ph_te.drop_duplicates("rdom", keep="first"))
    names.append("TEST")
    med = [b.date.median() for b in blocks]
    rows = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            X = pd.concat([blocks[i], blocks[j]])[feats].to_numpy(float)
            y = np.r_[np.zeros(len(blocks[i])), np.ones(len(blocks[j]))].astype(int)
            oof = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y,
                                    cv=5, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, oof)
            dd = abs((med[j] - med[i]).days)
            rows.append({"a": names[i], "b": names[j], "auc": auc, "delta_days": dd,
                         "n_a": len(blocks[i]), "n_b": len(blocks[j])})
            print(f"[E2] {names[i]:>4} vs {names[j]:<4} AUC={auc:.3f} (dt={dd}d, "
                  f"n={len(blocks[i])}/{len(blocks[j])})")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_MATRIX, index=False)
    internal = out[out.a.str.startswith("W") & out.b.str.startswith("W")]
    adj_int = internal[internal.b.str[1].astype(int)
                       - internal.a.str[1].astype(int) == 1]["auc"].mean()
    bound = out[(out.a == "W4") & (out.b == "TEST")]["auc"].iloc[0]
    print(f"[E2] adjacent internal mean AUC={adj_int:.3f}  vs  W4->TEST boundary AUC={bound:.3f}")
    print(f"[+] {len(out)} pairs -> {OUT_MATRIX}")


def e3_novelty(feats, ph_tr, ph_te, be, seeds):
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    rows = []
    for s in seeds:
        rng = np.random.RandomState(s)
        bmask = rng.rand(len(be)) < 0.70
        tr = pd.concat([ph_tr, be[bmask]])
        te = pd.concat([ph_te, be[~bmask]])
        Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
        sc = StandardScaler().fit(Xtr)
        nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(Xtr))
        u = nn.kneighbors(sc.transform(Xte))[0].mean(axis=1)
        for fam in ("CatBoost", "LogReg"):
            m = make_any_model(fam, s)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            wrong = ((p >= 0.5).astype(int) != yte)
            ph_side = yte == 1
            rows.append({"seed": s, "family": fam,
                         "auc_all": roc_auc_score(wrong, u),
                         "auc_phish": roc_auc_score(wrong[ph_side], u[ph_side]),
                         "err_rate": float(wrong.mean())})
            print(f"[E3] seed={s} {fam:<9} AUC(u;wrong) all={rows[-1]['auc_all']:.3f} "
                  f"phish-only={rows[-1]['auc_phish']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_NOVELTY, index=False)
    m = out.groupby("family")[["auc_all", "auc_phish"]].mean().round(3)
    print(f"[E3] means:\n{m.to_string()}")
    print("[E3] pre-set rule: gated backoff stays alive only if CatBoost AUC >= 0.65")
    print(f"[+] {len(out)} rows -> {OUT_NOVELTY}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", choices=["e1", "e2", "e3"], default=None)
    args = ap.parse_args()
    feats, ph_tr, ph_te, be = split_canonical(load())
    print(f"canonical split: ph_tr={len(ph_tr)} ph_te={len(ph_te)} benign={len(be)}")
    seeds = range(args.seeds)
    if args.only in (None, "e2"):
        e2_matrix(feats, ph_tr, ph_te)
    if args.only in (None, "e3"):
        e3_novelty(feats, ph_tr, ph_te, be, seeds)
    if args.only in (None, "e1"):
        e1_decay(feats, ph_tr, be, seeds)


if __name__ == "__main__":
    main()
