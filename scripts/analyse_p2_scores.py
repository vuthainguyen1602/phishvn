#!/usr/bin/env python3
"""
analyse_p2_scores.py — what the per-row scores of the primary protocol contrast say about
calibration, deployable operating points, domain-resampled uncertainty and plausible label error
(revision 1, 2026-10-31). Reads run_p2_scores.py's output; fits nothing.

Every number comes from models fitted on the training window minus its validation slice (see
run_p2_scores.py), so levels differ slightly from Table tab:strict; contrasts are like for like.

  calibration.csv   Brier score and expected calibration error (15 equal-width bins) on the test
                    window, per family, protocol and seed.
  operating.csv     at the threshold chosen on the validation slice (max F1 there): precision,
                    recall, FPR, false positives per 10,000 benign test URLs, and the precision
                    and F1 the same TPR/FPR would give at a phishing prevalence of 1%, 5% and 10%;
                    and, for a false-positive budget of 0.1% and 1%, the threshold set on the
                    validation benign scores, with the recall and FPR it then gets on test.
  bootstrap.csv     95% intervals from resampling registrable domains of the test window (1,000
                    replicates; every row of a drawn domain enters together) for the protocol
                    contrast of each family and for logistic regression minus CatBoost under each
                    protocol, F1 at 0.5 and at the validation threshold, averaged over seeds.
  label_noise.csv   the same contrasts when a share of the bronze-tier phishing rows in the test
                    window are in truth benign: 5%, 12.1% (the audited positive-label error rate)
                    and 20% of them, chosen at random (200 draws) or, as a worst case for each
                    model, the ones it scores lowest.

RUN: python scripts/analyse_p2_scores.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = "data/processed/p2/revision"
PREVALENCES = (0.01, 0.05, 0.10)
FPR_BUDGETS = (0.001, 0.01)
NOISE_RATES = (0.05, 0.121, 0.20)
B, NOISE_DRAWS, SEED = 1000, 200, 0
CONTRASTS_FAM = ("LogReg", "CatBoost")


def f1_counts(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def ece(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def at_prevalence(tpr, fpr, pi):
    prec = pi * tpr / (pi * tpr + (1 - pi) * fpr) if (pi * tpr + (1 - pi) * fpr) > 0 else 0.0
    return prec, (2 * prec * tpr / (prec + tpr) if prec + tpr > 0 else 0.0)


def main() -> int:
    sc = pd.read_csv(os.path.join(D, "p2_scores.csv.gz"))
    runs = pd.read_csv(os.path.join(D, "p2_scores_runs.csv")).set_index(["family", "protocol", "seed"])
    cal, op = [], []
    for (fam, proto, s), g in sc.groupby(["family", "protocol", "seed"]):
        te, va = g[g.part == "test"], g[g.part == "val"]
        y, p = te.y.to_numpy(int), te.score.to_numpy(float)
        cal.append({"family": fam, "protocol": proto, "seed": s,
                    "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
                    "roc_auc": roc_auc_score(y, p)})
        tau = float(runs.loc[(fam, proto, s), "tau_val"])
        pred = p >= tau
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, nb = int((~pred & (y == 1)).sum()), int((y == 0).sum())
        tpr, fpr = tp / (tp + fn), fp / nb
        row = {"family": fam, "protocol": proto, "seed": s, "tau_val": tau,
               "precision": tp / max(tp + fp, 1), "recall": tpr, "fpr": fpr,
               "f1": float(f1_counts(tp, fp, fn)), "fp_per_10k_benign": 1e4 * fpr}
        for pi in PREVALENCES:
            row[f"precision_at_{pi:g}"], row[f"f1_at_{pi:g}"] = at_prevalence(tpr, fpr, pi)
        vb = np.sort(va.loc[va.y == 0, "score"].to_numpy(float))
        for b in FPR_BUDGETS:
            thr = float(np.quantile(vb, 1 - b)) if len(vb) else 1.0
            pb = p > thr
            row[f"recall_at_fpr{b:g}"] = float((pb & (y == 1)).sum() / (y == 1).sum())
            row[f"realised_fpr_at_fpr{b:g}"] = float((pb & (y == 0)).sum() / nb)
        op.append(row)
    cal, op = pd.DataFrame(cal), pd.DataFrame(op)
    cal.to_csv(os.path.join(D, "calibration.csv"), index=False)
    op.to_csv(os.path.join(D, "operating.csv"), index=False)

    # Per-domain confusion counts, so one bootstrap replicate is a weighted sum, not a refit.
    rng = np.random.RandomState(SEED)
    te = sc[sc.part == "test"].copy()
    te = te.join(runs["tau_val"], on=["family", "protocol", "seed"])
    te["p05"], te["pv"] = te.score >= 0.5, te.score >= te.tau_val
    te["rdom"] = te.rdom.astype(str)
    stats = {}
    for proto, g in te.groupby("protocol"):
        doms = np.sort(g.rdom.unique())
        code = pd.Categorical(g.rdom, categories=doms).codes
        g = g.assign(d=code)
        W = rng.multinomial(len(doms), np.full(len(doms), 1 / len(doms)), size=B).astype(float)
        for (fam, s), h in g.groupby(["family", "seed"]):
            for thr in ("p05", "pv"):
                pos, pr = h.y.to_numpy(int) == 1, h[thr].to_numpy(bool)
                tp = np.bincount(h.d[pos & pr], minlength=len(doms))
                fp = np.bincount(h.d[~pos & pr], minlength=len(doms))
                fn = np.bincount(h.d[pos & ~pr], minlength=len(doms))
                point = float(f1_counts(tp.sum(), fp.sum(), fn.sum()))
                boot = f1_counts(W @ tp, W @ fp, W @ fn)
                stats.setdefault((proto, fam, thr), []).append((point, boot))
    mean = {k: (np.mean([a for a, _ in v]), np.mean([b for _, b in v], axis=0)) for k, v in stats.items()}
    boot = []
    fams = sorted(te.family.unique())
    for thr in ("p05", "pv"):
        for fam in fams:
            a, b = mean[("temporal_strict", fam, thr)], mean[("random_same_rows_guarded", fam, thr)]
            d = a[1] - b[1]
            boot.append({"contrast": "temporal_minus_random", "family": fam, "threshold": thr,
                         "estimate": a[0] - b[0], "lo": float(np.quantile(d, .025)),
                         "hi": float(np.quantile(d, .975))})
        for proto in ("temporal_strict", "random_same_rows_guarded"):
            a, b = mean[(proto, CONTRASTS_FAM[0], thr)], mean[(proto, CONTRASTS_FAM[1], thr)]
            d = a[1] - b[1]
            boot.append({"contrast": f"LogReg_minus_CatBoost_{proto}", "family": "", "threshold": thr,
                         "estimate": a[0] - b[0], "lo": float(np.quantile(d, .025)),
                         "hi": float(np.quantile(d, .975))})
    boot = pd.DataFrame(boot)
    boot.to_csv(os.path.join(D, "bootstrap.csv"), index=False)

    # Label-error scenarios: flip a share of bronze phishing TEST rows to benign, rescore.
    noise = []
    for rate in NOISE_RATES:
        for mode in ("random", "lowest_score"):
            draws = NOISE_DRAWS if mode == "random" else 1
            res = {}
            for (fam, proto, s), h in te.groupby(["family", "protocol", "seed"]):
                y = h.y.to_numpy(int)
                cand = np.flatnonzero((y == 1) & (h.tier.to_numpy() == "bronze"))
                k = int(round(rate * len(cand)))
                sco, pr = h.score.to_numpy(float), h.p05.to_numpy(bool)
                f1s, aucs = [], []
                for _ in range(draws):
                    flip = (rng.choice(cand, k, replace=False) if mode == "random"
                            else cand[np.argsort(sco[cand])[:k]])
                    yy = y.copy()
                    yy[flip] = 0
                    tp, fp = (pr & (yy == 1)).sum(), (pr & (yy == 0)).sum()
                    fn = (~pr & (yy == 1)).sum()
                    f1s.append(float(f1_counts(tp, fp, fn)))
                    aucs.append(roc_auc_score(yy, sco))
                res[(fam, proto, s)] = (np.mean(f1s), np.mean(aucs))
            r = pd.DataFrame([(f, p, s, a, b) for (f, p, s), (a, b) in res.items()],
                             columns=["family", "protocol", "seed", "f1", "roc_auc"])
            m = r.groupby(["family", "protocol"])[["f1", "roc_auc"]].mean()
            for fam in fams:
                for met in ("f1", "roc_auc"):
                    noise.append({"rate": rate, "mode": mode, "family": fam, "metric": met,
                                  "temporal": m.loc[(fam, "temporal_strict"), met],
                                  "random": m.loc[(fam, "random_same_rows_guarded"), met]})
    noise = pd.DataFrame(noise)
    noise["delta_proto"] = noise.temporal - noise.random
    noise.to_csv(os.path.join(D, "label_noise.csv"), index=False)

    pd.set_option("display.width", 200)
    print(cal.groupby(["protocol", "family"])[["brier", "ece", "roc_auc"]].mean().round(4).to_string())
    print(op.groupby(["protocol", "family"])[["f1", "fp_per_10k_benign", "precision_at_0.01",
                                              "recall_at_fpr0.001", "recall_at_fpr0.01",
                                              "realised_fpr_at_fpr0.01"]].mean().round(4).to_string())
    print(boot.round(4).to_string(index=False))
    print(noise[noise.metric == "f1"].pivot_table(index=["rate", "mode"], columns="family",
                                                  values="delta_proto").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
