#!/usr/bin/env python3
"""
analyse_p2_noise_ordering.py — does the family ordering under the phishing-temporal protocol
survive plausible label error? (revision 1, 2026-11-12). Reads run_p2_scores.py's output; fits
nothing.

WHY. label_noise.csv (analyse_p2_scores.py) asks whether the protocol step survives label error,
and its worst case relabels, for each model, the rows THAT model scores lowest. That is the right
worst case for one model's step, but not for comparing two models: each would be scored against a
different ground truth. To compare families, the same rows have to be wrong for every family.

What it does. On each seed's phishing-temporal test window (validation-carved fits, as in
label_noise.csv), a share of the bronze-tier phishing rows (5%, 12.1% = the audited positive-label
error rate, 20%) is relabelled benign, the SAME rows for all seven families, chosen
  random  — uniformly (NOISE_DRAWS draws, F1 averaged over draws), or
  hard    — the rows with the lowest mean score across the seven families: label error
            concentrated where the models agree a row looks least like phishing.
Each family is then scored at tau = 0.5 against that one relabelled truth.

Writes data/processed/p2/revision/noise_ordering.csv: rate, mode, seed, family, f1 (rate 0 is the
clean window).

RUN: python scripts/analyse_p2_noise_ordering.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

D = "data/processed/p2/revision"
OUT = os.path.join(D, "noise_ordering.csv")
NOISE_RATES = (0.05, 0.121, 0.20)
NOISE_DRAWS, SEED = 200, 0


def f1_at_half(y: np.ndarray, score: np.ndarray) -> float:
    p = score >= 0.5
    tp = int((p & (y == 1)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main() -> int:
    s = pd.read_csv(os.path.join(D, "p2_scores.csv.gz"))
    s = s[(s.protocol == "temporal_strict") & (s.part == "test")]
    families = sorted(s.family.unique())
    rows = []
    for seed, g in s.groupby("seed"):
        per = {f: gg.reset_index(drop=True) for f, gg in g.groupby("family")}
        ref = per[families[0]]
        for f in families:  # every family must score the same rows in the same order
            if not (per[f].rdom.to_numpy() == ref.rdom.to_numpy()).all() or len(per[f]) != len(ref):
                raise SystemExit(f"seed {seed}: {f} does not score the rows {families[0]} scores")
        y = ref.y.to_numpy()
        scores = {f: per[f].score.to_numpy() for f in families}
        bronze = np.flatnonzero((y == 1) & (ref.tier.to_numpy() == "bronze"))
        mean_score = np.mean([scores[f] for f in families], axis=0)
        for f in families:
            rows.append((0.0, "clean", seed, f, f1_at_half(y, scores[f])))
        rng = np.random.RandomState(SEED + int(seed))
        for rate in NOISE_RATES:
            k = int(round(rate * len(bronze)))
            hard = bronze[np.argsort(mean_score[bronze], kind="stable")[:k]]
            yy = y.copy()
            yy[hard] = 0
            for f in families:
                rows.append((rate, "hard", seed, f, f1_at_half(yy, scores[f])))
            acc = {f: 0.0 for f in families}
            for _ in range(NOISE_DRAWS):
                pick = rng.choice(bronze, k, replace=False)
                yy = y.copy()
                yy[pick] = 0
                for f in families:
                    acc[f] += f1_at_half(yy, scores[f])
            for f in families:
                rows.append((rate, "random", seed, f, acc[f] / NOISE_DRAWS))
    out = pd.DataFrame(rows, columns=["rate", "mode", "seed", "family", "f1"])
    out.to_csv(OUT, index=False)
    piv = out.pivot_table(index=["rate", "mode"], columns="family", values="f1")
    print((piv["LogReg"] - piv["CatBoost"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
