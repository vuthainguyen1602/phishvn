#!/usr/bin/env python3
"""
audit_label_noise.py — Confident-learning label-noise audit of the corpus provenance tiers (P2).

WHY: P2's decomposition shows the boosters' full-corpus advantage lives almost entirely on the
undated bronze phishing mass — the tier with the weakest provenance (single-feed, unverified, no
detection date). The natural next question is ALGORITHMIC and the 2026 literature does not ask it:
how much of that mass even looks mislabelled to a calibrated model? Every public phishing corpus
treats its feeds as ground truth; none ships a label-noise estimate.

METHOD (Confident Learning, Northcutt et al., JAIR 2021 — binary form): out-of-fold predicted
probabilities for every row (stratified K folds, HistGB calibrated with isotonic regression so the
probabilities are usable as confidences, repeated over --seeds); per-class thresholds t_c = mean
p_c over rows labelled c; a row labelled phishing is FLAGGED when p_benign >= t_benign (and vice
versa); flag rate reported per provenance tier with across-seed spread and a Wilson CI.

READING THE NUMBERS HONESTLY: the scorer sees only URL-lexical features, so a flag means
"lexically indistinguishable from benign" — an UPPER bound on label error that includes genuine
phishing on benign-looking domains (e.g. compromised sites). The calibrated quantity is the
CONTRAST between tiers: gold is human-verified, so its flag rate is the method's floor on labels
known to be right, and the EXCESS of bronze over gold is the share attributable to provenance.

OUTPUT: data/processed/p2/label_noise_audit.csv (per-tier, per-seed rates)
        papers/P2_url_benchmark/sections/tab_label_noise.tex (auto-generated table)

RUN:  python scripts/audit_label_noise.py            # 5 seeds, ~minutes
"""
from __future__ import annotations
import argparse
import math
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
from genfile import write_generated
from train_url_baseline import COMPPHISH, add_label
from paired_eval import wilson

CORPUS = os.path.join(ROOT, "data", "processed", "vn_compphish.csv")
OUT_CSV = os.path.join(ROOT, "data", "processed", "p2", "label_noise_audit.csv")
OUT_TEX = os.path.join(ROOT, "papers", "P2_url_benchmark", "sections", "tab_label_noise.tex")




def oof_proba(X, y, seed: int, folds: int):
    """Out-of-fold calibrated P(phishing) for every row. Isotonic-calibrated HistGB: fast on
    50k rows and the calibration is what makes the confident-learning thresholds meaningful."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = CalibratedClassifierCV(
            HistGradientBoostingClassifier(random_state=seed), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def flags(y, p_phish):
    """Confident-learning flags: t_c = mean confidence over rows labelled c; a row is flagged
    when its confidence for the OPPOSITE class clears that class's threshold."""
    p_benign = 1 - p_phish
    t_phish = p_phish[y == 1].mean()
    t_benign = p_benign[y == 0].mean()
    flagged = np.where(y == 1, p_benign >= t_benign, p_phish >= t_phish)
    return flagged, t_phish, t_benign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = add_label(pd.read_csv(CORPUS, low_memory=False))
    for c in COMPPHISH:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = df[COMPPHISH].to_numpy(float)
    y = df["y"].to_numpy(int)
    tier = df["tier"].fillna("?").to_numpy()
    print(f"[i] corpus: {len(df)} rows; phishing tiers: "
          + ", ".join(f"{t}={int(((tier == t) & (y == 1)).sum())}"
                      for t in ("gold", "silver", "bronze")))

    # groups audited: phishing per tier, plus all benign as the opposite-direction reference
    groups = {t: (tier == t) & (y == 1) for t in ("gold", "silver", "bronze")}
    groups["benign (all)"] = y == 0
    rates = {g: [] for g in groups}
    per_seed_rows = []
    flag_count = np.zeros(len(df))
    for s in range(args.seeds):
        p = oof_proba(X, y, s, args.folds)
        fl, t_p, t_b = flags(y, p)
        flag_count += fl
        for g, mask in groups.items():
            r = float(fl[mask].mean())
            rates[g].append(r)
            per_seed_rows.append({"group": g, "seed": s, "n": int(mask.sum()),
                                  "flag_rate": round(r, 4)})
        print(f"[i] seed {s}: thresholds t_phish={t_p:.3f} t_benign={t_b:.3f}; "
              + " ".join(f"{g}={np.mean(fl[m]):.3f}" for g, m in groups.items()))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(per_seed_rows).to_csv(OUT_CSV, index=False)
    print(f"[+] {OUT_CSV}")

    # a majority-of-seeds flag is the stable per-row verdict; used for the qualitative sample
    stable = flag_count >= (args.seeds / 2)
    ex = df.loc[stable & groups["bronze"], "dom" if "dom" in df else "url"].astype(str)
    print("[i] example stable-flagged bronze domains:", ", ".join(ex.head(12)))

    label = {"gold": "Gold phishing (human-verified)", "silver": "Silver phishing",
             "bronze": "Bronze phishing (single-feed)",
             "benign (all)": "Benign, all tiers (reference)"}
    body_rows = []
    stats = {}
    for g in groups:
        n = int(groups[g].sum())
        mean_r = float(np.mean(rates[g]))
        std_r = float(np.std(rates[g]))
        lo, hi = wilson(mean_r * n, n)
        stats[g] = (n, mean_r, std_r, lo, hi)
        body_rows.append(f"{label[g]} & {n:,} & {100 * mean_r:.1f}\\,$\\pm$\\,{100 * std_r:.1f} & "
                         f"[{100 * lo:.1f}, {100 * hi:.1f}] \\\\")
    excess = stats["bronze"][1] - stats["gold"][1]
    # normal-approximation CI on the difference of the two proportions
    se = math.sqrt(stats["bronze"][1] * (1 - stats["bronze"][1]) / stats["bronze"][0]
                   + stats["gold"][1] * (1 - stats["gold"][1]) / stats["gold"][0])
    body = "\n".join(body_rows)
    # \caption BEFORE the tabular: IEEEtran prints table captions above the table, and the
    # caption renders wherever it is issued — after the tabular it lands underneath, the one
    # table in P2 breaking the convention (caught in the 2026-08-17 format pass).
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Confident-learning label-noise audit by provenance tier: rows flagged as
lexically indistinguishable from the opposite class (mean\\,$\\pm$\\,std over {args.seeds}
fold seeds).}}
\\label{{tab:labelnoise}}
\\small
\\begin{{tabular}}{{l r c c}}
\\toprule
Group & $n$ & flagged (\\%) & Wilson 95\\% CI \\\\
\\midrule
{body}
\\midrule
\\multicolumn{{4}}{{l}}{{\\footnotesize Bronze excess over gold: ${100 * excess:.1f}$\\,pp
(95\\% CI $[{100 * (excess - 1.96 * se):.1f}, {100 * (excess + 1.96 * se):.1f}]$).}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    write_generated(OUT_TEX, tex)
    print(f"[+] {OUT_TEX}")
    print(f"[+] bronze {100 * stats['bronze'][1]:.1f}% vs gold {100 * stats['gold'][1]:.1f}% "
          f"-> excess {100 * excess:.1f}pp")


if __name__ == "__main__":
    main()
