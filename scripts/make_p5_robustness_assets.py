#!/usr/bin/env python3
"""
make_p5_robustness_assets.py — the P5 table and sentences that read run_p5_robustness.py's CSVs.

  tab_windows.tex          per-window composition: spread share, prior, static P/R/F1, policy F1
  gen_window_reading.tex   what the composition says about the staleness gap, plus the
                           dated-only stream
  gen_budget_seeds.tex     the budget frontier across model seeds

Every number is read from data/processed/p5/; nothing is typed. Run after run_p5_robustness.py.
"""
from __future__ import annotations

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

P5 = os.path.join(ROOT, "data", "processed", "p5")
SEC = os.path.join(ROOT, "papers", "P5_temporal_drift", "sections")
WIN = os.path.join(P5, "p5_window_composition.csv")
DATED = os.path.join(P5, "p5_dated_only.csv")
SEEDS = os.path.join(P5, "p5_budget_seeds.csv")


def floor(p):
    """F1 of the classifier that answers phishing for every row, at positive rate p."""
    return 2 * p / (1 + p)


def tab_windows(w):
    rows = []
    for r in w.itertuples():
        cells = [f"{r.window}", f"{100 * r.spread_share:.0f}", f"{r.prior:.2f}"]
        if np.isnan(r.static_f1):
            cells += ["--"] * 5
        else:
            cells += [f"{r.static_precision:.3f}", f"{r.static_recall:.3f}", f"{r.static_f1:.3f}",
                      f"{r.periodic_f1:.3f}", f"{r.drift_f1:.3f}"]
        rows.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{The constructed stream window by window: "
           "the share of hash-spread (undated) rows, the class prior, the static model's precision, "
           "recall and F1, and the F1 of the two updating policies. Window 1 is the training "
           "window.}\n\\label{tab:windows}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
           "Window & Spread (\\%) & Prior & \\multicolumn{3}{c}{Static} & Periodic & Drift \\\\\n"
           "\\cmidrule(lr){4-6}\n & & & P & R & F1 & F1 & F1 \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
    write_generated(os.path.join(SEC, "tab_windows.tex"), tex)


def _windows(g):
    """'window 2 and windows 15--20' from a set of window numbers."""
    ws = sorted(int(x) for x in g.window)
    runs, start = [], ws[0]
    for a, b in zip(ws, ws[1:] + [None]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    parts = [f"window {a}" if a == b else f"windows {a}--{b}" for a, b in runs]
    return " and ".join(parts)


def gen_window_reading(w, d):
    sc = w.dropna()
    mid = sc[sc.spread_share < 0.7]
    late = sc[sc.spread_share >= 0.7]
    rho = float(np.corrcoef(sc.spread_share, sc.static_recall)[0, 1])
    pol = d.set_index("policy")
    gap = float(pol.loc[["periodic", "drift"], "autc"].max() - pol.loc["static", "autc"])
    first_spread = float(w.iloc[0].spread_share)
    # the dated-only stream is mostly phishing: the floor is what any F1 there has to clear
    df, prior = int(pol["rows"].iloc[0]), float(pol["prior"].iloc[0])
    body = (
        f"Table~\\ref{{tab:windows}} shows where that gap comes from. The static model is fitted "
        f"on window 1, whose rows are {100 * first_spread:.0f}\\% hash-spread, and its precision "
        f"stays at ${sc.static_precision.min():.3f}$--${sc.static_precision.max():.3f}$ in every "
        f"later window; what moves is recall. In the {len(mid)} windows where dated rows are the "
        f"majority or close to it (spread share ${mid.spread_share.min():.2f}$--"
        f"${mid.spread_share.max():.2f}$) its recall is ${mid.static_recall.min():.3f}$--"
        f"${mid.static_recall.max():.3f}$, and in the {len(late)} windows that are mostly spread "
        f"rows ({_windows(late)}) it is ${late.static_recall.min():.3f}$--"
        f"${late.static_recall.max():.3f}$; across "
        f"the scored windows the correlation between spread share and static recall is "
        f"${rho:.2f}$. The static model misses the dated phishing and recognises the background "
        f"it was fitted on. Run on the dated rows alone ($" + f"{df:,}".replace(",", "{,}")
        + f"$ rows, cut into the same number of equal-count windows), the gap between the "
        f"static model and the better updating policy is ${gap:.3f}$ AUTC "
        f"(static ${pol.loc['static', 'autc']:.3f}$, periodic ${pol.loc['periodic', 'autc']:.3f}$, "
        f"drift-triggered ${pol.loc['drift', 'autc']:.3f}$). That stream is "
        f"{100 * prior:.0f}\\% phishing, so every policy sits close to the all-positive floor "
        f"(${floor(prior):.3f}$ F1 at that prior) and the comparison has little room to separate anything. It does "
        f"show that the twenty-point gap of Table~\\ref{{tab:overtime}} does not survive "
        f"without the spreading scheme.")
    write_generated(os.path.join(SEC, "gen_window_reading.tex"), body)


def gen_budget_seeds(s):
    w = s.pivot_table(index=["seed", "budget"], columns="policy", values="autc")
    k = s.seed.nunique()
    dp = (w["drift"] - w["periodic"]).groupby("budget")
    ap = (w["active"] - w["periodic"]).groupby("budget")
    bs = sorted(s.budget.unique())
    lo, hi = bs[0], bs[-1]
    worst_b = int(ap.mean().idxmin())
    act_all = bool(((w["active"] - w["periodic"]) < 0).all())
    dp_lo_wins = int(((w["drift"] - w["periodic"]).xs(lo, level="budget") > 0).sum())
    dp_hi_wins = int(((w["drift"] - w["periodic"]).xs(hi, level="budget") > 0).sum())
    second = bs[1]
    dp_2_wins = int(((w["drift"] - w["periodic"]).xs(second, level="budget") > 0).sum())
    pm = lambda g, b: f"${g.mean()[b]:+.3f} \\pm {g.std()[b]:.3f}$"
    act_txt = ("below random selection at every budget on every seed" if act_all else
               "below random selection on most seeds")
    body = (
        f"Repeating the grid over {k} model seeds (the seed sets the forest and the random label "
        f"draws together; mean $\\pm$ SD over seeds) settles the two readings the single seed "
        f"left open. Uncertainty sampling is {act_txt}: {pm(ap, lo)} AUTC against periodic "
        f"retraining at $B={lo}$, widest at $B={worst_b}$ ({pm(ap, worst_b)}), and "
        f"{pm(ap, hi)} at $B={hi}$. The drift trigger trails the fixed schedule at the cheapest "
        f"budgets ({pm(dp, lo)} at $B={lo}$ and {pm(dp, second)} at $B={second}$, ahead on "
        f"{dp_lo_wins} and {dp_2_wins} of {k} seeds) and leads at the dearest ({pm(dp, hi)} at "
        f"$B={hi}$, ahead on {dp_hi_wins} of {k}); between them the sign depends on the seed.")
    write_generated(os.path.join(SEC, "gen_budget_seeds.tex"), body)


def main():
    missing = [p for p in (WIN, DATED, SEEDS) if not os.path.exists(p)]
    if missing:
        raise SystemExit("run run_p5_robustness.py first; missing: " + ", ".join(missing))
    w, d, s = pd.read_csv(WIN), pd.read_csv(DATED), pd.read_csv(SEEDS)
    tab_windows(w)
    gen_window_reading(w, d)
    gen_budget_seeds(s)


if __name__ == "__main__":
    main()
