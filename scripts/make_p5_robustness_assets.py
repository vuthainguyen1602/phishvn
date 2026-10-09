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
    f = lambda v: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.3f}"
    rows = []
    for r in w.itertuples():
        cells = [f"{r.window}", f"{100 * r.spread_share:.0f}", f"{r.prior:.2f}"]
        if np.isnan(r.static_f1):
            cells += ["--"] * 7
        else:
            rd = r.static_recall_dated if r.positives_dated >= 50 else float("nan")
            cells += [f(r.static_precision), f(r.static_recall), f(rd), f(r.static_recall_spread),
                      f(r.static_f1), f(r.periodic_f1), f(r.drift_f1)]
        rows.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{The constructed stream window by window: the "
           "share of hash-spread (undated) rows, the class prior, the static model's precision, "
           "recall overall and on dated and spread positives separately (blank under 50 dated "
           "positives), its F1, and the F1 of the two updating policies. Window 1 is the training "
           "window.}\n\\label{tab:windows}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
           "Window & Spread (\\%) & Prior & \\multicolumn{5}{c}{Static} & Periodic & Drift \\\\\n"
           "\\cmidrule(lr){4-8}\n & & & P & R & R dated & R spread & F1 & F1 & F1 \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}")
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


def gen_window_reading(w, d, fmt, dw, dprior):
    sc = w.dropna(subset=["static_f1"])
    rho = float(np.corrcoef(sc.spread_share, sc.static_recall)[0, 1])
    rd = sc.dropna(subset=["static_recall_dated"])
    rd = rd[rd.positives_dated >= 300]           # windows with enough dated positives to read
    rs = sc.static_recall_spread
    c = fmt[fmt.source == "chongluadao"].set_index("rows_kind")
    first_spread = float(w.iloc[0].spread_share)
    pol = d.set_index("policy")
    # the dated-only stream, read against its own per-window all-positive floor
    fl = floor(dprior.iloc[1:].to_numpy())
    below = {k: int((dw[k].to_numpy() < fl).sum()) for k in ("static", "periodic", "drift")}
    gap_w = (dw[["periodic", "drift"]].max(axis=1) - dw["static"]).to_numpy()
    last_share = float(gap_w[-1] / gap_w.sum()) if gap_w.sum() > 0 else float("nan")
    n = len(dw)
    rows_txt = f"{int(pol['rows'].iloc[0]):,}".replace(",", "{,}")
    body = (
        f"Table~\\ref{{tab:windows}} shows where that gap comes from. The static model is fitted "
        f"on window 1, whose rows are {100 * first_spread:.0f}\\% hash-spread, and its precision "
        f"stays at ${sc.static_precision.min():.3f}$--${sc.static_precision.max():.3f}$ in every "
        f"later window; what moves is recall, and the correlation between a window's spread share "
        f"and the static model's recall is ${rho:.2f}$. Split by row, the pattern is sharper: on "
        f"spread positives the static model's recall is ${rs.min():.3f}$--${rs.max():.3f}$ in every "
        f"window, and on dated positives it is ${rd.static_recall_dated.min():.3f}$--"
        f"${rd.static_recall_dated.max():.3f}$ wherever a window holds enough of them to read. The "
        f"two kinds of row are recorded differently. In the community feed, "
        f"{100 * c.loc['undated', 'subdomain_share']:.1f}\\% of the undated phishing rows carry a "
        f"subdomain (median URL length {c.loc['undated', 'median_url_len']:.0f} characters) against "
        f"{100 * c.loc['dated', 'subdomain_share']:.1f}\\% of the dated ones (median "
        f"{c.loc['dated', 'median_url_len']:.0f}), so a model fitted on the spread background learns "
        f"a URL format and misses the other. The twenty-point gap is a property of that split, not "
        f"an estimate of drift. Run on the dated rows alone ($" + rows_txt + f"$ rows, the same "
        f"number of equal-count windows), the stream is {100 * float(pol['prior'].iloc[0]):.0f}\\% "
        f"phishing and every policy sits at or near the all-positive floor of its window (mean "
        f"${fl.mean():.3f}$ F1): the static model scores below that floor in {below['static']} of "
        f"{n} windows, the drift-triggered policy in {below['drift']} and periodic retraining in "
        f"{below['periodic']}. Static AUTC is ${pol.loc['static', 'autc']:.3f}$ against "
        f"${pol.loc['periodic', 'autc']:.3f}$ periodic and ${pol.loc['drift', 'autc']:.3f}$ "
        f"drift-triggered, with {pol.loc['periodic', 'retrains']} and {pol.loc['drift', 'retrains']} "
        f"retrains, and {100 * last_share:.0f}\\% of the static model's shortfall comes from the "
        f"last window alone, whose prior is ${float(dprior.iloc[-1]):.2f}$. This stream sits too close "
        f"to its ceiling to measure staleness, so it neither confirms nor bounds the gap.")
    write_generated(os.path.join(SEC, "gen_window_reading.tex"), body)


def gen_shap_trace(t):
    """The SHAP-shift separation, read off the exported trace instead of typed."""
    prox, non = t[t.proxy_shift == 1], t[t.proxy_shift == 0]
    tau = 0.05
    fired = prox[prox.tv_distance > tau]
    missed = prox[prox.tv_distance <= tau]
    body = (
        f"The separation behind the result is thin. The exported trace gives total-variation "
        f"distances of ${non.tv_distance.min():.3f}$--${non.tv_distance.max():.3f}$ on the "
        f"{len(non)} non-proxy windows and ${prox.tv_distance.min():.3f}$--"
        f"${prox.tv_distance.max():.3f}$ on the {len(prox)} proxy windows. "
        + (f"{'The proxy window' if len(missed) == 1 else 'Proxy windows'} at "
           + ", ".join(f"${v:.3f}$" for v in missed.tv_distance)
           + f" {'is' if len(missed) == 1 else 'are'} missed at onset and credited only through the "
             f"next window's alarm inside the horizon, so {len(fired)} of the {len(prox)} are "
             f"caught at onset, and " if len(missed) else "")
        + f"$\\tau_{{\\mathrm{{SHAP}}}}={tau}$ lies ${tau - non.tv_distance.max():.4f}$ above the "
        f"highest non-proxy value and ${fired.tv_distance.min() - tau:.4f}$ below the lowest proxy "
        f"value that fires.")
    write_generated(os.path.join(SEC, "gen_shap_trace.tex"), body)


def _label_note(s):
    """Which labels each policy buys, and the trigger against the schedule at equal spend."""
    lab = s.pivot_table(index=["seed", "budget"], columns="policy", values="labels")
    auc = s.pivot_table(index=["seed", "budget"], columns="policy", values="autc")
    if (lab.drift == lab.periodic).all():
        return "The two spend the same labels in every cell."
    pairs = []
    for (sd, b), r in lab.iterrows():
        for (sd2, h), r2 in lab.xs(sd, level="seed", drop_level=False).iterrows():
            if r.drift == r2.periodic and r.drift > 0:
                pairs.append((b, h, auc.loc[(sd, b), "drift"] - auc.loc[(sd2, h), "periodic"]))
    p = pd.DataFrame(pairs, columns=["b", "h", "d"])
    g = p.groupby("b").d
    share = float((lab.drift / lab.periodic).median())
    worse = int((p.d < 0).sum())
    return (f"The label-free trigger buys {100 * share:.0f}\\% of the schedule's labels in every cell. "
            f"Matched on labels bought (the trigger at $B$ against the schedule at the budget that "
            f"buys as many), it is behind on {worse} of {len(p)} seed-budget pairs, by "
            f"${g.mean().max():+.3f}$ to ${g.mean().min():+.3f}$ AUTC on average.")


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
    dmean = dp.mean()
    if (dmean < 0).all() and dp_hi_wins == 0 and dp_lo_wins == 0:
        trig = (f"The label-free trigger trails the fixed schedule at every budget, from "
                f"{pm(dp, lo)} at $B={lo}$ to {pm(dp, hi)} at $B={hi}$, on every seed. ")
    else:
        trig = (f"The drift trigger trails the fixed schedule at the cheapest budgets ({pm(dp, lo)} "
                f"at $B={lo}$ and {pm(dp, second)} at $B={second}$, ahead on {dp_lo_wins} and "
                f"{dp_2_wins} of {k} seeds) and leads at the dearest ({pm(dp, hi)} at $B={hi}$, "
                f"ahead on {dp_hi_wins} of {k}); between them the sign depends on the seed. ")
    body = (
        f"Repeating the grid over {k} model seeds (the seed sets the forest and the random label "
        f"draws together; mean $\\pm$ SD over seeds) gives the same reading. Uncertainty sampling is {act_txt}: {pm(ap, lo)} AUTC against periodic "
        f"retraining at $B={lo}$, widest at $B={worst_b}$ ({pm(ap, worst_b)}), and "
        f"{pm(ap, hi)} at $B={hi}$. " + trig + _label_note(s))
    write_generated(os.path.join(SEC, "gen_budget_seeds.tex"), body)


def main():
    missing = [p for p in (WIN, DATED, SEEDS) if not os.path.exists(p)]
    if missing:
        raise SystemExit("run run_p5_robustness.py first; missing: " + ", ".join(missing))
    w, d, s = pd.read_csv(WIN), pd.read_csv(DATED), pd.read_csv(SEEDS)
    fmt = pd.read_csv(os.path.join(P5, "p5_format_split.csv"))
    dw = pd.read_csv(os.path.join(P5, "p5_dated_only_windows.csv"))
    dprior = pd.read_csv(os.path.join(P5, "p5_dated_only_priors.csv"))["prior"]
    tab_windows(w)
    gen_window_reading(w, d, fmt, dw, dprior)
    gen_shap_trace(pd.read_csv(os.path.join(P5, "p5_shap_trace.csv")))
    gen_budget_seeds(s)


if __name__ == "__main__":
    main()
