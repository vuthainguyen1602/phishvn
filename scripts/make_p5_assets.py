#!/usr/bin/env python3
"""
make_p5_assets.py — Regenerate P5's PRELIMINARY drift-study assets from the dataset, so the numbers
in the manuscript never drift from the code. Produces (all on ONE 20-window, PSI-$\\tau$=0.2 stream):

  papers/P5_temporal_drift/figures/drift_f1_over_time.pdf  — F1-over-time per strategy + PSI signal
  papers/P5_temporal_drift/sections/tab_overtime.tex       — retraining strategies (End F1/AUTC/retrains)
  papers/P5_temporal_drift/sections/tab_drift.tex          — drift-detection accuracy & lag
  papers/P5_temporal_drift/sections/tab_budget.tex         — labelling-budget frontier

Preliminary: RandomForest on the CompPhish URL features, streamed over the temporally-ordered
PhishVN+feeds data (NCSC + reconstructed-date ChongLuaDao phishing; benign held stationary).

RUN:  python scripts/make_p5_assets.py
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
from retrain_drift import load, run
from drift_detectors import run_detection, score_detector, run_budget

P5 = os.path.join(ROOT, "papers", "P5_temporal_drift")
FIG = os.path.join(P5, "figures")
SEC = os.path.join(P5, "sections")
DRIFT_CSV = os.path.join(ROOT, "data", "interim", "drift_compphish.csv")
WINDOWS, PERIOD, PSI_TAU, F1_DROP, ADWIN_DELTA = 20, 3, 0.2, 0.1, 0.002
BUDGETS = [50, 200, 800]


def build_input():
    """CompPhish features (row-aligned with dataset_url) + collected_at/id/tier -> drift input."""
    u = pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"), low_memory=False)
    c = pd.read_csv(os.path.join(ROOT, "data", "processed", "vn_compphish.csv"), low_memory=False)
    if len(u) != len(c):
        raise SystemExit(f"row mismatch {len(u)} vs {len(c)} — rebuild vn_compphish.csv")
    c = c.copy()
    c["collected_at"] = u["collected_at"].values
    c["id"] = u["id"].values
    c["tier"] = u["tier"].values
    os.makedirs(os.path.dirname(DRIFT_CSV), exist_ok=True)
    c.to_csv(DRIFT_CSV, index=False)


def make_figure(strat, psi_trace):
    from figstyle import apply, ORANGE, BLUE, TEAL, GRAY, INK
    plt = apply()

    x = list(range(2, 2 + len(strat["static"])))
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    # House hues by role (was three greys 0.6/0.32/black — a sequential ramp on a categorical job).
    styles = {"static": ("o-", BLUE), "periodic": ("s--", TEAL), "drift": ("^-", ORANGE)}
    labels = {"static": "static (train once)", "periodic": f"periodic (every {PERIOD})",
              "drift": "drift-triggered (PSI $\\vee$ F1-drop)"}
    for k in ("static", "periodic", "drift"):
        m, col = styles[k]
        ax.plot(x, strat[k], m, color=col, lw=1.3, ms=3.5, label=labels[k])
    ax.set_xlabel("stream window (time-ordered, 2020$\\to$2025)")
    ax.set_ylabel("F1 on the incoming window")
    # the window index is an integer: fractional ticks (2.5, 5.0, ...) named windows that do
    # not exist
    ax.set_xticks([w for w in x if w % 2 == 0])
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # Floor from the data, not pinned: a 0.30 pin clipped static's trough (0.209 in window 11,
    # 0.287 in window 12) — the deepest decay left the plot.
    lo = min(min(strat[k]) for k in ("static", "periodic", "drift"))
    ax.set_ylim(lo - 0.05, 1.03)
    ax.spines[["top"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.bar(x, psi_trace, width=0.55, color=GRAY, alpha=0.28, zorder=0)
    ax2.set_ylabel("mean PSI (drift signal)")
    ax2.set_ylim(0, max(psi_trace) * 3 if psi_trace else 1)
    # tau is a RIGHT-axis quantity but the rule lands at F1~0.49 on the left axis; grey + direct
    # label tie it to the PSI bars so it doesn't read as an F1 threshold.
    ax2.axhline(PSI_TAU, ls=":", color=GRAY, lw=0.9)
    ax2.annotate(f"$\\tau={PSI_TAU}$", (1.0, PSI_TAU), xycoords=("axes fraction", "data"),
                 xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=7.5, color=INK)
    ax2.spines[["top"]].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    # Out of the data area entirely: at center left it sat on static's descent.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8,
              frameon=False, columnspacing=1.6, handlelength=2.2)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "drift_f1_over_time.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def make_overtime_table(strat, counts, n, spans, dated_pct):
    rows = [("Static RF (no update)", "static"),
            ("Static CatBoost (no update)", "static_cb"),
            ("Static Stack[CB{+}LR] (no update)", "static_cblr"),
            (f"Periodic RF (every {PERIOD})", "periodic"),
            ("Drift-triggered RF (PSI $\\vee$ F1-drop)", "drift")]
    body = "\n".join(
        f"{lab} & {strat[k][-1]:.3f} & {np.mean(strat[k]):.3f} & {counts[k]} \\\\"
        for lab, k in rows if k in strat)
    n_pretty = f"{n:,}".replace(",", "{,}")  # LaTeX thin thousands separator, matches budget table
    tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{\\textbf{{Preliminary}} per-strategy performance over {WINDOWS} windows of the
temporally-ordered stream ({n_pretty} records, {dated_pct:.0f}\\% event-dated). AUTC = mean
per-window F1; static rows never retrain.}}
\\label{{tab:overtime}}
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Strategy}} & \\textbf{{End F1}} & \\textbf{{AUTC (mean F1)}} & \\textbf{{Retrains}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_overtime.tex"), tex)


def make_drift_table(truth, fires):
    order = ["ADWIN (error stream)", "PSI (feature distribution)",
             "SHAP-shift (attribution)", "ADWIN + PSI (combined)", "Fixed-schedule (naive)"]
    lines = []
    for name in (n for n in order if n in fires):
        dr, fa, lag, pr = score_detector(truth, fires[name])
        esc = name.replace("+", "{+}")
        lines.append(f"{esc} & {dr:.2f} & {fa:.2f} & {lag:.2f} & {pr:.2f} \\\\")
    body = "\n".join(lines)
    npos = sum(truth)
    tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{\\textbf{{Preliminary}} drift-detection accuracy and lag (in windows). Ground truth:
consecutive-window KS shift (Bonferroni $p<0.01$, $D>0.15$), giving
{npos}/{len(truth)} drift windows; the PSI detector fires at $0.10$.}}
\\label{{tab:drift}}
\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Detector}} & \\textbf{{Detection rate}} & \\textbf{{False-alarm rate}} & \\textbf{{Mean lag}} & \\textbf{{Precision}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_drift.tex"), tex)


def make_detector_figure(truth, fires):
    """Detection rate vs false-alarm rate per drift monitor, Wilson intervals on BOTH axes, mean
    lag in each label. The intervals are the point: on 7 drift events / 12 non-drift windows,
    7/7 detection is Wilson [0.65, 1.00] and 0/12 false alarms is [0.00, 0.24] — wide enough to
    overlap most of the field, so the figure must not assert a ranking (paper is preliminary,
    awaiting the 2027 live stream)."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    from paired_eval import wilson

    order = ["SHAP-shift (attribution)", "PSI (feature distribution)", "ADWIN (error stream)",
             "ADWIN + PSI (combined)", "Fixed-schedule (naive)"]
    short = {"SHAP-shift (attribution)": "SHAP-shift", "PSI (feature distribution)": "PSI",
             "ADWIN (error stream)": "ADWIN", "ADWIN + PSI (combined)": "ADWIN+PSI",
             "Fixed-schedule (naive)": "fixed schedule"}
    style = {"SHAP-shift (attribution)": (ORANGE, "D"), "PSI (feature distribution)": (BLUE, "o"),
             "ADWIN (error stream)": (BLUE, "s"), "ADWIN + PSI (combined)": (BLUE, "^"),
             "Fixed-schedule (naive)": (GRAY, "v")}
    # Labels go BELOW their points (axis stops at 1.0, where four of five monitors sit), at
    # staggered depths with leader lines. ADWIN (0.67) and ADWIN+PSI (0.75): the combined
    # monitor's label goes right — leftward it lands on ADWIN's marker.
    offset = {"SHAP-shift (attribution)": (10, -6), "Fixed-schedule (naive)": (8, -24),
              "ADWIN (error stream)": (-8, -10), "ADWIN + PSI (combined)": (12, -10),
              "PSI (feature distribution)": (10, 0)}
    # va="top" is load-bearing: with the default baseline anchor a negative y-offset still leaves
    # the text sitting ABOVE its own anchor, which put the SHAP-shift label through the title.
    align = {"ADWIN (error stream)": "right", "ADWIN + PSI (combined)": "left"}
    n_pos = int(sum(truth))
    n_neg = len(truth) - n_pos
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    drawn = []
    for name in (n for n in order if n in fires):
        dr, fa, lag, _ = score_detector(truth, fires[name])
        dlo, dhi = wilson(round(dr * n_pos), n_pos)
        flo, fhi = wilson(round(fa * n_neg), n_neg)
        drawn.append(dlo)
        c, mk = style[name]
        ax.errorbar(fa, dr, xerr=[[fa - flo], [fhi - fa]], yerr=[[dr - dlo], [dhi - dr]],
                    fmt=mk, color=c, markersize=7, capsize=2, elinewidth=0.8, alpha=0.95, zorder=3)
        ax.annotate(f"{short[name]}\nlag {lag:.2f}", (fa, dr), textcoords="offset points",
                    xytext=offset.get(name, (8, -14)), fontsize=7.5, color=c,
                    ha=align.get(name, "left"), va="top",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.5,
                                    shrinkA=0, shrinkB=4))
    ax.set_xlabel("false-alarm rate (alarms crediting no event, over non-drift windows)")
    ax.set_ylabel("detection rate")
    ax.set_xlim(-0.06, 1.0)
    # 1.02 is marker-glyph padding, not data headroom (ticks stop at 1.0; the axis used to run to
    # 1.14, making perfect detection look mid-range). The floor must clear the widest interval's
    # lower bound (PSI reaches 0.250) or the figure clips the width the caption relies on.
    # Floor at 0.2 or lower: with 0.04 of padding PSI's lower bound (0.250) sat on the axis edge.
    y_lo = min(0.2, min(drawn) - 0.04)
    ax.set_ylim(y_lo, 1.02)
    ax.set_yticks([t / 10 for t in range(int(round(y_lo * 10)), 11)])
    ax.axhline(1.0, color=INK, lw=0.5, ls=":")
    # No "ideal" marker: the ideal corner (0, 1.0) is exactly where SHAP-shift sits; the
    # direction goes in the title instead.
    ax.spines[["top", "right"]].set_visible(False)
    # plain text, not LaTeX — matplotlib renders an escaped percent literally
    ax.set_title(f"top-left is ideal;  bars: Wilson 95% CI on {n_pos} drift events / "
                 f"{n_neg} non-drift windows", fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "papers", "P5_temporal_drift", "figures", "fig_detectors.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def make_budget_table(bud):
    names = {"static": "Static", "periodic": "Periodic", "drift": "Drift-triggered", "active": "Active learning"}
    order = ["static", "periodic", "drift", "active"]
    lines = []
    for B in BUDGETS:
        cells = " & ".join(f"{bud[B][k][0]:.3f} ({bud[B][k][1]:,})".replace(",", "{,}") for k in order)
        lines.append(f"{B} & {cells} \\\\")
    body = "\n".join(lines)
    head = " & ".join(f"\\textbf{{{names[k]}}}" for k in order)
    tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{\\textbf{{Preliminary}} labelling-budget frontier: AUTC (total labels spent in
parentheses) per strategy and per-retrain budget $B$.}}
\\label{{tab:budget}}
\\small
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Budget $B$ (labels/retrain)}} & {head} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_budget.tex"), tex)


def main():
    build_input()
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    # (1) retraining strategies + figure
    strat, counts, psi_trace = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP)
    make_figure(strat, psi_trace)
    # Every caption quantity computed, not hardcoded: a literal "38%" and a "PhishVN v2" label
    # once survived a corpus recut, and on 2026-08-15 a hand-edit split the date span into
    # phishing-first-seen vs benign-certification (benign certification dates are not attacker
    # volume) without porting it here — the next regen clobbered it.
    import pandas as _pd
    _u = _pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"), low_memory=False)
    _s = _u["collected_at"].astype(str).str.strip().str.slice(0, 10)
    _d = _pd.to_datetime(_s, format="%d/%m/%Y", errors="coerce").fillna(
        _pd.to_datetime(_s, format="%Y-%m-%d", errors="coerce"))
    _ph, _be = (_u["label"] == "phishing") & _d.notna(), (_u["label"] == "benign") & _d.notna()
    spans = {"ph": f"{_d[_ph].min():%Y-%m}--{_d[_ph].max():%Y-%m}",
             "n_be": f"{int(_be.sum()):,}".replace(",", "{,}"),
             "be_max": f"{_d[_be].max():%Y-%m}"}
    make_overtime_table(strat, counts, len(df), spans, 100.0 * _d.notna().sum() / len(_u))
    # (2) drift detection
    truth, fires = run_detection(df, feats, WINDOWS, PERIOD, ADWIN_DELTA)
    make_drift_table(truth, fires)
    make_detector_figure(truth, fires)  # same truth/fires as the table — they cannot diverge
    # (3) labelling-budget frontier
    bud = run_budget(df, feats, WINDOWS, BUDGETS, period=PERIOD, psi_tau=PSI_TAU, f1_drop=F1_DROP)
    make_budget_table(bud)
    print("Done. Recompile the P5 manuscript to pick up the regenerated assets.")


if __name__ == "__main__":
    main()
