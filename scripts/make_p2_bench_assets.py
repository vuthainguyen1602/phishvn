#!/usr/bin/env python3
"""
make_p2_bench_assets.py — Generate P2's results tables from the experiment CSVs.

Tables are GENERATED, never hand-typed (the drift between a typed number and its source is the
exact failure check_paper_claims.py exists to catch — see the P8 217:217 story). Reads:
  data/processed/p2_benchmark.csv        (family x {random, temporal-of-normalize_merge})
  data/processed/p2_temporal_strict.csv  (family x {random_same_rows, temporal_strict})
  data/processed/cross_dataset_F1_CatBoost.csv + cross_dataset_F1.csv (RF, for the gap contrast)
  data/processed/p2_hpo.csv              (optional — emits a placeholder until the run lands)
  data/processed/p2_stacking_baseline.csv + p2_stacking_combos.csv    (ensemble table)
  data/processed/p2_temporal_strict_cb_k20.csv + p2_stacking_cblr_k20.csv (crossover verdict)
  data/processed/p2_forecastability_{decay,shiftmatrix,novelty}.csv   (forecastability macros)
Writes papers/P2_url_benchmark/sections/tab_*.tex + gen_*.tex verdict macros.

RUN:  python scripts/assets/make_p2_bench_assets.py
"""
from __future__ import annotations

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
from genfile import write_generated  # noqa: E402

SEC = "papers/P2_url_benchmark/sections"
FIG = "papers/P2_url_benchmark/figures"
ORDER = ["CatBoost", "MLP", "XGBoost", "LightGBM", "RandomForest", "HistGB", "LogReg"]


def fig_eval_design():
    """The central claim, visual: the F1 span BETWEEN evaluation designs dwarfs the span BETWEEN
    families (between-family spread collapses 0.053 at full-corpus random -> 0.014 on dated rows).
    CatBoost and LogReg drawn as the envelope so the booster's early lead visibly evaporates; the
    other five families recessive grey. Values read from the same CSVs the tab_* tables use.
    """
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()

    bench = pd.read_csv("data/processed/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2_temporal_strict.csv")

    def per_family(df, proto):
        return df[df["protocol"] == proto].groupby("family")["F1"].mean()

    stages = [
        ("Full corpus\nrandom", per_family(bench, "random")),
        ("Dated subset\nrandom", per_family(strict, "random_same_rows")),
        ("Dated subset\ntemporal", per_family(strict, "temporal_strict")),
    ]
    # Cross-dataset transfer: off-diagonal mean of each available family's matrix.
    xdata = {}
    for path, fam in [("data/processed/cross_dataset_F1.csv", "RandomForest"),
                      ("data/processed/cross_dataset_F1_CatBoost.csv", "CatBoost")]:
        if os.path.exists(path):
            m = pd.read_csv(path, index_col=0).values.astype(float)
            xdata[fam] = (m.sum() - np.trace(m)) / (m.size - len(m))

    fams = sorted(bench["family"].unique())
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    xs = list(range(len(stages)))
    # recessive families first, so the highlighted envelope sits on top
    for xi, (_, series) in zip(xs, stages):
        for fam in fams:
            if fam in ("CatBoost", "LogReg"):
                continue
            ax.plot(xi, series[fam], "o", ms=5, color=GRAY, alpha=0.55,
                    mec="white", mew=0.6, zorder=2)
    # envelope lines + dots for the two extremes
    for fam, col in [("CatBoost", ORANGE), ("LogReg", BLUE)]:
        ys = [s[fam] for _, s in stages]
        ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=3)
        ax.plot(xs, ys, "o", ms=7, color=col, mec="white", mew=0.8, zorder=4)

    # Cross-dataset stage. Only CatBoost and Random Forest have a transfer matrix, so the blue
    # envelope simply ends at stage 3 — which reads as a dropped series unless the grey dot that
    # DOES continue is named. Label it and give it its own dashed run-in.
    xc = len(stages)
    for fam, col in [("CatBoost", ORANGE), ("RandomForest", GRAY)]:
        if fam in xdata:
            ax.plot(xc, xdata[fam], "o", ms=7 if fam == "CatBoost" else 5,
                    color=col, mec="white", mew=0.8,
                    alpha=1.0 if fam == "CatBoost" else 0.55, zorder=4)
            ax.plot([xs[-1], xc], [stages[-1][1][fam], xdata[fam]], "--", color=col,
                    lw=1.4 if fam == "CatBoost" else 1.0,
                    alpha=0.7 if fam == "CatBoost" else 0.5, zorder=3)
    if "RandomForest" in xdata:
        ax.annotate("RandomForest", (xc, xdata["RandomForest"]), textcoords="offset points",
                    xytext=(-6, -12), ha="right", fontsize=7.5, color=GRAY)

    # between-family spread annotations, clear below the lowest dot of each cluster
    for xi, (_, series) in zip(xs, stages):
        sp = series.max() - series.min()
        ax.annotate(f"spread {sp:.3f}", (xi, series.min()), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=7.5, color=INK)

    # Direct labels for the envelope. LogReg is the MINIMUM of the first cluster, so the spread
    # annotation sits directly under its dot and the label cannot go below; it used to go right,
    # where its own horizontal line ran straight through the text. Left is the only free side.
    ax.annotate("CatBoost", (0, stages[0][1]["CatBoost"]), textcoords="offset points",
                xytext=(8, 5), fontsize=8, color=ORANGE, fontweight="bold")
    ax.annotate("LogReg", (0, stages[0][1]["LogReg"]), textcoords="offset points",
                xytext=(-9, -3), ha="right", fontsize=8, color=BLUE, fontweight="bold")
    if "CatBoost" in xdata:
        ax.annotate("cross-dataset\ntransfer", (xc, xdata["CatBoost"]),
                    textcoords="offset points", xytext=(-4, 12), ha="center",
                    fontsize=7.5, color=INK)

    ax.set_xticks(xs + [xc])
    ax.set_xticklabels([s[0] for s in stages] + ["Cross-\ndataset"])
    ax.set_xlim(-0.4, xc + 0.45)
    ax.set_ylim(0.58, 0.95)
    ax.set_ylabel("F1")
    ax.margins(x=0.02)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "eval_design.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


PR_BENCH = "data/processed/p2_pr_curves.csv"
PR_STRICT = "data/processed/p2_pr_curves_strict.csv"


def fig_family_pr():
    """Seven families, one protocol, seven curves on top of each other: the family result is a
    NULL (CatBoost/MLP/XGBoost agree to three decimals in F1 and PR-AUC), and curves make a null
    convincing where a column of 0.977s reads as a formatting accident. Only LogReg and Random
    Forest separate, low. Curves are persisted by run_p2_benchmark from the SAME fits as
    tab_families — a fresh run would resample the splits and put figure and table a hair apart."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    if not os.path.exists(PR_STRICT):
        print(f"[i] {PR_STRICT} absent — run run_p2_temporal_strict.py; skipping family PR figure")
        return
    cur = pd.read_csv(PR_STRICT)
    bench = pd.read_csv("data/processed/p2_temporal_strict.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    # Families that separate get identity; the coinciding five share one colour (naming five
    # overlapping curves implies a distinction to look for). RandomForest sits below the field,
    # so INK dashed — at GRAY against the GRAY pack it was invisible as a distinct series.
    hero = {"CatBoost": (ORANGE, "-", 1.9), "LogReg": (BLUE, "-", 1.9),
            "RandomForest": (INK, "--", 1.5)}
    drawn_pack = False
    floor = 1.0
    for fam in ORDER:
        sub = cur[(cur.family == fam) & (cur.protocol == "temporal_strict")].sort_values("recall")
        if sub.empty:
            continue
        auc = bench[(bench.family == fam) &
                    (bench.protocol == "temporal_strict")]["PR-AUC"].mean()
        floor = min(floor, float(sub.precision.min()))
        if fam in hero:
            col, ls, lw = hero[fam]
            ax.plot(sub.recall, sub.precision, ls, color=col, lw=lw, zorder=3,
                    label=f"{fam} — {auc:.3f}")
        else:
            ax.plot(sub.recall, sub.precision, "-", color=GRAY, lw=1.0, alpha=0.8,
                    zorder=2,
                    label="HistGB, MLP, XGBoost, LightGBM" if not drawn_pack else None)
            drawn_pack = True
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.0)
    # Floor = lowest precision reached, not a round number: at a fixed 0.55 every terminal point
    # (precision at recall 1.0) fell just underneath and the curves ran off the bottom edge.
    ax.set_ylim(min(0.55, floor - 0.02), 1.01)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left", title="PR-AUC", title_fontsize=8)
    ax.set_title("strict-temporal protocol, mean over 5 seeds", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, "pr_families.pdf")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def fig_design_pr():
    """One family, three evaluation designs, three curves far apart — the thesis in one frame:
    hold the model fixed, vary only train/test assignment, and the curve moves further than any
    family choice moved it. Cross-dataset is not drawn: a pooled PR curve across corpora would
    compare precisions against different base rates; Table~\\ref{tab:xdataset} carries it."""
    from figstyle import apply, ORANGE, BLUE, GRAY
    plt = apply()
    if not (os.path.exists(PR_BENCH) and os.path.exists(PR_STRICT)):
        print("[i] PR curve caches absent — skipping design PR figure")
        return
    fam = "CatBoost"
    stages = [
        (pd.read_csv(PR_BENCH), "random", "data/processed/p2_benchmark.csv",
         "full corpus, random", BLUE, "-"),
        (pd.read_csv(PR_STRICT), "random_same_rows", "data/processed/p2_temporal_strict.csv",
         "dated subset, random", GRAY, "--"),
        (pd.read_csv(PR_STRICT), "temporal_strict", "data/processed/p2_temporal_strict.csv",
         "dated subset, temporal", ORANGE, "-"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    floor = 1.0
    for cur, proto, metpath, lab, col, ls in stages:
        sub = cur[(cur.family == fam) & (cur.protocol == proto)].sort_values("recall")
        if sub.empty:
            continue
        floor = min(floor, float(sub.precision.min()))
        met = pd.read_csv(metpath)
        auc = met[(met.family == fam) & (met.protocol == proto)]["PR-AUC"].mean()
        ax.plot(sub.recall, sub.precision, ls, color=col, lw=1.9, zorder=3,
                label=f"{lab} — {auc:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.0)
    # Floor = lowest precision reached, not a round number: at a fixed 0.55 every terminal point
    # (precision at recall 1.0) fell just underneath and the curves ran off the bottom edge.
    ax.set_ylim(min(0.55, floor - 0.02), 1.01)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left", title="PR-AUC", title_fontsize=8)
    ax.set_title(f"{fam} held fixed; only the evaluation design changes", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, "pr_designs.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def fig_prauc_designs():
    """The headline claim on a threshold-free metric, answering "is it a thresholding artefact?"
    Under PR-AUC the seven families span only 0.009 at full-corpus random — the 5.3-point F1
    booster lead is almost entirely calibration — while the DESIGN steps move the level by 0.076.
    That is 8.4x the full-corpus-random spread but only 2.9x the strict-stage one (0.027), so
    "an order of magnitude more than the spread at any stage", which this docstring asserted and
    which reached the abstract and §6, was true of the widest stage and not of the narrowest.
    """
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    bench = pd.read_csv("data/processed/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2_temporal_strict.csv")

    def per_family(df, proto):
        return df[df["protocol"] == proto].groupby("family")["PR-AUC"].mean()

    stages = [("Full corpus\nrandom", per_family(bench, "random")),
              ("Dated subset\nrandom", per_family(strict, "random_same_rows")),
              ("Dated subset\ntemporal", per_family(strict, "temporal_strict"))]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    xs = list(range(len(stages)))
    fams = sorted(bench["family"].unique())
    for xi, (_, series) in zip(xs, stages):
        for fam in fams:
            if fam in ("CatBoost", "LogReg"):
                continue
            ax.plot(xi, series[fam], "o", ms=5, color=GRAY, alpha=0.55, mec="white", mew=0.6,
                    zorder=2)
    for fam, col in (("CatBoost", ORANGE), ("LogReg", BLUE)):
        ys = [st[fam] for _, st in stages]
        ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=3)
        ax.plot(xs, ys, "o", ms=7, color=col, mec="white", mew=0.8, zorder=4)
    # Above the cluster, not below it: below the last cluster the annotation landed on the
    # x tick label, and the level falls stage to stage so the space overhead is always free.
    for xi, (_, series) in zip(xs, stages):
        # The first cluster's overhead is taken by the CatBoost series label, so its spread
        # annotation moves left of centre; the rest stay centred.
        dx, ha = (-8, "right") if xi == 0 else (0, "center")
        ax.annotate(f"spread {series.max() - series.min():.3f}", (xi, series.max()),
                    textcoords="offset points", xytext=(dx, 9), ha=ha, fontsize=7.5, color=INK)
    ax.annotate("CatBoost", (0, stages[0][1]["CatBoost"]), textcoords="offset points",
                xytext=(8, 4), fontsize=8, color=ORANGE, fontweight="bold")
    ax.annotate("LogReg", (0, stages[0][1]["LogReg"]), textcoords="offset points",
                xytext=(-9, -3), ha="right", fontsize=8, color=BLUE, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([st[0] for st in stages])
    ax.set_xlim(-0.45, len(stages) - 0.55)
    lo = min(st.min() for _, st in stages)
    hi = max(st.max() for _, st in stages)
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.16 * (hi - lo))   # headroom for the annotations
    ax.set_ylabel("PR-AUC")
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "pr_auc_designs.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def fig_prauc_cost():
    """What the family choice costs vs what it buys: on the honest protocol the whole field lies
    inside 0.021 PR-AUC while training cost spans two orders of magnitude — the cheapest
    competitive model is a rational default, not a compromise.
    """
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    df = pd.read_csv("data/processed/p2_temporal_strict.csv")
    d = df[df.protocol == "temporal_strict"].groupby("family").agg(
        prauc=("PR-AUC", "mean"), secs=("fit_seconds", "mean"))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, lw=0.6)
    # Label side decided from the data: fixed right-side labels printed LogReg through LightGBM
    # (near-identical y is the NORMAL case here — families sit 0.001 PR-AUC apart). A family whose
    # near-twin sits to its right within 3x in fit time takes its label on the left.
    yr = d.prauc.max() - d.prauc.min()
    for fam, r in d.iterrows():
        col = ORANGE if fam == "CatBoost" else BLUE if fam == "LogReg" else GRAY
        crowded = any(abs(o.prauc - r.prauc) < 0.08 * yr and 1.0 < o.secs / r.secs < 3.0
                      for f2, o in d.iterrows() if f2 != fam)
        ax.plot(r.secs, r.prauc, "o", ms=8 if col != GRAY else 6, color=col,
                mec="white", mew=0.8, zorder=3)
        ax.annotate(fam, (r.secs, r.prauc), textcoords="offset points",
                    xytext=(-10, -3) if crowded else (10, -3),
                    ha="right" if crowded else "left",
                    fontsize=7.5, color=col if col != GRAY else INK)
    ax.set_xscale("log")
    ax.set_xlabel("mean fit time per run (s, log scale)")
    ax.set_ylabel("PR-AUC (strict-temporal)")
    ax.set_xlim(d.secs.min() * 0.36, d.secs.max() * 3.2)
    span = d.prauc.max() - d.prauc.min()
    ax.set_ylim(d.prauc.min() - 0.25 * span, d.prauc.max() + 0.25 * span)
    ax.axhspan(d.prauc.min(), d.prauc.max(), color=GRAY, alpha=0.12, zorder=0)
    ax.annotate(f"whole field: {span:.3f} PR-AUC", (d.secs.max() * 2.6, d.prauc.max()),
                ha="right", va="bottom", fontsize=7.5, color=INK)
    out = os.path.join(FIG, "pr_auc_cost.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def ms(g, k):
    # std rides in \scriptsize so six mean+-std columns fit the 466pt text block (the full-size
    # form overflowed the margin and the reader anyway compares means first)
    return f"{g[k].mean():.3f}{{\\scriptsize\\,$\\pm${g[k].std() if len(g) > 1 else 0:.3f}}}"


def tab_families():
    df = pd.read_csv("data/processed/p2_benchmark.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Family benchmark under the corpus' bundled splits, mean$\\pm$std over 5 seeds."
        " FPR@0.90 = false-positive rate at 0.90 phishing recall.}",
        "\\label{tab:families}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        # ROC-AUC always sat in p2_benchmark.csv but was never printed; a benchmark reviewer
        # will ask for it.
        "\\begin{tabular}{lcccccccc}\\toprule",
        " & \\multicolumn{4}{c}{Random} & \\multicolumn{4}{c}{Partly-temporal} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Family & F1 & PR-AUC & ROC-AUC & FPR@0.90 & F1 & PR-AUC & ROC-AUC & FPR@0.90 \\\\ \\midrule",
    ]
    for f in ORDER:
        r = df[(df.family == f) & (df.protocol == "random")]
        t = df[(df.family == f) & (df.protocol == "temporal")]
        lines.append(f"{f} & {ms(r,'F1')} & {ms(r,'PR-AUC')} & {ms(r,'ROC-AUC')} & "
                     f"{ms(r,'FPR@R0.90')} & {ms(t,'F1')} & {ms(t,'PR-AUC')} & "
                     f"{ms(t,'ROC-AUC')} & {ms(t,'FPR@R0.90')} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def tab_strict():
    df = pd.read_csv("data/processed/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{The seven families on the same dated row set under the two protocols;"
        " $\\Delta$F1 = strict-temporal minus random.}",
        "\\label{tab:strict}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccccccc}\\toprule",
        " & \\multicolumn{3}{c}{Random (same rows)} & \\multicolumn{3}{c}{Strict-temporal} & \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Family & F1 & PR-AUC & ROC-AUC & F1 & PR-AUC & ROC-AUC & $\\Delta$F1 \\\\ \\midrule",
    ]
    for f in ORDER:
        r = df[(df.family == f) & (df.protocol == "random_same_rows")]
        t = df[(df.family == f) & (df.protocol == "temporal_strict")]
        d = t["F1"].mean() - r["F1"].mean()
        lines.append(f"{f} & {ms(r,'F1')} & {ms(r,'PR-AUC')} & {ms(r,'ROC-AUC')} & "
                     f"{ms(t,'F1')} & {ms(t,'PR-AUC')} & {ms(t,'ROC-AUC')} & {d:+.3f} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


STACKS = [("Stacking", "Stack[RF+XGB]"), ("Stacking+HFS", "Stack[RF+XGB]+HFS"),
          ("Stack[XGB+LGBM]", "Stack[XGB+LGBM]"), ("Stack[CB+HGB]", "Stack[CB+HGB]"),
          ("Stack[CB+MLP]", "Stack[CB+MLP]"), ("Stack[CB+HGB+LR+MLP]", "Stack[CB+HGB+LR+MLP]"),
          ("Stack[CB+LR]", "Stack[CB+LR]")]


def tab_stacking():
    """Ensemble baselines on the SAME rows/protocols as tab_strict. Sources:
    p2_stacking_baseline.csv (the JNCA recipe RF+XGB, +HFS) and p2_stacking_combos.csv
    (the base-learner sweep), both 5 seeds; CatBoost reference row from p2_temporal_strict.csv."""
    df = pd.concat([pd.read_csv("data/processed/p2_stacking_baseline.csv"),
                    pd.read_csv("data/processed/p2_stacking_combos.csv")])
    ref = pd.read_csv("data/processed/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Stacked ensembles (Algorithm~\\ref{alg:stacking}) on the rows and protocols of"
        " Table~\\ref{tab:strict}. Gap = random minus strict-temporal PR-AUC; +HFS = hybrid"
        " feature selection.}",
        "\\label{tab:stacking}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccccc}\\toprule",
        " & \\multicolumn{2}{c}{Random (same rows)} & \\multicolumn{2}{c}{Strict-temporal} & & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Ensemble & F1 & PR-AUC & F1 & PR-AUC & Gap & FPR@0.90 (temp.) \\\\ \\midrule",
    ]
    cb_r = ref[(ref.family == "CatBoost") & (ref.protocol == "random_same_rows")]
    cb_t = ref[(ref.family == "CatBoost") & (ref.protocol == "temporal_strict")]
    lines.append(f"CatBoost (reference) & {ms(cb_r,'F1')} & {ms(cb_r,'PR-AUC')} & "
                 f"{ms(cb_t,'F1')} & {ms(cb_t,'PR-AUC')} & "
                 f"{cb_r['PR-AUC'].mean() - cb_t['PR-AUC'].mean():.3f} & "
                 f"{ms(cb_t,'FPR@R0.90')} \\\\ \\midrule")
    for key, label in STACKS:
        r = df[(df.family == key) & (df.protocol == "random_same_rows")]
        t = df[(df.family == key) & (df.protocol == "temporal_strict")]
        lines.append(f"{label} & {ms(r,'F1')} & {ms(r,'PR-AUC')} & {ms(t,'F1')} & "
                     f"{ms(t,'PR-AUC')} & {r['PR-AUC'].mean() - t['PR-AUC'].mean():.3f} & "
                     f"{ms(t,'FPR@R0.90')} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def tab_shiftmatrix():
    """The E2 shift-localization matrix as a compact upper-triangle table."""
    df = pd.read_csv("data/processed/p2_forecastability_shiftmatrix.csv")
    names = ["W1", "W2", "W3", "W4", "TEST"]
    cell = {(r.a, r.b): (r.auc, r.delta_days) for r in df.itertuples()}
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Out-of-fold ROC-AUC of a phishing-only discriminator (HistGB, 5-fold) between"
        " time blocks; small figures: distance between block median dates.}",
        "\\label{tab:shiftmatrix}",
        "\\small\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{l" + "c" * (len(names) - 1) + "}\\toprule",
        " & " + " & ".join(names[1:]) + " \\\\ \\midrule",
    ]
    for i, a in enumerate(names[:-1]):
        cells = []
        for b in names[i + 1:]:
            auc, dd = cell[(a, b)]
            cells.append(f"{auc:.3f}{{\\scriptsize\\,({int(dd)}d)}}")
        pad = [""] * i
        lines.append(a + " & " + " & ".join(pad + cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


def tab_confusion():
    """CatBoost confusion counts at the paper's operating point (recall 0.90), both same-row
    protocols. Derived, not re-measured: TP/FN follow from the target recall and the guarded
    phishing test count; FP/TN from the seed-mean FPR@0.90 in p2_temporal_strict.csv and the
    seed-mean benign test size (the per-seed masks are RandomState(s), so reproduced exactly).
    A 0.5-threshold matrix is deliberately absent: Section~V argues that operating point is a
    calibration artefact."""
    from run_p2_temporal_strict import load as _load
    df = _load()
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date")
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    n_ph = int((~ph_te.rdom.isin(set(ph_tr.rdom))).sum())
    n_be_pool = int((df.y == 0).sum())
    n_be = int(np.mean([(np.random.RandomState(s).rand(n_be_pool) >= 0.70).sum()
                        for s in range(5)]))
    strict = pd.read_csv("data/processed/p2_temporal_strict.csv")
    tp = round(0.90 * n_ph)
    fn = n_ph - tp
    cells = {}
    for proto in ("random_same_rows", "temporal_strict"):
        fpr = strict[(strict.family == "CatBoost")
                     & (strict.protocol == proto)]["FPR@R0.90"].mean()
        fp = round(fpr * n_be)
        cells[proto] = (fp, n_be - fp)
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{CatBoost confusion counts at the FPR@0.90 operating point (recall fixed at"
        " $0.90$), derived from the runs of Table~\\ref{tab:strict}; benign test size is the"
        " seed mean.}",
        "\\label{tab:confusion}",
        "\\small\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lcccc}\\toprule",
        " & \\multicolumn{2}{c}{Random (same rows)} & \\multicolumn{2}{c}{Strict-temporal} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Actual $\\backslash$ Pred. & phish & benign & phish & benign \\\\ \\midrule",
        f"Phishing ($n = {n_ph:,}$) & {tp:,} & {fn:,} & {tp:,} & {fn:,} \\\\",
        f"Benign ($n = {n_be:,}$) & {cells['random_same_rows'][0]:,} & "
        f"{cells['random_same_rows'][1]:,} & {cells['temporal_strict'][0]:,} & "
        f"{cells['temporal_strict'][1]:,} \\\\",
        "\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


def gen_forecastability():
    """Macros for the drift-forecastability subsection, computed from the three diagnostic
    CSVs written by run_p2_drift_forecastability.py."""
    from scipy.stats import spearmanr

    # (a) internal decay vs canonical gap, per origin x metric
    dec = pd.read_csv("data/processed/p2_forecastability_decay.csv")
    canon = pd.read_csv("data/processed/p2_temporal_strict.csv")
    g = canon.groupby(["family", "protocol"])["PR-AUC"].mean().unstack()
    gap = g["random_same_rows"] - g["temporal_strict"]
    rhos = []
    for origin in sorted(dec.origin.unique()):
        for metric in ("PR-AUC", "ROC-AUC"):
            slopes = {}
            sub = dec[dec.origin == origin]
            for fam in sub.family.unique():
                gg = sub[sub.family == fam].groupby("win")[["horizon_d", metric]].mean()
                slopes[fam] = np.polyfit(gg["horizon_d"], gg[metric], 1)[0]
            s_ = pd.Series(slopes)
            rho, p = spearmanr(-s_[gap.index], gap)
            rhos.append((origin, metric, rho, p))
    rho_txt = "; ".join(f"origin {o:.2f}/{m}: $\\rho = {r:+.2f}$ ($p = {p:.2f}$)"
                        for o, m, r, p in rhos)
    # By MAGNITUDE. max() over signed rhos returned +0.32 and printed "never exceeds $\\rho=+0.32$"
    # three words before listing $\\rho=-0.46$ -- a correlation-strength claim read as a signed one.
    worst = max((r for _, _, r, _ in rhos), key=abs)
    # Two-sided as well: the tripwire only fired on positive rhos, so a strong NEGATIVE correlation
    # -- equally fatal to a "no relationship" sentence -- would have passed silently.
    if any(abs(r) > 0.5 and p < 0.05 for _, _, r, p in rhos):
        raise SystemExit("gen_forecastability: an internal-decay correlation came out "
                         "strong and significant in either direction — the 'does not reproduce' "
                         "sentence is no longer true; reword the subsection before regenerating.")
    win_means = (dec[(dec.origin == 0.50) & (dec.family == "CatBoost")]
                 .groupby("win")["PR-AUC"].mean())
    swing = win_means.max() - win_means.min()
    decay = (f"Across both origins and both metrics the correlation between a family's "
             f"internal decay slope and its canonical random-minus-temporal gap never "
             f"reaches significance, and the strongest of them is "
             f"$\\rho = {worst:+.2f}$ ({rho_txt}). "
             f"The obstacle is one of scale: window-to-window difficulty moves CatBoost by "
             f"${swing:.2f}$ PR-AUC across the four forward windows --- "
             f"{swing / (gap['CatBoost']):.0f}$\\times$ the ${gap['CatBoost']:.3f}$ gap the "
             f"signal would need to resolve"
             + (" --- and the movement is upward, not the decay the extrapolation methods "
                "assume." if win_means.iloc[-1] > win_means.iloc[0] else "."))

    # (b) the matrix summary
    mx = pd.read_csv("data/processed/p2_forecastability_shiftmatrix.csv")
    internal = mx[mx.a.str.startswith("W") & mx.b.str.startswith("W")]
    adj = internal[internal.b.str[1].astype(int) - internal.a.str[1].astype(int) == 1]
    bd = mx[(mx.a == "W4") & (mx.b == "TEST")].iloc[0]
    w1w4 = mx[(mx.a == "W1") & (mx.b == "W4")].iloc[0]
    w1te = mx[(mx.a == "W1") & (mx.b == "TEST")].iloc[0]
    matrix = (f"Adjacent internal pairs average AUC ${adj.auc.mean():.3f}$, while the "
              f"boundary pair (W4 vs.\\ test) reaches ${bd.auc:.3f}$ over the shortest "
              f"median-date distance in the table ({int(bd.delta_days)} days); and the "
              f"matrix is not monotone in time --- the test window resembles the "
              f"\\emph{{earliest}} training block (${w1te.auc:.3f}$ at "
              f"{int(w1te.delta_days)} days) more than the last training block does "
              f"(${w1w4.auc:.3f}$ at {int(w1w4.delta_days)} days).")

    # (c) the novelty probe
    nov = pd.read_csv("data/processed/p2_forecastability_novelty.csv")
    cb = nov[nov.family == "CatBoost"]
    novelty = (f"Train-only lexical novelty does not predict where CatBoost fails forward "
               f"in time: AUC(novelty; error) $= {cb.auc_all.mean():.2f}$ over the full "
               f"test window and ${cb.auc_phish.mean():.2f}$ on its phishing side "
               f"(pre-set keep-alive bar: $0.65$), so instance-level backoff has nothing "
               f"to key on.")

    return ("\\newcommand{\\ForecastDecayVerdict}{" + decay + "}\n"
            "\\newcommand{\\ForecastMatrixVerdict}{" + matrix + "}\n"
            "\\newcommand{\\ForecastNoveltyVerdict}{" + novelty + "}")


def gen_stacking_verdict():
    """The stacking significance sentences, computed from the CSVs so printed p/q-values can
    never drift from the runs. Two parts: (a) the K=5 family of ensemble-vs-CatBoost paired
    tests under strict-temporal, BH-adjusted; (b) the K=20 crossover for Stack[CB+LR].
    Regenerate the K=20 sources with:
      run_p2_temporal_strict.py --families CatBoost --seeds 20 --out .../p2_temporal_strict_cb_k20.csv
      run_p2_stacking_baseline.py --seeds 20 --bases CatBoost+LogReg --out .../p2_stacking_cblr_k20.csv"""
    from paired_eval import bh_adjust, corrected_paired_t, fmt_p
    df5 = pd.concat([pd.read_csv("data/processed/p2_stacking_baseline.csv"),
                     pd.read_csv("data/processed/p2_stacking_combos.csv")])
    ref5 = pd.read_csv("data/processed/p2_temporal_strict.csv")

    def paired(a_df, b_df, fam_a, fam_b, proto, k_col="PR-AUC"):
        a = a_df[(a_df.family == fam_a) & (a_df.protocol == proto)].set_index("seed")[k_col]
        b = b_df[(b_df.family == fam_b) & (b_df.protocol == proto)].set_index("seed")[k_col]
        return corrected_paired_t((b.sort_index() - a.sort_index()).dropna().values,
                                  test_frac=0.30)

    fams = [k for k, _ in STACKS if k != "Stacking+HFS"]  # HFS is an ablation, not a member
    res = {f: paired(ref5, df5, "CatBoost", f, "temporal_strict") for f in fams}
    qs, _ = bh_adjust([res[f]["p"] for f in fams])
    q = dict(zip(fams, qs))
    tree = (f"Under the strict-temporal protocol the literature's recipe is not merely "
            f"unhelpful but significantly \\emph{{worse}} than its strongest constituent's "
            f"family peer: Stack[RF+XGB] trails CatBoost by "
            f"${abs(res['Stacking']['mean']):.4f}$ PR-AUC "
            f"(corrected, BH-adjusted ${fmt_p(q['Stacking'], 'q')}$) and Stack[XGB+LGBM] by "
            f"${abs(res['Stack[XGB+LGBM]']['mean']):.4f}$ (${fmt_p(q['Stack[XGB+LGBM]'], 'q')}$), "
            f"while the CatBoost-anchored mixed stacks without LogReg are statistically "
            f"indistinguishable from CatBoost alone.")

    cb = pd.read_csv("data/processed/p2_temporal_strict_cb_k20.csv")
    st = pd.read_csv("data/processed/p2_stacking_cblr_k20.csv")
    t = paired(cb, st, "CatBoost", "Stack[CB+LR]", "temporal_strict")
    r = paired(cb, st, "CatBoost", "Stack[CB+LR]", "random_same_rows")
    cross = (f"Stack[CB+LR] beats CatBoost under the strict-temporal protocol "
             f"(paired $\\Delta$PR-AUC $= {t['mean']:+.4f}$, {t['wins']}/{t['k']} seeds, "
             f"corrected ${fmt_p(t['p'])}$) and loses to it under random-same-rows "
             f"($\\Delta = {r['mean']:+.4f}$, {r['k'] - r['wins']}/{r['k']} seeds against, "
             f"corrected ${fmt_p(r['p'])}$): the crossover is significant in \\emph{{both}} "
             f"directions.")
    return ("\\newcommand{\\StackTreeVerdict}{" + tree + "}\n"
            "\\newcommand{\\StackCrossoverVerdict}{" + cross + "}")


def tab_decomp():
    """Decompose the random-full -> strict-temporal drop into its two steps: composition
    (full corpus -> dated-row subset, protocol still random) and protocol (random -> temporal
    on the SAME dated rows). This is the table the headline claim must rest on."""
    full = pd.read_csv("data/processed/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Decomposing the \\textbf{F1} drop. $\\Delta_{comp}$ = dated-subset random"
        " minus full-corpus random (\\emph{composition}, protocol unchanged); $\\Delta_{proto}$"
        " = strict-temporal minus dated-subset random (\\emph{protocol}, rows unchanged).}",
        "\\label{tab:decomp}",
        "\\begin{tabular}{lccccc}\\toprule",
        "Family & Random (full) & Random (dated rows) & Strict-temporal & $\\Delta_{comp}$ &"
        " $\\Delta_{proto}$ \\\\ \\midrule",
    ]
    for f in ORDER:
        a = full[(full.family == f) & (full.protocol == "random")]["F1"].mean()
        b = strict[(strict.family == f) & (strict.protocol == "random_same_rows")]["F1"].mean()
        c = strict[(strict.family == f) & (strict.protocol == "temporal_strict")]["F1"].mean()
        lines.append(f"{f} & {a:.3f} & {b:.3f} & {c:.3f} & {b-a:+.3f} & {c-b:+.3f} \\\\")

    def spread(vals):
        return max(vals) - min(vals)

    sa = spread([full[(full.family == f) & (full.protocol == "random")]["F1"].mean()
                 for f in ORDER])
    sb = spread([strict[(strict.family == f) &
                        (strict.protocol == "random_same_rows")]["F1"].mean() for f in ORDER])
    sc = spread([strict[(strict.family == f) &
                        (strict.protocol == "temporal_strict")]["F1"].mean() for f in ORDER])
    lines.append("\\midrule")
    lines.append(f"family spread & {sa:.3f} & {sb:.3f} & {sc:.3f} & & \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def tab_gwo_cv():
    """The GWO-vs-random-search CV head-to-head the adoption rule was applied to. Printing it
    is the difference between reporting a null and burying one."""
    df = pd.read_csv("data/processed/p2_hpo.csv")
    df = df[df.config != "default"].copy()
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Best 3-fold CV PR-AUC on the strict-temporal train window: Grey Wolf Optimizer"
        " vs.\\ equal-budget uniform random search.}",
        "\\label{tab:gwocv}",
        "\\begin{tabular}{lccc}\\toprule",
        "Family & CV PR-AUC (GWO) & CV PR-AUC (random) & budget \\\\ \\midrule",
    ]
    for _, r in df.iterrows():
        if pd.isna(r.get("cv_prauc_gwo")):
            continue
        lines.append(f"{r.family} & {r.cv_prauc_gwo:.4f} & {r.cv_prauc_random:.4f} & "
                     f"{int(r.budget)} evals \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


def gen_decomp_macros():
    """Emit the numbers Figure~\\ref{fig:decomp} prints, as \\newcommand macros.

    The figure is hand-drawn TikZ (figures/decomposition.tex) because its content is structural,
    but every NUMBER comes from here: a number typed into TikZ would be the one place a stale
    figure could survive a data change unnoticed (check_paper_claims reads .tex, not compiled
    PDFs). Macros mean the figure cannot disagree with tab_decomp — both read the same CSVs."""
    full = pd.read_csv("data/processed/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2_temporal_strict.csv")

    def mean_f1(df, proto, fam):
        return float(df[(df.family == fam) & (df.protocol == proto)]["F1"].mean())

    stage = {
        "Full": [mean_f1(full, "random", f) for f in ORDER],
        "Dated": [mean_f1(strict, "random_same_rows", f) for f in ORDER],
        "Temp": [mean_f1(strict, "temporal_strict", f) for f in ORDER],
    }
    comp = [stage["Dated"][i] - stage["Full"][i] for i in range(len(ORDER))]
    proto = [stage["Temp"][i] - stage["Dated"][i] for i in range(len(ORDER))]

    # Row counts come from the experiment's OWN definition of a dated row, not a re-derivation.
    rows_full = rows_dated = None
    try:
        from run_p2_temporal_strict import load as _load  # noqa: E402
        df = _load()
        # The composition step drops UNDATED PHISHING only, keeping the benign pool whole (what
        # run_p2_temporal_strict does). Counting dated rows across both classes gave a corpus-wide
        # total no experiment was ever run on, printed beside an F1 computed on the right one.
        is_ph = df["label"].astype(int).eq(1)      # the loader codes phishing as 1, not a string
        rows_full = len(df)
        rows_dated = int((~is_ph).sum() + (is_ph & df["date"].notna()).sum())
    except Exception as e:
        print(f"[i] row counts unavailable ({type(e).__name__}) — figure will omit them.")

    def num(v):
        return f"{v:.3f}".lstrip("0") if abs(v) < 1 else f"{v:.3f}"

    m = {
        "FullBest": num(max(stage["Full"])), "FullSpread": num(max(stage["Full"]) - min(stage["Full"])),
        "DatedBest": num(max(stage["Dated"])), "DatedSpread": num(max(stage["Dated"]) - min(stage["Dated"])),
        "TempBest": num(max(stage["Temp"])), "TempSpread": num(max(stage["Temp"]) - min(stage["Temp"])),
        "CompWorst": f"{min(comp):+.3f}", "CompBest": f"{max(comp):+.3f}",
        "ProtoWorst": f"{min(proto):+.3f}", "ProtoBest": f"{max(proto):+.3f}",
    }
    if rows_full:
        m.update(RowsFull=f"{rows_full:,}", RowsDated=f"{rows_dated:,}",
                 RowsUndated=f"{rows_full - rows_dated:,}")
    body = "\n".join(f"\\newcommand{{\\decomp{k}}}{{{v}}}" for k, v in m.items())
    write_generated("papers/P2_url_benchmark/sections/gen_decomp_macros.tex",
                    body,
                    f"(spread {m['FullSpread']} -> {m['DatedSpread']} -> {m['TempSpread']})")


def gen_strict_verdict():
    """Generated prose: paired-by-seed CatBoost-vs-LogReg under strict-temporal, t-based 95% CI.
    Same corrected resampled t as the content experiments; only the BENIGN complement is redrawn
    per seed, so charging the full 0.30 test fraction is conservative — safe for a reported null,
    since the correction can only widen the interval and strengthen the tie."""
    from paired_eval import corrected_paired_t
    from scipy import stats

    df = pd.read_csv("data/processed/p2_temporal_strict.csv")
    t = df[df.protocol == "temporal_strict"]
    cb = t[t.family == "CatBoost"].sort_values("seed")["F1"].to_numpy()
    lr = t[t.family == "LogReg"].sort_values("seed")["F1"].to_numpy()
    s = corrected_paired_t(cb - lr)
    n, mean, p = s["k"], s["mean"], s["p"]
    ci = stats.t.ppf(0.975, n - 1) * s["se"] if s["se"] else 0.0
    return (
            f"Paired by seed (the benign split mask is shared within a seed), CatBoost minus"
            f" logistic regression under the strict-temporal protocol is "
            f"${mean:+.4f}$ F1 (95\\% CI $\\pm{ci:.4f}$, corrected resampled $t$-test "
            f"$p={p:.2f}$, $n={n}$ seeds): the two families are statistically tied.")


def tab_xdataset():
    cb = pd.read_csv("data/processed/cross_dataset_F1_CatBoost.csv", index_col=0)
    rf = pd.read_csv("data/processed/cross_dataset_F1.csv", index_col=0)

    def gap(m):
        v = m.to_numpy(float)
        d = np.diag(v)
        off = v[~np.eye(len(v), dtype=bool)]
        return d.mean(), off.mean(), d.mean() - off.mean()

    cd, co, cg = gap(cb)
    rd, ro, rg = gap(rf)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Cross-dataset F1 for CatBoost (rows = train corpus, columns = test corpus);"
        f" generalisation gap {cg:.3f}.}}",
        "\\label{tab:xdataset}",
        "\\begin{tabular}{l" + "c" * len(cb.columns) + "}\\toprule",
        "Train $\\backslash$ Test & " + " & ".join(cb.columns) + " \\\\ \\midrule",
    ]
    for name, row in cb.iterrows():
        cells = [f"\\textbf{{{v:.3f}}}" if c == name else f"{v:.3f}"
                 for c, v in row.items()]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def tab_hpo():
    path = "data/processed/p2_hpo.csv"
    if not os.path.exists(path):
        return "% p2_hpo.csv not generated yet -- run scripts/train/run_p2_hpo.py\n[HPO table to be filled]"
    df = pd.read_csv(path)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Tuned vs.\\ default configurations, tuned on the strict-temporal train"
        " window only and evaluated once on the test window; mean$\\pm$std over 3 model"
        " seeds.}",
        "\\label{tab:hpo}",
        "\\begin{tabular}{llcccc}\\toprule",
        "Family & Config & F1 & PR-AUC & ROC-AUC & FPR@0.90 \\\\ \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r.family} & {r.config} & {r['F1']:.3f}$\\pm${r['F1_std']:.3f} & "
                     f"{r['PR-AUC']:.3f}$\\pm${r['PR-AUC_std']:.3f} & "
                     f"{r['ROC-AUC']:.3f}$\\pm${r['ROC-AUC_std']:.3f} & "
                     f"{r['FPR@R0.90']:.3f}$\\pm${r['FPR@R0.90_std']:.3f} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def main():
    os.makedirs(SEC, exist_ok=True)
    for name, fn in [("tab_families", tab_families), ("tab_strict", tab_strict),
                     ("tab_decomp", tab_decomp), ("tab_gwo_cv", tab_gwo_cv),
                     ("gen_strict_verdict", gen_strict_verdict),
                     ("tab_stacking", tab_stacking),
                     ("gen_stacking_verdict", gen_stacking_verdict),
                     ("tab_shiftmatrix", tab_shiftmatrix),
                     ("tab_confusion", tab_confusion),
                     ("gen_forecastability", gen_forecastability),
                     ("tab_xdataset", tab_xdataset), ("tab_hpo", tab_hpo)]:
        # fn() is evaluated BEFORE the target is opened; the old line truncated first and an
        # exception in fn left the asset empty. See scripts/lib/genfile.py.
        write_generated(os.path.join(SEC, name + ".tex"), fn())
    gen_decomp_macros()  # writes its own file (macros for the decomposition figure)
    fig_eval_design()    # writes figures/eval_design.pdf
    fig_family_pr()      # writes figures/pr_families.pdf
    fig_design_pr()      # writes figures/pr_designs.pdf
    fig_prauc_designs()  # writes figures/pr_auc_designs.pdf
    fig_prauc_cost()     # writes figures/pr_auc_cost.pdf


if __name__ == "__main__":
    main()
