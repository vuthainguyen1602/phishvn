#!/usr/bin/env python3
"""
make_p2_bench_assets.py — Generate P2's results tables from the experiment CSVs.

Tables are GENERATED, never hand-typed (the drift between a typed number and its source is the
exact failure check_paper_claims.py exists to catch — see the P8 217:217 story). Reads:
  data/processed/p2/p2_benchmark.csv        (family x {random, temporal-of-normalize_merge})
  data/processed/p2/p2_temporal_strict.csv  (family x {random_same_rows, temporal_strict})
  data/processed/p2/cross_dataset_F1_CatBoost.csv + cross_dataset_F1.csv (RF, for the gap contrast)
  data/processed/p2/p2_hpo.csv              (optional — emits a placeholder until the run lands)
  data/processed/p2/p2_stacking_baseline.csv + p2_stacking_combos.csv    (ensemble table)
  data/processed/p2/p2_temporal_strict_cb_k20.csv + p2_stacking_cblr_k20.csv (crossover verdict)
  data/processed/p2/p2_forecastability_{decay,shiftmatrix,novelty}.csv   (forecastability macros)
  data/processed/p2/cross_dataset_{F1,ROC-AUC}[_pruned|_coral|_LogReg|_HistGB].csv (interventions)
  data/processed/p2/combined_training.csv   (pooling, for the repair-attempts figure)
Writes papers/P2_url_benchmark/sections/tab_*.tex + gen_*.tex verdict macros, and
papers/P2_url_benchmark/figures/*.pdf.

RUN:  python scripts/make_p2_bench_assets.py [--shap]
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

SEC = "papers/P2_url_benchmark/sections"
FIG = "papers/P2_url_benchmark/figures"
ORDER = ["CatBoost", "MLP", "XGBoost", "LightGBM", "RandomForest", "HistGB", "LogReg"]


def fig_eval_design():
    """The central claim, visual: the F1 span BETWEEN evaluation designs dwarfs the span BETWEEN
    families (0.053 at full-corpus random -> 0.014 on dated rows). CatBoost and LogReg drawn as the
    envelope so the booster's early lead visibly evaporates, the other five recessive grey. Values
    read from the same CSVs the tab_* tables use."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()

    bench = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

    def per_family(df, proto):
        return df[df["protocol"] == proto].groupby("family")["F1"].mean()

    stages = [
        ("Full corpus\nrandom", per_family(bench, "random")),
        ("Dated subset\nrandom", per_family(strict, "random_same_rows")),
        ("Dated subset\ntemporal", per_family(strict, "temporal_strict")),
    ]
    # Cross-dataset transfer: off-diagonal mean of each available family's matrix.
    xdata = {}
    for path, fam in [("data/processed/p2/cross_dataset_F1.csv", "RandomForest"),
                      ("data/processed/p2/cross_dataset_F1_CatBoost.csv", "CatBoost")]:
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

    # Cross-dataset stage: only CatBoost and Random Forest have a transfer matrix, so the blue envelope
    # ends at stage 3 -- name the grey dot that DOES continue, or it reads as a dropped series
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
        # Right of its dot (to the left it read as a third design annotation), and in INK not series grey:
        # text never wears the series colour, and at 7.5 pt the grey was too faint against white
        ax.annotate("RandomForest", (xc, xdata["RandomForest"]), textcoords="offset points",
                    xytext=(7, -3), ha="left", va="center", fontsize=7.5, color=INK)

    # Between-family spread annotations, below the lowest dot of each cluster and given an opaque
    # box: at the third cluster the two dashed cross-dataset lines pass exactly through the text.
    for xi, (_, series) in zip(xs, stages):
        sp = series.max() - series.min()
        ax.annotate(f"spread {sp:.3f}", (xi, series.min()), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=7.5, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    # LogReg is the MINIMUM of the first cluster, so its spread annotation sits directly under the dot
    # and the label cannot go below; right puts its own horizontal line through the text
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


PR_BENCH = "data/processed/p2/p2_pr_curves.csv"
PR_STRICT = "data/processed/p2/p2_pr_curves_strict.csv"


def fig_family_pr():
    """Seven families, one protocol, seven curves on top of each other: the family result is a NULL
    (CatBoost/MLP/XGBoost agree to three decimals), and curves make a null convincing where a column
    of 0.977s reads as a formatting accident. Curves are persisted by run_p2_benchmark from the SAME
    fits as tab_families -- a fresh run would put figure and table a hair apart."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    if not os.path.exists(PR_STRICT):
        print(f"[i] {PR_STRICT} absent — run run_p2_temporal_strict.py; skipping family PR figure")
        return
    cur = pd.read_csv(PR_STRICT)
    bench = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    # Families that separate get identity; the coinciding five share one colour. RandomForest sits
    # below the field, so INK dashed -- at GRAY against the GRAY pack it vanished as a series
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
    ax.set_title("phishing-temporal protocol, mean over 5 seeds", fontsize=8.5)
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
        (pd.read_csv(PR_BENCH), "random", "data/processed/p2/p2_benchmark.csv",
         "full corpus, random", BLUE, "-"),
        (pd.read_csv(PR_STRICT), "random_same_rows", "data/processed/p2/p2_temporal_strict.csv",
         "dated subset, random", GRAY, "--"),
        (pd.read_csv(PR_STRICT), "temporal_strict", "data/processed/p2/p2_temporal_strict.csv",
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
    """The headline claim on a threshold-free metric: is it a thresholding artefact?

    Under PR-AUC the seven families span only 0.009 at full-corpus random -- the 5.3-point F1 lead
    is almost entirely calibration -- while the DESIGN steps move the level by 0.076. That is 8.4x
    the full-corpus-random spread but only 2.9x the strict-stage one, so "an order of magnitude more
    than the spread at any stage" was true of the widest stage and not of the narrowest."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()
    bench = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

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
        # The first cluster's overhead is taken by the CatBoost series label, so its spread annotation moves
        # left; labels are lifted clear with an opaque box where the descending line would cross them
        dx, ha = (-8, "right") if xi == 0 else (0, "center")
        ax.annotate(f"spread {series.max() - series.min():.3f}", (xi, series.max()),
                    textcoords="offset points", xytext=(dx, 13), ha=ha, fontsize=7.5, color=INK,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
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
    df = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    d = df[df.protocol == "temporal_strict"].groupby("family").agg(
        prauc=("PR-AUC", "mean"), secs=("fit_seconds", "mean"))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, lw=0.6)
    # Label side decided from the data: families sit 0.001 PR-AUC apart, so a family whose near-twin is
    # to its right within 3x in fit time takes its label on the left
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
    ax.set_ylabel("PR-AUC (phishing-temporal)")
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
    df = pd.read_csv("data/processed/p2/p2_benchmark.csv")
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
    """All FOUR metrics the experiment records, not the three that suited the headline.

    ROUND-2 REVIEW, M4. p2_temporal_strict.csv has always carried FPR@0.90 -- the metric Section VII
    names as the deployment one -- and this table printed the other three. The paper's "statistically
    tied" verdict is true of threshold-fixed F1 and false, unanimously across seeds, of FPR@0.90;
    selecting which of four recorded metrics to print is what the paper accuses random-split
    benchmarks of, so the column now prints whether or not it flatters the argument."""
    df = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Seven families, same dated rows, both protocols, every recorded metric;"
        " $\\Delta$F1 = phishing-temporal minus random; FPR@0.90 = false-positive rate at $0.90$"
        " phishing recall (lower is better).}",
        "\\label{tab:strict}",
        "\\small\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lccccccccc}\\toprule",
        " & \\multicolumn{4}{c}{Random (same rows)} & \\multicolumn{4}{c}{Phishing-temporal}"
        " & \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Family & F1 & PR-AUC & ROC-AUC & FPR@0.90 & F1 & PR-AUC & ROC-AUC & FPR@0.90 &"
        " $\\Delta$F1 \\\\ \\midrule",
    ]
    for f in ORDER:
        r = df[(df.family == f) & (df.protocol == "random_same_rows")]
        t = df[(df.family == f) & (df.protocol == "temporal_strict")]
        d = t["F1"].mean() - r["F1"].mean()
        lines.append(f"{f} & {ms(r,'F1')} & {ms(r,'PR-AUC')} & {ms(r,'ROC-AUC')} & "
                     f"{ms(r,'FPR@R0.90')} & {ms(t,'F1')} & {ms(t,'PR-AUC')} & "
                     f"{ms(t,'ROC-AUC')} & {ms(t,'FPR@R0.90')} & {d:+.3f} \\\\")
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
    df = pd.concat([pd.read_csv("data/processed/p2/p2_stacking_baseline.csv"),
                    pd.read_csv("data/processed/p2/p2_stacking_combos.csv")])
    ref = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Stacked ensembles (Algorithm~\\ref{alg:stacking}) on the rows and protocols of"
        " Table~\\ref{tab:strict}. Gap = random minus phishing-temporal PR-AUC; +HFS = hybrid"
        " feature selection.}",
        "\\label{tab:stacking}",
        "\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccccc}\\toprule",
        " & \\multicolumn{2}{c}{Random (same rows)} & \\multicolumn{2}{c}{Phishing-temporal} & & \\\\",
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
    df = pd.read_csv("data/processed/p2/p2_forecastability_shiftmatrix.csv")
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
    """CatBoost confusion counts at the paper's operating point (recall 0.90), both same-row protocols.

    Derived, not re-measured: TP/FN follow from the target recall and the guarded phishing test
    count, FP/TN from the seed-mean FPR@0.90 and benign test size (per-seed masks are RandomState(s),
    so reproduced exactly). A 0.5-threshold matrix is deliberately absent -- Section~V argues that
    operating point is a calibration artefact."""
    n_ph, n_be = _dated_test_counts()
    n_ph_rand = _random_arm_phish_test_mean()
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
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
        "\\caption{CatBoost confusion counts at recall $0.90$, derived from"
        " Table~\\ref{tab:strict}. Benign size is the seed mean; the phishing count is the"
        " temporal arm's (the random arm re-draws it per seed, mean"
        f" ${n_ph_rand:,}$).}}",
        "\\label{tab:confusion}",
        "\\small\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lcccc}\\toprule",
        " & \\multicolumn{2}{c}{Random (same rows)} & \\multicolumn{2}{c}{Phishing-temporal} \\\\",
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
    dec = pd.read_csv("data/processed/p2/p2_forecastability_decay.csv")
    canon = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
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
             f"${swing:.2f}$ PR-AUC across the four forward windows "
             f"({swing / (gap['CatBoost']):.0f}$\\times$ the ${gap['CatBoost']:.3f}$ gap the "
             f"signal would need to resolve)"
             + (", and the movement is upward, not the decay the extrapolation methods "
                "assume." if win_means.iloc[-1] > win_means.iloc[0] else "."))

    # (b) the matrix summary
    mx = pd.read_csv("data/processed/p2/p2_forecastability_shiftmatrix.csv")
    internal = mx[mx.a.str.startswith("W") & mx.b.str.startswith("W")]
    adj = internal[internal.b.str[1].astype(int) - internal.a.str[1].astype(int) == 1]
    bd = mx[(mx.a == "W4") & (mx.b == "TEST")].iloc[0]
    w1w4 = mx[(mx.a == "W1") & (mx.b == "W4")].iloc[0]
    w1te = mx[(mx.a == "W1") & (mx.b == "TEST")].iloc[0]
    matrix = (f"Adjacent internal pairs average AUC ${adj.auc.mean():.3f}$, while the "
              f"boundary pair (W4 vs.\\ test) reaches ${bd.auc:.3f}$ over the shortest "
              f"median-date distance in the table ({int(bd.delta_days)} days); and the "
              f"matrix is not monotone in time: the test window resembles the "
              f"\\emph{{earliest}} training block (${w1te.auc:.3f}$ at "
              f"{int(w1te.delta_days)} days) more than the last training block does "
              f"(${w1w4.auc:.3f}$ at {int(w1w4.delta_days)} days).")

    # (c) the novelty probe, two-sided on purpose: it is below chance on every seed, so novelty is
    # ANTI-predictive and the bar is max(AUC, 1-AUC). kept in the development repository, not shipped in this mirror
    nov = pd.read_csv("data/processed/p2/p2_forecastability_novelty.csv")
    cb = nov[nov.family == "CatBoost"]
    a_all, a_ph = cb.auc_all.mean(), cb.auc_phish.mean()
    usable = max(a_all, 1 - a_all)
    below = int((cb.auc_all < 0.5).sum())
    novelty = (f"Train-only lexical novelty does not give a usable gate on where CatBoost fails "
               f"forward in time, and the reason is not that the probe is silent. "
               f"AUC(novelty; error) $= {a_all:.3f}$ over the full test window "
               f"(${cb.auc_all.min():.3f}$--${cb.auc_all.max():.3f}$, below $0.5$ on "
               f"{below}/{len(cb)} seeds) and ${a_ph:.3f}$ on its phishing side. Read as "
               f"Section~\\ref{{ssec:xdata}} reads a below-chance AUC, that is an "
               f"\\emph{{inverted}} signal, not an absent one: the errors concentrate on the "
               f"rows most \\emph{{like}} the training window, which is the opposite of the "
               f"premise a novelty gate is built on. Either way no gate clears the pre-set "
               f"keep-alive bar of $0.65$: the usable discrimination is "
               f"$\\max(\\mathrm{{AUC}}, 1{{-}}\\mathrm{{AUC}}) = {usable:.3f}$, and a "
               f"backoff rule keyed on \\emph{{familiarity}} would defer exactly the traffic a "
               f"deployment least wants deferred.")

    return ("\\newcommand{\\ForecastDecayVerdict}{" + decay + "}\n"
            "\\newcommand{\\ForecastMatrixVerdict}{" + matrix + "}\n"
            "\\newcommand{\\ForecastNoveltyVerdict}{" + novelty + "}")


def gen_fcts():
    """The forward-chained-OOF diagnostic (E5), from run_p2_fcts.py.

    The subsection used to assert that forward-chained meta-learning spends its machinery on a
    signal that is not there; this measures it. The shuffled-order control is what makes the
    comparison mean anything: same blocked folds and meta-sample size, differing only in time."""
    from paired_eval import corrected_paired_t

    d = pd.read_csv("data/processed/p2/p2_fcts.csv")
    t = d[d.protocol == "temporal_strict"]
    piv = t.pivot(index="seed", columns="variant", values="PR-AUC")
    vs_shuf = corrected_paired_t(piv.forward_chained - piv.shuffled_order)
    vs_rand = corrected_paired_t(piv.forward_chained - piv.random_fold)
    if vs_shuf["mean"] > 0 and vs_shuf["p"] <= 0.05:
        raise SystemExit("gen_fcts: forward chaining beat its shuffled control — the "
                         "'no signal inside the window' reading no longer holds, rewrite it")
    ratio = (t.groupby("variant").coef_CB.mean() / t.groupby("variant").coef_LR.mean())
    return ("\\newcommand{\\ForecastFCTSVerdict}{"
            f"Fitting it on forward-chained ones instead---bases fitted on each block of the "
            f"training window and scored on the next---moves temporal PR-AUC by "
            f"${vs_shuf['mean']:+.4f}$ ({vs_shuf['wins']}/{vs_shuf['k']} seeds, corrected "
            f"$p = {vs_shuf['p']:.2f}$) against a shuffled-order control sharing its blocked "
            f"folds and meta-sample size, and ${vs_rand['mean']:+.4f}$ "
            f"({vs_rand['wins']}/{vs_rand['k']}, $p = {vs_rand['p']:.2f}$) against the canonical "
            f"stack. The blend does move---the weight ratio between the booster and the logistic "
            f"member falls from ${ratio['random_fold']:.1f}$:$1$ under random folds to "
            f"${ratio['forward_chained']:.1f}$:$1$ under chaining, the shuffled control between "
            f"them at ${ratio['shuffled_order']:.1f}$:$1$---so the tilt toward the shift-stable "
            f"member is bought mostly by the smaller fold-training sets, not by the order, and "
            f"none of it reaches the score."
            "}")



def gen_polarity_verdict():
    """Emit the sign-stability sentence for P2's below-chance transfer reading.

    The section excluded one alternative -- an inverted label convention -- and concluded the
    inversions are "a property of the models". That is a false dichotomy: it never tested the third
    possibility, that the inversion belongs to the CORPUS PAIR. It does. Under four learners the
    below/above-chance verdict is identical for 9 of 12 cells and the three that disagree sit
    against 0.5; a property invariant to the learner is not a property of the learner. No new
    experiment -- all four matrices were already on disk -- and generated so it cannot go stale."""
    import numpy as np
    files = {"RandomForest": "cross_dataset_ROC-AUC.csv",
             "LogReg": "cross_dataset_ROC-AUC_LogReg.csv",
             "HistGB": "cross_dataset_ROC-AUC_HistGB.csv",
             "RandomForest (pruned)": "cross_dataset_ROC-AUC_pruned.csv"}
    mats = {}
    for k, f in files.items():
        path = os.path.join("data/processed/p2", f)
        if not os.path.exists(path):
            print(f"[i] {f} absent — skipping polarity verdict."); return
        mats[k] = pd.read_csv(path, index_col=0)
    names = list(mats["RandomForest"].index)
    stable = unstable = 0
    worst_margin = 0.0
    for i in names:
        for j in names:
            if i == j:
                continue
            vals = [mats[k].loc[i, j] for k in files]
            inv = [v < 0.5 for v in vals]
            if all(inv) or not any(inv):
                stable += 1
            else:
                unstable += 1
                # How CLOSE does the nearest learner come to 0.5 -- reporting the FARTHEST would overstate how
                # marginal the disagreeing cells are, which an earlier draft did
                worst_margin = max(worst_margin, min(abs(v - 0.5) for v in vals))
    total = stable + unstable
    verdict = (
        f"Re-running the identical design under {len(files)} learners (random forest, logistic "
        f"regression, histogram gradient boosting, and the random forest with the artefact "
        f"features pruned) leaves the below-chance verdict unchanged for ${stable}$ of the "
        f"${total}$ transfer cells; each of the ${unstable}$ that disagree has at least one learner "
        f"within ${worst_margin:.3f}$ of $0.5$, so none of them is placed decisively on both sides"
    )
    write_generated("papers/P2_url_benchmark/sections/gen_polarity_verdict.tex", verdict + ".\n")

def gen_stacking_verdict():
    """The stacking significance sentences, computed from the CSVs so printed p/q-values cannot drift
    from the runs: (a) the K=5 family of ensemble-vs-CatBoost paired tests under phishing-temporal,
    BH-adjusted; (b) the K=20 crossover for Stack[CB+LR]. Regenerate the K=20 sources with:
      run_p2_temporal_strict.py --families CatBoost --seeds 20 --out .../p2_temporal_strict_cb_k20.csv
      run_p2_stacking_baseline.py --seeds 20 --bases CatBoost+LogReg --out .../p2_stacking_cblr_k20.csv"""
    from paired_eval import bh_adjust, corrected_paired_t, fmt_p
    df5 = pd.concat([pd.read_csv("data/processed/p2/p2_stacking_baseline.csv"),
                     pd.read_csv("data/processed/p2/p2_stacking_combos.csv")])
    ref5 = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

    def paired(a_df, b_df, fam_a, fam_b, proto, k_col="PR-AUC"):
        a = a_df[(a_df.family == fam_a) & (a_df.protocol == proto)].set_index("seed")[k_col]
        b = b_df[(b_df.family == fam_b) & (b_df.protocol == proto)].set_index("seed")[k_col]
        return corrected_paired_t((b.sort_index() - a.sort_index()).dropna().values,
                                  test_frac=0.30)

    fams = [k for k, _ in STACKS if k != "Stacking+HFS"]  # HFS is an ablation, not a member
    res = {f: paired(ref5, df5, "CatBoost", f, "temporal_strict") for f in fams}
    qs, _ = bh_adjust([res[f]["p"] for f in fams])
    q = dict(zip(fams, qs))
    tree = (f"Under the phishing-temporal protocol the literature's recipe is significantly "
            f"\\emph{{worse}} than its strongest constituent's "
            f"family peer: Stack[RF+XGB] trails CatBoost by "
            f"${abs(res['Stacking']['mean']):.4f}$ PR-AUC "
            f"(corrected, BH-adjusted ${fmt_p(q['Stacking'], 'q')}$) and Stack[XGB+LGBM] by "
            f"${abs(res['Stack[XGB+LGBM]']['mean']):.4f}$ (${fmt_p(q['Stack[XGB+LGBM]'], 'q')}$), "
            f"while the CatBoost-anchored mixed stacks without LogReg are statistically "
            f"indistinguishable from CatBoost alone.")

    # Report the K=5 BH family first (positive, does not clear BH), then the pre-planned K=20
    # extension adjusted over its own two tests (temporal, random), printed as q.
    cb = pd.read_csv("data/processed/p2/p2_temporal_strict_cb_k20.csv")
    st = pd.read_csv("data/processed/p2/p2_stacking_cblr_k20.csv")
    t = paired(cb, st, "CatBoost", "Stack[CB+LR]", "temporal_strict")
    r = paired(cb, st, "CatBoost", "Stack[CB+LR]", "random_same_rows")
    (tq, rq), _ = bh_adjust([t["p"], r["p"]])
    cross = (f"At the five seeds of Table~\\ref{{tab:stacking}} the two LogReg-anchored stacks "
             f"are ahead of CatBoost but do not clear the family-wise bar: Stack[CB+LR] "
             f"${res['Stack[CB+LR]']['mean']:+.4f}$ PR-AUC "
             f"(BH-adjusted ${fmt_p(q['Stack[CB+LR]'], 'q')}$) and Stack[CB+HGB+LR+MLP] "
             f"${res['Stack[CB+HGB+LR+MLP]']['mean']:+.4f}$ "
             f"(${fmt_p(q['Stack[CB+HGB+LR+MLP]'], 'q')}$). The significance claim therefore "
             f"rests on the pre-planned 20-seed extension for the focal pair, adjusted over its "
             f"own two tests: Stack[CB+LR] beats CatBoost under the phishing-temporal protocol "
             f"(paired $\\Delta$PR-AUC $= {t['mean']:+.4f}$, {t['wins']}/{t['k']} seeds, "
             f"corrected ${fmt_p(tq, 'q')}$) and loses to it under random-same-rows "
             f"($\\Delta = {r['mean']:+.4f}$, {r['k'] - r['wins']}/{r['k']} seeds against, "
             f"corrected ${fmt_p(rq, 'q')}$): at 20 seeds the crossover is significant in "
             f"\\emph{{both}} directions.")

    # Fit cost, measured rather than guessed: the stacks' fit_seconds against the boosters'.
    def fit_s(df, fam):
        return float(df[(df.family == fam) & (df.protocol == "temporal_strict")]
                     ["fit_seconds"].mean())
    s_rfxgb, s_xl, s_cbmlp = (fit_s(df5, "Stacking"), fit_s(df5, "Stack[XGB+LGBM]"),
                              fit_s(df5, "Stack[CB+MLP]"))
    s_cb, s_xgb = fit_s(ref5, "CatBoost"), fit_s(ref5, "XGBoost")
    cost = (f"costs between ${s_rfxgb / s_cb:.0f}\\times$ and ${s_rfxgb / s_xgb:.0f}\\times$ a "
            f"single booster's fit time for the cheapest stack (Stack[RF+XGB] ${s_rfxgb:.1f}$\\,s "
            f"against CatBoost ${s_cb:.2f}$\\,s and XGBoost ${s_xgb:.2f}$\\,s), "
            f"${s_xl:.0f}$\\,s for Stack[XGB+LGBM] and ${s_cbmlp:.0f}$\\,s once the MLP is a "
            f"member")
    return ("\\newcommand{\\StackTreeVerdict}{" + tree + "}\n"
            "\\newcommand{\\StackCrossoverVerdict}{" + cross + "}\n"
            "\\newcommand{\\StackCostVerdict}{" + cost + "}")


# Tests in the refresh-window confirmatory family: 1 since E4 forfeited the residual-lambda
# slot (PREREG_refresh_window.md, amendment 2026-08-22).
REFRESH_BH_M = 1
REFRESH_CB = "data/processed/p2/p2_refresh_cb_k20.csv"
REFRESH_CBLR = "data/processed/p2/p2_refresh_cblr_k20.csv"


def gen_refresh_verdict():
    """The time-stamped pre-specified Test 1 verdict (PREREG_refresh_window.md): Stack[CB+LR] vs CatBoost on the
    refresh window `ph_te2`, K seeds, Nadeau--Bengio corrected paired t on PR-AUC, BH over
    m=REFRESH_BH_M. Reads the two --test-after CSVs written by
      run_p2_temporal_strict.py --families CatBoost --seeds 20 --test-after <max current date>
      run_p2_stacking_baseline.py --bases CatBoost+LogReg --seeds 20 --test-after <same date>
    and returns "" until both exist: a section file the manuscript does not \\input can rot
    unnoticed. Only Test 1's p enters the family here, so its BH adjustment is p*REFRESH_BH_M."""
    from paired_eval import QValue, corrected_paired_t, fmt_p
    if not (os.path.exists(REFRESH_CB) and os.path.exists(REFRESH_CBLR)):
        return ""
    cb, st = pd.read_csv(REFRESH_CB), pd.read_csv(REFRESH_CBLR)

    def pick(df, fam, col):
        return df[(df.family == fam) & (df.protocol == "temporal_strict")] \
            .set_index("seed")[col].sort_index()

    d = (pick(st, "Stack[CB+LR]", "PR-AUC") - pick(cb, "CatBoost", "PR-AUC")).dropna().values
    t = corrected_paired_t(d, test_frac=0.30)
    q = QValue(min(1.0, t["p"] * REFRESH_BH_M))  # BH with one observed test of m
    ok = q <= 0.05 and t["mean"] > 0
    f1 = float((pick(st, "Stack[CB+LR]", "F1") - pick(cb, "CatBoost", "F1")).mean())
    fpr = float((pick(st, "Stack[CB+LR]", "FPR@R0.90") - pick(cb, "CatBoost", "FPR@R0.90")).mean())
    cbm = float(pick(cb, "CatBoost", "PR-AUC").mean())
    txt = (f"On the time-stamped pre-specified refresh window ($K={t['k']}$ seeds, CatBoost PR-AUC "
           f"${cbm:.4f}$) Stack[CB+LR] {'beats' if ok else 'does not significantly beat'} "
           f"CatBoost: paired $\\Delta$PR-AUC $= {t['mean']:+.4f}$ ({t['wins']}/{t['k']} seeds, "
           f"corrected ${fmt_p(t['p'])}$, BH over $m={REFRESH_BH_M}$ ${fmt_p(q, 'q')}$); "
           f"descriptively $\\Delta$F1 $= {f1:+.4f}$ and $\\Delta$FPR@R0.90 $= {fpr:+.4f}$.")
    return ("\\newcommand{\\RefreshVerdict}{" + txt + "}\n"
            f"\\newcommand{{\\RefreshVerdictPass}}{{{'yes' if ok else 'no'}}}")


def tab_decomp():
    """Decompose the random-full -> phishing-temporal drop into its two steps: composition
    (full corpus -> dated-row subset, protocol still random) and protocol (random -> temporal
    on the SAME dated rows). This is the table the headline claim must rest on."""
    full = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Decomposing the \\textbf{F1} drop. $\\Delta_{comp}$ = dated-subset random"
        " minus full-corpus random (\\emph{composition}, protocol unchanged); $\\Delta_{proto}$"
        " = phishing-temporal minus dated-subset random (\\emph{protocol}, rows unchanged).}",
        "\\label{tab:decomp}",
        "\\begin{tabular}{lccccc}\\toprule",
        "Family & Random (full) & Random (dated rows) & Phishing-temporal & $\\Delta_{comp}$ &"
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
    df = pd.read_csv("data/processed/p2/p2_hpo.csv")
    df = df[df.config != "default"].copy()
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Best 3-fold CV PR-AUC on the phishing-temporal train window: Grey Wolf Optimizer"
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

    The figure is hand-drawn TikZ because its content is structural, but a number typed into TikZ
    would be the one place a stale figure could survive a data change unnoticed (check_paper_claims
    reads .tex, not compiled PDFs). Macros mean it cannot disagree with tab_decomp."""
    full = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

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
        from run_p2_temporal_strict import load as _load
        df = _load()
        # The composition step drops UNDATED PHISHING only, keeping the benign pool whole; counting dated
        # rows across both classes gives a corpus-wide total no experiment was ever run on
        is_ph = df["label"].astype(int).eq(1)      # the loader codes phishing as 1, not a string
        rows_full = len(df)
        rows_dated = int((~is_ph).sum() + (is_ph & df["date"].notna()).sum())
    except Exception as e:
        print(f"[i] row counts unavailable ({type(e).__name__}) — figure will omit them.")

    def num(v):
        return f"{v:.3f}"  # leading zero kept: the prose writes 0.053, the figure must match

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
    """Paired-by-seed CatBoost-vs-LogReg under phishing-temporal, on ALL FOUR recorded metrics.

    ROUND-2 REVIEW, M4. This sentence used to report F1 alone and conclude "statistically tied" --
    true of threshold-fixed F1 and false of the other three, most sharply of FPR@0.90 where CatBoost
    wins on every seed. A tie asserted from one of four recorded metrics is the selective reporting
    this paper spends five pages criticising. Only the BENIGN complement is redrawn per seed, so
    charging the full 0.30 test fraction to the Nadeau-Bengio correction is conservative."""
    from paired_eval import corrected_paired_t, fmt_p
    from scipy import stats

    df = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    t = df[df.protocol == "temporal_strict"]

    def pair(metric):
        cb = t[t.family == "CatBoost"].sort_values("seed")[metric].to_numpy()
        lr = t[t.family == "LogReg"].sort_values("seed")[metric].to_numpy()
        return corrected_paired_t(cb - lr)

    f1 = pair("F1")
    ci = stats.t.ppf(0.975, f1["k"] - 1) * f1["se"] if f1["se"] else 0.0
    n = f1["k"]
    # FPR is a loss: a negative difference is CatBoost winning, so the seed count is (k - wins).
    # Backwards, this prints "0 of 5 seeds" for a unanimous effect
    parts = []
    for metric, label, lower_better in (("PR-AUC", "PR-AUC", False),
                                        ("ROC-AUC", "ROC-AUC", False),
                                        ("FPR@R0.90", "FPR@0.90", True)):
        s = pair(metric)
        w = (s["k"] - s["wins"]) if lower_better else s["wins"]
        parts.append(f"{label} ${s['mean']:+.4f}$ (${fmt_p(s['p'])}$, {w}/{s['k']} seeds)")

    return (
        f"Paired by seed (the benign split mask is shared within a seed), CatBoost minus"
        f" logistic regression under the phishing-temporal protocol is "
        f"${f1['mean']:+.4f}$ F1 (95\\% CI $\\pm{ci:.4f}$, corrected resampled $t$-test "
        f"${fmt_p(f1['p'])}$, $n={n}$ seeds). On threshold-fixed F1 the two families are"
        f" therefore tied, but that verdict is a property of the metric, not of the"
        f" families. On the other three quantities the same runs record, the booster is ahead"
        f" on every seed: " + "; ".join(parts) + ". The ranking advantage is real and small;"
        f" what the threshold-fixed comparison shows is that it does not convert into a"
        f" better operating point at $\\tau = 0.5$.")


CURVES_FULL = "data/processed/p2/p2_pr_curves.csv"
CURVES_STRICT = "data/processed/p2/p2_pr_curves_strict.csv"
# (label, curve file, protocol) for the three same-schema designs, in decomposition order.
MAXF1_DESIGNS = [("Random (full)", CURVES_FULL, "random"),
                 ("Random (dated rows)", CURVES_STRICT, "random_same_rows"),
                 ("Phishing-temporal", CURVES_STRICT, "temporal_strict")]


def _max_f1(curves: pd.DataFrame) -> dict:
    """Best F1 reachable anywhere on each family's stored precision--recall curve.

    A descriptive quantity read off the published curve (seed-mean precision at a fixed recall grid,
    persisted by run_p2_benchmark from the same fitted models), not a re-run and not a per-seed
    statistic that could carry a paired test. It answers what the tables cannot: how much of the
    between-family spread is separation, and how much is where tau=0.5 happens to fall."""
    out = {}
    for fam, g in curves.groupby("family"):
        r = g["recall"].to_numpy(float)
        p = g["precision"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(r + p > 0, 2 * r * p / (r + p), 0.0)
        out[fam] = float(np.nanmax(f1))
    return out


def _maxf1_stages():
    stages = []
    for label, path, proto in MAXF1_DESIGNS:
        df = pd.read_csv(path)
        stages.append((label, _max_f1(df[df.protocol == proto])))
    return stages


def tab_maxf1():
    """The decomposition again, with the threshold chosen instead of fixed at 0.5.

    ROUND-2 REVIEW, M1. tab_decomp decomposes a THRESHOLDED F1 drop, and the paper read one column
    as capability: the composition step costs the boosters 0.060-0.063 and logistic regression only
    0.012, therefore "the boosters' margin lives on the undated portion". At each family's own best
    operating point the same step costs EVERY family 0.058-0.062 -- the differential is entirely
    threshold placement. The step sizes do not change; what changes is the mechanism the paper is
    entitled to assert, and which family is ahead at the end of it."""
    stages = _maxf1_stages()
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Table~\\ref{tab:decomp} at each family's \\emph{chosen} operating point:"
        " the best F1 on the seed-mean precision--recall curve of the Table~\\ref{tab:strict}"
        " models; $\\Delta_{comp}$ and $\\Delta_{proto}$ as in Table~\\ref{tab:decomp}.}",
        "\\label{tab:maxf1}",
        "\\begin{tabular}{lccccc}\\toprule",
        "Family & " + " & ".join(lbl for lbl, _ in stages)
        + " & $\\Delta_{comp}$ & $\\Delta_{proto}$ \\\\ \\midrule",
    ]
    for f in ORDER:
        a, b, c = (s[1][f] for s in stages)
        lines.append(f"{f} & {a:.3f} & {b:.3f} & {c:.3f} & {b-a:+.3f} & {c-b:+.3f} \\\\")
    lines.append("\\midrule")
    sp = [max(s[1].values()) - min(s[1].values()) for s in stages]
    lines.append("family spread & " + " & ".join(f"{v:.3f}" for v in sp) + " & & \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def gen_maxf1_verdict():
    """The sentence Section~\\ref{ssec:decomp} has to carry once Table~\\ref{tab:maxf1} exists.

    Generated rather than typed because it states a SIGN -- which family is ahead at a chosen
    threshold under the deployment protocol -- and a sign that flips on a re-run would otherwise sit
    in the manuscript indefinitely. The tripwire below refuses a sentence its own arithmetic
    contradicts."""
    stages = _maxf1_stages()
    (_, full), (_, dated), (_, temp) = stages
    comp = {f: dated[f] - full[f] for f in full}
    cb_lr = [d["CatBoost"] - d["LogReg"] for _, d in stages]
    sp = [max(d.values()) - min(d.values()) for _, d in stages]
    lo, hi = min(comp.values()), max(comp.values())
    best_temp = max(temp, key=temp.get)
    # The thresholded counterparts come from the same CSVs tab_decomp reads, never retyped:
    # this sentence's whole point is the contrast between the two columns.
    tf = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    ts = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    fixed_full = tf[tf.protocol == "random"].groupby("family")["F1"].mean()
    fixed_dated = ts[ts.protocol == "random_same_rows"].groupby("family")["F1"].mean()
    fixed_spread = float(fixed_full.max() - fixed_full.min())
    fixed_comp = fixed_dated - fixed_full

    # The claim is that the composition step is UNIFORM across families: a re-run that made it
    # differential again would falsify the sentence while every number in it stayed true
    if hi - lo > 0.02:
        raise SystemExit(
            "gen_maxf1_verdict: the composition step is no longer uniform across families "
            f"(range {lo:+.3f} to {hi:+.3f}) — the 'charges every family alike' reading in "
            "Section VI-B is no longer supported; rewrite it before regenerating.")

    # The table prints three decimals; the sentence beside it must not print four, or a reader
    # comparing the two finds "0.0055" next to "0.006" and has to work out that they agree.
    prose_name = {"LogReg": "logistic regression", "RandomForest": "random forest",
                  "MLP": "the MLP"}.get(best_temp, best_temp)
    return (
        f"Table~\\ref{{tab:maxf1}} repeats the decomposition with the operating point chosen "
        f"rather than fixed, and two things change. First, the between-family spread does not "
        f"collapse at the composition step, because at a chosen threshold there was no spread "
        f"to collapse: it is ${sp[0]:.3f}$ F1 at full-corpus random (against "
        f"${fixed_spread:.3f}$ at $\\tau = 0.5$), and it \\emph{{widens}} to ${sp[1]:.3f}$ and "
        f"then ${sp[2]:.3f}$ as the design tightens. Second, the composition step charges every "
        f"family alike (${lo:+.3f}$ to ${hi:+.3f}$ F1, against "
        f"${fixed_comp['CatBoost']:+.3f}$ for CatBoost and ${fixed_comp['LogReg']:+.3f}$ for "
        f"logistic regression at the fixed threshold): the differential the thresholded table "
        f"shows is where $\\tau = 0.5$ falls for each family, not what each family learned from "
        f"the undated rows. The booster's advantage over logistic regression at a chosen "
        f"threshold is ${cb_lr[0]:+.3f}$ F1 on the full corpus, ${cb_lr[1]:+.3f}$ on the dated "
        f"rows and ${cb_lr[2]:+.3f}$ under the temporal protocol, where {prose_name} is the "
        f"highest-scoring family in the table. Every one of those differences is smaller than "
        f"the ${fixed_spread:.3f}$ the fixed threshold reports, and the last has the opposite "
        f"sign.")


def tab_trivial_floor():
    """The all-positive classifier, against which every F1 in this paper has to be read.

    ROUND-2 REVIEW, M5. The paper reports F1 of 0.922, 0.853 and 0.628 across designs and reads the
    sequence as degradation. Two of the three are near or below what predicting phishing for every
    row scores on the same test set, 2p/(1+p); ten of the twelve transfer cells are below it. It
    also disciplines the within-corpus levels, which is why the second block exists: the composition
    step moves the prior 0.691 -> 0.546, so F1 0.922 -> 0.853 is a drop of 0.069 against a floor
    that drops 0.111 -- measured as headroom, the honest design does BETTER. Rates are read from the
    corpora, and the same table is generated for the companion detection paper."""
    from train_url_baseline import add_label
    files = {"PhishVN": "vn_compphish.csv", "PhiUSIIL": "external/phiusiil_compphish.csv",
             "ISCXURL2016": "external/iscx_compphish.csv",
             "PhishStorm": "external/phishstorm_compphish.csv"}
    # The Random Forest matrix is the headline since 2026-08-21 -- the learner every intervention and
    # the ROC-AUC view run on -- so the floor is priced against it. CatBoost's gives the same count
    mat = pd.read_csv("data/processed/p2/cross_dataset_F1.csv", index_col=0)
    names = [n for n in mat.columns]
    rates, floor = {}, {}
    for n in names:
        y = add_label(pd.read_csv(os.path.join("data/processed", files[n])))["y"]
        rates[n] = float(y.mean())
        floor[n] = 2 * rates[n] / (1 + rates[n])
    off = [(i, j) for i in names for j in names if i != j]
    below = [(i, j) for i, j in off if float(mat.loc[i, j]) < floor[j]]
    off_mean = float(np.mean([float(mat.loc[i, j]) for i, j in off]))
    # The mean of the per-target floors, NOT the floor of the mean prior: cells are scored one target at
    # a time. The two differ in the third decimal and only one is the comparison the sentence makes
    floor_mean = float(np.mean([floor[j] for _, j in off]))

    # The within-corpus designs: prior of the TEST window each design scores on, derived the same way as
    # tab_confusion's count rather than re-split here -- two derivations of one row count drift
    strict = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    bench = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    n_ph, n_be = _dated_test_counts()
    dated_p = n_ph / (n_ph + n_be)
    designs = [("Full corpus random", rates["PhishVN"],
                float(bench[bench.protocol == "random"].groupby("family")["F1"].mean().max())),
               ("Dated subset random", dated_p,
                float(strict[strict.protocol == "random_same_rows"]
                      .groupby("family")["F1"].mean().max())),
               ("Dated subset temporal", dated_p,
                float(strict[strict.protocol == "temporal_strict"]
                      .groupby("family")["F1"].mean().max())),
               ]

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{The all-positive classifier's F1, $2p/(1{+}p)$ at test positive rate $p$,"
        " per evaluation design (top) and transfer target (bottom); headroom = best family"
        " minus floor; transfer row = mean off-diagonal of Table~\\ref{tab:xdatasetrf}.}",
        "\\label{tab:trivialfloor}", "\\small",
        "\\begin{tabular}{l ccc}", "\\toprule",
        "Evaluation design & test prior $p$ & all-positive F1 & best family (headroom)"
        " \\\\", "\\midrule",
    ]
    for label, p, best in designs:
        fl = 2 * p / (1 + p)
        lines.append(f"{label} & {p:.3f} & {fl:.3f} & {best:.3f} ({best - fl:+.3f}) \\\\")
    lines.append(f"Cross-dataset (mean off-diagonal) & n/a & {floor_mean:.3f} & "
                 f"{off_mean:.3f} ({off_mean - floor_mean:+.3f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\\\[4pt]",
              "\\begin{tabular}{l cccc}", "\\toprule",
              "Transfer target & " + " & ".join(names) + " \\\\", "\\midrule",
              "positive rate & " + " & ".join(f"{rates[n]:.3f}" for n in names) + " \\\\",
              "all-positive F1 & " + " & ".join(f"{floor[n]:.3f}" for n in names) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _dated_test_counts():
    """(guarded phishing test rows, seed-mean benign test rows) of the dated-row designs.

    Shared by tab_confusion and tab_trivial_floor: the same two counts derived twice, in two
    functions, is exactly how a table and the paragraph beside it come to disagree."""
    from run_p2_temporal_strict import load as _load
    df = _load()
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date")
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    n_ph = int((~ph_te.rdom.isin(set(ph_tr.rdom))).sum())
    n_be_pool = int((df.y == 0).sum())
    n_be = int(np.mean([(np.random.RandomState(s).rand(n_be_pool) >= 0.70).sum()
                        for s in range(5)]))
    return n_ph, n_be


def _random_arm_phish_test_mean():
    """Seed-mean phishing test size of the random-same-rows arm. The temporal arm's window is fixed
    (5,879 guarded rows); the random arm re-draws a 70/30 mask per seed after the benign mask has
    consumed the same RandomState, so its count differs from 5,879 and from seed to seed.
    tab_confusion prints the temporal count and must say so."""
    from run_p2_temporal_strict import load as _load
    df = _load()
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date")
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    n_be_pool = int((df.y == 0).sum())
    n = len(ph_tr) + len(ph_te)
    sizes = []
    for s in range(5):
        rng = np.random.RandomState(s)
        rng.rand(n_be_pool)  # the benign mask is drawn first in run_p2_temporal_strict.py
        sizes.append(int((~(rng.rand(n) < len(ph_tr) / n)).sum()))
    return int(round(np.mean(sizes)))


def gen_leakage_verdict():
    """What the full-corpus random split's 0.922 is worth (round-2 review, M3).

    The opening number comes from a plain train_test_split with no domain guard and no
    de-duplication, over a schema on which distinct URLs collapse onto identical points. The census
    and the memorisation oracle come from scripts/p2_dup_leakage.py; this only renders them.
    The sentence stops where the evidence does: the oracle scores BELOW every fitted family, so the
    leak supports only the narrower claim that the opening level is not a clean estimate."""
    d = pd.read_csv("data/processed/p2/p2_dup_leakage.csv")
    r = d.iloc[0]
    bench = pd.read_csv("data/processed/p2/p2_benchmark.csv")
    best = bench[bench.protocol == "random"].groupby("family")["F1"].mean()
    top = best.idxmax()
    oracle, allpos = d.lookup_oracle_f1.mean(), d.all_positive_f1.mean()
    if oracle >= best.max():
        raise SystemExit(
            f"gen_leakage_verdict: the memorisation oracle ({oracle:.3f}) now matches or beats "
            f"the best fitted family ({best.max():.3f}) — the 'the models do more than "
            "memorise' clause is no longer true; rewrite Section VI-A before regenerating.")
    return (
        f"\\textbf{{What the full-corpus random split leaks.}} The $0.922$ is produced by a "
        f"plain stratified split with neither a domain guard nor de-duplication, and the "
        f"$21$-feature schema is coarse: ${int(r.rows):,}$ rows occupy only "
        f"${int(r.unique_featvec):,}$ distinct feature vectors over ${int(r.unique_regdom):,}$ "
        f"registrable domains, and ${int(r.rows_in_mixed_groups):,}$ of those rows sit in "
        f"${int(r.mixed_featvec_groups):,}$ vector groups carrying both labels, a schema "
        f"ceiling of ${r.schema_ceiling_f1:.3f}$ F1 that no model on these features can pass. "
        f"Under the benchmark's own splitter, ${100*d.test_twin_rate.mean():.0f}\\%$ of test "
        f"rows have an exact feature-vector twin in training and "
        f"${100*d.test_regdom_shared_rate.mean():.0f}\\%$ share a registrable domain with it. "
        f"We price that directly with a lookup table (store the majority label of each exact "
        f"training vector, answer the training majority class otherwise), which learns nothing "
        f"and scores ${oracle:.3f}$ F1, against ${allpos:.3f}$ for the all-positive classifier "
        f"(Table~\\ref{{tab:trivialfloor}}) and ${best.max():.3f}$ for {top}. Two readings "
        f"follow, and only the second is licensed. The models do \\emph{{more}} than memorise: "
        f"they clear the memorisation oracle by "
        f"${best.max() - oracle:.3f}$ F1. But ${100*d.test_twin_rate.mean():.0f}\\%$ of the "
        f"test window being a copy of training means this design does not estimate "
        f"generalisation to unseen URLs at all, and the level it reports should not be read as "
        f"if it did: the same objection this paper raises to single-corpus benchmarks, owed "
        f"here to its own opening number. The dated-row designs are built on the registrable "
        f"domain and do not have this property to the same degree "
        f"(Section~\\ref{{ssec:decomp}}).")


def gen_charcnn():
    """The character-CNN arm, and the source-swap stress test on its margin.

    §II answers three deep-learning citations by scope rather than by measurement, one of them a
    CNN over URL strings on the same task this benchmark evaluates. run_p2_charcnn.py supplies the
    number; run_p2_source_probe.py measures sensitivity to the benign-source family. Because
    every PhishVN source is single-class, that diagnostic cannot identify provenance separately
    from label and must never be rendered as ruling provenance out.

    Every quantity here is a PAIRED difference on identical rows -- the arm imports the split and
    the per-seed benign mask rather than re-drawing them -- so the corrected resampled t applies
    directly, with BH across the family of comparisons reported in the sentence."""
    cp = "data/processed/p2/p2_charcnn.csv"
    sp = "data/processed/p2/p2_source_probe.csv"
    if not os.path.exists(cp):
        return "% the char-CNN arm has not been run; see scripts/run_p2_charcnn.py\n"
    from paired_eval import corrected_paired_t, bh_adjust
    cnn = pd.read_csv(cp)
    base = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

    def paired(metric, proto, fam):
        c = cnn[cnn.protocol == proto].set_index("seed")[metric]
        t = base[(base.protocol == proto) & (base.family == fam)].set_index("seed")[metric]
        seeds = sorted(set(c.index) & set(t.index))
        d = np.array([c[i] - t[i] for i in seeds], float)
        return d.mean(), corrected_paired_t(d)["p"]

    fams = sorted(base[base.protocol == "temporal_strict"].family.unique())
    res, ps = [], []
    for fam in fams:
        for m in ("F1", "PR-AUC", "FPR@R0.90"):
            for proto in ("temporal_strict", "random_same_rows"):
                delta, p = paired(m, proto, fam)
                res.append((fam, m, proto, delta))
                ps.append(p)
    qs, rej = bh_adjust(ps)
    n_sig = sum(rej)
    # a q that rounds to 0.000 reads as a missing number rather than as a small one
    qfmt = "$q < 0.001$" if max(qs) < 0.001 else f"$q \\leq {max(qs):.3f}$"

    best = max(fams, key=lambda f: base[(base.protocol == "temporal_strict")
                                        & (base.family == f)]["F1"].mean())
    d_f1, _ = paired("F1", "temporal_strict", best)
    d_pr, _ = paired("PR-AUC", "temporal_strict", best)
    d_fpr, _ = paired("FPR@R0.90", "temporal_strict", best)

    def dproto(fam):
        r = base[(base.protocol == "random_same_rows") & (base.family == fam)]["F1"].mean() \
            if fam != "CharCNN" else cnn[cnn.protocol == "random_same_rows"]["F1"].mean()
        t = base[(base.protocol == "temporal_strict") & (base.family == fam)]["F1"].mean() \
            if fam != "CharCNN" else cnn[cnn.protocol == "temporal_strict"]["F1"].mean()
        return r - t

    dp_cnn = dproto("CharCNN")
    dp_tab = {f: dproto(f) for f in fams}
    dp_best = max(dp_tab.values())
    dp_worst = min(dp_tab.values())

    out = [
        f"Against the strongest tabular family ({best}) on identical rows and seeds, the "
        f"character-CNN gains ${d_f1:+.3f}$ F1 and ${d_pr:+.3f}$ PR-AUC under the direct-dated "
        f"phishing rolling-origin "
        f"protocol, and reduces FPR at $90\\%$ recall by ${abs(d_fpr):.3f}$; all "
        f"{n_sig} paired comparisons across the seven families, three metrics and two protocols "
        f"survive Benjamini--Hochberg ({qfmt}). "
        f"The gain is largest on the deployment metric, which is the one Section~"
        f"\\ref{{sec:discussion}} argues a benchmark should be read on.",
        f"Capacity is not free of the protocol, however: the character-CNN carries the largest "
        f"$\\Delta_{{\\mathrm{{proto}}}}$ in the study (${dp_cnn:+.3f}$ F1 against "
        f"${dp_best:+.3f}$ for the most protocol-sensitive tabular family and "
        f"${dp_worst:+.3f}$ for the least), so the model that gains most from a random split is "
        f"also the model that loses most when the split is made honest.",
    ]

    if os.path.exists(sp):
        pr = pd.read_csv(sp)

        def swap(fam, tb, metric="F1"):
            return float(pr[(pr.family == fam) & (pr.train_benign == tb)][metric].mean())

        def canon(fam, metric="F1"):
            src = cnn[cnn.protocol == "temporal_strict"] if fam == "CharCNN" else \
                base[(base.protocol == "temporal_strict") & (base.family == fam)]
            return float(src[metric].mean())

        cnn_tr = swap("CharCNN", "tranco") - canon("CharCNN")
        cb_tr = swap("CatBoost", "tranco") - canon("CatBoost")
        lr_tr = swap("LogReg", "tranco") - canon("LogReg")
        cb_pr = swap("CatBoost", "tranco", "PR-AUC")
        cnn_ti = swap("CharCNN", "tinnhiem") - canon("CharCNN")
        cb_ti = swap("CatBoost", "tinnhiem") - canon("CatBoost")
        out.append(
            f"Because a model reading the raw string could key on the benign feed rather than on "
            f"phishing, we swapped the benign source: trained against one benign family and tested "
            f"against the other, with the phishing side held at its direct-dated rolling-origin "
            f"split. Trained "
            f"on the broad list the character-CNN loses ${abs(cnn_tr):.3f}$ F1, inside its own seed "
            f"spread, while the tabular families lose ${abs(cb_tr):.3f}$ and ${abs(lr_tr):.3f}$ and "
            f"the gradient booster falls to ${cb_pr:.3f}$ PR-AUC on a test set whose positive rate "
            f"is higher than that: the ranking inverts. The "
            f"21~features are aggregate string statistics, and the two benign populations differ in "
            f"exactly those coordinates. The effect is one-directional, and we report it as such: "
            f"trained on the narrow registry population every model collapses "
            f"(${cnn_ti:+.3f}$ and ${cb_ti:+.3f}$ F1 respectively) and the character-CNN holds no "
            f"advantage. Because every source in PhishVN is single-class, this one-sided swap does "
            f"not identify provenance separately from label; it is a sensitivity analysis, not "
            f"evidence that benign provenance has been ruled out.")

    return "\n\n".join(out) + "\n"


def gen_guard_control():
    """The M7 control: phishing-temporal against a random control carrying the SAME domain guard.

    Section~\\ref{sec:method} claimed protocol was the only variable between the two arms. It was
    not -- the guard was applied on one side only, so part of Delta_proto was the guard. The arm is
    run by `run_p2_temporal_strict.py --guard-control`, the asymmetry it closes is measured by
    `scripts/p2_dup_leakage.py`, and both are read here rather than retyped."""
    gp = "data/processed/p2/p2_temporal_strict_guarded.csv"
    lp = "data/processed/p2/p2_dup_leakage_protocols.csv"
    if not (os.path.exists(gp) and os.path.exists(lp)):
        return ("% the M7 guarded-control arm has not been run; see"
                " run_p2_temporal_strict.py --guard-control\n")
    g = pd.read_csv(gp)
    rates = pd.read_csv(lp)
    base = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")

    def mean_f1(df, proto, fam):
        return float(df[(df.family == fam) & (df.protocol == proto)]["F1"].mean())

    def mean_m(df, proto, fam, metric):
        return float(df[(df.family == fam) & (df.protocol == proto)][metric].mean())

    unguarded = {f: mean_f1(base, "random_same_rows", f) for f in ORDER}
    guarded = {f: mean_f1(g, "random_same_rows_guarded", f) for f in ORDER}
    temporal = {f: mean_f1(base, "temporal_strict", f) for f in ORDER}
    d_old = {f: temporal[f] - unguarded[f] for f in ORDER}
    d_new = {f: temporal[f] - guarded[f] for f in ORDER}
    shared = rates[rates.protocol == "random_same_rows"]["phish_test_regdom_shared_rate"].mean()
    shared_t = rates[rates.protocol == "temporal_strict"]["phish_test_regdom_shared_rate"].mean()
    shift = float(np.mean([guarded[f] - unguarded[f] for f in ORDER]))
    pr_old = mean_m(base, "temporal_strict", "CatBoost", "PR-AUC") \
        - mean_m(base, "random_same_rows", "CatBoost", "PR-AUC")
    pr_new = mean_m(base, "temporal_strict", "CatBoost", "PR-AUC") \
        - mean_m(g, "random_same_rows_guarded", "CatBoost", "PR-AUC")
    worst = min(d_new, key=d_new.get)
    worst_name = {"RandomForest": "random forest", "LogReg": "logistic regression",
                  "MLP": "the MLP"}.get(worst, worst)

    # Delta_proto reads as "protocol plus a constant" only if the family ORDERING survives even though
    # the level does not. Rank CORRELATION, not equality (a 0.0000 tie breaks either way); the
    # extremes carry the claim and are pinned exactly. kept in the development repository, not shipped in this mirror
    from scipy.stats import spearmanr
    rho = float(spearmanr([d_old[f] for f in ORDER], [d_new[f] for f in ORDER]).statistic)
    if rho < 0.9 or min(d_old, key=d_old.get) != worst \
            or max(d_old, key=d_old.get) != max(d_new, key=d_new.get):
        raise SystemExit(
            "gen_guard_control: the guard-matched control reorders the families by "
            f"Delta_proto (Spearman {rho:+.2f}; extremes "
            f"{min(d_old, key=d_old.get)}/{max(d_old, key=d_old.get)} -> "
            f"{worst}/{max(d_new, key=d_new.get)}) — the 'ordering is unchanged' reading in "
            "Section VI-B no longer holds; rewrite it before regenerating.")

    return (
        f"\\textbf{{The control's own leakage, and the arm that removes it.}} One asymmetry has "
        f"to be closed before $\\Delta_{{proto}}$ can be called a protocol effect. The "
        f"phishing-temporal arm applies the registrable-domain guard; the random control re-splits "
        f"the pooled rows without it, so ${100*shared:.1f}\\%$ of the control's phishing test "
        f"window shares a domain with its own training window against "
        f"${100*shared_t:.1f}\\%$ under the temporal arm: the guard, not the protocol. We "
        f"therefore re-ran the control with the identical guard applied after the random "
        f"re-split, and the correction is not cosmetic. The guard costs the control "
        f"${shift:+.3f}$ F1, near-uniformly across families, and $\\Delta_{{proto}}$ falls from "
        f"${min(d_old.values()):+.3f}$--${max(d_old.values()):+.3f}$ to "
        f"${min(d_new.values()):+.4f}$--${max(d_new.values()):+.4f}$: roughly half of the "
        f"published protocol step was the guard. For the leading families it is now "
        f"indistinguishable from zero (CatBoost ${d_new['CatBoost']:+.4f}$ F1, "
        f"${pr_new:+.3f}$ PR-AUC against ${pr_old:+.3f}$ unguarded; logistic regression "
        f"${d_new['LogReg']:+.4f}$, the wrong sign for a temporal penalty), and what remains is "
        f"carried by the tail: {worst_name} at ${d_new[worst]:+.4f}$. Two things survive "
        f"intact. "
        f"The guard's cost is a near-constant offset, so the \\emph{{ordering}} of families by "
        f"$\\Delta_{{proto}}$ is preserved (Spearman $\\rho = {rho:.2f}$ between the two "
        f"controls, logistic regression smallest and {worst_name} largest under both); and the "
        f"direction of the paper's argument is unaffected, since a "
        f"smaller protocol step makes the composition and threshold steps larger relative to it, "
        f"not smaller. The published table keeps the unguarded control, because that is the "
        f"comparison the benchmark literature runs and the quantity a reader of that literature "
        f"needs; this arm is what says how much of it is protocol.")


def tab_xdataset():
    cb = pd.read_csv("data/processed/p2/cross_dataset_F1_CatBoost.csv", index_col=0)
    rf = pd.read_csv("data/processed/p2/cross_dataset_F1.csv", index_col=0)

    def gap(m):
        v = m.to_numpy(float)
        d = np.diag(v)
        off = v[~np.eye(len(v), dtype=bool)]
        return d.mean(), off.mean(), d.mean() - off.mean()

    cd, co, cg = gap(cb)
    rd, ro, rg = gap(rf)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Table~\\ref{tab:xdatasetrf}'s four-corpus design refitted with CatBoost,"
        " F1 only (rows = train corpus, columns = test corpus; diagonal cells bold): a second"
        " learner, the same pattern.}",
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


# --------------------------------------------------------------- transfer interventions (ex-P3)
# Moved here from make_p3_assets.py on 2026-08-19: P2 owns the four-corpus matrix, so the
# cell-level ROC-AUC view and the four interventions run on it came with it. Not one number changed.


def _matrix_stats(path):
    """(mean diagonal, mean off-diagonal, gap) for a saved train x test matrix, or None."""
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path, index_col=0)
    v = m.to_numpy(float)
    diag = np.diag(v).mean()
    off = float(np.nanmean(v[~np.eye(len(v), dtype=bool)]))
    return diag, off, diag - off


def tab_xdataset_rf():
    """The Random Forest transfer matrix, both metrics, cell by cell.

    The ROC-AUC panel is the one a round-2 reviewer asked for: the strongest empirical claim --
    eight of twelve cells below chance, down to 0.200 -- reached the reader only as a summary
    sentence, and whether they cluster on one corpus matters because PhiUSIIL ships an inverted
    label convention. The F1 panel stays because every intervention below is a Random Forest run and
    quotes its cells. Since 2026-08-21 this is the matrix the transfer section opens with, so the
    headline, the threshold-free view and the interventions share one learner."""
    f1p = "data/processed/p2/cross_dataset_F1.csv"
    rocp = "data/processed/p2/cross_dataset_ROC-AUC.csv"
    if not (os.path.exists(f1p) and os.path.exists(rocp)):
        return "% cross_dataset_F1.csv / cross_dataset_ROC-AUC.csv absent -- run run_cross_dataset.py\n"
    f1 = pd.read_csv(f1p, index_col=0)
    roc = pd.read_csv(rocp, index_col=0)
    names = list(f1.index)
    fd, fo, fg = _matrix_stats(f1p)
    off = roc.to_numpy(float)[~np.eye(len(names), dtype=bool)]
    sub = off[off < 0.5]

    def f1cell(i, j):
        v = f1.loc[i, j]
        return f"\\textbf{{{v:.3f}}}" if i == j else f"{v:.3f}"

    def roccell(i, j):
        v = roc.loc[i, j]
        if i == j:
            return f"\\textbf{{{v:.3f}}}"
        return f"\\underline{{{v:.3f}}}" if v < 0.5 else f"{v:.3f}"

    head = "Train $\\backslash$ Test & " + " & ".join(names) + " \\\\"
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Random-forest cross-dataset transfer, F1 and ROC-AUC (rows = train corpus,"
        " columns = test corpus); the matrix every later intervention re-runs. Diagonal bold;"
        " \\underline{underlined} ROC-AUC cells are below chance ($<0.5$).}",
        "\\label{tab:xdatasetrf}", "\\small",
        "\\begin{tabular}{l cccc}", "\\toprule",
        "\\multicolumn{5}{l}{\\emph{F1}} \\\\",
        head, "\\midrule",
    ]
    lines += [f"{i} & " + " & ".join(f1cell(i, j) for j in names) + " \\\\" for i in names]
    lines += [
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\footnotesize Mean diagonal $={fd:.3f}$;\\quad mean"
        f" off-diagonal $={fo:.3f}$;\\quad gap $={fg:.3f}$.}} \\\\",
        "\\midrule", "\\multicolumn{5}{l}{\\emph{ROC-AUC}} \\\\", head, "\\midrule",
    ]
    lines += [f"{i} & " + " & ".join(roccell(i, j) for j in names) + " \\\\" for i in names]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def tab_pruned():
    """The artefact-pruning intervention: the matrix re-run without the three TreeSHAP-named
    features (run_cross_dataset.py --drop tld_len,subdom_cnt,dot_cnt)."""
    runs = {(metric, cfg): _matrix_stats(f"data/processed/p2/cross_dataset_{metric}{suf}.csv")
            for metric in ("F1", "ROC-AUC")
            for cfg, suf in (("full", ""), ("pruned", "_pruned"))}
    if any(v is None for v in runs.values()):
        return "% pruned/full matrices incomplete -- run run_cross_dataset.py (+/- --drop)\n"

    def row(label, idx):
        cells = " & ".join(f"{runs[(m, c)][idx]:.3f}"
                           for m in ("F1", "ROC-AUC") for c in ("full", "pruned"))
        return f"{label} & {cells} \\\\"

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{The artefact-pruning intervention: the four-corpus matrix (random forest, five seeds,
protocol of Table~\\ref{{tab:xdatasetrf}}) without the three TreeSHAP-named features
\\texttt{{tld\\_len}}, \\texttt{{subdom\\_cnt}}, \\texttt{{dot\\_cnt}}, against the full
21-feature schema.}}
\\label{{tab:pruned}}
\\small
\\begin{{tabular}}{{l cc cc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{F1}} & \\multicolumn{{2}}{{c}}{{ROC-AUC}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
 & 21 features & 18 (pruned) & 21 features & 18 (pruned) \\\\
\\midrule
{row("Mean diagonal (in-distribution)", 0)}
{row("Mean off-diagonal (transfer)", 1)}
{row("Generalisation gap", 2)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def tab_pruned_models():
    """Per-family robustness check for the pruning intervention: the gap, full schema vs pruned,
    for each of the three families the matrix was run with."""
    files = {("RandomForest", metric, cfg): f"cross_dataset_{metric}{suf}.csv"
             for metric in ("F1", "ROC-AUC") for cfg, suf in (("full", ""), ("pruned", "_pruned"))}
    for model in ("LogReg", "HistGB"):
        for metric in ("F1", "ROC-AUC"):
            for cfg, suf in (("full", ""), ("pruned", "_pruned")):
                files[(model, metric, cfg)] = f"cross_dataset_{metric}_{model}{suf}.csv"
    gaps = {k: _matrix_stats(os.path.join("data/processed/p2", f)) for k, f in files.items()}
    if any(v is None for v in gaps.values()):
        return "% per-model matrices incomplete -- run run_cross_dataset.py --model ...\n"
    label = {"LogReg": "Logistic regression", "RandomForest": "Random Forest", "HistGB": "HistGB"}
    rows = "\n".join(
        f"{label[m]} & " + " & ".join(f"{gaps[(m, met, cfg)][2]:.3f}"
                                      for met in ("F1", "ROC-AUC") for cfg in ("full", "pruned"))
        + " \\\\" for m in ("LogReg", "RandomForest", "HistGB"))
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Generalisation gap (mean diagonal $-$ mean off-diagonal) on the four-corpus matrix,
full schema vs.\\ artefact-pruned, across model families.}}
\\label{{tab:pruned_models}}
\\small
\\begin{{tabular}}{{l cc cc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{F1 gap}} & \\multicolumn{{2}}{{c}}{{ROC-AUC gap}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Model family & 21 features & 18 (pruned) & 21 features & 18 (pruned) \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def tab_adaptation():
    """The CORAL domain-adaptation intervention (run_cross_dataset.py --adapt coral) against the
    canonical baseline matrices, all regenerated in the same library environment."""
    files = {("RandomForest", met, cfg): f"cross_dataset_{met}{suf}.csv"
             for met in ("F1", "ROC-AUC")
             for cfg, suf in (("base", ""), ("coral", "_coral"))}
    for met in ("F1", "ROC-AUC"):
        files[("LogReg", met, "base")] = f"cross_dataset_{met}_LogReg.csv"
        files[("LogReg", met, "coral")] = f"cross_dataset_{met}_LogReg_coral.csv"
    runs = {k: _matrix_stats(os.path.join("data/processed/p2", f)) for k, f in files.items()}
    if any(v is None for v in runs.values()):
        return "% CORAL/base matrices incomplete -- run run_cross_dataset.py --adapt coral\n"
    label = {"RandomForest": "Random Forest", "LogReg": "Logistic regression"}
    rows = "\n".join(
        f"{label[m]} & " + " & ".join(f"{runs[(m, met, cfg)][2]:.3f}"
                                      for met in ("F1", "ROC-AUC") for cfg in ("base", "coral"))
        + " \\\\" for m in ("RandomForest", "LogReg"))
    rf_f1_off = (runs[("RandomForest", "F1", "base")][1], runs[("RandomForest", "F1", "coral")][1])
    rf_roc_off = (runs[("RandomForest", "ROC-AUC", "base")][1],
                  runs[("RandomForest", "ROC-AUC", "coral")][1])
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{The domain-adaptation intervention: generalisation gap (mean diagonal $-$ mean
off-diagonal) without and with CORAL alignment of source features to the unlabelled target,
off-diagonal cells only; protocol of Table~\\ref{{tab:xdatasetrf}}.}}
\\label{{tab:adapt}}
\\small
\\begin{{tabular}}{{l cc cc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{F1 gap}} & \\multicolumn{{2}}{{c}}{{ROC-AUC gap}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Model family & no adapt.\\ & CORAL & no adapt.\\ & CORAL \\\\
\\midrule
{rows}
\\midrule
\\multicolumn{{5}}{{l}}{{\\footnotesize RF off-diagonal means: F1 ${rf_f1_off[0]:.3f} \\to
{rf_f1_off[1]:.3f}$;\\quad ROC-AUC ${rf_roc_off[0]:.3f} \\to {rf_roc_off[1]:.3f}$.}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def gen_coral_degeneracy():
    """The sentence that stops CORAL's F1 gain from being read as a repair.

    The subsection once credited CORAL as the one intervention that improves transfer, on a mean
    off-diagonal F1 rising 0.629 -> 0.673. On a balanced target the all-positive classifier scores
    exactly 0.667, and on PhishVN exactly 0.817 -- and most of CORAL's off-diagonal cells sit ON
    those values, several while their ROC-AUC is BELOW chance. F1 at a fixed threshold cannot tell
    "learned to transfer" from "gave up and predicted one class"."""
    f1p = "data/processed/p2/cross_dataset_F1_coral.csv"
    rocp = "data/processed/p2/cross_dataset_ROC-AUC_coral.csv"
    if not (os.path.exists(f1p) and os.path.exists(rocp)):
        return "% CORAL matrices absent -- run run_cross_dataset.py --adapt coral\n"
    f = pd.read_csv(f1p, index_col=0)
    r = pd.read_csv(rocp, index_col=0)
    # Positive rate per target corpus: PhishVN is read from the corpus, the external three are
    # balanced by construction. Never hard-code 0.691 -- it moves with the deposit.
    rates = {}
    for name in f.columns:
        path = "data/processed/vn_compphish.csv" if name == "PhishVN" else None
        if path and os.path.exists(path):
            from train_url_baseline import add_label
            rates[name] = float(add_label(
                pd.read_csv(path, usecols=lambda c: c in ("label", "y")))["y"].mean())
        else:
            rates[name] = 0.5
    const = {k: 2 * v / (1 + v) for k, v in rates.items()}
    deg, sub = [], 0
    for i in f.index:
        for j in f.columns:
            if i == j:
                continue
            if abs(f.loc[i, j] - const[j]) < 0.005:
                deg.append((i, j, f.loc[i, j], r.loc[i, j]))
                sub += r.loc[i, j] < 0.5
    n_off = len(f.index) * (len(f.columns) - 1)
    worst = min(deg, key=lambda t: t[3]) if deg else None
    return (
        f"${len(deg)}$ of the ${n_off}$ adapted transfer cells sit within $0.005$ of the "
        f"all-positive classifier's F1 on their target "
        f"(${const['PhiUSIIL']:.3f}$ on the balanced corpora, ${const['PhishVN']:.3f}$ on "
        f"PhishVN at its ${rates['PhishVN']:.3f}$ positive rate), and ${sub}$ of those "
        f"do so while ranking below chance: "
        f"{worst[0]}$\\to${worst[1]} posts F1 ${worst[2]:.3f}$ against ROC-AUC "
        f"${worst[3]:.3f}$.\n")


def fig_repair_attempts():
    """Every attempt this paper makes at the generalisation gap on one axis -- the transfer section's
    negative-result spine, otherwise legible only by reading four subsections in sequence.

    A DUMBBELL AND NOT A BAR OF THE GAP, because pruning "closes" the gap and a gap-only chart would
    draw that as an improvement: the in-distribution end fell too, so the gap narrowed by getting
    worse at the task. An earlier version of this claim got exactly that backwards. Pooling's
    cross-corpus end is the union-trained held-out score, the number that design produces."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()

    def stats(fname):
        s = _matrix_stats(os.path.join("data/processed", fname))
        return None if s is None else (s[0], s[1])

    base = stats("cross_dataset_F1.csv")
    pruned = stats("cross_dataset_F1_pruned.csv")
    coral = stats("cross_dataset_F1_coral.csv")
    lr_base = stats("cross_dataset_F1_LogReg.csv")
    lr_coral = stats("cross_dataset_F1_LogReg_coral.csv")
    comb_path = "data/processed/p2/combined_training.csv"
    if any(v is None for v in (base, pruned, coral, lr_base, lr_coral)) \
            or not os.path.exists(comb_path):
        print("[i] repair-attempt inputs incomplete — run run_cross_dataset.py (+ --drop/--adapt) "
              "and run_combined_training.py; skipping figure.")
        return
    comb = pd.read_csv(comb_path)
    # combined_training.csv stores the 21-feature diagonal as indist_F1 for both configs; the
    # pruned row must use the pruned matrix's own diagonal, as tab_pruned does.
    pool = {"full": (base[0], comb[comb.config == "full"].F1.mean()),
            "pruned": (pruned[0], comb[comb.config == "pruned"].F1.mean())}
    rows = [
        ("Baseline\n21 features, single source", base),
        ("Artefact pruning\n18 features", pruned),
        ("CORAL alignment\nunlabelled target", coral),
        ("Pooled training\nunion of the other three", pool["full"]),
        ("Pooled $+$ pruned", pool["pruned"]),
    ]
    # Sized for a single IEEE Access column (3.5 in): at the old 6.4 in the text printed at ~5 pt.
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ys = list(range(len(rows)))[::-1]
    for y, (label, (diag, off)) in zip(ys, rows):
        ax.plot([off, diag], [y, y], color=GRAY, lw=2.0, solid_capstyle="round", zorder=1)
        ax.scatter([off], [y], s=52, color=ORANGE, zorder=3)
        ax.scatter([diag], [y], s=52, color=BLUE, zorder=3)
        ax.text((off + diag) / 2, y + 0.20, f"gap {diag - off:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    # the baseline's transfer end, so every later row is read as a move against it
    ax.axvline(base[1], color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=0)
    ax.text(base[1] - 0.004, -0.62, "baseline transfer", fontsize=7.5, color=INK,
            ha="right", va="bottom", alpha=0.75)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("F1 (four-corpus setup, Random Forest, five seeds)", fontsize=8.5)
    ax.set_xlim(0.52, 0.93)
    ax.set_ylim(-0.75, len(rows) - 0.45)
    ax.grid(axis="x", zorder=0)
    # Legend ABOVE the axes: placed inside, its proxy markers sit on the value axis and read as a
    # sixth row of data.
    ax.scatter([], [], s=52, color=ORANGE, label="cross-corpus (transfer)")
    ax.scatter([], [], s=52, color=BLUE, label="in-distribution")
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2,
              bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_repair_attempts.pdf")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def tab_shap(n_sample=2000, top_k=10, seeds=5):
    """TreeSHAP attribution for the PhishVN in-distribution Random Forest (the transfer matrix's
    diagonal cell): which lexical features carry the decision and how concentrated it is. Mean
    |SHAP| per feature on a stratified subsample of each seed's test split.

    NOT called by main(): it refits the forest five times and runs an exact TreeSHAP pass, minutes
    rather than milliseconds, and audit_stale_assets.py runs main() on every commit. Regenerate
    deliberately with `python scripts/make_p2_bench_assets.py --shap`."""
    try:
        import shap
    except ImportError:
        print("[i] shap not installed — skipping TreeSHAP table."); return
    corpus = "data/processed/vn_compphish.csv"
    if not os.path.exists(corpus):
        print("[i] vn_compphish.csv absent — run align_compphish.py first; skipping TreeSHAP.")
        return
    from run_cross_dataset import load_corpus, in_dataset_split
    from train_url_baseline import COMPPHISH, make_model

    df = load_corpus(corpus)
    imps, signs = [], []
    for s in range(seeds):
        tr, te = in_dataset_split(df, s)
        m = make_model("RandomForest", s).fit(tr[COMPPHISH], tr["y"])
        sub = te.groupby("y", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n_sample // 2), random_state=s))
        sv = shap.TreeExplainer(m).shap_values(sub[COMPPHISH])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imps.append(np.abs(sv).mean(axis=0))
        # direction: does a HIGH feature value push toward phishing? (sign of corr(x, shap))
        signs.append([np.corrcoef(sub[c], sv[:, i])[0, 1] if sub[c].nunique() > 1 else np.nan
                      for i, c in enumerate(COMPPHISH)])
    imp_m, imp_s = np.mean(imps, axis=0), np.std(imps, axis=0)
    sign_m = np.nanmean(np.array(signs, dtype=float), axis=0)
    share = imp_m / imp_m.sum()
    order = np.argsort(-imp_m)
    top = order[:top_k]
    top3_share = share[order[:3]].sum()

    def arrow(v):
        return "n/a" if np.isnan(v) else ("$\\uparrow$ phishing" if v > 0 else "$\\uparrow$ benign")
    body = "\n".join(
        f"\\texttt{{{COMPPHISH[i].replace('_', chr(92) + '_')}}} & "
        f"{imp_m[i]:.4f}\\,$\\pm$\\,{imp_s[i]:.4f} & {100 * share[i]:.1f}\\% & {arrow(sign_m[i])} \\\\"
        for i in top)
    rest = 100 * share[order[top_k:]].sum()
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{TreeSHAP attribution for the PhishVN in-distribution Random Forest: mean absolute SHAP
per lexical feature (mean\\,$\\pm$\\,std over {seeds} seeds) and its share of total attribution.}}
\\label{{tab:shap}}
\\small
\\begin{{tabular}}{{l c c l}}
\\toprule
Feature & mean $|$SHAP$|$ & share & high value pushes \\\\
\\midrule
{body}
\\midrule
\\multicolumn{{4}}{{l}}{{\\footnotesize remaining {len(COMPPHISH) - top_k} features together: {rest:.1f}\\%;\\quad top-3 share $={100 * top3_share:.1f}$\\%.}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_shap.tex"), tex,
                    f"(top-3 share {100 * top3_share:.1f}%)")
    _fig_shap(COMPPHISH, imp_m, imp_s, sign_m, top)


def _fig_shap(feats, imp_m, imp_s, sign_m, top):
    """Horizontal bar chart of the same TreeSHAP attribution shown in tab_shap, coloured by the
    direction a high value pushes (validated blue=benign, orange=phishing; CVD dE 24.7)."""
    from figstyle import apply, ORANGE, BLUE, INK
    plt = apply()
    from matplotlib.patches import Patch

    order = list(top)[::-1]  # largest at the top of the barh
    names = [feats[i].replace("_", r"\_") for i in order]
    vals = [imp_m[i] for i in order]
    errs = [imp_s[i] for i in order]
    cols = [ORANGE if sign_m[i] > 0 else BLUE for i in order]  # phishing / benign
    # Sized for a single column (3.5 in); at 5.6 in the tick labels printed at ~5 pt.
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    y = np.arange(len(order))
    ax.barh(y, vals, xerr=errs, color=cols, edgecolor=INK, linewidth=0.5,
            error_kw=dict(ecolor=INK, lw=0.7, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\mathtt{{{n}}}$" for n in names], fontsize=8.5)
    # The axis is mean |SHAP|; the share of total attribution is the per-bar annotation, not
    # a second axis quantity, so it is named there and not in the label.
    ax.set_xlabel(r"mean $|$SHAP$|$", fontsize=8.5)
    for yi, (v, e, i) in enumerate(zip(vals, errs, order)):
        ax.text(v + e + max(vals) * 0.02, yi, f"{100 * imp_m[i] / imp_m.sum():.1f}% of total",
                va="center", fontsize=7.5)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.legend(handles=[Patch(fc=ORANGE, ec=INK, label="high value $\\to$ phishing"),
                       Patch(fc=BLUE, ec=INK, label="high value $\\to$ benign")],
              frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.02, 0.10))  # clear of the short lower annotations
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_shap.pdf")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def tab_hpo():
    # The optimiser comparison is withdrawn.  The corrected GWO rerun remains an audit artifact,
    # but the manuscript's HPO control is random-search-only until a pre-specified, adequately
    # repeated equal-budget optimiser study is run.
    path = "data/processed/p2/p2_hpo.csv"
    if not os.path.exists(path):
        return "% p2_hpo.csv not generated yet -- run scripts/run_p2_hpo.py\n[HPO table to be filled]"
    df = pd.read_csv(path)
    if "search_seed" in df.columns and df["search_seed"].notna().any():
        lines = [
            "\\begin{table*}[t]\\centering",
            "\\scriptsize",
            "\\caption{Corrected GWO versus equal-budget random search on the phishing-temporal"
            " train window. Default rows are evaluated over 3 model seeds; tuned rows are shown"
            " for 3 independent search seeds, with each test metric reported as mean$\\pm$std over"
            " the same 3 model seeds.}",
            "\\label{tab:hpo}",
            "\\begin{tabular}{llrccccc}\\toprule",
            "Family & Config & Search & CV GWO & CV random & F1 & PR-AUC & FPR@0.90 \\\\ \\midrule",
        ]
        for _, r in df.iterrows():
            seed = "--" if pd.isna(r.get("search_seed")) else f"{int(r.search_seed)}"
            cv_gwo = "--" if pd.isna(r.get("cv_prauc_gwo")) else f"{r['cv_prauc_gwo']:.4f}"
            cv_random = "--" if pd.isna(r.get("cv_prauc_random")) else f"{r['cv_prauc_random']:.4f}"
            lines.append(f"{r.family} & {r.config} & {seed} & {cv_gwo} & {cv_random} & "
                         f"{r['F1']:.3f}$\\pm${r['F1_std']:.3f} & "
                         f"{r['PR-AUC']:.3f}$\\pm${r['PR-AUC_std']:.3f} & "
                         f"{r['FPR@R0.90']:.3f}$\\pm${r['FPR@R0.90_std']:.3f} \\\\")
        lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
        return "\n".join(lines)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Tuned vs.\\ default configurations, tuned on the direct-dated phishing"
        " rolling-origin train window only and evaluated once on the test window; mean$\\pm$std over 3 model"
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


def tab_official_split_priors():
    """Audit the immutable P1 split without treating it as a P2 evaluation arm."""
    df = pd.read_csv("data/processed/dataset_url.csv", usecols=["label", "tier", "split"])
    df["is_phish"] = df["label"].astype(str).eq("phishing")
    populations = [
        ("All tiers", df),
        ("Gold$+$silver", df[df["tier"].astype(str).isin(["gold", "silver"])]),
        ("Gold only", df[df["tier"].astype(str).eq("gold")]),
    ]

    def cell(d, split):
        s = d[d["split"].astype(str).eq(split)]
        n, p = len(s), int(s["is_phish"].sum())
        return f"{p:,}/{n:,} ({100 * p / n:.1f}\\%)" if n else "--"

    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Class support in PhishVN's published hybrid group split (phishing/total and"
        " phishing share). This is a split audit, not our evaluation arm. Gold-only validation"
        " has no positive row and therefore cannot calibrate or select a binary detector.}",
        "\\label{tab:official-priors}",
        "\\small",
        "\\begin{tabular}{lccc}\\toprule",
        "Population & Train & Validation & Test \\\\ \\midrule",
    ]
    for name, d in populations:
        lines.append(f"{name} & {cell(d, 'train')} & {cell(d, 'val')} & {cell(d, 'test')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main(shap_too: bool = False):
    os.makedirs(SEC, exist_ok=True)
    for name, fn in [("tab_families", tab_families), ("tab_strict", tab_strict),
                     ("tab_decomp", tab_decomp),
                     ("tab_maxf1", tab_maxf1),
                     ("gen_maxf1_verdict", gen_maxf1_verdict),
                     ("tab_trivial_floor", tab_trivial_floor),
                     ("gen_leakage_verdict", gen_leakage_verdict),
                     ("gen_guard_control", gen_guard_control),
                     ("gen_charcnn", gen_charcnn),
                     ("gen_strict_verdict", gen_strict_verdict),
                     ("tab_stacking", tab_stacking),
                     ("gen_stacking_verdict", gen_stacking_verdict),
                     ("tab_shiftmatrix", tab_shiftmatrix),
                     ("tab_confusion", tab_confusion),
                     ("gen_forecastability", gen_forecastability),
                     ("gen_fcts", gen_fcts),
                     ("tab_xdataset", tab_xdataset), ("tab_hpo", tab_hpo),
                     ("tab_official_split_priors", tab_official_split_priors),
                     # transfer interventions, moved from P3 on 2026-08-19
                     ("tab_xdataset_rf", tab_xdataset_rf),
                     ("tab_pruned", tab_pruned),
                     ("tab_pruned_models", tab_pruned_models),
                     ("tab_adaptation", tab_adaptation),
                     ("gen_coral_degeneracy", gen_coral_degeneracy)]:
        # fn() is evaluated BEFORE the target is opened; the old line truncated first and an
        # exception in fn left the asset empty. See scripts/genfile.py.
        write_generated(os.path.join(SEC, name + ".tex"), fn())
    # The refresh-window verdict ships only once its CSVs exist (PREREG_refresh_window.md);
    # until then the file is absent rather than a placeholder no section reads.
    refresh_tex = os.path.join(SEC, "gen_refresh_verdict.tex")
    refresh = gen_refresh_verdict()
    if refresh:
        write_generated(refresh_tex, refresh)
    elif os.path.exists(refresh_tex):
        os.remove(refresh_tex)
        print(f"[i] {refresh_tex} removed — the refresh window has not arrived")
    gen_decomp_macros()  # writes its own file (macros for the decomposition figure)
    fig_eval_design()    # writes figures/eval_design.pdf
    fig_family_pr()      # writes figures/pr_families.pdf
    fig_design_pr()      # writes figures/pr_designs.pdf
    fig_prauc_designs()  # writes figures/pr_auc_designs.pdf
    fig_prauc_cost()     # writes figures/pr_auc_cost.pdf
    fig_repair_attempts()  # writes figures/fig_repair_attempts.pdf
    # tab_shap refits the forest and runs TreeSHAP (minutes, and needs the shap package), so it
    # is opt-in: audit_stale_assets.py calls main() on every commit and must stay cheap.
    if shap_too:
        tab_shap()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shap", action="store_true",
                    help="also re-measure the TreeSHAP attribution table and figure (minutes)")
    main(shap_too=ap.parse_args().shap)
