#!/usr/bin/env python3
"""
make_p6_xai_assets.py — Generate the XAI paper's results tables from the experiment CSVs
(never hand-typed; papers/P6_xai).

Reads  data/processed/p6/p6_protocol_shap.csv, p6_tld_shap.csv, p6_brand_hits.csv
Writes papers/P6_xai/sections/tab_shap_contrast.tex, tab_tld.tex, tab_brands.tex
RUN:  python scripts/make_p6_xai_assets.py
Why each table, figure and guard is shaped the way it is: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

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

SEC = "papers/P6_xai/sections"
FIG = "papers/P6_xai/figures"
# registry tokens that are generic Vietnamese words / places, not impersonatable brands --
# excluded from the NAMED-BRAND count; the any-token count is reported separately, unfiltered
GENERIC = {"dientu", "taichinh", "tieudu", "quocte", "didong", "phong", "hanoi", "duong",
           "truyenhinh", "saigon", "binhduong", "vietnam", "online", "nguyen", "thanh",
           # review 2026-07-31: generic words/places that survived the first list
           "phattrien", "thinhvuong", "chung", "tphcm", "baoha"}
N_DATED_PH_DOMAINS = 18543  # dated phishing registrable domains (denominator, from the run log)


def n_dated_ph_domains() -> int:
    """The brand-study denominator: DISTINCT dated phishing registrable domains. The locus CSV
    (run_p6_suffix_blindspot.py) is one row per (registrable domain, path) and so holds a few
    more rows than there are domains (ten rows carry a path); its distinct `rdom` count is the
    same population run_p6_vn_reading.py's brand matcher iterates, and is used here so the two
    generated sentences cannot disagree on the denominator again (18,543 vs 18,547 shipped)."""
    f = "data/processed/p6/p6_brand_locus.csv"
    if os.path.exists(f):
        return int(pd.read_csv(f, usecols=["rdom"]).rdom.nunique())
    return N_DATED_PH_DOMAINS


def num(x) -> str:
    """Thousands separator that survives math mode. A bare "$3,017$" renders as "3, 017" and a
    guard reading the PDF for "3,017" fails on the spacing, not on the number."""
    return f"{int(round(x)):,}".replace(",", "{,}")


def esc(s):
    return str(s).replace("_", "\\_")


def tab_shap_contrast():
    df = pd.read_csv("data/processed/p6/p6_protocol_shap.csv").head(10)
    rho = pd.read_csv("data/processed/p6/p6_protocol_shap_rho.csv")
    r0 = rho.iloc[0]
    lines = [
        "\\begin{table*}[t]\\centering",
        # Caption kept to the read-out; the re-seeding band, the zero-attribution tie and the
        # XGBoost exception live in gen_rho.tex and the prose of Section ssec:rho.
        "\\caption{Global attribution (mean $|$SHAP$|$ over "
        f"{int(r0.n_seeds)} independently-masked seeds) of the CatBoost detector under the"
        " random and phishing-temporal protocols, top-10 features by temporal attribution;"
        f" between-protocol Spearman $\\rho = {r0.rho_between_mean:.3f}"
        f"\\pm{r0.rho_between_sd:.3f}$ (mean\\,$\\pm$\\,sd, all 21 features).}}",
        "\\label{tab:shapcontrast}",
        "\\begin{tabular}{lcccc}\\toprule",
        "Feature & $|$SHAP$|$ random & $|$SHAP$|$ temporal & rank$_{rnd}$ & rank$_{tmp}$"
        " \\\\ \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"\\texttt{{{esc(r.feature)}}} & {r.shap_random:.3f} & "
                     f"{r.shap_temporal:.3f} & {int(r.rank_random)} & {int(r.rank_temporal)}"
                     f" \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def tab_prospective_ablation():
    """Locked forward holdout and refit ablations.  The table intentionally prints both sides
    of the redistribution: a lower short-suffix FNR is not a repair when short-suffix false
    alarms and long-suffix misses rise sharply at the same calibration budget."""
    d = pd.read_csv("data/processed/p6/p6_prospective_ablation.csv")
    labels = {
        "full": "Full 21-feature model",
        "minus_tld_len": r"Remove \texttt{tld\_len}",
        "minus_length_family": r"Remove length family$^{\dagger}$",
    }

    def ms(arm, group, col):
        x = d[(d.arm == arm) & (d.stratum == group)][col]
        return f"{x.mean():.3f}$\\pm${x.std(ddof=1):.3f}"

    lines = [
        r"\begin{table*}[t]\centering",
        r"\caption{Locked forward (prospective-style) holdout, five CatBoost seeds. Models fit",
        r"on the oldest 60\% of dated phishing and calibrate on the next 20\%; the newest 20\%",
        r"is held out from fitting and calibration in this experiment. All arms use the full model's calibration-benign",
        r"false-positive budget. Values are mean$\pm$sd across model seeds.}",
        r"\label{tab:prospectiveablation}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r"Refit arm & Overall AUC & Overall MCC & FNR$_{short}$ & FPR$_{short}$ & FNR$_{long}$ & FPR$_{long}$ \\ \midrule",
    ]
    for arm in labels:
        lines.append(
            f"{labels[arm]} & {ms(arm, 'overall', 'roc_auc')} & "
            f"{ms(arm, 'overall', 'mcc')} & {ms(arm, 'short_all', 'fnr')} & "
            f"{ms(arm, 'short_all', 'fpr')} & {ms(arm, 'long', 'fnr')} & "
            f"{ms(arm, 'long', 'fpr')} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\begin{minipage}{0.97\linewidth}\vspace{2pt}\footnotesize",
        r"Short: all two-character public suffixes; long: non-\texttt{.vn} suffixes of length",
        r"three or more. $^{\dagger}$Removes \texttt{tld\_len}, \texttt{url\_len}, and",
        r"\texttt{dom\_len}; this is a length-family sensitivity arm, not a suffix-only effect.",
        r"\end{minipage}\end{table*}",
    ]
    return "\n".join(lines)


def gen_prospective_ablation():
    d = pd.read_csv("data/processed/p6/p6_prospective_ablation.csv")
    p = pd.read_csv("data/processed/p6/p6_prospective_protocol.csv").iloc[0]

    def mean(arm, group, col):
        return d[(d.arm == arm) & (d.stratum == group)][col].mean()

    return (
        f"the locked {p.holdout_start}--{p.holdout_end} holdout contains "
        f"${num(p.n_holdout_phishing)}$ phishing and ${num(p.n_holdout_benign)}$ benign rows. "
        f"The full model retains the two-character deficit (FNR ${mean('full', 'short_all', 'fnr'):.3f}$ "
        f"against ${mean('full', 'long', 'fnr'):.3f}$ on longer non-\\texttt{{.vn}} suffixes). "
        f"Removing \\texttt{{tld\\_len}} narrows that gap to "
        f"${mean('minus_tld_len', 'short_all', 'fnr'):.3f}$ against "
        f"${mean('minus_tld_len', 'long', 'fnr'):.3f}$, but lowers overall ROC area from "
        f"${mean('full', 'overall', 'roc_auc'):.3f}$ to "
        f"${mean('minus_tld_len', 'overall', 'roc_auc'):.3f}$ and raises the two-character benign "
        f"false-positive rate from ${mean('full', 'short_all', 'fpr'):.3f}$ to "
        f"${mean('minus_tld_len', 'short_all', 'fpr'):.3f}$"
    )


MIN_PHISH = 20   # the decomposition's inclusion floor, stated in the methodology


def _suffix_len(tld: pd.Series) -> pd.Series:
    """Character length of the public suffix, i.e. the `tld_len` feature. The `.vn(+2LD)` label
    pools two of them (2 and 6) and is therefore printed as a range, not a number."""
    return tld.astype(str).str.lstrip(".").str.len()


def tab_tld():
    """EVERY group above the inclusion floor, ordered by miss rate -- not the twelve largest by row
    count. `.head(12)` hid `.id` (31 of 31 missed, the worst group in the decomposition) and `.me`,
    `.co`, `.cn`, `.us` with it. Sort by FNR and the two-character suffixes come out together at the
    top, which is what the reframed paper is about."""
    df = pd.read_csv("data/processed/p6/p6_tld_shap.csv")
    df = df[df.n_phish >= MIN_PHISH].dropna(subset=["fnr_phish@0.5"])
    df = df.sort_values(["fnr_phish@0.5", "n_phish"], ascending=[False, False])
    n_grp = len(df)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Per-suffix decomposition, phishing-temporal test, single seed-0 split:"
        f" \\emph{{all}} {n_grp} groups with at least {MIN_PHISH} phishing rows, by miss rate"
        " (left block, then right). `ph.': phishing count; `SHAP': mean \\emph{signed}"
        " \\texttt{tld\\_len} contribution; FNR at threshold 0.5.}",
        "\\label{tab:tld}",
        "\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        # Two blocks side by side rather than one 28-row column: printing every group is the
        # point of the 2026-08-19 fix, and a single column of 28 rows cost a page.
        "\\begin{tabular}{lrrrrr@{\\hspace{10pt}}lrrrrr}\\toprule",
        ("Suffix & len & $n$ & ph. & SHAP & FNR & "
         "Suffix & len & $n$ & ph. & SHAP & FNR \\\\ \\midrule"),
    ]

    def cell(r):
        L = len(str(r.tld).lstrip("."))
        lab = "2/6" if "(" in str(r.tld) else str(L)
        return (f"\\texttt{{{esc(r.tld)}}} & {lab} & {int(r.n)} & {int(r.n_phish)} & "
                f"{r.mean_shap_tld_len:+.2f} & {r['fnr_phish@0.5']:.3f}")

    rows = [r for _, r in df.iterrows()]
    half = (len(rows) + 1) // 2
    for i in range(half):
        left = cell(rows[i])
        right = cell(rows[i + half]) if i + half < len(rows) else " & & & & & "
        lines.append(f"{left} & {right} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


# --------------------------------------------------------------------- the 2026-08-19 reframing
STRATUM_LABEL = {
    "vn_short": "\\texttt{.vn} (len 2)",
    "cc_short": "other two-letter ccTLDs",
    "vn_compound": "\\texttt{.com.vn} family (len 6)",
    "long": "suffixes of 3+ characters",
}
STRATUM_ORDER = ["vn_short", "cc_short", "vn_compound", "long"]


def _strata():
    d = pd.read_csv("data/processed/p6/p6_suffix_strata.csv")
    g = d.groupby("stratum")
    m, s = g.mean(numeric_only=True), g.std(numeric_only=True)
    return m.reindex(STRATUM_ORDER), s.reindex(STRATUM_ORDER)


def tab_suffix():
    """The reframed paper's central table: the four suffix-length strata, each with its own
    ranking quality. The `.vn(+2LD)` group of Table~\\ref{tab:tld} is rows 1 and 3 added together,
    and adding them is what produced the retracted "the blind group ranks better" reading."""
    m, s = _strata()
    # Single-column `table`, not `table*`: four rows do not need the full width, and the paper
    # already carries six full-width floats.
    lines = [
        "\\begin{table}[t]\\centering",
        # The per-stratum signed SHAP of tld_len (seed-0 model) is stated in the prose of
        # ssec:suffixprior, not here; the caption only says which column is which population.
        "\\caption{The blind spot by public-suffix length (CatBoost, phishing-temporal test):"
        " $n$, phish., FNR, FPR and ROC area (mean\\,$\\pm$\\,sd) over 5 independently-masked"
        " seeds at threshold $0.5$; FNR$_1$ on the single seed-0 split of Table~\\ref{tab:tld}.}",
        "\\label{tab:suffix}",
        "\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{l r r r r r c}\\toprule",
        "Stratum & $n$ & phish. & FNR & FNR$_1$ & FPR & ROC area \\\\ \\midrule",
    ]
    short = {"vn_short": "\\texttt{.vn}, len 2", "cc_short": "other len-2 ccTLDs",
             "vn_compound": "\\texttt{.com.vn}, len 6", "long": "other, len $\\geq 3$"}
    for k in STRATUM_ORDER:
        r, sd = m.loc[k], s.loc[k]
        lines.append(
            f"{short[k]} & {r.n:,.0f} & {r.n_phish:,.0f} & "
            f"{r.fnr:.3f} & {r.fnr_single:.3f} & {r.fpr:.3f} & "
            f"{r.auc:.3f}{{\\scriptsize\\,$\\pm${sd.auc:.3f}}} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


def gen_model_strata():
    d = pd.read_csv("data/processed/p6/p6_suffix_strata_stackcblr.csv")
    m = d.groupby("stratum").mean(numeric_only=True)
    return (
        f"the stack's FNR / within-stratum ROC area is "
        f"${m.loc['vn_short', 'fnr']:.3f}/{m.loc['vn_short', 'auc']:.3f}$ on bare "
        f"\\texttt{{.vn}}, ${m.loc['cc_short', 'fnr']:.3f}/{m.loc['cc_short', 'auc']:.3f}$ on "
        f"the other two-character ccTLDs, "
        f"${m.loc['vn_compound', 'fnr']:.3f}/{m.loc['vn_compound', 'auc']:.3f}$ on the "
        f"\\texttt{{.com.vn}} family, and ${m.loc['long', 'fnr']:.3f}/"
        f"{m.loc['long', 'auc']:.3f}$ on longer non-\\texttt{{.vn}} suffixes"
    )


def gen_balanced_strata():
    """The corrected training-time intervention: eight (stratum,class) cells rather than the
    four pooled (.vn/non-.vn,class) cells the paper's own Simpson audit invalidates."""
    d = pd.read_csv("data/processed/p6/p6_group_threshold.csv")
    base = d[d.condition == "default"]
    bal = d[d.condition == "balanced"]

    def mean(frame, col):
        return float(frame[col].mean())

    auc_parts = []
    for key in STRATUM_ORDER:
        auc_parts.append(f"${mean(base, 'auc_' + key):.3f} \\to "
                         f"{mean(bal, 'auc_' + key + '_bal'):.3f}$")
    return (
        f"at the same calibration-benign budget, the eight-cell objective lowers FNR from "
        f"${mean(base, 'fnr_vn_short'):.3f}$ to ${mean(bal, 'fnr_vn_short'):.3f}$ on bare "
        f"\\texttt{{.vn}} and from ${mean(base, 'fnr_cc_short'):.3f}$ to "
        f"${mean(bal, 'fnr_cc_short'):.3f}$ on the other two-character ccTLDs, but raises "
        f"long-suffix FNR from ${mean(base, 'fnr_long'):.3f}$ to "
        f"${mean(bal, 'fnr_long'):.3f}$ and lowers corpus F1 from "
        f"${mean(base, 'f1'):.3f}$ to ${mean(bal, 'f1'):.3f}$. Within-stratum ROC area "
        f"improves nowhere (bare \\texttt{{.vn}}, other two-character ccTLDs, "
        f"\\texttt{{.com.vn}}, long: {', '.join(auc_parts)})"
    )


RULE_LABEL = {
    "default": "Global $0.5$ (baseline)",
    "global": "Global, matched FPR budget",
    "per-group": "Per-group (\\texttt{.vn} vs other), the pooled-group remedy",
    "per-suffixlen": "Per-suffix-length (2 vs $3+$)",
    "per-stratum": "Per-stratum (all four)",
    "vn-only": "\\texttt{.vn}-only (others' op.\\ point)",
    "all-positive": "\\emph{Trivial:} flag everything",
    "all-negative": "\\emph{Trivial:} flag nothing",
}
RULE_ORDER = ["all-negative", "all-positive", "default", "global", "per-group",
              "per-suffixlen", "per-stratum", "vn-only"]


def tab_rules():
    """Every decision rule against the strata it is supposed to repair, plus the two trivial
    classifiers. F1 cannot separate the published per-group remedy from flagging every row (they
    differ in the third decimal here), so MCC and balanced accuracy are printed too; and the remedy's
    headline is bought inside one stratum, raising the miss rate on the ccTLD stratum that carries
    most of the missed phishing."""
    d = pd.read_csv("data/processed/p6/p6_suffix_threshold.csv")
    g = d.groupby("rule").mean(numeric_only=True)
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{Decision rules re-priced against the strata (CatBoost, phishing-temporal"
        " test, mean over 5 seeds), with the two trivial classifiers that fix the floor.}",
        "\\label{tab:rules}",
        "\\footnotesize\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{l ccc c ccc}\\toprule",
        "& \\multicolumn{3}{c}{FNR on} & & \\multicolumn{3}{c}{corpus-wide} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}",
        "Decision rule & \\texttt{.vn} len 2 & other 2-char & \\texttt{.com.vn} & FPR"
        " & F1 & MCC & bal.\\ acc. \\\\ \\midrule",
    ]
    for k in RULE_ORDER:
        if k not in g.index:
            continue
        r = g.loc[k]
        lines.append(
            f"{RULE_LABEL[k]} & {r.fnr_vn_short:.3f} & {r.fnr_cc_short:.3f} & "
            f"{r.fnr_vn_compound:.3f} & {r.fpr:.3f} & {r.f1:.3f} & {r.mcc:.3f} & "
            f"{r.bal_acc:.3f} \\\\")
        if k == "all-positive":
            lines.append("\\midrule")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def gen_missshare():
    """The reframing's headline: who the misses actually belong to, and which group is worst."""
    df = pd.read_csv("data/processed/p6/p6_tld_shap.csv")
    df = df[(df.n_phish > 0)].dropna(subset=["fnr_phish@0.5"]).copy()
    df["miss"] = (df["fnr_phish@0.5"] * df.n_phish).round().astype(int)
    df["L"] = _suffix_len(df.tld)
    vn = df[df.tld.str.contains("vn")]
    cc = df[(df.L == 2) & (~df.tld.str.contains("vn"))]
    rest = df[(df.L != 2) & (~df.tld.str.contains("vn"))]
    tot = int(df.miss.sum())

    def part(sub):
        k, n = int(sub.miss.sum()), int(sub.n_phish.sum())
        return k, n, 100.0 * k / tot, k / n

    kv, nv, sv, fv = part(vn)
    kc, nc, sc, fc = part(cc)
    kr, nr, sr, fr = part(rest)
    big = df[df.n_phish >= MIN_PHISH].sort_values("fnr_phish@0.5", ascending=False)
    w = big.iloc[0]
    lo_c, hi_c = wilson(kc, nc)
    return (
        f"of the ${tot}$ phishing rows the deployed detector misses on the phishing-temporal test"
        f" set, \\texttt{{.vn(+2LD)}} accounts for ${kv}$ (${sv:.1f}\\%$, FNR ${fv:.3f}$ on"
        f" $n={nv}$), \\emph{{other}} two-letter ccTLDs for ${kc}$ (${sc:.1f}\\%$, FNR"
        f" ${fc:.3f}$, Wilson 95\\% CI ${lo_c:.3f}$--${hi_c:.3f}$, $n={nc}$), and every other suffix"
        f" of three characters or more (the \\texttt{{.vn}} registry excluded) for the remaining ${kr}$ (${sr:.1f}\\%$, FNR"
        f" ${fr:.3f}$, $n={num(nr)}$). The worst-hit group is not \\texttt{{.vn}} but"
        f" \\texttt{{{esc(w.tld)}}}, at FNR ${w['fnr_phish@0.5']:.3f}$"
        f" (${int(round(w['fnr_phish@0.5'] * w.n_phish))}$ of ${int(w.n_phish)}$). Taken"
        f" together the \\texttt{{.vn}} registry and the other two-letter ccTLDs hold"
        f" ${num(nv + nc)}$ of the test set's ${num(nv + nc + nr)}$ phishing rows"
        f" (${100 * (nv + nc) / (nv + nc + nr):.1f}\\%$) and account for"
        f" ${100 * (kv + kc) / tot:.1f}\\%$ of everything it misses")


def gen_simpson():
    """M1: the retracted ROC-area reading, and the decomposition that retracts it."""
    a = pd.read_csv("data/processed/p6/p6_suffix_auc.csv")
    m, s = a.mean(numeric_only=True), a.std(numeric_only=True)
    return (
        f"pooled over both suffix lengths the \\texttt{{.vn(+2LD)}} group's ROC area is"
        f" ${m.auc_vn_pooled:.3f} \\pm {s.auc_vn_pooled:.3f}$, against"
        f" ${m.auc_other_pooled:.3f} \\pm {s.auc_other_pooled:.3f}$ elsewhere. Split by suffix"
        f" length it is ${m.auc_vn_short:.3f} \\pm {s.auc_vn_short:.3f}$ on bare \\texttt{{.vn}}"
        f" (length 2, $n_{{phish}}={int(m.n_vn_short_phish)}$) and"
        f" ${m.auc_vn_compound:.3f} \\pm {s.auc_vn_compound:.3f}$ on the \\texttt{{.com.vn}}"
        f" family (length 6, $n_{{phish}}={int(m.n_vn_compound_phish)}$), with"
        f" ${100 * m.cross_pair_share:.1f}\\%$ of the pooled group's (phishing, benign) pairs"
        f" straddling the two, pairs the pooled area scores and neither stratum's own area"
        f" does. The other two-letter ccTLDs rank at"
        f" ${m.auc_cc_short:.3f} \\pm {s.auc_cc_short:.3f}$ and the $3+$-character suffixes at"
        f" ${m.auc_long:.3f} \\pm {s.auc_long:.3f}$")


def gen_sweep():
    """M3: the mechanism, from the paper's own neutralisation sweep. Each .vn stratum's misses
    flip when it is given the OTHER stratum's suffix length and nothing else changes."""
    sw = pd.read_csv("data/processed/p6/p6_vn_deficit_sweep.csv").set_index("tld_len")
    b2, c2 = sw.loc[2.0], sw.loc[2.0]
    b6 = sw.loc[6.0]
    nb, nc = int(b2.n_bare), int(b2.n_compound)
    return (
        f"re-scoring the ${int(b2.n_fn)}$ missed \\texttt{{.vn}} rows with \\texttt{{tld\\_len}}"
        f" pinned to $2$ (bare \\texttt{{.vn}}'s own value) moves"
        f" ${int(b2.crossed_bare)}$ of the ${nb}$ bare rows across the threshold and"
        f" ${int(c2.crossed_compound)}$ of the ${nc}$ compound ones; pinning it to $6$"
        f" (the \\texttt{{.com.vn}} family's own value) moves ${int(b6.crossed_bare)}$ bare"
        f" rows and ${int(b6.crossed_compound)}$ compound ones. Each stratum's misses flip when"
        f" it is handed the \\emph{{other}} stratum's suffix length and nothing else is touched,"
        f" and neither flips at its own")


def gen_leakage():
    """Limitation (v), re-derived at the three granularities that differ, and the direction the
    correction actually runs."""
    # Seed 0's split, not the five-seed mean: these are counts of ONE partition, the one the
    # published limitation quotes. The across-seed spread is printed with the reversal.
    lkall = pd.read_csv("data/processed/p6/p6_leakage.csv")
    lk = lkall[lkall.seed == 0].iloc[0]
    g_lo, g_hi = lkall.gap_pooled_domclean.min(), lkall.gap_pooled_domclean.max()
    return (
        f"${num(lk.n_benign_test_domains_shared)}$ of the"
        f" ${num(lk.n_benign_test_domains)}$ benign registrable domains in the test"
        f" window also appear in the training window (${100 * lk.share_domains:.1f}\\%$), but a"
        f" domain count is not the exposure a ROC curve has: those domains carry"
        f" ${num(lk.n_benign_test_rows_domain_twin)}$ of the"
        f" ${num(lk.n_benign_test_rows)}$ benign test \\emph{{rows}}"
        f" (${100 * lk.share_rows:.1f}\\%$), and"
        f" ${100 * lk.share_vectors:.1f}\\%$ of all ${num(lk.n_benign_test_rows)}$ benign test"
        f" rows have a training row with an identical 21-feature vector, which needs no shared"
        f" domain at all. Deleting every benign test row"
        f" whose registrable domain recurs in training does not narrow the ROC-area ordering, it"
        f" \\emph{{reverses}} it: \\texttt{{.vn(+2LD)}} falls from"
        f" ${lk.auc_vn_pooled:.3f}$ to ${lk.auc_vn_pooled_domclean:.3f}$ while the other group"
        f" holds at ${lk.auc_other_pooled_domclean:.3f}$, and almost all of the fall is in the"
        f" \\texttt{{.com.vn}} stratum (${lk.auc_vn_compound:.3f} \\to"
        f" {lk.auc_vn_compound_domclean:.3f}$) while bare \\texttt{{.vn}}"
        f" (${lk.auc_vn_short:.3f} \\to {lk.auc_vn_short_domclean:.3f}$), the other two-letter"
        f" ccTLDs (${lk.auc_cc_short:.3f} \\to {lk.auc_cc_short_domclean:.3f}$) and the long"
        f" suffixes (${lk.auc_long:.3f} \\to {lk.auc_long_domclean:.3f}$) barely move. The"
        f" reversal is not a one-seed effect: the twin-free gap runs from ${g_lo:.3f}$ to"
        f" ${g_hi:.3f}$ across the five seeds, never touching zero")


def gen_collide():
    """Section 5.3's closing sentence, which until 2026-08-21 said "ninety per cent" from memory:
    how much of the benign test set the aggressive (domain twin OR feature-vector twin) deletion
    removes, on the seed-0 split the leakage sentence quotes and as the five-seed mean."""
    lkall = pd.read_csv("data/processed/p6/p6_leakage.csv")
    lk = lkall[lkall.seed == 0].iloc[0]
    return (
        f"The feature-vector figure (${100 * lk.share_vectors:.1f}\\%$) is an outer bound, not an"
        f" estimate: with twenty-one coarse integer features distinct benign domains collide by"
        f" coincidence as well as by memorisation, and deleting every benign test row with"
        f" either kind of twin leaves ${num(lk.n_benign_after)}$ of the"
        f" ${num(lk.n_benign_test_rows)}$ benign test rows on the seed-0 split"
        f" (${100 * lk.share_rows_any_twin:.1f}\\%$ removed;"
        f" ${100 * lkall.share_rows_any_twin.mean():.1f}\\%$ as the five-seed mean) and drives"
        f" every area toward chance")


def gen_brand_locus():
    """The conclusion asserted the lure sits "in page content, paths or platform subdomains".
    Two thirds of that is checkable here, and neither holds."""
    lo = pd.read_csv("data/processed/p6/p6_brand_locus.csv")
    n_dom = int(lo.rdom.nunique())
    n_pairs = len(lo)
    n_path = int(lo.has_path.sum())
    n_sub = int((lo.in_subdomain > 0).sum())
    n_hit = int(lo.drop_duplicates("rdom").in_rdom.sum())
    return (
        f"the corpus cannot check the first of those three places and does not support the other"
        f" two: of the ${num(n_dom)}$ dated phishing registrable domains"
        f" (${num(n_pairs)}$ distinct domain-and-path pairs), ${n_hit}$ carry a registry"
        f" token in the registrable domain, \\emph{{none}} has a platform subdomain at all"
        f" (${n_sub}$ token hits there), and only ${n_path}$ carry a path component, none of"
        f" which contains a registry token. What the collection records is a list of registrable"
        f" domains, so ``the lure is elsewhere in the URL'' is a claim this corpus is not"
        f" competent to make, and ``the lure is in page content'' is a claim no experiment here"
        f" tested")


def tab_brands():
    hb = pd.read_csv("data/processed/p6/p6_brand_hits.csv")
    hb = hb[~hb.brand.isin(GENERIC)]
    agg = (hb.groupby("brand").agg(domains=("domain", "nunique"), mangled=("mangled", "sum"))
             .sort_values("domains", ascending=False))
    top = agg.head(10)
    # The caption must say this is a top-10 cut and carry the totals it is a cut OF, or a reader
    # summing the ten rows lands 64 domains short of the named-brand count. Generated, not typed.
    n_tok, n_dom, n_man = len(agg), int(hb.domain.nunique()), int(hb.mangled.sum())
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Top ten of"
        f" ${n_tok}$ registry brand tokens in dated phishing registrable domains (generic"
        " tokens excluded); `mangled': matched only after removing dashes/digits. Together"
        f" ${top.domains.sum()}$ of ${n_dom}$ named-brand domains,"
        f" ${int(top.mangled.sum())}$ of ${n_man}$ mangled.}}",
        "\\label{tab:brands}",
        "\\begin{tabular}{lrr}\\toprule",
        "Brand token & domains & mangled \\\\ \\midrule",
    ]
    for b, r in top.iterrows():
        lines.append(f"\\texttt{{{esc(b)}}} & {int(r.domains)} & {int(r.mangled)} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines)


def gen_charcnn_strata():
    """Does the blind spot survive a representation with no suffix feature?

    The paper explains the miss by a `tld_len` prior, which is a column of the hand-built schema.
    A character-CNN reading the raw string has no such column, so this is the test of whether the
    mechanism is the cause or one model's expression of it. Emitted only when
    run_p6_charcnn_strata.py has been run; every number is recomputed here from its CSV."""
    src = "data/processed/p6/p6_charcnn_strata.csv"
    d = pd.read_csv(src)
    piv = d.pivot_table(index="stratum", columns="model", values="FNR", aggfunc="mean")
    n = d.groupby("stratum")["n_phish"].mean()
    seeds = d.seed.nunique()

    def cell(st, m):
        return float(piv.loc[st, m])

    vn_cnn = (cell("vn_short", "CharCNN") * n["vn_short"]
              + cell("vn_compound", "CharCNN") * n["vn_compound"]) / (n["vn_short"] + n["vn_compound"])
    vn_ref = (cell("vn_short", "CatBoost") * n["vn_short"]
              + cell("vn_compound", "CatBoost") * n["vn_compound"]) / (n["vn_short"] + n["vn_compound"])
    return (
        f"The prior is expressed through \\texttt{{tld\\_len}}, which is a column of the "
        f"hand-built schema, so we asked whether a representation without it inherits the blind "
        f"spot. A character-CNN over the raw URL string, trained on the identical rows and seeds "
        f"and read at the same $\\tau = 0.5$, still misses ${cell('vn_short','CharCNN'):.3f}$ of "
        f"bare \\texttt{{.vn}} phishing and ${cell('cc_short','CharCNN'):.3f}$ of the other "
        f"two-character ccTLDs, against ${cell('long','CharCNN'):.3f}$ on suffixes of three "
        f"characters or more ({seeds} seeds; the same harness reproduces the tabular model's "
        f"${cell('cc_short','CatBoost'):.3f}$ and ${cell('long','CatBoost'):.3f}$). The blind spot "
        f"is therefore not an artefact of the feature schema: it survives a model that never sees "
        f"a suffix length. What the representation buys is a smaller version of it --- pooled "
        f"\\texttt{{.vn}} falls from ${vn_ref:.3f}$ to ${vn_cnn:.3f}$ --- bought by tripling the "
        f"miss rate on the long-suffix stratum that holds most of the phishing, so it redistributes "
        f"the error rather than repairing it, which is the budget the rest of this section prices.\n")


def gen_rho():
    """Generated sentence: the calibrated rho claim, both families."""
    rho = pd.read_csv("data/processed/p6/p6_protocol_shap_rho.csv")
    parts = []
    for _, r in rho.iterrows():
        parts.append(f"{r.family} $\\rho_{{between}} = {r.rho_between_mean:.3f}\\pm"
                     f"{r.rho_between_sd:.3f}$ vs.\\ within-protocol re-seeding baselines"
                     f" ${r.rho_within_random_mean:.3f}$ (random) and"
                     f" ${r.rho_within_temporal_mean:.3f}$ (temporal)")
    # The closing claim, tested rather than asserted: it held for CatBoost and not for XGBoost.
    # Flagged when between-protocol agreement falls below a within-protocol baseline by more than
    # that baseline's own seed spread -- a shortfall inside the noise is not evidence against it.
    worse = [r.family for _, r in rho.iterrows()
             if r.rho_between_mean < r.rho_within_random_mean - r.rho_within_random_sd
             or r.rho_between_mean < r.rho_within_temporal_mean - r.rho_within_temporal_sd]
    if not worse:
        tail = ("Switching the protocol perturbs the reliance ranking no more than re-training "
                "under the same protocol does.")
    elif len(worse) == len(rho):
        tail = ("Switching the protocol perturbs the reliance ranking more than re-training under "
                "the same protocol does, for every family measured.")
    else:
        # Name the baseline it actually falls below; "both" was wrong for the one family flagged.
        detail = []
        for _, r in rho.iterrows():
            if r.family not in worse:
                continue
            under = [(n, m_, sd) for n, m_, sd in
                     (("random", r.rho_within_random_mean, r.rho_within_random_sd),
                      ("temporal", r.rho_within_temporal_mean, r.rho_within_temporal_sd))
                     if r.rho_between_mean < m_ - sd]
            # Print the baseline's own sd beside the claim, so "more than the baseline's own
            # spread" is checkable on the page rather than only in the CSV.
            detail.append(f"{r.family} (${r.rho_between_mean:.3f}$ against its "
                          + " and ".join(f"{n} baseline of ${m_:.3f} \\pm {sd:.3f}$"
                                         for n, m_, sd in under) + ")")
        tail = (f"Switching the protocol perturbs the reliance ranking no more than re-training "
                f"under the same protocol does, except for {', '.join(detail)}, where the "
                f"between-protocol agreement falls short by more than the baseline's own "
                f"seed-to-seed standard deviation.")
    return ("Across " + str(int(rho.iloc[0].n_seeds)) + " seeds with independently-drawn benign"
            " masks per run: " + "; ".join(parts) + ". " + tail + " " + _rho_degenerate())


def _rho_degenerate():
    """The correlation is over all 21 features, and five of them are dead: their mean $|$SHAP$|$
    is identically zero in every run of every family, so they sit at a shared tie in both
    profiles and agree by construction. A rank correlation cannot tell a tie it created from
    agreement it measured, and five of twenty-one is a quarter of the vector. Recomputed on the
    features that carry any attribution at all, so the headline is not resting on them."""
    from scipy.stats import spearmanr
    import itertools
    runs = pd.read_csv("data/processed/p6/p6_protocol_shap_runs.csv")
    feats = runs.feature.unique().tolist()
    dead = [f for f in feats if (runs[runs.feature == f].shap.abs() < 1e-12).all()]
    live = [f for f in feats if f not in dead]
    seeds = sorted(runs.seed.unique())

    def prof(fam, proto, s):
        return (runs[(runs.family == fam) & (runs.protocol == proto) & (runs.seed == s)]
                .set_index("feature").shap.reindex(live))

    out, flagged = [], []
    for fam in runs.family.unique():
        bet = [spearmanr(prof(fam, "random_same_rows", s),
                         prof(fam, "temporal_strict", s)).statistic for s in seeds]
        wit = {}
        for p, key in (("random_same_rows", "random"), ("temporal_strict", "temporal")):
            wit[key] = [spearmanr(prof(fam, p, i), prof(fam, p, j)).statistic
                        for i, j in itertools.combinations(seeds, 2)]
        b = float(np.mean(bet))
        if any(b < float(np.mean(v)) - float(np.std(v, ddof=1)) for v in wit.values()):
            flagged.append(fam)
        out.append(f"{fam} ${b:.3f}\\pm{np.std(bet, ddof=1):.3f}$ against"
                   f" ${np.mean(wit['random']):.3f}$ and ${np.mean(wit['temporal']):.3f}$")
    # Whether the verdict survives the recomputation is itself tested, not asserted: the same
    # rule the 21-feature sentence uses, re-applied to the live-feature correlations.
    rho = pd.read_csv("data/processed/p6/p6_protocol_shap_rho.csv")
    worse21 = sorted(r.family for _, r in rho.iterrows()
                     if r.rho_between_mean < r.rho_within_random_mean - r.rho_within_random_sd
                     or r.rho_between_mean < r.rho_within_temporal_mean - r.rho_within_temporal_sd)
    verdict = ("the comparison to the null reaches the same verdict on the same families"
               if sorted(flagged) == worse21
               else "and the verdict changes: "
                    f"{', '.join(flagged) if flagged else 'no family'} now falls below its own "
                    f"re-seeding baseline, against {', '.join(worse21) if worse21 else 'none'} "
                    "over all 21")
    dead_tex = ", ".join("\\texttt{" + esc(f) + "}" for f in dead)
    return (f"One caveat the headline number hides: {len(dead)} of the {len(feats)} features"
            f" ({dead_tex}) carry mean"
            f" $|$SHAP$|$ identically zero in every run of both families, so they occupy the same"
            f" tie in both profiles and agree by construction. Recomputed over the"
            f" {len(live)} features that carry any attribution at all, the correlations fall by"
            f" roughly $0.04$ (" + "; ".join(out) + f"); {verdict}.")


def wilson(k, n, z=1.96):
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def gen_vn_ci():
    """Generated sentence: the .vn vs elsewhere FNR with Wilson 95% CIs, from the full per-TLD
    table (not just the printed top rows)."""
    df = pd.read_csv("data/processed/p6/p6_tld_shap.csv")
    vn = df[df.tld.str.startswith(".vn")]
    ot = df[~df.tld.str.startswith(".vn")].dropna(subset=["fnr_phish@0.5"])
    k_vn = int(round((vn["fnr_phish@0.5"] * vn.n_phish).sum()))
    n_vn = int(vn.n_phish.sum())
    k_ot = int(round((ot["fnr_phish@0.5"] * ot.n_phish).sum()))
    n_ot = int(ot.n_phish.sum())
    lo_v, hi_v = wilson(k_vn, n_vn)
    lo_o, hi_o = wilson(k_ot, n_ot)
    # The "all other TLDs" aggregate is what the earlier framing rested on, and it pools the two
    # populations this paper is about. Print it, then split it in the same breath.
    ot = ot.copy()
    ot["L"] = _suffix_len(ot.tld)
    cc, lg = ot[ot.L == 2], ot[ot.L != 2]
    k_cc = int(round((cc["fnr_phish@0.5"] * cc.n_phish).sum()))
    n_cc = int(cc.n_phish.sum())
    k_lg = int(round((lg["fnr_phish@0.5"] * lg.n_phish).sum()))
    n_lg = int(lg.n_phish.sum())
    lo_c, hi_c = wilson(k_cc, n_cc)
    return (
            f"the detector misses ${k_vn}/{n_vn} = {k_vn / n_vn:.3f}$ of \\texttt{{.vn}}-registered"
            f" phishing (Wilson 95\\% CI ${lo_v:.3f}$--${hi_v:.3f}$), versus"
            f" ${k_ot / n_ot:.3f}$ (${lo_o:.3f}$--${hi_o:.3f}$, $n={num(n_ot)}$) on all other TLDs"
            f", and that second figure is an average over two populations that have nothing"
            f" in common but their exclusion from the first: ${k_cc}/{n_cc} = {k_cc / n_cc:.3f}$"
            f" (${lo_c:.3f}$--${hi_c:.3f}$) on the other two-character ccTLDs against"
            f" ${k_lg}/{num(n_lg)} = {k_lg / n_lg:.3f}$ on every other suffix of three characters or more")


def gen_brand_stats():
    """Generated sentence: BOTH brand-token counts — raw any-token and named-brand
    (post generic-word filter) — so the prose can never slide between them again."""
    hb = pd.read_csv("data/processed/p6/p6_brand_hits.csv")
    raw = hb.domain.nunique()
    named = hb[~hb.brand.isin(GENERIC)].domain.nunique()
    mangled = int(hb[~hb.brand.isin(GENERIC)].mangled.sum())
    n = n_dated_ph_domains()
    return (
            f"${raw}$ of ${n:,}$ dated phishing registrable domains (${100 * raw / n:.1f}\\%$)"
            f" contain \\emph{{any}} registry token; after excluding generic-word tokens"
            f" ({len(GENERIC)} listed in the released code), ${named}$ domains"
            f" (${100 * named / n:.1f}\\%$) contain a \\emph{{named brand}}, ${mangled}$ of them"
            f" mangled")


def _figstyle():
    from figstyle import apply
    return apply()


def fig_attribution_drift():
    """Experiment 3: does attribution drift track performance drift? Two stacked panels on one time
    axis -- the deployed model's recall on freshly-arrived phishing above, the |SHAP| of the three key
    features below, indexed to the first window so different magnitudes are comparable. The co-movement
    is the point: attribution magnitude is a live-readable correlate of performance the operator cannot
    yet see. Read from the same CSV as the generated drift verdict."""
    plt = _figstyle()
    from figstyle import ORANGE, BLUE, TEAL, PURPLE, INK, GRID
    d = pd.read_csv("data/processed/p6/p6_attribution_drift.csv")
    x = list(range(len(d)))
    xlab = [s[:7] for s in d["date_mid"]]      # YYYY-MM

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.12})
    # top: recall
    ax1.plot(x, d["recall"], "-o", color=ORANGE, lw=1.9, ms=6, mec="white", mew=0.8)
    ax1.set_ylabel("recall on\nnew phishing")
    ax1.set_ylim(0.40, 0.72)
    ax1.yaxis.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    # Inside the axes, not straddling them: right-aligned at x=0 the left spine cut the label.
    ax1.annotate(f"{d['recall'].iloc[0]:.2f}", (0, d["recall"].iloc[0]),
                 textcoords="offset points", xytext=(3, 8), fontsize=7.5, color=INK, ha="left")
    ax1.annotate(f"{d['recall'].iloc[-1]:.2f}", (x[-1], d["recall"].iloc[-1]),
                 textcoords="offset points", xytext=(2, -12), fontsize=7.5, color=INK)

    # bottom: |SHAP| indexed to window 0 = 100
    series = [("dot_cnt", BLUE), ("dash_cnt", TEAL), ("tld_len", PURPLE)]
    for feat, col in series:
        base = d[f"shap_{feat}"].iloc[0]
        idx = 100.0 * d[f"shap_{feat}"] / base
        ax2.plot(x, idx, "-o", color=col, lw=1.8, ms=5, mec="white", mew=0.7)
        ax2.annotate(feat, (x[-1], idx.iloc[-1]), textcoords="offset points",
                     xytext=(6, 0), fontsize=7.5, color=col, va="center", fontweight="bold")
    ax2.axhline(100, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.5)
    ax2.set_ylabel("|SHAP| indexed\n(window 0 = 100)")
    ax2.set_xlabel("forward time window (mid-date)")
    ax2.yaxis.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlab, rotation=30, ha="right", fontsize=7.5)
    ax2.set_xlim(-0.3, x[-1] + 1.1)
    ax1.set_xlim(-0.3, x[-1] + 1.1)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "attribution_drift.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def gen_drift_verdict():
    """Experiment 3's generated sentence: the recall drift and the sign+strength of the
    recall-vs-attribution co-variation, honestly labelled as descriptive (too few windows for a
    significance claim)."""
    from scipy.stats import spearmanr
    d = pd.read_csv("data/processed/p6/p6_attribution_drift.csv")
    corr = pd.read_csv("data/processed/p6/p6_attribution_drift_corr.csv").set_index("feature")
    nwin = len(d)
    r0, r1 = d["recall"].iloc[0], d["recall"].iloc[-1]
    c = corr["recall_shap_spearman"]
    # At n=6 the two-sided Spearman p cannot go below 0.0028 and needs |rho| >= 0.886 to clear 0.05,
    # which none of these does. Printing the p-values is the honest fix.
    p = {f: spearmanr(d.recall, d["shap_" + f]).pvalue for f in ("dot_cnt", "dash_cnt", "tld_len")}
    return (
            f"Applied forward over {nwin} equal-count time windows, the deployed detector's recall "
            f"on freshly-arrived phishing falls from ${r0:.3f}$ to ${r1:.3f}$, and the attribution "
            f"profile moves with it: over the windows, recall's Spearman correlation with "
            f"$|$SHAP$|$ is ${c['dot_cnt']:+.2f}$ for \\texttt{{dot\\_cnt}} and "
            f"${c['dash_cnt']:+.2f}$ for \\texttt{{dash\\_cnt}} (both rise as recall falls) and "
            f"${c['tld_len']:+.2f}$ for \\texttt{{tld\\_len}} (the dominant prior weakens). "
            f"\\emph{{None of the three is significant}}: two-sided $p = {p['dot_cnt']:.3f}$, "
            f"${p['dash_cnt']:.3f}$ and ${p['tld_len']:.3f}$ respectively, and with $n={nwin}$ "
            f"windows no Spearman correlation below $|\\rho| = 0.886$ can reach $p<0.05$ at all. "
            f"With only {nwin} windows this is a descriptive co-movement, not a powered lead/lag "
            f"test, and we report it as one: the candidate the discussion floated is not "
            f"refuted and not established, and what it earns is a powered replication in the "
            f"study designed for it rather than a place among this paper's findings")


def tab_case_studies():
    """Experiment 4's table: named missed .vn phishing with their model score and top signed SHAP,
    plus caught contrasts. Generated from run_p6_case_studies.py."""
    df = pd.read_csv("data/processed/p6/p6_case_studies.csv")
    lines = [
        "\\begin{table*}[t]\\centering",
        "\\caption{The \\texttt{.vn} blind spot in named cases (phishing-temporal test):"
        " blocklisted phishing domains, detector score, and top-3 signed SHAP features,"
        " largest first (positive = phishing-ward).}",
        "\\label{tab:cases}",
        "\\footnotesize\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{l l c l}\\toprule",
        "Case & Domain & Score & Top-3 signed SHAP (largest $|\\cdot|$ first) \\\\ \\midrule",
    ]
    for _, r in df.iterrows():
        dom = esc(r["domain"])
        top = esc(r["top_signed_shap"]).replace("; ", ",\\, ")
        lines.append(f"{r['kind']} & \\texttt{{{dom}}} & {r['score']:.3f} & "
                     f"\\texttt{{{top}}} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(lines)


def fig_tld_blindspot():
    """The .vn blind spot as a mechanism, not just an outcome (C4). Each TLD is a dot at its mean
    signed SHAP(tld_len) against its phishing FNR, area scaled by phishing count. The story is the
    diagonal: the TLDs whose tld_len SHAP is NEGATIVE are exactly the ones the model misses, and .vn
    sits at the extreme -- the attribution predicts the failure. Read from Table~\\ref{tab:tld}'s CSV."""
    plt = _figstyle()
    from figstyle import ORANGE, BLUE, INK
    # Same inclusion floor as Table~\ref{tab:tld}: without it ~100 singleton suffixes whose FNR is
    # 0 or 1 by construction are drawn at the same weight as the groups the claim is about.
    df = pd.read_csv("data/processed/p6/p6_tld_shap.csv").dropna(subset=["fnr_phish@0.5"])
    df = df[df.n_phish >= MIN_PHISH].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.set_axisbelow(True)
    ax.grid(True, lw=0.6)
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.6)

    # Colour by SUFFIX LENGTH, not by the sign of the attribution: length is what sorts the miss
    # rates, and encoding the sign let the figure read as if the registry were doing the work.
    two = _suffix_len(df.tld).eq(2) | df.tld.str.contains(r"\(")   # .vn(+2LD) pools 2 and 6
    sizes = 20 + 360 * (df["n_phish"] / df["n_phish"].max())
    ax.scatter(df.loc[~two, "mean_shap_tld_len"], df.loc[~two, "fnr_phish@0.5"],
               s=sizes[~two], color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.8,
               zorder=3, label="suffix of 3+ characters")
    ax.scatter(df.loc[two, "mean_shap_tld_len"], df.loc[two, "fnr_phish@0.5"],
               s=sizes[two], color=ORANGE, alpha=0.8, edgecolor="white", linewidth=0.8,
               zorder=4, label="two-character suffix")

    # Offsets hand-placed: .com at (0, -14) sat on the x-axis spine, and .cc at (9, -3) landed on
    # top of .me. .com goes above its bubble, .cc goes left of its point -- and 11pt left was not
    # far enough: at print size the label's last character still touched the marker (2026-09-02).
    label_these = {".vn(+2LD)": (9, 4), ".de": (9, -1), ".cc": (-16, -12),
                   ".id": (9, -2), ".co": (-9, 5), ".me": (9, 2),
                   ".com": (0, 13), ".online": (7, 3)}
    for _, r in df.iterrows():
        if r["tld"] in label_these:
            dx, dy = label_these[r["tld"]]
            two_r = len(str(r["tld"]).lstrip(".")) == 2 or "(" in str(r["tld"])
            col = ORANGE if two_r else INK
            ax.annotate(r["tld"], (r["mean_shap_tld_len"], r["fnr_phish@0.5"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.5,
                        color=col, ha="left" if dx > 0 else "right" if dx < 0 else "center",
                        fontweight="bold" if r["tld"] == ".vn(+2LD)" else "normal")

    # The separating miss rate, read off the data: the lowest FNR at which every group above it is
    # two-character (a hard-coded "above 0.2" was wrong by one group). Rounded UP to 2 dp --
    # 0.222 -> 0.23, because rounding to nearest printed "above 0.22" while .website sits above.
    lo = math.ceil(float(df.loc[~two, "fnr_phish@0.5"].max()) * 100) / 100
    ax.annotate(f"above FNR {lo:.2f}, every group\nhas a two-character suffix", (-2.04, 0.904),
                textcoords="offset points", xytext=(26, -36), fontsize=7.5, color=INK)
    ax.set_xlabel("mean signed SHAP(tld_len)")
    ax.set_ylabel("false-negative rate on phishing")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-2.55, 1.95)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5,
                    frameon=False, handletextpad=0.3, columnspacing=1.6)
    for h in leg.legend_handles:      # uniform legend marker size (scatter picks the first point's)
        h.set_sizes([42])
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "tld_blindspot.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def fig_shap_null():
    """The protocol-contrast null, calibrated (C3). Per family, the between-protocol Spearman rho of
    the attribution ranking against the WITHIN-protocol re-seeding baselines -- the null band for "how
    much does reliance move if you change nothing but the seed". rho_between sitting inside that band
    is the finding: the forward-in-time collapse is feature-value drift, not the model reading
    different signals. Read from the same CSV as the gen_rho sentence."""
    plt = _figstyle()
    from figstyle import ORANGE, INK, GRAY
    rho = pd.read_csv("data/processed/p6/p6_protocol_shap_rho.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, lw=0.6)

    fams = list(rho["family"])
    xs = range(len(fams))
    for xi, (_, r) in zip(xs, rho.iterrows()):
        lo = min(r["rho_within_random_mean"], r["rho_within_temporal_mean"])
        hi = max(r["rho_within_random_mean"], r["rho_within_temporal_mean"])
        # null band: the within-protocol reseeding baselines
        ax.add_patch(plt.Rectangle((xi - 0.22, lo), 0.44, hi - lo,
                                   facecolor=GRAY, alpha=0.28, edgecolor="none", zorder=1))
        ax.plot([xi - 0.22, xi + 0.22], [r["rho_within_random_mean"]] * 2,
                color=GRAY, lw=1.2, zorder=2)
        ax.plot([xi - 0.22, xi + 0.22], [r["rho_within_temporal_mean"]] * 2,
                color=GRAY, lw=1.2, zorder=2)
        # between-protocol rho with its sd
        ax.errorbar(xi, r["rho_between_mean"], yerr=r["rho_between_sd"], fmt="o", ms=8,
                    color=ORANGE, ecolor=ORANGE, elinewidth=1.4, capsize=4,
                    mec="white", mew=0.9, zorder=4)

    ax.annotate("between-protocol $\\rho$\n(mean $\\pm$ sd)",
                (0, rho["rho_between_mean"].iloc[0]), textcoords="offset points",
                xytext=(16, -22), fontsize=7.5, color=ORANGE, va="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=6,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6, alpha=0.55,
                                shrinkA=0, shrinkB=6))
    ax.annotate("within-protocol\nre-seeding band", (len(fams) - 1, 0.9975),
                textcoords="offset points", xytext=(-6, 0), fontsize=7.5, color=INK,
                ha="right", va="top")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(fams)
    ax.set_xlim(-0.5, len(fams) - 0.5)
    ax.set_ylim(0.94, 1.001)
    ax.set_ylabel("Spearman $\\rho$ of attribution ranking")
    out = os.path.join(FIG, "shap_null.pdf")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def main():
    os.makedirs(SEC, exist_ok=True)
    assets = [("tab_shap_contrast", tab_shap_contrast), ("tab_tld", tab_tld),
              ("tab_brands", tab_brands), ("gen_rho", gen_rho),
              ("gen_vn_ci", gen_vn_ci), ("gen_brand_stats", gen_brand_stats),
              ("gen_missshare", gen_missshare), ("gen_sweep", gen_sweep)]
    # The 2026-08-19 reframing's assets, each gated on the run that produces its CSV so a fresh
    # checkout without run_p6_suffix_blindspot.py still regenerates the rest of the paper.
    if os.path.exists("data/processed/p6/p6_suffix_strata.csv"):
        assets += [("tab_suffix", tab_suffix), ("gen_simpson", gen_simpson)]
    if os.path.exists("data/processed/p6/p6_suffix_strata_stackcblr.csv"):
        assets.append(("gen_model_strata", gen_model_strata))
    if os.path.exists("data/processed/p6/p6_suffix_threshold.csv"):
        assets.append(("tab_rules", tab_rules))
    if os.path.exists("data/processed/p6/p6_group_threshold.csv"):
        assets.append(("gen_balanced_strata", gen_balanced_strata))
    if os.path.exists("data/processed/p6/p6_leakage.csv"):
        assets += [("gen_leakage", gen_leakage), ("gen_collide", gen_collide)]
    if os.path.exists("data/processed/p6/p6_brand_locus.csv"):
        assets.append(("gen_brand_locus", gen_brand_locus))
    if os.path.exists("data/processed/p6/p6_charcnn_strata.csv"):
        assets.append(("gen_charcnn_strata", gen_charcnn_strata))
    if os.path.exists("data/processed/p6/p6_prospective_ablation.csv"):
        assets += [("tab_prospective_ablation", tab_prospective_ablation),
                   ("gen_prospective_ablation", gen_prospective_ablation)]
    # experiments 3-4 depend on their run_* scripts having been executed; emit their assets only
    # when the data exists, so a fresh checkout without those runs still regenerates the core paper
    if os.path.exists("data/processed/p6/p6_attribution_drift.csv"):
        assets.append(("gen_drift_verdict", gen_drift_verdict))
    if os.path.exists("data/processed/p6/p6_case_studies.csv"):
        assets.append(("tab_case_studies", tab_case_studies))
    for name, fn in assets:
        body = fn()
        # A gen_* file holds a sentence FRAGMENT the prose \input{}s mid-sentence, so its trailing
        # newline becomes a space and the caller's full stop prints detached ("on all other TLDs .").
        # A trailing %% swallows it. Tables are whole environments and must keep their newline.
        if name.startswith("gen_") and not body.rstrip().endswith("%"):
            body = body.rstrip() + "%"
        write_generated(os.path.join(SEC, name + ".tex"), body + "\n")
    fig_tld_blindspot()
    fig_shap_null()
    if os.path.exists("data/processed/p6/p6_attribution_drift.csv"):
        fig_attribution_drift()


if __name__ == "__main__":
    main()
