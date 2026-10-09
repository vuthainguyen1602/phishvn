#!/usr/bin/env python3
"""
make_p5_sweep_assets.py — the construction sweep's table and paragraph, read off
data/processed/p5/dedup/p5_construction_sweep.csv (written by run_p5_dedup.py).

  tab_sweep.tex   AUTC, end F1 and retrains per policy under each of the three constructions
  gen_sweep.tex   what each construction choice does to the staleness gap
  gen_construction.tex  the corpus facts the construction rests on (dated share, www. copies)
  gen_psi_blind.tex     PSI against the drift trigger's reference while the static model misses
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated

CSV = os.path.join(ROOT, "data", "processed", "p5", "dedup", "p5_construction_sweep.csv")
SEC = os.path.join(ROOT, "papers", "P5_temporal_drift", "sections")
LABEL = {"released": "Released rows", "nowww": "\\texttt{www.} removed",
         "dedup": "One row per URL"}


def num(v):
    return f"{int(v):,}".replace(",", "{,}")


def construction():
    u = pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"), low_memory=False,
                    usecols=["label", "collected_at", "url_norm"])
    s = u.collected_at.astype(str).str.strip().str.slice(0, 10)
    dated = (pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
             .fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))).notna()
    host = u.url_norm.astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
    bare = host.str.replace(r"^www\.", "", regex=True)
    www = host.str.startswith("www.")
    ph = u.label.astype(str).str.lower() == "phishing"
    und = (~dated) & ph
    copy = und & bare.isin(set(bare[dated])) & www
    removed = len(u) - bare.nunique()
    body = (
        f"Of the corpus's ${num(len(u))}$ rows, only ${num(dated.sum())}$, the "
        f"{100 * dated.mean():.0f}\\% of records that carry an event date, can be placed in time. "
        f"The undated remainder includes ${num(und.sum())}$ phishing rows, and ${num(copy.sum())}$ of "
        f"them ({100 * copy.sum() / und.sum():.1f}\\%) are the \\texttt{{www.}}-prefixed copy of a "
        f"host that also appears as a dated row; keeping one row per URL removes ${num(removed)}$ rows "
        f"in all.")
    write_generated(os.path.join(SEC, "gen_construction.tex"), body)


ORDER = ("released", "nowww", "dedup", "dated")
LABEL["dated"] = "Dated rows only"
SALTS_CSV = os.path.join(os.path.dirname(CSV), "p5_construction_salts.csv")


def url_twin_share():
    """Share of phishing rows in windows 2..K of the released stream whose URL (scheme and a
    leading www. removed) also occurs in window 1."""
    import numpy as np
    from make_p5_assets import DRIFT_CSV, WINDOWS
    from retrain_drift import load
    df, _ = load(DRIFT_CSV, "collected_at", spread_undated=True)
    u = pd.read_csv(os.path.join(ROOT, "data", "processed", "dataset_url.csv"), low_memory=False,
                    usecols=["id", "url_norm"])
    bare = (u.url_norm.astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True))
    b = df.id.astype(str).map(dict(zip(u.id.astype(str), bare)))
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
    w1 = set(b.iloc[idx[0]])
    later = np.concatenate(idx[1:])
    ph = df.y.iloc[later].to_numpy() == 1
    return float(b.iloc[later][ph].isin(w1).mean())


def gaps(x, col):
    """Better updating policy minus the static model, on the AUTC of metric `col`."""
    return max(x.loc["periodic", col], x.loc["drift", col]) - x.loc["static", col]


def main():
    construction()
    d = pd.read_csv(CSV)
    g = {c: x.set_index("policy") for c, x in d.groupby("construction")}
    rows = []
    for c in ORDER:
        x = g[c]
        rows.append(f"{LABEL[c]} & ${num(x['rows'].iloc[0])}$ & "
                    f"{x.loc['static', 'autc']:.3f} & {x.loc['periodic', 'autc']:.3f} & "
                    f"{x.loc['drift', 'autc']:.3f} & {gaps(x, 'autc'):.3f} & {int(x.loc['drift', 'retrains'])} & "
                    f"{x['w1_prior'].iloc[0]:.2f} & {x['prior_min'].iloc[0]:.2f}--{x['prior_max'].iloc[0]:.2f} \\\\")
    tab = ("\\begin{table*}[t]\n\\centering\n\\caption{One corpus, one set of policies, four "
           "constructions of the stream. AUTC is the mean per-window F1 over windows 2--20; the gap is "
           "the better updating policy minus the static model; the drift trigger is the over-time "
           "PSI-or-F1-drop trigger; priors are per-window phishing shares. The dated-only stream drops "
           "every undated row and is not placed by hash.}\n"
           "\\label{tab:sweep}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{3}{c}{AUTC} & & Drift & \\multicolumn{2}{c}{Prior} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){8-9}\n"
           "Construction & Rows & Static & Periodic & Drift & Gap & retrains & Window 1 & Range "
           "\\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}")
    write_generated(os.path.join(SEC, "tab_sweep.tex"), tab)

    sal = pd.read_csv(SALTS_CSV) if os.path.exists(SALTS_CSV) else None
    srange = {}
    if sal is not None:
        for c, x in sal.groupby("construction"):
            per = [y.set_index("policy") for _, y in x.groupby(x["salt"].fillna(""))]
            gs = [gaps(y, "autc") for y in per]
            ga = [gaps(y, "auc_autc") for y in per]
            srange[c] = (min(gs), max(gs), len(gs), min(ga), max(ga))
    rows = []
    for c in ORDER:
        x = g[c]
        sr = (f"{srange[c][0]:.3f}--{srange[c][1]:.3f}" if c in srange else "--")
        rows.append(f"{LABEL[c]} & {gaps(x, 'autc'):.3f} & {gaps(x, 'auc_autc'):.3f} & "
                    f"{gaps(x, 'bacc_autc'):.3f} & {100 * x['novel_pos_share'].iloc[0]:.0f} & "
                    f"{gaps(x, 'novel_autc'):.3f} & {sr} \\\\")
    nsalt = max((v[2] for v in srange.values()), default=0)
    tab2 = ("\\begin{table}[t]\n\\centering\n\\caption{The staleness gap of Table~\\ref{tab:sweep} "
            "measured four ways: on F1 at $0.5$, on ROC-AUC and balanced accuracy (neither depends on "
            "the window's class prior), and on F1 restricted to rows whose 21-feature vector never "
            "occurs in window 1 (``unseen'', with the share of each window's phishing rows that are "
            "unseen). The last column re-draws the hash placement under "
            f"{nsalt} salts, the released placement included, and gives the range of the F1 gap.}}\n"
            "\\label{tab:gapways}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Gap, all rows} & \\multicolumn{2}{c}{Unseen vectors} & F1 gap over \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "Construction & F1 & ROC-AUC & Bal.\\ acc. & Share (\\%) & F1 gap & salts \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
    write_generated(os.path.join(SEC, "tab_gapways.tex"), tab2)

    r, w, u, dt = g["released"], g["nowww"], g["dedup"], g["dated"]
    body = (
        f"Table~\\ref{{tab:sweep}} holds the corpus and the policies fixed and changes only how the "
        f"stream is built. On the released rows the static model trails the better updating policy by "
        f"${gaps(r, 'autc'):.3f}$ AUTC. Removing a leading \\texttt{{www.}} from every URL, and nothing "
        f"else, closes that to ${gaps(w, 'autc'):.3f}$: the static model rises from "
        f"${r.loc['static', 'autc']:.3f}$ to ${w.loc['static', 'autc']:.3f}$, because it can no longer "
        f"tell the two copies of a host apart by their format. Keeping one row per URL instead opens "
        f"the gap to ${gaps(u, 'autc'):.3f}$, for a different reason: the undated background that is "
        f"left is mostly benign, so the per-window phishing share swings from "
        f"${u['prior_min'].iloc[0]:.2f}$ to ${u['prior_max'].iloc[0]:.2f}$ and the static model is "
        f"fitted on a window that is {100 * u['w1_prior'].iloc[0]:.0f}\\% phishing and scored on "
        f"windows that are mostly phishing. Dropping the undated rows altogether leaves "
        f"${num(dt['rows'].iloc[0])}$ rows and a gap of ${gaps(dt, 'autc'):.3f}$, on a stream that is "
        f"{100 * dt['w1_prior'].iloc[0]:.0f}\\% phishing in its first window and near its F1 ceiling "
        f"throughout. Four constructions of one corpus therefore give F1 gaps from "
        f"${min(gaps(g[c], 'autc') for c in ORDER):.3f}$ to ${max(gaps(g[c], 'autc') for c in ORDER):.3f}$ "
        f"AUTC, so none of them can be read as the rate at which a detector ages.")
    write_generated(os.path.join(SEC, "gen_sweep.tex"), body)

    # Table 4's three checks on that range: a prior-free metric, the rows the static model cannot
    # have memorised, and the placement itself re-drawn.
    auc = {c: gaps(g[c], "auc_autc") for c in ORDER}
    nov = {c: gaps(g[c], "novel_autc") for c in ORDER}
    seen = {c: 1 - g[c]["novel_pos_share"].iloc[0] for c in ORDER}
    txt = (
        f"Table~\\ref{{tab:gapways}} tests that range three ways. F1 at a fixed threshold moves with "
        f"the class prior, and on ROC-AUC the one-row-per-URL gap falls from ${gaps(u, 'autc'):.3f}$ to "
        f"${auc['dedup']:.3f}$, so most of its F1 gap is the prior swing; the released stream keeps a "
        f"ranking gap of ${auc['released']:.3f}$, and ROC-AUC gaps still run from "
        f"${min(auc.values()):.3f}$ to ${max(auc.values()):.3f}$ across the four constructions. The 21 "
        f"lexical features are coarse: {100 * seen['released']:.0f}\\% of the later windows' phishing "
        f"rows in the released stream, and {100 * seen['nowww']:.0f}\\% once \\texttt{{www.}} is "
        f"removed, repeat a feature vector that window 1 already holds, against "
        f"{100 * url_twin_share():.1f}\\% that repeat one of its URLs. Restricted to the rows whose vector window 1 never held, the gap is "
        f"${nov['released']:.3f}$ on the released stream and ${nov['nowww']:.3f}$ with \\texttt{{www.}} "
        f"removed, so the small residual of the stripped stream is not an artefact of rows the static "
        f"model has already seen.")
    if srange:
        txt += (f" Re-drawing the hash placement under {nsalt} salts moves the released gap between "
                f"${srange['released'][0]:.3f}$ and ${srange['released'][1]:.3f}$, the stripped one between "
                f"${srange['nowww'][0]:.3f}$ and ${srange['nowww'][1]:.3f}$ and the one-row-per-URL one "
                f"between ${srange['dedup'][0]:.3f}$ and ${srange['dedup'][1]:.3f}$ (on ROC-AUC "
                f"${srange['released'][3]:.3f}$--${srange['released'][4]:.3f}$, "
                f"${srange['nowww'][3]:.3f}$--${srange['nowww'][4]:.3f}$ and "
                f"${srange['dedup'][3]:.3f}$--${srange['dedup'][4]:.3f}$); ")
        def apart(k):  # the three placed constructions' salt ranges are pairwise disjoint on metric k
            lo, hi = (0, 1) if k == "f1" else (3, 4)
            iv = sorted((srange[c][lo], srange[c][hi]) for c in ("released", "nowww", "dedup"))
            return all(iv[i][1] < iv[i + 1][0] for i in range(len(iv) - 1))
        f1_apart, auc_apart = apart("f1"), apart("auc")
        txt += ("on both metrics the constructions separate by more than the placement moves any of them."
                if f1_apart and auc_apart else
                "on F1 the constructions separate by more than the placement moves any of them, on ROC-AUC "
                "they do not." if f1_apart else
                "the placement alone moves the gap by as much as the choice of construction does.")
    write_generated(os.path.join(SEC, "gen_gapways.tex"), txt)

    # PSI against the drift trigger's reference window on the released stream, beside the
    # static model's recall on dated positives: the monitor reads the inputs, and the inputs of
    # the two URL formats differ only in a few counts that its binning averages away.
    tr = pd.read_csv(os.path.join(os.path.dirname(CSV), "p5_psi_traces.csv"))
    tr = tr[tr.construction == "released"].set_index("window").psi_vs_reference
    win = pd.read_csv(os.path.join(ROOT, "data", "processed", "p5", "p5_window_composition.csv"))
    win = win.set_index("window")
    low = win[(win.positives_dated >= 300) & (win.static_recall_dated <= 0.25)].index
    first_fire = int(tr[tr > 0.2].index.min()) if (tr > 0.2).any() else None
    w0 = int(min(low))
    w1 = (first_fire - 1) if first_fire else int(max(low))
    quiet = tr.loc[w0:w1]
    if quiet.max() >= 0.10:
        raise SystemExit(f"PSI reaches {quiet.max():.3f} before it first fires; reword gen_psi_blind")
    psi_body = (
        f"PSI does not see the split that drives the gap. Against the drift trigger's reference "
        f"window, the mean per-feature PSI stays between ${quiet.min():.3f}$ and ${quiet.max():.3f}$ "
        f"across windows {w0}--{w1}, below both the detector's $0.10$ cut and the trigger's $0.2$, "
        f"while the static model recalls at most "
        f"{100 * win.loc[w0:w1, 'static_recall_dated'].max():.0f}\\% of the dated phishing in "
        f"those windows"
        + (f"; it first crosses $0.2$ in window {first_fire}." if first_fire else ".")
        + " The two URL formats differ in a handful of counts (subdomains, dots, length), and an "
          "average over twenty-one binned features dilutes them.")
    write_generated(os.path.join(SEC, "gen_psi_blind.tex"), psi_body)


if __name__ == "__main__":
    main()
