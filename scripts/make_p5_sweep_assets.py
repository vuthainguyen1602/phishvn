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


def main():
    construction()
    d = pd.read_csv(CSV)
    g = {c: x.set_index("policy") for c, x in d.groupby("construction")}
    rows = []
    for c in ("released", "nowww", "dedup"):
        x = g[c]
        gap = max(x.loc["periodic", "autc"], x.loc["drift", "autc"]) - x.loc["static", "autc"]
        rows.append(f"{LABEL[c]} & ${num(x['rows'].iloc[0])}$ & "
                    f"{x.loc['static', 'autc']:.3f} & {x.loc['periodic', 'autc']:.3f} & "
                    f"{x.loc['drift', 'autc']:.3f} & {gap:.3f} & {int(x.loc['drift', 'retrains'])} & "
                    f"{x['w1_prior'].iloc[0]:.2f} & {x['prior_min'].iloc[0]:.2f}--{x['prior_max'].iloc[0]:.2f} \\\\")
    tab = ("\\begin{table*}[t]\n\\centering\n\\caption{One corpus, one set of policies, three "
           "constructions of the stream. AUTC is the mean per-window F1 over windows 2--20; the gap is "
           "the better updating policy minus the static model; the drift trigger is the over-time "
           "PSI-or-F1-drop trigger; priors are per-window phishing shares.}\n"
           "\\label{tab:sweep}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{3}{c}{AUTC} & & Drift & \\multicolumn{2}{c}{Prior} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){8-9}\n"
           "Construction & Rows & Static & Periodic & Drift & Gap & retrains & Window 1 & Range "
           "\\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}")
    write_generated(os.path.join(SEC, "tab_sweep.tex"), tab)

    def gap(c):
        x = g[c]
        return max(x.loc["periodic", "autc"], x.loc["drift", "autc"]) - x.loc["static", "autc"]
    r, w, u = g["released"], g["nowww"], g["dedup"]
    body = (
        f"Table~\\ref{{tab:sweep}} holds the corpus and the policies fixed and changes only how the "
        f"stream is built. On the released rows the static model trails the better updating policy by "
        f"${gap('released'):.3f}$ AUTC. Removing a leading \\texttt{{www.}} from every URL, and nothing "
        f"else, closes that to ${gap('nowww'):.3f}$: the static model rises from "
        f"${r.loc['static', 'autc']:.3f}$ to ${w.loc['static', 'autc']:.3f}$, because it can no longer "
        f"tell the two copies of a host apart by their format. Keeping one row per URL instead opens "
        f"the gap to ${gap('dedup'):.3f}$, for a different reason: the undated background that is "
        f"left is mostly benign, so the per-window phishing share swings from "
        f"${u['prior_min'].iloc[0]:.2f}$ to ${u['prior_max'].iloc[0]:.2f}$ and the static model is "
        f"fitted on a window that is {100 * u['w1_prior'].iloc[0]:.0f}\\% phishing and scored on "
        f"windows that are mostly phishing. Three defensible constructions of one corpus therefore "
        f"give staleness gaps from ${min(gap(c) for c in g):.3f}$ to ${max(gap(c) for c in g):.3f}$ "
        f"AUTC, so none of them can be read as the rate at which a detector ages.")
    write_generated(os.path.join(SEC, "gen_sweep.tex"), body)

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
