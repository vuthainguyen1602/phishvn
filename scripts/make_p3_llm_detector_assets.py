#!/usr/bin/env python3
r"""
make_p3_llm_detector_assets.py — P1b's zero-shot MLLM-detector table.

Reads every data/processed/p3/llm_content_baseline_<model>_k<shots>.csv (written by
llm_content_baseline.py) and emits papers/P3_multimodal/sections/tab_llm_detector.tex. Built for
SEVERAL models on purpose: one model confounds "an MLLM cannot do this" with "this model cannot",
and the refusal rate belongs to a vendor's safety policy rather than to the task. A refusal is
excluded from the metrics but reported in its own column. With no gradable results on disk a
clearly-marked placeholder table is written, so the paper still compiles.

RUN:  python scripts/make_p3_llm_detector_assets.py
The filename contract for dropping in results scored elsewhere, and the required columns:
kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import csv
import glob
import os
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated

RES = os.path.join(ROOT, "data", "processed", "p3", "llm_content_baseline.csv")
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
PROC = os.path.join(ROOT, "data", "processed")
# The minimum a dropped-in results file must carry to be scoreable.
REQUIRED = {"label", "verdict"}
OUT = os.path.join(SEC, "tab_llm_detector.tex")

# Used only once P3 \\input{}s the table: it keeps the manuscript compiling between runs.
PLACEHOLDER = r"""\begin{table*}[t]
\small
\begin{tabular}{l c c c c}
\toprule
Detector & F1 & Precision & Recall & Accuracy \\
\midrule
Zero-shot MLLM (screenshot $+$ HTML) & \multicolumn{4}{c}{\emph{[to be filled]}} \\
\bottomrule
\end{tabular}
\caption{\textbf{Preliminary} zero-shot multimodal-LLM phishing detector on the captured
landing-page subset (screenshot-first, HTML auxiliary).}
\label{tab:llmdetector}
\end{table*}"""


def runs():
    """Every per-condition results file, newest first, one row group per (model, shots).

    Filenames carry the condition because the runs must sit side by side; a fixed output path
    was the reason this comparison could only ever hold one model."""
    out = []
    for path in sorted(glob.glob(os.path.join(PROC, "llm_content_baseline_*_k*.csv"))):
        base = os.path.basename(path)[len("llm_content_baseline_"):-len(".csv")]
        model, _, shots = base.rpartition("_k")
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Say what is wrong with a hand-placed file rather than reporting an empty table: a
        # silently ignored CSV looks exactly like a model that scored nothing.
        missing = REQUIRED - set(rows[0]) if rows else REQUIRED
        if missing:
            print(f"[!] {os.path.basename(path)}: missing column(s) "
                  f"{', '.join(sorted(missing))} — skipped")
            continue
        gradable = [r for r in rows if not r.get("error")
                    and r.get("verdict") in ("phishing", "benign")]
        refused = sum(1 for r in rows if (r.get("error") or "").lower().startswith("refus"))
        if not gradable:
            print(f"[!] {os.path.basename(path)}: {len(rows)} rows, none with a phishing/benign "
                  f"verdict ({refused} refusals) — skipped")
            continue
        bad = {r.get("label") for r in rows} - {"0", "1", 0, 1}
        if bad:
            print(f"[!] {os.path.basename(path)}: label column holds {sorted(map(str, bad))[:3]}; "
                  "expected 1 = phishing, 0 = benign — skipped")
            continue
        out.append({"model": model, "shots": int(shots or 0), "rows": gradable,
                    "n_seen": len(rows), "refused": refused})
    return out


def main():
    os.makedirs(SEC, exist_ok=True)
    got = runs()
    if not got:
        # Deliberately write NOTHING. The placeholder exists so a manuscript that already
        # \\input{}s this table still compiles; P3 does not input it yet, so writing the file
        # would only plant an orphan float that `make claims` correctly refuses. It is written
        # the moment there is something to say, and the paper wires it in then.
        if os.path.exists(OUT):
            print(f"[i] no gradable results — leaving the existing {OUT} untouched")
        else:
            print("[i] no gradable results yet — nothing written. Run "
                  "scripts/llm_content_baseline.py first (needs ANTHROPIC_API_KEY).")
        return
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    lines, totals = [], []
    for run in sorted(got, key=lambda r: (r["model"], r["shots"])):
        y = [int(r["label"]) for r in run["rows"]]
        p = [1 if r["verdict"] == "phishing" else 0 for r in run["rows"]]
        f1 = f1_score(y, p, zero_division=0)
        pr = precision_score(y, p, zero_division=0)
        rc = recall_score(y, p, zero_division=0)
        ac = accuracy_score(y, p)
        shot = "zero-shot" if run["shots"] == 0 else f"{run['shots']}-shot"
        name = run["model"].replace("_", "-")
        refused_pct = 100.0 * run["refused"] / max(run["n_seen"], 1)
        lines.append(rf"\texttt{{{name}}}, {shot} & {f1:.3f} & {pr:.3f} & {rc:.3f} & "
                     rf"{ac:.3f} & {refused_pct:.1f} \\")
        totals.append((name, shot, f1, len(run["rows"]), sum(y), refused_pct))

    n_pages = max(len(r["rows"]) for r in got)
    n_ph = max(sum(int(x["label"]) for x in r["rows"]) for r in got)
    tex = ("\\begin{table*}[t]\n"
           "\\caption{Zero-shot multimodal LLMs as the detector, on the captured landing-page "
           f"subset ({n_pages} pages, {n_ph} phishing; screenshot-first, HTML auxiliary), scored "
           "over the entire labelled subset rather than a held-out split. \\emph{Refused} is the "
           "share of pages the model declined to classify; those pages count as undetected.}\n"
           "\\label{tab:llmdetector}\n\\small\n"
           "\\begin{tabular}{l c c c c c}\n\\toprule\n"
           "Detector & F1 & Precision & Recall & Accuracy & Refused (\\%) \\\\\n\\midrule\n"
           + "\n".join(lines) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}")
    write_generated(OUT, tex)
    for name, shot, f1, n, nph, ref in totals:
        print(f"    {name:26s} {shot:10s} F1={f1:.3f} n={n} phish={nph} refused={ref:.1f}%")


if __name__ == "__main__":
    main()
