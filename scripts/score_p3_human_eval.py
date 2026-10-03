#!/usr/bin/env python3
"""
score_p3_human_eval.py — scores the two-annotator rating of band-controlled paraphrases (revision R2.8 / R3.6).

Implements the reporting fixed in papers/P3_multimodal/REVISION1/HUMAN_EVAL_README.md (protocol
2026-09-28, amendment 1 2026-09-29), and nothing beyond it. Written and committed BEFORE any
label was collected; the README records this file's SHA-256.

Per annotator and per binary question (meaning_preserved, lure_intact): proportion positive with
a 95% Wilson interval. Per binary question: raw agreement and unweighted Cohen's kappa between A
and B. For the three-way persuasiveness column (A / B / ngang): per annotator the share of each
category and the share "rewrite at least as persuasive" (B or ngang), each with a 95% Wilson
interval; raw agreement and unweighted Cohen's kappa over the three categories. No test is run.

Fail-closed: every row's two messages must match human_eval_sheet.csv in order, every rating
cell must be filled with an allowed value, or the script exits non-zero and writes nothing.
`--status` only counts filled cells.

Reads REVISION1/cham_A.xlsx and cham_B.xlsx (sheet "CHAM DIEM", rows 2-51, columns B-F).
Emits data/processed/p3/p3_human_eval.csv (merged labels) and sections/gen_human_eval.tex
(macros only; the paper text that cites them is written after scoring).

RUN: python scripts/score_p3_human_eval.py [--status] [--a PATH --b PATH]
"""
import argparse
import csv
import math
import os
import sys

import re

import openpyxl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "core", "lib"))
from paired_eval import wilson  # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REV = os.path.join(ROOT, "papers", "P3_multimodal", "REVISION1")
SHEET_CSV = os.path.join(REV, "human_eval_sheet.csv")
OUT_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_human_eval.csv")
OUT_TEX = os.path.join(ROOT, "papers", "P3_multimodal", "sections", "gen_human_eval.tex")
SHEET = "CHAM DIEM"
N_PAIRS = 50
BINARY = ("meaning_preserved", "lure_intact")
PERSUASIVE = ("A", "B", "ngang")


_LINK = re.compile(r"https?://\S*?(?=[.,;:!?)]?(?:\s|$))")


def _norm_text(s) -> str:
    """The rating sheets show every placeholder link as [link]; compare on that form."""
    return " ".join(_LINK.sub("[link]", str(s or "")).split())


def _binary(v):
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    if s in ("1", "1.0"):
        return 1
    if s in ("0", "0.0"):
        return 0
    raise ValueError(s)


def _three(v):
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    for cat in PERSUASIVE:
        if s == cat.lower():
            return cat
    raise ValueError(str(v))


def load_pairs():
    with open(SHEET_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != N_PAIRS:
        sys.exit(f"{SHEET_CSV}: expected {N_PAIRS} pairs, found {len(rows)}")
    return rows


def read_annotator(path, pairs, who):
    """Returns (ratings list, problems list). ratings[i] = (meaning, lure, persuasive)."""
    ws = openpyxl.load_workbook(path, data_only=True)[SHEET]
    ratings, problems = [], []
    for i, pair in enumerate(pairs):
        r = i + 2
        if _norm_text(ws[f"B{r}"].value) != _norm_text(pair["source"]) or \
                _norm_text(ws[f"C{r}"].value) != _norm_text(pair["rewrite"]):
            problems.append(f"{who} row {r}: messages do not match pair {pair['pair_id']}")
            ratings.append((None, None, None))
            continue
        vals = []
        for col, parse in (("D", _binary), ("E", _binary), ("F", _three)):
            try:
                v = parse(ws[f"{col}{r}"].value)
            except ValueError as e:
                problems.append(f"{who} {col}{r}: value {e} not allowed")
                v = None
            vals.append(v)
        ratings.append(tuple(vals))
    return ratings, problems


def kappa(x, y, cats):
    """Unweighted Cohen's kappa; nan when chance agreement is 1 (both raters used one category)."""
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, (float("nan") if math.isclose(pe, 1.0) else (po - pe) / (1 - pe))


def _f(v, d=3):
    return "n/a" if v != v else f"{v:.{d}f}"


def _macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(REV, "cham_A.xlsx"))
    ap.add_argument("--b", default=os.path.join(REV, "cham_B.xlsx"))
    ap.add_argument("--status", action="store_true", help="count filled cells only")
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-tex", default=OUT_TEX)
    args = ap.parse_args()

    pairs = load_pairs()
    rated = {}
    problems = []
    for who, path in (("A", args.a), ("B", args.b)):
        rated[who], p = read_annotator(path, pairs, who)
        problems += p
        filled = sum(v is not None for row in rated[who] for v in row)
        print(f"annotator {who}: {filled}/{3 * N_PAIRS} cells filled ({path})")
    if args.status:
        for p in problems:
            if "not allowed" in p or "do not match" in p:
                print("  " + p)
        return
    missing = [f"{w} pair {i + 1:02d}" for w in "AB" for i, row in enumerate(rated[w]) if None in row]
    if problems or missing:
        for p in problems:
            print("ERROR " + p, file=sys.stderr)
        if missing:
            print(f"ERROR {len(missing)} incomplete rows, e.g. {', '.join(missing[:5])}", file=sys.stderr)
        sys.exit("not scored: the sheets are incomplete or invalid; nothing written")

    n = N_PAIRS
    macros, lines = [], []
    for qi, q in enumerate(BINARY):
        tag = {"meaning_preserved": "Meaning", "lure_intact": "Lure"}[q]
        for w in "AB":
            k = sum(row[qi] for row in rated[w])
            lo, hi = wilson(k, n)
            macros += [_macro(f"HE{tag}Count{w}", k), _macro(f"HE{tag}Prop{w}", _f(k / n)),
                       _macro(f"HE{tag}Lo{w}", _f(lo)), _macro(f"HE{tag}Hi{w}", _f(hi))]
            lines.append(f"{q:18s} {w}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]")
        po, kap = kappa([r[qi] for r in rated["A"]], [r[qi] for r in rated["B"]], (0, 1))
        # PABAK = 2*p_o - 1: Cohen's kappa collapses when one rater accepts (almost) every pair,
        # so the prevalence-adjusted bias-adjusted kappa is reported beside it (Byrt et al. 1993).
        macros += [_macro(f"HE{tag}Agree", _f(po)), _macro(f"HE{tag}Kappa", _f(kap)),
                   _macro(f"HE{tag}Pabak", _f(2 * po - 1))]
        lines.append(f"{q:18s} agreement {po:.3f}, kappa {_f(kap)}, PABAK {2 * po - 1:.3f}")

    for w in "AB":
        col = [row[2] for row in rated[w]]
        for cat in PERSUASIVE:
            k = col.count(cat)
            lo, hi = wilson(k, n)
            tag = {"A": "Source", "B": "Rewrite", "ngang": "Equal"}[cat]
            macros += [_macro(f"HEPers{tag}Count{w}", k), _macro(f"HEPers{tag}Prop{w}", _f(k / n)),
                       _macro(f"HEPers{tag}Lo{w}", _f(lo)), _macro(f"HEPers{tag}Hi{w}", _f(hi))]
            lines.append(f"more_persuasive={cat:5s} {w}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]")
        k = col.count("B") + col.count("ngang")
        lo, hi = wilson(k, n)
        macros += [_macro(f"HEPersAtLeastCount{w}", k), _macro(f"HEPersAtLeastProp{w}", _f(k / n)),
                   _macro(f"HEPersAtLeastLo{w}", _f(lo)), _macro(f"HEPersAtLeastHi{w}", _f(hi))]
        lines.append(f"rewrite at least as persuasive {w}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]")
    po, kap = kappa([r[2] for r in rated["A"]], [r[2] for r in rated["B"]], PERSUASIVE)
    macros += [_macro("HEPersAgree", _f(po)), _macro("HEPersKappa", _f(kap)), _macro("HEPairs", n)]
    lines.append(f"more_persuasive agreement {po:.3f}, kappa {_f(kap)}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pair_id", "jaccard"] + [f"{q}_{w}" for w in "AB" for q in BINARY + ("more_persuasive",)])
        for i, pair in enumerate(pairs):
            wr.writerow([pair["pair_id"], pair["jaccard"]] + [v for w in "AB" for v in rated[w][i]])
    with open(args.out_tex, "w", encoding="utf-8") as fh:
        fh.write("% generated by scripts/score_p3_human_eval.py — do not edit\n")
        fh.write("\n".join(macros) + "\n")
    print("\n".join(lines))
    print(f"wrote {args.out_csv}\nwrote {args.out_tex}")


if __name__ == "__main__":
    main()
