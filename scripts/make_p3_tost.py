#!/usr/bin/env python3
"""
make_p3_tost.py — TOST equivalence bounds for the unresolved sweep direction (revision R3.2).

The reviewers are right that a difference in significance is not a significant difference: the
null verdicts of the URL-into-content direction do not establish equivalence. This emits the
equivalence reading explicitly: for each encoder, a two-one-sided corrected resampled t-test
(the same Nadeau–Bengio variance the paper uses everywhere; each one-sided test at alpha=0.05)
of the paired (content+url) − (content) F1 difference against symmetric margins.

POST-HOC: this analysis was added at revision and the margins are declared, not derived —
delta=0.02 (a two-point F1 margin, below every resolved effect in the other direction) and
delta=0.05 (chosen when the smallest resolved effect was +0.056 under the raw-scale head; under
the standardised head of revision 2 it is +0.038, so the margin is now only a convention).

Reads the sweep cache (data/processed/p3/p3_encoder_sweep.csv) written by make_p3_assets.py;
never re-runs models. Emits sections/gen_tost.tex.

RUN: python scripts/make_p3_tost.py
"""
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "core", "lib"))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Revision 2: the sweep cache of the standardised linear head (make_p3_assets.STANDARDISE).
CACHE = os.path.join(ROOT, "data", "processed", "p3", "p3_encoder_sweep_std.csv")
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
TEST_FRAC = 0.30
MARGINS = (0.02, 0.05)
ENC_LABEL = {"tfidf": "char-$n$-gram TF-IDF", "phobert": "PhoBERT",
             "phobert-v2": "PhoBERT-v2", "visobert": "ViSoBERT", "xlm-r": "XLM-R"}
ENC_ORDER = ("tfidf", "phobert", "phobert-v2", "visobert", "xlm-r")


def load_cache():
    by = {}
    with open(CACHE, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["encoder"], r["config"], r["metric"]), {})[int(r["split"])] = float(r["value"])
    out = {}
    for (enc, cfg, met), d in by.items():
        out[(enc, cfg, met)] = np.array([d[i] for i in sorted(d)])
    return out


def tost(d, margin):
    """Two one-sided corrected resampled t-tests of H0: |mean| >= margin."""
    k = len(d)
    mean = float(np.mean(d))
    var = float(np.var(d, ddof=1))
    se = math.sqrt((1.0 / k + TEST_FRAC / (1.0 - TEST_FRAC)) * var)
    t_lo = (mean + margin) / se   # H0: mean <= -margin
    t_hi = (mean - margin) / se   # H0: mean >= +margin
    p = max(float(stats.t.sf(t_lo, k - 1)), float(stats.t.cdf(t_hi, k - 1)))
    return mean, se, p


def main():
    cache = load_cache()
    rows, eq_counts = [], {m: 0 for m in MARGINS}
    for enc in ENC_ORDER:
        a = cache.get((enc, "content+url", "F1"))
        b = cache.get((enc, "content", "F1"))
        if a is None or b is None:
            continue
        d = a - b
        cells = []
        for m in MARGINS:
            mean, se, p = tost(d, m)
            eq = p < 0.05
            eq_counts[m] += int(eq)
            cells.append((m, p, eq))
        rows.append((enc, float(np.mean(d)), cells))
    if not rows:
        raise SystemExit("no cache rows — run make_p3_assets.py first")

    frag_rows = []
    for enc, mean, cells in rows:
        frag_rows.append(
            f"{ENC_LABEL.get(enc, enc)} ${mean:+.3f}$ ("
            + ", ".join(f"${p:.3f}" + ("^{*}" if eq else "") + "$" for m, p, eq in cells) + ")")
    n = len(rows)
    frag = (
        "Because a difference in significance is not a significant difference, the unresolved "
        "direction is also tested for equivalence. This analysis is exploratory: it was added post hoc, "
        "and its margins are conventions rather than derived from an operational cost "
        "(two and five F1 points), so no conclusion of the paper rests on it. "
        "The test is a two-one-sided corrected resampled $t$ (TOST) on the paired F1 difference "
        "(content$+$URL) $-$ (content), each side at $\\alpha=0.05$ with the same Nadeau--Bengio "
        "variance as every other test here. "
        f"At $\\delta=0.02$ equivalence is established for {eq_counts[0.02]} of {n} encoders, and "
        f"for {eq_counts[0.05]} of {n} at $\\delta=0.05$. "
        "Per encoder (mean difference; $p_{\\mathrm{TOST}}$ at $\\delta=0.02$ and $0.05$, "
        "$^{*}$ for $p<0.05$): " + "; ".join(frag_rows) + ". "
        "The URL channel's incremental F1 contribution on top of content is therefore neither "
        "resolved as positive nor, for every encoder, bounded inside $\\pm 0.02$. Where the TOST "
        "rejects, the data bound it below that margin, a statement that is only as meaningful as the "
        "margin"
    )
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "core", "lib"))
    from genfile import write_generated, write_results
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results", "tost.json"), {
        enc: {"mean_diff": mean, **{f"p_tost_{m}": p for m, p, _ in cells}} for enc, mean, cells in rows})
    write_generated(os.path.join(SEC, "gen_tost.tex"), frag + ".\n",
                    f"(eq at 0.02: {eq_counts[0.02]}/{n}, at 0.05: {eq_counts[0.05]}/{n})")
    for enc, mean, cells in rows:
        print(f"[i] {enc:11s} mean={mean:+.4f} " +
              " ".join(f"d={m:.2f}:p={p:.3f}{'*' if eq else ''}" for m, p, eq in cells))


if __name__ == "__main__":
    main()
