#!/usr/bin/env python3
"""
p3_paired_test.py — Is content-only really better than the content+URL fusion, or is it noise?

The P1b ablation's mean +/- std is the wrong statistic for the question: it describes how much a
single run moves between splits, not how uncertain the DIFFERENCE between two configurations is.
Since evaluate() scores every configuration on the SAME split, the scores are paired and the
difference cancels split-to-split difficulty. Reports a paired t-test plus a Wilcoxon signed-rank,
under two conditions — a fixed benign pool (is the gap robust to how we split?) and a resampled one
(robust to WHICH benign pages we balanced against?). A disagreement between them is itself the
finding.

RUN:  python scripts/p3_paired_test.py --splits 20
The flip that prompted it and both conditions in full: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_content_fusion import evaluate, load_pages
from paired_eval import corrected_paired_t

MANIFEST = os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv")
CONFIGS = ["url", "content", "content+url"]
PAIRS = [("content", "content+url"), ("content", "url"), ("content+url", "url")]


def balanced(rows, seed):
    rng = random.Random(seed)
    ph = [r for r in rows if r["y"] == 1]
    be = [r for r in rows if r["y"] == 0]
    k = min(len(ph), len(be))
    return rng.sample(ph, k) + rng.sample(be, k), k


def paired_stats(d: np.ndarray) -> dict:
    """d = per-split (a - b). The headline test is the CORRECTED resampled t (paired_eval): the
    splits overlap, so the ordinary paired t understates the variance of the mean difference.
    Wilcoxon is kept as a distribution-free cross-check but is reported as INDICATIVE ONLY — it
    assumes independent observations too, and no standard correction exists for it, so it stays
    anti-conservative here and must not carry a verdict on its own."""
    n = len(d)
    mean, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    s = corrected_paired_t(d)
    se = s["se"]
    out = {"n": n, "mean": mean, "sd": sd, "se": se,
           "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
           "wins": int(np.sum(d > 0)), "losses": int(np.sum(d < 0)),
           "t_p": s["p"], "t_p_naive": s["p_naive"]}
    try:
        from scipy import stats
        out["w_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except Exception:
        out["w_p"] = float("nan")
    return out


def run(rows_all, splits, resample_pool, metric="F1", text_enc="tfidf"):
    """resample_pool=False -> one fixed balanced pool, `splits` train/test splits over it.
    resample_pool=True  -> a fresh pool AND a fresh split per iteration (1 split each)."""
    per = {c: [] for c in CONFIGS}
    if not resample_pool:
        rows, k = balanced(rows_all, 42)
        agg = evaluate(rows, CONFIGS, text_enc, "lightweight", False, splits)
        for c in CONFIGS:
            per[c] = list(agg[c][metric])
    else:
        k = 0
        for i in range(splits):
            rows, k = balanced(rows_all, 1000 + i)
            agg = evaluate(rows, CONFIGS, text_enc, "lightweight", False, 1)
            for c in CONFIGS:
                per[c].append(agg[c][metric][0])
    return {c: np.asarray(v, dtype=float) for c, v in per.items()}, k


def report(title, per, k, metric):
    print(f"\n=== {title}  (n={len(next(iter(per.values())))} iterations, {k} pages/class) ===")
    for c in CONFIGS:
        v = per[c]
        print(f"  {c:14s} {metric} {v.mean():.4f} +/- {v.std(ddof=1):.4f}")
    for a, b in PAIRS:
        s = paired_stats(per[a] - per[b])
        # The verdict rides on the corrected t alone. It used to ride on max(t, wilcoxon), which
        # let the anti-conservative Wilcoxon rescue a difference the t could not support.
        verdict = ("separated" if s["t_p"] < 0.05 else "NOT separated")
        print(f"  {a} - {b}: {s['mean']:+.4f} (95% CI {s['ci_lo']:+.4f}..{s['ci_hi']:+.4f}) "
              f"| {s['wins']}W-{s['losses']}L | corrected t p={s['t_p']:.4f} "
              f"(naive {s['t_p_naive']:.4f}) | wilcoxon p={s['w_p']:.4f} [indicative] "
              f"-> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--metric", default="F1", choices=["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"])
    ap.add_argument("--skip-resampled", action="store_true", help="condition A only (faster)")
    from train_content_fusion import HF_ENCODERS
    ap.add_argument("--text-encoder", default="tfidf", choices=["tfidf"] + sorted(HF_ENCODERS))
    args = ap.parse_args()

    rows_all = load_pages(os.path.normpath(MANIFEST))
    print(f"manifest: {len(rows_all)} usable pages")
    per, k = run(rows_all, args.splits, resample_pool=False, metric=args.metric,
                 text_enc=args.text_encoder)
    report(f"A. fixed pool (seed 42), {args.splits} splits [{args.text_encoder}]",
           per, k, args.metric)
    if not args.skip_resampled:
        per2, k2 = run(rows_all, args.splits, resample_pool=True, metric=args.metric,
                       text_enc=args.text_encoder)
        report(f"B. resampled pool + split, {args.splits} draws [{args.text_encoder}]",
               per2, k2, args.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
