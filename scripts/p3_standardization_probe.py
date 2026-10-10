#!/usr/bin/env python3
"""
p3_standardization_probe.py — per-block standardisation and late-fusion probe (revision R3.3).

The reviewer's hypothesis: the three-channel linear stack falling below URL-only (0.811 vs
0.836 F1) while recovering under scale-invariant XGBoost (0.918) is a feature-scaling artefact
— raw URL counts (url_len up to hundreds) sit beside unit-scale TF-IDF/embedding values in one
L2-regularised logistic regression. This probe answers it on the same subset, splits, encoders
and heads as the published run (POST-HOC, added at revision):

  raw    — the published pipeline, re-run as-is (must reproduce the published numbers);
  std    — identical, except every DENSE block (URL 21-dim, JS, transformer embedding) is
           per-block standardised by a StandardScaler fit on the split's training portion
           (sparse TF-IDF is left untouched);
  late   — stacked-probability late fusion: one logistic model per channel, out-of-fold
           training probabilities (5-fold within the training portion), a logistic meta-head
           over the per-channel probabilities.

RUN:  python scripts/p3_standardization_probe.py [--encoders tfidf,xlm-r]
OUT:  data/processed/p3/p3_standardization_probe.csv
      papers/P3_multimodal/sections/gen_standardization.tex
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "core", "lib"))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

import train_content_fusion as tcf
from paired_eval import corrected_paired_t
from genfile import write_generated, write_results

OUT_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_standardization_probe.csv")
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
SEEDS = 20
CONFIGS = ("url", "content", "content+url", "content+url+js")


def fpr_at_recall(y, sc, target=0.90):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, sc)
    ok = tpr >= target
    return float(fpr[ok][0]) if ok.any() else 1.0


def balanced_rows(encoder):
    rows = tcf.load_pages(os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv"))
    rng = random.Random(42)
    ph = [r for r in rows if r["y"] == 1]
    be = [r for r in rows if r["y"] == 0]
    k = min(len(ph), len(be))
    rows = rng.sample(ph, k) + rng.sample(be, k)
    tcf.precompute_js(rows, "lightweight")
    return rows


def head():
    return LogisticRegression(max_iter=4000, class_weight="balanced")


def _fmt(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$"


def early_fit(txt, blocks_tr, blocks_te, kind, ytr, standardise):
    if standardise:
        std = []
        for btr, bte in zip(blocks_tr, blocks_te):
            sc = StandardScaler().fit(btr)
            std.append((sc.transform(btr), sc.transform(bte)))
        blocks_tr = [a for a, _ in std]
        blocks_te = [b for _, b in std]
        if kind == "dense" and txt[0] is not None:
            sc = StandardScaler().fit(txt[0])
            txt = (sc.transform(txt[0]), sc.transform(txt[1]), kind)
    Xtr = tcf.stack(txt[0], blocks_tr, kind)
    Xte = tcf.stack(txt[1], blocks_te, kind)
    clf = head().fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def late_fit(channels, ytr):
    """channels: list of (Xtr, Xte) single-channel matrices. Returns test meta-probability."""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    n = len(ytr)
    meta_tr = np.zeros((n, len(channels)))
    meta_te = np.zeros((channels[0][1].shape[0], len(channels)))
    for j, (Xtr, Xte) in enumerate(channels):
        oof = np.zeros(n)
        for tr_i, va_i in kf.split(np.zeros(n), ytr):
            m = head().fit(Xtr[tr_i], ytr[tr_i])
            oof[va_i] = m.predict_proba(Xtr[va_i])[:, 1]
        meta_tr[:, j] = oof
        meta_te[:, j] = head().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    meta = head().fit(meta_tr, ytr)
    return meta.predict_proba(meta_te)[:, 1]


def run(encoder):
    rows = balanced_rows(encoder)
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))
    res = {}  # (variant, cfg) -> {"F1": [...], "FPR": [...]}

    def rec(variant, cfg, yte, sc):
        d = res.setdefault((variant, cfg), {"F1": [], "FPR": []})
        d["F1"].append(f1_score(yte, (sc >= 0.5).astype(int), zero_division=0))
        d["FPR"].append(fpr_at_recall(yte, sc))

    for s in range(SEEDS):
        tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=s)
        trr = [rows[i] for i in tr]
        ter = [rows[i] for i in te]
        ytr, yte = y[tr], y[te]
        tmat_tr, tmat_te, kind = tcf.text_matrix(
            [r["text"] for r in trr], [r["text"] for r in ter], encoder)
        js_tr, js_te = tcf.js_matrix(trr, ter)
        url_tr = np.array([r["url"] for r in trr])
        url_te = np.array([r["url"] for r in ter])

        def parts(cfg):
            txt = (tmat_tr, tmat_te, kind) if "content" in cfg else (None, None, kind)
            btr, bte = [], []
            if "url" in cfg:
                btr.append(url_tr); bte.append(url_te)
            if "js" in cfg:
                btr.append(js_tr); bte.append(js_te)
            return txt, btr, bte

        for cfg in CONFIGS:
            txt, btr, bte = parts(cfg)
            rec("raw", cfg, yte, early_fit(txt, btr, bte, kind, ytr, standardise=False))
            rec("std", cfg, yte, early_fit(txt, btr, bte, kind, ytr, standardise=True))
            if "+" in cfg:  # late fusion is defined for multi-channel configs only
                chans = []
                if "content" in cfg:
                    if kind == "sparse":
                        chans.append((tmat_tr.tocsr(), tmat_te.tocsr()))
                    else:
                        chans.append((tmat_tr, tmat_te))
                if "url" in cfg:
                    chans.append((url_tr, url_te))
                if "js" in cfg:
                    chans.append((js_tr, js_te))
                rec("late", cfg, yte, late_fit(chans, ytr))
        print(f"[i] {encoder} split {s + 1}/{SEEDS}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="tfidf,xlm-r")
    args = ap.parse_args()
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    all_rows, frag_parts, any_std_gain = [], [], False
    results = {}
    split_rows, headline = [], {}
    for enc in encoders:
        res = run(enc)
        for (variant, cfg), d in res.items():
            all_rows.append({
                "encoder": enc, "variant": variant, "config": cfg,
                "f1_mean": np.mean(d["F1"]), "f1_std": np.std(d["F1"]),
                "fpr90_mean": np.mean(d["FPR"]), "fpr90_std": np.std(d["FPR"]),
            })
        # The reviewer's question, answered with the paper's own paired discipline:
        # does standardisation (or late fusion) move the three-channel stack, and does the
        # stack still trail URL-only once scales are equalised?
        stk = "content+url+js"
        t_std = corrected_paired_t(np.array(res[("std", stk)]["F1"]) - np.array(res[("raw", stk)]["F1"]))
        t_late = corrected_paired_t(np.array(res[("late", stk)]["F1"]) - np.array(res[("raw", stk)]["F1"]))
        t_vs_url = corrected_paired_t(np.array(res[("std", stk)]["F1"]) - np.array(res[("raw", "url")]["F1"]))
        if (t_std["mean"] > 0 and t_std["p"] < 0.05) or (t_late["mean"] > 0 and t_late["p"] < 0.05):
            any_std_gain = True
        # The paper's headline contrast, (content+url) - (content), re-asked under each variant:
        # if equalising scales or late fusion resolves a URL contribution the raw stack could not,
        # the main claim changes and the paper must say so.
        head_c = {}
        for variant in ("raw", "std", "late"):
            a = np.array(res[(variant, "content+url")]["F1"])
            b = np.array(res[("raw" if variant == "late" else variant, "content")]["F1"])
            head_c[variant] = corrected_paired_t(a - b)
            for metric in ("F1", "FPR"):
                for i, v in enumerate(res[(variant, "content+url")][metric]):
                    split_rows.append({"encoder": enc, "variant": variant, "config": "content+url",
                                       "metric": metric, "split": i, "value": v})
        for i, v in enumerate(res[("raw", "content")]["F1"]):
            split_rows.append({"encoder": enc, "variant": "raw", "config": "content",
                               "metric": "F1", "split": i, "value": v})
        headline[enc] = head_c
        results[enc] = {"per_split": {f"{v}|{c}": d for (v, c), d in res.items()},
                        "std_vs_raw_stack": t_std, "late_vs_raw_stack": t_late,
                        "std_stack_vs_url": t_vs_url, "headline": head_c}
        lab = {"tfidf": "char-$n$-gram TF-IDF", "xlm-r": "XLM-R"}.get(enc, enc)
        frag_parts.append(
            f"Under {lab}, per-block standardisation lifts the raw three-channel stack from "
            f"${np.mean(res[('raw', stk)]['F1']):.3f}$ to ${np.mean(res[('std', stk)]['F1']):.3f}$ F1 "
            f"(paired ${t_std['mean']:+.3f}$, corrected {_fmt(t_std['p'])}), and stacked late fusion "
            f"to ${np.mean(res[('late', stk)]['F1']):.3f}$ (${t_late['mean']:+.3f}$, {_fmt(t_late['p'])}). "
            f"The standardised stack sits ${t_vs_url['mean']:+.3f}$ from URL-only "
            f"(corrected {_fmt(t_vs_url['p'])}). The headline contrast, content$+$URL minus content, is "
            f"${head_c['raw']['mean']:+.3f}$ raw ({_fmt(head_c['raw']['p'])}), "
            f"${head_c['std']['mean']:+.3f}$ standardised ({_fmt(head_c['std']['p'])}) and "
            f"${head_c['late']['mean']:+.3f}$ under late fusion ({_fmt(head_c['late']['p'])}).")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"[+] {OUT_CSV}")
    split_csv = OUT_CSV.replace(".csv", "_splits.csv")
    with open(split_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
        w.writeheader()
        w.writerows(split_rows)
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results", "standardization.json"),
                  {"encoders": results, "any_std_gain": any_std_gain})
    for enc, hc in headline.items():
        print(f"[i] HEADLINE {enc}: " + " ".join(
            f"{v}={hc[v]['mean']:+.4f}(p={hc[v]['p']:.4f})" for v in ("raw", "std", "late")))

    frag = (
        "Is the linear-head dilution a feature-scaling artefact? The URL counts enter the "
        "logistic head raw, at scales far from the unit-scale content features. A post-hoc "
        "probe (same subset, splits and heads) "
        "re-runs the stack with every dense block per-block standardised on the training "
        "portion, and with a stacked-probability late fusion in place of concatenation. "
        "Its six contrasts are post hoc, outside any pre-specified family, and carry corrected "
        "$p$-values without Benjamini--Hochberg adjustment. "
        + " ".join(frag_parts) + " "
        + ("The scaling hypothesis is therefore borne out at least in part: equalising the "
           "block scales moves the linear stack, and the head-capacity reading of "
           "Section~\\ref{ssec:indataset} must be read alongside a scaling account"
           if any_std_gain else
           "Standardisation is therefore not what the missing fusion effect was hiding behind "
           "on this subset, and the head-capacity reading of Section~\\ref{ssec:indataset} "
           "stands")
        + (". More consequentially, the headline contrast itself is fusion-architecture "
           "dependent: under stacked late fusion the URL channel's incremental contribution on "
           "top of content \\emph{is} resolved for "
           + ", ".join({"tfidf": "char-$n$-gram TF-IDF", "xlm-r": "XLM-R"}.get(e, e)
                       for e, hc in headline.items()
                       if hc["late"]["mean"] > 0 and hc["late"]["p"] < 0.05)
           + ", so the directional asymmetry of Table~\\ref{tab:encsweep} is a property of "
           "early concatenation into a linear head, not of the channels alone"
           if any(hc["late"]["mean"] > 0 and hc["late"]["p"] < 0.05 for hc in headline.values())
           else ". The headline contrast, however, is resolved under none of the three fusion "
                "variants for either encoder, so the paper's central finding does not depend on "
                "how the blocks are scaled or combined")
    )
    write_generated(os.path.join(SEC, "gen_standardization.tex"), frag + ".\n")


if __name__ == "__main__":
    main()
