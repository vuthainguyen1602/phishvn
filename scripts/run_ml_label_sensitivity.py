#!/usr/bin/env python3
"""Tier/noise sensitivity analyses for the four ML papers (P2, P3, P5, P6).

The corpus audit estimated that 12.1% of resolvable sampled positives were legitimate, with
11/12 errors in bronze.  That uncertainty is larger than several reported model contrasts.
This script therefore uses the strongest analysis supported by each experiment:

* P2: recall by provenance tier under both headline protocols and both headline families.
* P3: recall by provenance tier on the existing repeated content-fusion splits.
* P5: Monte Carlo relabelling of 12.1% of bronze positives, then a full stream-policy rerun.
* P6: provenance tier x suffix stratum on the locked forward holdout.

The P5 arm is a sensitivity analysis, not latent-label recovery: the corrected rows are sampled
at random because the audit estimates a rate but does not identify every erroneous corpus row.
It also imports that study's own asset builder and drift runner, which the public code mirror
does not carry: there, run `--papers p2 p3 p6`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

from genfile import write_generated
from paired_eval import wilson
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load as load_url, split_phishing
from run_p6_prospective_ablation import split as split_p6, stratum as suffix_stratum
from train_url_baseline import COMPPHISH

RATE = 0.121
SEEDS = 5


def _pm(values, digits=3):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return "--"
    return f"{a.mean():.{digits}f}$\\pm${a.std(ddof=0):.{digits}f}"


def _save(df, relative):
    path = os.path.join(ROOT, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[+] {path}")


def p2(df):
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    rows = []
    for family in ("CatBoost", "LogReg"):
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            bmask = rng.rand(len(be)) < 0.70
            pool = pd.concat([ph_tr, ph_te]).reset_index(drop=True)
            pmask = rng.rand(len(pool)) < len(ph_tr) / len(pool)
            protocols = {
                "phishing-temporal": (pd.concat([ph_tr, be[bmask]]),
                                       pd.concat([ph_te, be[~bmask]])),
                "random-same-rows": (pd.concat([pool[pmask], be[bmask]]),
                                     pd.concat([pool[~pmask], be[~bmask]])),
            }
            for protocol, (tr, te) in protocols.items():
                model = make_any_model(family, seed)
                model.fit(tr[feats].to_numpy(float), tr.y.to_numpy(int))
                score = model.predict_proba(te[feats].to_numpy(float))[:, 1]
                pred = score >= 0.5
                rows.append({"family": family, "protocol": protocol, "seed": seed,
                             "tier": "overall", "n": len(te),
                             "metric": "F1", "value": f1_score(te.y, pred)})
                for tier in ("bronze", "silver", "gold"):
                    mask = (te.y.to_numpy(int) == 1) & te.tier.astype(str).eq(tier).to_numpy()
                    rows.append({"family": family, "protocol": protocol, "seed": seed,
                                 "tier": tier, "n": int(mask.sum()), "metric": "recall",
                                 "value": float(pred[mask].mean()) if mask.any() else np.nan})
    out = pd.DataFrame(rows)
    _save(out, "data/processed/p2/p2_tier_sensitivity.csv")
    lines = []
    for (fam, proto), g in out.groupby(["family", "protocol"], sort=False):
        vals = {tier: _pm(g[g.tier == tier].value) for tier in ("overall", "bronze", "silver", "gold")}
        lines.append(f"{fam} & {proto} & {vals['overall']} & {vals['bronze']} & "
                     f"{vals['silver']} & {vals['gold']} \\\\")
    tex = """\\begin{table*}[t]
\\centering
\\caption{Tier-stratified sensitivity on the same dated rows as the P2 protocol contrast.
Overall is F1; tier columns are positive-class recall, mean$\\pm$sd over five shared benign
splits. Tier is label provenance and is confounded with source/time; these are diagnostics,
not causal tier effects.}
\\label{tab:tier-sensitivity}
\\small
\\begin{tabular}{llrrrr}
\\toprule Family & Protocol & Overall F1 & Bronze recall & Silver recall & Gold recall \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table*}"""
    write_generated(os.path.join(ROOT, "papers/P2_url_benchmark/sections/tab_tier_sensitivity.tex"), tex)


def p3():
    # Importing here avoids the expensive HTML read for --papers selections that exclude P3.
    from make_p3_assets import _balanced_rows
    from train_content_fusion import evaluate

    configs = ["url", "content", "content+url"]
    rows = _balanced_rows()
    agg, curves = evaluate(rows, configs, "tfidf", "lightweight", False, 20,
                           return_scores=True, return_test_metadata=True)
    records = []
    for config in configs:
        for seed, (y, score, meta) in enumerate(curves[config]):
            pred = score >= 0.5
            records.append({"config": config, "seed": seed, "tier": "overall", "n": len(y),
                            "metric": "F1", "value": f1_score(y, pred)})
            for tier in ("bronze", "silver", "gold", "unassigned"):
                mask = (y == 1) & (meta["tier"] == tier)
                records.append({"config": config, "seed": seed, "tier": tier,
                                "n": int(mask.sum()), "metric": "recall",
                                "value": float(pred[mask].mean()) if mask.any() else np.nan})
    out = pd.DataFrame(records)
    _save(out, "data/processed/p3/p3_tier_sensitivity.csv")
    labels = {"url": "URL only", "content": "Content only", "content+url": "Content + URL"}
    lines = []
    for config, g in out.groupby("config", sort=False):
        vals = {tier: _pm(g[g.tier == tier].value) for tier in
                ("overall", "bronze", "silver", "gold", "unassigned")}
        lines.append(f"{labels[config]} & {vals['overall']} & {vals['bronze']} & "
                     f"{vals['silver']} & {vals['gold']} & {vals['unassigned']} \\\\")
    tex = """\\begin{table}[t]
\\centering
\\caption{Tier-stratified sensitivity on the balanced Vietnamese-content subset. Overall is
F1; tier columns are phishing recall, mean$\\pm$sd over the same twenty 70/30 splits. The
content-quality filters remain applied before splitting.}
\\label{tab:tier-sensitivity}
\\small
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{lrrrrr}
\\toprule Configuration & Overall F1 & Bronze (84) & Silver (2) & Gold (0) & Unassigned (121) \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}%
}
\\end{table}"""
    write_generated(os.path.join(ROOT, "papers/P3_multimodal/sections/tab_tier_sensitivity.tex"), tex)


def p5(reps):
    from make_p5_assets import build_input, DRIFT_CSV, WINDOWS, PERIOD, PSI_TAU, F1_DROP
    from retrain_drift import load, run

    build_input()
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    base, _, _ = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP,
                     include_static_arch=False)
    baseline = {k: float(np.mean(v)) for k, v in base.items()}
    eligible = df.index[(df.y == 1) & df.tier.astype(str).str.lower().eq("bronze")].to_numpy()
    n_flip = int(round(RATE * len(eligible)))
    records = []
    for seed in range(reps):
        perturbed = df.copy()
        chosen = np.random.RandomState(seed).choice(eligible, n_flip, replace=False)
        perturbed.loc[chosen, "y"] = 0
        result, counts, _ = run(perturbed, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP,
                                include_static_arch=False)
        for policy, values in result.items():
            autc = float(np.mean(values))
            records.append({"seed": seed, "policy": policy, "noise_rate": RATE,
                            "n_bronze_positive": len(eligible), "n_relabelled": n_flip,
                            "baseline_autc": baseline[policy], "corrected_autc": autc,
                            "delta_autc": autc - baseline[policy], "retrains": counts[policy]})
        print(f"[i] P5 correction replicate {seed + 1}/{reps}", flush=True)
    out = pd.DataFrame(records)
    _save(out, "data/processed/p5/p5_label_noise_sensitivity.csv")
    labels = {"static": "Static RF", "periodic": "Periodic RF", "drift": "Drift-triggered RF"}
    lines = []
    for policy, g in out.groupby("policy", sort=False):
        lines.append(f"{labels[policy]} & {baseline[policy]:.3f} & {_pm(g.corrected_autc)} & "
                     f"{_pm(g.delta_autc)} & {_pm(g.retrains, 1)} \\\\")
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Label-noise sensitivity: in each of {reps} Monte Carlo reruns, 12.1\\% of
bronze-positive stream rows are randomly relabelled benign before fitting and scoring. This
audit-rate stress test does not identify which individual rows are wrong. AUTC is mean
per-window F1.}}
\\label{{tab:noise-sensitivity}}
\\small
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrr}}
\\toprule Policy & Original AUTC & Corrected AUTC & $\\Delta$AUTC & Retrains \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}%
}
\\end{table}"""
    write_generated(os.path.join(ROOT, "papers/P5_temporal_drift/sections/tab_noise_sensitivity.tex"), tex)


def p6(df):
    feats = [c for c in COMPPHISH if c in df.columns]
    ph_fit, ph_cal, ph_test, be_fit, be_cal, be_test, _, _ = split_p6(df)
    fit = pd.concat([ph_fit, be_fit]).reset_index(drop=True)
    cal = pd.concat([ph_cal, be_cal]).reset_index(drop=True)
    test = pd.concat([ph_test, be_test]).reset_index(drop=True)
    test["suffix_stratum"] = suffix_stratum(test)
    records = []
    for seed in range(SEEDS):
        model = make_any_model("CatBoost", seed)
        model.fit(fit[feats].to_numpy(float), fit.y.to_numpy(int))
        cal_score = model.predict_proba(cal[feats].to_numpy(float))[:, 1]
        cal_be = cal_score[cal.y.to_numpy(int) == 0]
        alpha = float((cal_be >= 0.5).mean())
        threshold = float(np.quantile(cal_be, 1 - alpha))
        pred = model.predict_proba(test[feats].to_numpy(float))[:, 1] >= threshold
        for tier in ("bronze", "silver", "gold"):
            for suffix in ("vn_short", "cc_short"):
                mask = ((test.y.to_numpy(int) == 1) & test.tier.astype(str).eq(tier).to_numpy()
                        & test.suffix_stratum.eq(suffix).to_numpy())
                miss = int((~pred[mask]).sum())
                n = int(mask.sum())
                lo, hi = wilson(miss, n) if n else (np.nan, np.nan)
                records.append({"seed": seed, "tier": tier, "suffix_stratum": suffix,
                                "n": n, "misses": miss, "fnr": miss / n if n else np.nan,
                                "ci_low": lo, "ci_high": hi, "threshold": threshold})
    out = pd.DataFrame(records)
    _save(out, "data/processed/p6/p6_tier_sensitivity.csv")
    # Counts are fixed across model seeds; FNR varies only where CatBoost's scores cross threshold.
    lines = []
    for tier in ("bronze", "silver", "gold"):
        cells = []
        for suffix in ("vn_short", "cc_short"):
            g = out[(out.tier == tier) & (out.suffix_stratum == suffix)]
            n = int(g.n.iloc[0])
            cells.append(f"{_pm(g.fnr)} ({n})" if n else "-- (0)")
        lines.append(f"{tier.title()} & {cells[0]} & {cells[1]} \\\\")
    tex = """\\begin{table}[t]
\\centering
\\caption{Tier sensitivity on the locked forward holdout: false-negative rate (positive count)
within the two-character suffix comparison, mean$\\pm$sd over five CatBoost seeds. Sparse
silver/gold cells are shown rather than pooled; tier and source/time remain confounded.}
\\label{tab:tier-sensitivity}
\\small
\\begin{tabular}{lrr}
\\toprule Tier & Bare \\texttt{.vn} & Other country-code suffix \\\\
\\midrule
""" + "\n".join(lines) + """
\\bottomrule
\\end{tabular}
\\end{table}"""
    write_generated(os.path.join(ROOT, "papers/P6_xai/sections/tab_tier_sensitivity.tex"), tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", choices=["p2", "p3", "p5", "p6"],
                    default=["p2", "p3", "p5", "p6"])
    ap.add_argument("--p5-reps", type=int, default=20)
    args = ap.parse_args()
    url_df = load_url() if set(args.papers) & {"p2", "p6"} else None
    if "p2" in args.papers:
        p2(url_df)
    if "p3" in args.papers:
        p3()
    if "p5" in args.papers:
        p5(args.p5_reps)
    if "p6" in args.papers:
        p6(url_df)


if __name__ == "__main__":
    main()
