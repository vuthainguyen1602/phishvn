#!/usr/bin/env python3
"""P6 confirmatory forward holdout and refit feature ablation.

This experiment is deliberately separate from the exploratory 70/30 analysis.  It fits on the
oldest 60% of dated phishing, calibrates on the next 20%, and scores the newest 20% once.  A test
registrable domain seen in either earlier window is removed.  Benign collection times are mostly
missing, so benign rows are assigned 60/20/20 by a stable registrable-domain hash; this limitation is emitted in
the protocol CSV and stated in the manuscript.

The ablations REFIT the detector.  They are therefore stronger than pinning one SHAP coordinate,
although the three-feature arm is only a length-family sensitivity analysis: url_len and dom_len
encode more than public-suffix length.

Outputs:
  data/processed/p6/p6_prospective_ablation.csv
  data/processed/p6/p6_prospective_protocol.csv

Run:
  python scripts/run_p6_prospective_ablation.py --seeds 5
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import COMPPHISH
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load

OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_prospective_ablation.csv")
PROTO = os.path.join(ROOT, "data", "processed", "p6", "p6_prospective_protocol.csv")
ARMS = {
    "full": (),
    "minus_tld_len": ("tld_len",),
    "minus_length_family": ("tld_len", "url_len", "dom_len"),
}
STRATA = ("vn_short", "cc_short", "vn_compound", "long")


def stable_fold(value: str) -> int:
    """A checked-in, implementation-independent 0..99 partition key."""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def stratum(frame: pd.DataFrame) -> pd.Series:
    tld = frame["tld"].astype(str).str.lower().str.lstrip(".")
    vn = tld.str.endswith("vn")
    short = tld.str.len().eq(2)
    return pd.Series(np.where(short, np.where(vn, "vn_short", "cc_short"),
                              np.where(vn, "vn_compound", "long")), index=frame.index)


def split(df: pd.DataFrame):
    ph = df[(df.y == 1) & df.date.notna()].sort_values(["date", "url"]).reset_index(drop=True)
    n = len(ph)
    k_fit, k_cal = int(0.60 * n), int(0.80 * n)
    # Pick calendar boundaries at the target quantiles and keep a boundary date wholly on the
    # earlier side. Positional slicing would let one day occur in two windows when dates tie.
    fit_end, cal_end = ph.iloc[k_fit - 1].date, ph.iloc[k_cal - 1].date
    ph_fit = ph[ph.date <= fit_end]
    ph_cal = ph[(ph.date > fit_end) & (ph.date <= cal_end)]
    ph_test_raw = ph[ph.date > cal_end]
    prior_domains = set(pd.concat([ph_fit.rdom, ph_cal.rdom]).astype(str))
    repeated = ph_test_raw.rdom.astype(str).isin(prior_domains)
    ph_test = ph_test_raw[~repeated]

    be = df[df.y == 0].copy()
    # Domain, rather than row order, URL, or Python's salted hash, makes the partition reproducible
    # and prevents a benign registrable domain from recurring across windows.
    fold = be["rdom"].astype(str).map(stable_fold)
    be_fit, be_cal, be_test = be[fold < 60], be[(fold >= 60) & (fold < 80)], be[fold >= 80]
    return ph_fit, ph_cal, ph_test, be_fit, be_cal, be_test, int(repeated.sum()), n


def safe_auc(y, score) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def metrics(y, score, threshold) -> dict[str, float]:
    y, score = np.asarray(y, int), np.asarray(score, float)
    pred = score >= threshold
    return {
        "n": len(y), "n_phish": int(y.sum()), "n_benign": int((y == 0).sum()),
        "fnr": float(1 - pred[y == 1].mean()) if (y == 1).any() else np.nan,
        "fpr": float(pred[y == 0].mean()) if (y == 0).any() else np.nan,
        "roc_auc": safe_auc(y, score),
        "pr_auc": float(average_precision_score(y, score)) if (y == 1).any() else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = load()
    all_features = [c for c in COMPPHISH if c in df.columns]
    ph_fit, ph_cal, ph_test, be_fit, be_cal, be_test, n_repeat, n_dated = split(df)
    fit = pd.concat([ph_fit, be_fit]).reset_index(drop=True)
    cal = pd.concat([ph_cal, be_cal]).reset_index(drop=True)
    test = pd.concat([ph_test, be_test]).reset_index(drop=True)
    test["stratum"] = stratum(test)

    protocol = pd.DataFrame([{
        "family": args.family,
        "phishing_fit_fraction": 0.60,
        "phishing_calibration_fraction": 0.20,
        "phishing_holdout_fraction": 0.20,
        "fit_start": ph_fit.date.min().date(), "fit_end": ph_fit.date.max().date(),
        "calibration_start": ph_cal.date.min().date(),
        "calibration_end": ph_cal.date.max().date(),
        "holdout_start": ph_test.date.min().date(), "holdout_end": ph_test.date.max().date(),
        "n_dated_phishing": n_dated, "n_fit_phishing": len(ph_fit),
        "n_calibration_phishing": len(ph_cal), "n_holdout_phishing": len(ph_test),
        "n_holdout_repeated_domains_removed": n_repeat,
        "n_fit_benign": len(be_fit), "n_calibration_benign": len(be_cal),
        "n_holdout_benign": len(be_test),
        "benign_partition": "stable SHA-256 registrable-domain hash; benign timestamps incomplete",
        "status": "locked forward (prospective-style), not a newly accrued prospective cohort",
    }])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    protocol.to_csv(PROTO, index=False)

    rows = []
    for seed in range(args.seeds):
        # Fix the operating budget before looking at any ablation's holdout scores: it is the
        # full model's calibration-benign FPR at the historical 0.5 threshold.
        full = make_any_model(args.family, seed)
        full.fit(fit[all_features].to_numpy(float), fit.y.to_numpy(int))
        full_cal = full.predict_proba(cal[all_features].to_numpy(float))[:, 1]
        alpha = float((full_cal[cal.y.to_numpy(int) == 0] >= 0.5).mean())
        q = 1 - alpha

        for arm, removed in ARMS.items():
            features = [f for f in all_features if f not in removed]
            model = full if arm == "full" else make_any_model(args.family, seed)
            if arm != "full":
                model.fit(fit[features].to_numpy(float), fit.y.to_numpy(int))
            cal_score = (full_cal if arm == "full" else
                         model.predict_proba(cal[features].to_numpy(float))[:, 1])
            cal_be_score = cal_score[cal.y.to_numpy(int) == 0]
            threshold = float(np.quantile(cal_be_score, q))
            score = model.predict_proba(test[features].to_numpy(float))[:, 1]

            for group in ("overall",) + STRATA + ("short_all",):
                if group == "overall":
                    mask = np.ones(len(test), dtype=bool)
                elif group == "short_all":
                    mask = test.stratum.isin(["vn_short", "cc_short"]).to_numpy()
                else:
                    mask = test.stratum.eq(group).to_numpy()
                result = metrics(test.y.to_numpy(int)[mask], score[mask], threshold)
                rows.append({
                    "seed": seed, "family": args.family, "arm": arm,
                    "removed_features": ";".join(removed) or "none",
                    "n_features": len(features), "stratum": group,
                    "alpha_calibration": alpha, "threshold": threshold, **result,
                })
        print(f"[i] seed={seed} alpha={alpha:.4f} threshold arms complete", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.6f")
    show = (out.groupby(["arm", "stratum"])[["fnr", "fpr", "roc_auc", "mcc"]]
               .mean().round(4))
    print(protocol.to_string(index=False))
    print(show.to_string())
    print(f"[+] {args.out}\n[+] {PROTO}")


if __name__ == "__main__":
    main()
