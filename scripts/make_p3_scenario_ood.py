#!/usr/bin/env python3
"""P3 leave-one-scenario-out (LOSO) evaluation.

The unit held out is one complete social-engineering scenario.  Both labels for that
scenario are test-only; the TF--IDF vocabulary and classifier are fitted on the other
scenarios.  A size- and class-matched stratified random-split comparator is repeated over
20 seeds for orientation.  It is an in-distribution comparator, not a second estimate of
LOSO uncertainty.

Writes
  data/processed/p3/p3_leave_one_scenario_out.csv
  data/processed/p3/p3_leave_one_scenario_out_summary.csv
  papers/P3_multimodal/sections/tab_scenario_ood.tex
  papers/P3_multimodal/sections/gen_scenario_ood_verdict.tex
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs

    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

from genfile import write_generated
from make_p3_paraphrase_assets import SEEDS, strip_url, vec

SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
OUT_DIR = os.path.join(ROOT, "data", "processed", "p3")
EXPECTED_SCENARIOS = (
    "bank",
    "delivery",
    "ecommerce",
    "gov",
    "social",
    "tax",
    "telecom",
)


def load_corpus() -> pd.DataFrame:
    frames = []
    for channel in ("sms", "email"):
        path = os.path.join(ROOT, "data", "processed", f"dataset_{channel}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No dataset_sms/email.csv; build the P3 author corpus first.")

    df = pd.concat(frames, ignore_index=True)
    required = {"id", "scenario", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"P3 corpus is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["scenario"] = df["scenario"].fillna("").astype(str).str.strip()
    df["clean_text"] = df["text"].fillna("").astype(str).map(strip_url)
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[(df["scenario"] != "") & (df["clean_text"].str.len() > 0)].reset_index(drop=True)

    scenarios = tuple(sorted(df["scenario"].unique()))
    if scenarios != EXPECTED_SCENARIOS:
        raise SystemExit(
            f"LOSO scenario set drifted: expected {EXPECTED_SCENARIOS}, observed {scenarios}"
        )
    counts = df.groupby(["scenario", "y"]).size().unstack(fill_value=0)
    bad = counts[(counts.get(0, 0) == 0) | (counts.get(1, 0) == 0)]
    if not bad.empty:
        raise SystemExit(f"Every LOSO test fold must contain both labels:\n{bad}")
    if df["id"].duplicated().any():
        raise SystemExit("P3 author-corpus IDs are not unique; LOSO units are ambiguous.")
    return df


def scenario_folds(df: pd.DataFrame):
    """Yield complete, disjoint scenario folds and fail closed on leakage."""
    all_idx = np.arange(len(df))
    for scenario in EXPECTED_SCENARIOS:
        test = all_idx[df["scenario"].to_numpy() == scenario]
        train = all_idx[df["scenario"].to_numpy() != scenario]
        train_scenarios = set(df.iloc[train]["scenario"])
        test_scenarios = set(df.iloc[test]["scenario"])
        if train_scenarios & test_scenarios or test_scenarios != {scenario}:
            raise AssertionError(f"scenario leakage in fold {scenario}")
        if set(df.iloc[train]["id"]) & set(df.iloc[test]["id"]):
            raise AssertionError(f"ID leakage in fold {scenario}")
        yield scenario, train, test


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    ph = y_true == 1
    be = y_true == 0
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "phishing_miss": float(np.mean(pred[ph] == 0)),
        "benign_fpr": float(np.mean(pred[be] == 1)),
    }


def _fit_predict(text: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray):
    vectorizer = vec()
    x_train = vectorizer.fit_transform(text[train])
    classifier = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    classifier.fit(x_train, y[train])
    return classifier.predict(vectorizer.transform(text[test]))


def _matched_random_split(y: np.ndarray, n_benign: int, n_phishing: int, seed: int):
    rng = np.random.default_rng(seed)
    benign = np.flatnonzero(y == 0)
    phishing = np.flatnonzero(y == 1)
    test = np.concatenate(
        [rng.choice(benign, n_benign, replace=False), rng.choice(phishing, n_phishing, replace=False)]
    )
    test = np.sort(test)
    train = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
    return train, test


def _cluster_bootstrap(values: np.ndarray, seed: int = 260821043, draws: int = 20000):
    """Percentile interval over the seven named scenarios, reported descriptively."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    """Descriptive exact paired randomisation over seven scenario-level differences."""
    observed = abs(float(np.mean(differences)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        null.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def evaluate(df: pd.DataFrame):
    text = df["clean_text"].to_numpy()
    y = df["y"].to_numpy(dtype=int)
    rows = []
    oof_true, oof_pred = [], []

    for fold_no, (scenario, train, test) in enumerate(scenario_folds(df)):
        pred = _fit_predict(text, y, train, test)
        metric = _metrics(y[test], pred)
        oof_true.extend(y[test])
        oof_pred.extend(pred)

        n_benign = int(np.sum(y[test] == 0))
        n_phishing = int(np.sum(y[test] == 1))
        id_scores = []
        for seed in range(SEEDS):
            id_train, id_test = _matched_random_split(
                y, n_benign=n_benign, n_phishing=n_phishing, seed=1000 * fold_no + seed
            )
            id_pred = _fit_predict(text, y, id_train, id_test)
            id_scores.append(_metrics(y[id_test], id_pred)["macro_f1"])

        rows.append(
            {
                "scenario": scenario,
                "n_benign": n_benign,
                "n_phishing": n_phishing,
                **metric,
                "matched_id_macro_f1_mean": float(np.mean(id_scores)),
                "matched_id_macro_f1_std": float(np.std(id_scores, ddof=1)),
                "macro_f1_gap": metric["macro_f1"] - float(np.mean(id_scores)),
                "train_scenarios": ";".join(sorted(set(df.iloc[train]["scenario"]))),
                "test_scenarios": scenario,
            }
        )

    per = pd.DataFrame(rows)
    pooled = _metrics(np.asarray(oof_true), np.asarray(oof_pred))
    ci_lo, ci_hi = _cluster_bootstrap(per["macro_f1"].to_numpy())
    differences = per["macro_f1_gap"].to_numpy()
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_scenarios": len(per),
                "scenario_balanced_macro_f1": per["macro_f1"].mean(),
                "scenario_balanced_macro_f1_std": per["macro_f1"].std(ddof=1),
                "scenario_bootstrap_ci_low": ci_lo,
                "scenario_bootstrap_ci_high": ci_hi,
                "pooled_oof_macro_f1": pooled["macro_f1"],
                "scenario_balanced_phishing_miss": per["phishing_miss"].mean(),
                "scenario_balanced_benign_fpr": per["benign_fpr"].mean(),
                "matched_id_macro_f1": per["matched_id_macro_f1_mean"].mean(),
                "macro_f1_gap": differences.mean(),
                "exact_sign_flip_p": _exact_sign_flip_p(differences),
            }
        ]
    )
    return per, summary


def _table(per: pd.DataFrame, summary: pd.DataFrame) -> str:
    s = summary.iloc[0]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Leave-one-scenario-out evaluation of the character-$n$-gram text detector. "
        r"Each row holds out every benign and phishing message in the named scenario; the "
        r"vocabulary and classifier are fitted on the other six scenarios. The matched ID "
        r"column is the mean of 20 class- and size-matched stratified random splits.}",
        r"\label{tab:scenarioood}",
        r"\small",
        r"\centering",
        r"\begin{tabular}{lrrcccc}",
        r"\toprule",
        r"Held-out scenario & Benign & Phish. & LOSO Macro-F1 & ID Macro-F1 & Miss (\%) & FPR (\%) \\",
        r"\midrule",
    ]
    for row in per.itertuples(index=False):
        label = row.scenario.capitalize()
        lines.append(
            f"{label} & {row.n_benign} & {row.n_phishing} & {row.macro_f1:.3f} & "
            f"{row.matched_id_macro_f1_mean:.3f} & {100 * row.phishing_miss:.1f} & "
            f"{100 * row.benign_fpr:.1f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Scenario-balanced mean & {int(per.n_benign.sum())} & {int(per.n_phishing.sum())} & "
        f"{s.scenario_balanced_macro_f1:.3f} & {s.matched_id_macro_f1:.3f} & "
        f"{100 * s.scenario_balanced_phishing_miss:.1f} & "
        f"{100 * s.scenario_balanced_benign_fpr:.1f} \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def _verdict(summary: pd.DataFrame) -> str:
    s = summary.iloc[0]
    return (
        f"Across the seven held-out scenarios, scenario-balanced Macro-F1 is "
        f"${s.scenario_balanced_macro_f1:.3f}$ (scenario-bootstrap 95\\% interval "
        f"${s.scenario_bootstrap_ci_low:.3f}$--${s.scenario_bootstrap_ci_high:.3f}$), "
        f"against ${s.matched_id_macro_f1:.3f}$ for the class- and size-matched random-split "
        f"comparator, a gap of ${s.macro_f1_gap:+.3f}$. Pooled out-of-fold Macro-F1 is "
        f"${s.pooled_oof_macro_f1:.3f}$; scenario-balanced phishing miss rate and benign "
        f"false-positive rate are ${100 * s.scenario_balanced_phishing_miss:.1f}\\%$ and "
        f"${100 * s.scenario_balanced_benign_fpr:.1f}\\%$, respectively%"
    )


def main():
    df = load_corpus()
    per, summary = evaluate(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    per_path = os.path.join(OUT_DIR, "p3_leave_one_scenario_out.csv")
    summary_path = os.path.join(OUT_DIR, "p3_leave_one_scenario_out_summary.csv")
    per.to_csv(per_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_generated(os.path.join(SEC, "tab_scenario_ood.tex"), _table(per, summary))
    write_generated(os.path.join(SEC, "gen_scenario_ood_verdict.tex"), _verdict(summary))

    print(per[["scenario", "n_benign", "n_phishing", "macro_f1", "matched_id_macro_f1_mean", "phishing_miss", "benign_fpr"]].to_string(index=False))
    print("\n", summary.to_string(index=False))
    print(f"[+] {per_path}")
    print(f"[+] {summary_path}")


if __name__ == "__main__":
    main()
