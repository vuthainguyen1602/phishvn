#!/usr/bin/env python3
"""Paired model-seed sensitivity on P5's fixed constructed stream (not temporal replication).

Run from any directory. Uses the same input builder and settings as make_p5_assets.
Records input SHA-256, per-window scores, per-seed summaries and paired differences.
Reported spread is sample SD across model seeds, not uncertainty across deployments.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT as _ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    _ROOT = os.path.dirname(_HERE)
ROOT = Path(_ROOT)
from make_p5_assets import build_input, DRIFT_CSV, WINDOWS, PERIOD, PSI_TAU, F1_DROP
from retrain_drift import load, run

SEEDS = (0, 1, 2, 3, 4)


def main():
    build_input()
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    out = ROOT / "data/processed/p5"
    out.mkdir(parents=True, exist_ok=True)
    windows, summaries = [], []
    for seed in SEEDS:
        scores, counts, _ = run(df, feats, WINDOWS, PERIOD, PSI_TAU, F1_DROP,
                               include_static_arch=False, seed=seed)
        for policy, values in scores.items():
            summaries.append(dict(seed=seed, policy=policy, autc=np.mean(values),
                                  end_f1=values[-1], retrains=counts[policy]))
            windows.extend(dict(seed=seed, policy=policy, window=w, f1=f1)
                           for w, f1 in enumerate(values, 2))
        print(f"seed {seed}: " + ", ".join(f"{p}={np.mean(v):.4f}" for p, v in scores.items()), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "p5_model_seed_sensitivity.csv", index=False)
    pd.DataFrame(windows).to_csv(out / "p5_model_seed_windows.csv", index=False)
    paired = summary.pivot(index="seed", columns="policy", values="autc")
    delta = paired["drift"] - paired["periodic"]
    pd.DataFrame({"seed": delta.index, "drift_minus_periodic_autc": delta.values}).to_csv(
        out / "p5_model_seed_paired.csv", index=False)
    metadata = dict(seeds=SEEDS, windows=WINDOWS, period=PERIOD, psi_tau=PSI_TAU,
                    f1_drop=F1_DROP, n_rows=len(df), features=feats,
                    input_sha256=hashlib.sha256(Path(DRIFT_CSV).read_bytes()).hexdigest(),
                    scope="model randomness conditional on one fixed constructed stream",
                    score_precision="per-window F1 rounded to 3 decimals by existing runner",
                    paired_delta_mean=float(delta.mean()), paired_delta_sd=float(delta.std(ddof=1)))
    (out / "p5_model_seed_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = [r"\begin{table}[ht]", r"\centering\small",
             r"\caption{Paired model-seed sensitivity on the fixed original-label stream: five Random Forest seeds (0--4), mean $\pm$ sample SD. This measures model randomness, not variation across temporal cuts or acquisition schedules.}",
             r"\label{tab:model-seeds}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Policy & AUTC & End F1 & Retrains \\", r"\midrule"]
    for policy in ("static", "periodic", "drift"):
        d = summary[summary.policy.eq(policy)]
        cells = [f"${d[c].mean():.4f} \\pm {d[c].std(ddof=1):.4f}$" for c in ("autc", "end_f1")]
        cells.append(f"${d.retrains.mean():.0f}$" if d.retrains.nunique() == 1
                     else f"${d.retrains.mean():.1f} \\pm {d.retrains.std(ddof=1):.1f}$")
        lines.append(policy.capitalize() + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{Paired AUTC, drift minus periodic: " +
              f"${delta.mean():+.4f} \\pm {delta.std(ddof=1):.4f}$" + r"} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "papers/P5_temporal_drift/sections/tab_model_seeds.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
