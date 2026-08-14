#!/usr/bin/env python3
"""
export_p1a_results.py — Release bundle for the P1a reference benchmark (reviewer #1: "No
per-seed results, no seeds, no prediction files, no fitted artefacts. Nobody can verify the
confidence intervals or the stratum breakdown without rerunning training").

Re-runs the exact canonical protocol of scripts/make_p1a_assets.py (gold+silver core,
official train/test splits, CompPhish features, threshold 0.5) and exports everything a
reader needs to verify Tables "benchmark", "difficulty" and "ablation" WITHOUT retraining:

  results/p1a_results_bundle/
    per_seed_metrics.csv        — one row per model x seed x metric set (core test split)
    predictions/<model>_seed<s>.csv — id, url, y_true, score for every core-test row
    predictions/difficulty_<best>_seed0.csv — best model refit on core, scored on the FULL
                                  test split (all tiers + benign strata; Table "difficulty")
    predictions/ablation_<best>_19feat_seed0.csv — same, path_len/query_len dropped
    models/<model>_seed0.joblib — fitted seed-0 estimators (plus the 19-feature ablation fit)
    environment.json            — python/library versions, platform, feature list, seeds
    MANIFEST.txt                — SHA-256 checksum of every file above
    README.md                   — how each file maps to a manuscript table

RUN:  python scripts/export_p1a_results.py
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_url_baseline import make_model, _metrics, DETERMINISTIC  # noqa: E402
from make_p1a_assets import _load_bench, SUITE, METRICS, DATASET  # noqa: E402

OUT = os.path.join(ROOT, "results", "p1a_results_bundle")
SEEDS = 5
THRESHOLD = 0.5


def main():
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)

    df, feats = _load_bench()
    raw = pd.read_csv(DATASET, low_memory=False)
    df["id"] = df["url"].map(raw.set_index("url_norm")["id"])

    core = df[df.tier.astype(str).isin(["gold", "silver"])]
    tr, te = core[core.split == "train"], core[core.split == "test"]
    yte = te.y.to_numpy()

    rows, mean_pr = [], {}
    for key, disp in SUITE:
        prs = []
        for s in range(1) if key in DETERMINISTIC else range(SEEDS):
            m = make_model(key, s).fit(tr[feats], tr.y)
            score = m.predict_proba(te[feats])[:, 1]
            met = _metrics(yte, score, THRESHOLD)
            prs.append(met["PR-AUC"])
            rows.append({"model": key, "display": disp, "seed": s,
                         "n_train": len(tr), "n_test": len(te), **met})
            pd.DataFrame({"id": te.id.values, "url": te.url.values,
                          "y_true": yte, "score": score}) \
                .to_csv(os.path.join(OUT, "predictions", f"{key}_seed{s}.csv"), index=False)
            if s == 0:
                joblib.dump(m, os.path.join(OUT, "models", f"{key}_seed0.joblib"))
            print(f"  [{key} seed {s}] " + "  ".join(f"{k}={met[k]:.3f}" for k in METRICS))
        mean_pr[key] = float(np.mean(prs))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_seed_metrics.csv"), index=False)

    # the difficulty table's model: best by mean PR-AUC, refit seed 0 on the core train split,
    # scored on the FULL test split so every tier/stratum rate is recomputable from this file
    best = max(mean_pr, key=mean_pr.get)
    te_all = df[df.split == "test"]
    abl_feats = [f for f in feats if f not in ("path_len", "query_len")]
    for tag, fts in ((f"difficulty_{best}_seed0", feats),
                     (f"ablation_{best}_19feat_seed0", abl_feats)):
        m = make_model(best, 0).fit(tr[fts], tr.y)
        score = m.predict_proba(te_all[fts])[:, 1]
        pd.DataFrame({"id": te_all.id.values, "url": te_all.url.values,
                      "y_true": te_all.y.values, "tier": te_all.tier.values,
                      "source": te_all.source.values, "score": score}) \
            .to_csv(os.path.join(OUT, "predictions", f"{tag}.csv"), index=False)
        if tag.startswith("ablation"):
            joblib.dump(m, os.path.join(OUT, "models", f"{best}_19feat_seed0.joblib"))
        print(f"  [{tag}] n={len(te_all)}")

    import sklearn
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
           "joblib": joblib.__version__, "features_21": feats, "features_19": abl_feats,
           "seeds": {k: (1 if k in DETERMINISTIC else SEEDS) for k, _ in SUITE},
           "threshold": THRESHOLD, "protocol": "tier in {gold,silver}; official split column; "
           "train on split=='train', test on split=='test'; difficulty/ablation scored on the "
           "full test split across all tiers", "best_model_by_mean_pr_auc": best}
    with open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    readme = """# PhishVN P1a reference-benchmark results bundle

Verification data for the manuscript's benchmark, difficulty and ablation tables — no
retraining needed.

- `per_seed_metrics.csv` — every model x seed with F1 / PR-AUC / ROC-AUC / FPR@90%recall on
  the core (gold+silver) temporal test split. Seed means/stds reproduce the benchmark table.
- `predictions/<model>_seed<s>.csv` — per-row scores on the core test split (`id` joins
  `dataset_url.csv`). Bootstrap/paired-bootstrap CIs are recomputable from these files.
- `predictions/difficulty_*_seed0.csv` — the benchmark's best model scored on the FULL test
  split with `tier` and `source` columns: every per-stratum recall/FPR and its Wilson CI in
  the difficulty table is recomputable by thresholding `score` at 0.5.
- `predictions/ablation_*_19feat_seed0.csv` — same rows, model refit without `path_len`,
  `query_len` (bare-domain ablation table).
- `models/*.joblib` — fitted seed-0 estimators (`joblib.load`; environment.json pins versions).
- `environment.json` — library versions, feature lists, seeds, protocol.

Generated by `scripts/export_p1a_results.py` in the PhishVN code repository.
"""
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    lines = []
    for dirpath, _dirs, files in sorted(os.walk(OUT)):
        for fn in sorted(files):
            if fn == "MANIFEST.txt":
                continue
            p = os.path.join(dirpath, fn)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append(f"{h}  {os.path.relpath(p, OUT)}  {os.path.getsize(p)}")
    with open(os.path.join(OUT, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] {OUT}  ({len(lines)} files in MANIFEST)")


if __name__ == "__main__":
    main()
