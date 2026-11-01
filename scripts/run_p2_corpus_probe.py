#!/usr/bin/env python3
"""
run_p2_corpus_probe.py — how much of a URL's 21 features says which corpus, or which feed, it came
from, holding the class fixed (revision 1, 2026-10-31).

WHY. Every PhishVN source is single-class, and the transfer matrix inverts across corpora. Both
leave open whether the features carry the way each corpus was assembled as much as they carry
phishing. A source classifier on PhishVN as a whole proves nothing (source predicts label by
construction; run_p2_source_probe.py says so). Within one class it does: if phishing rows from
PhiUSIIL and from PhishVN can be told apart on the 21 features, the features carry a corpus
signature that a phishing-vs-benign model trained on one corpus can learn and that the other
corpus does not share.

DESIGN. Three probes, each a gradient-boosted classifier (HistGB, defaults) scored by ROC-AUC
under 5-fold cross-validation grouped by registrable domain, so a domain is never on both sides of
a fold:
  corpus pair   for each pair of the four transfer corpora and each class: which corpus? (the
                featurised host-level files the transfer arm reads; each side subsampled to the
                smaller one so chance is 0.5)
  class         for each corpus: phishing or benign? (the same classifier and folds, as the
                yardstick the corpus AUCs are read against)
  PhishVN feed  within benign: Tranco family {tranco, tranco_vn} or tín nhiệm family
                {tinnhiem_org, tinnhiem_web}; within phishing: chongluadao or the other two feeds
Permutation importance (drop in ROC-AUC when one feature is shuffled within the held-out fold,
mean over folds and 5 shuffles) is computed for every probe. Shuffling inside a single-class
fold is conditional on the class by construction; it is not conditional on the other features.

Writes data/processed/p2/revision/corpus_probe.csv and corpus_probe_importance.csv.

RUN: python scripts/run_p2_corpus_probe.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label
from psl import registered_domain
from audit_p2_harmonisation import ALIGNED, host_of

OUT = "data/processed/p2/revision"
URL_CSV = "data/processed/dataset_url.csv"
FOLDS, SHUFFLES, SEED = 5, 5, 0


def load(path: str) -> pd.DataFrame:
    d = add_label(pd.read_csv(path))
    for c in COMPPHISH:
        d[c] = pd.to_numeric(d.get(c, 0), errors="coerce").fillna(0)
    d["rdom"] = d.url.map(host_of).map(registered_domain)
    return d


def probe(X: np.ndarray, t: np.ndarray, groups: np.ndarray, feats: list[str], rng) -> tuple[float, dict]:
    """Grouped-CV ROC-AUC of predicting binary target t, and per-feature permutation importance."""
    aucs, imp = [], {f: [] for f in feats}
    for tr, te in GroupKFold(FOLDS).split(X, t, groups):
        if len(np.unique(t[te])) < 2:
            continue
        m = HistGradientBoostingClassifier(random_state=SEED).fit(X[tr], t[tr])
        base = roc_auc_score(t[te], m.predict_proba(X[te])[:, 1])
        aucs.append(base)
        for j, f in enumerate(feats):
            drops = []
            for _ in range(SHUFFLES):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base - roc_auc_score(t[te], m.predict_proba(Xp)[:, 1]))
            imp[f].append(float(np.mean(drops)))
    return float(np.mean(aucs)), {f: float(np.mean(v)) for f, v in imp.items()}


def balanced(a: pd.DataFrame, b: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(len(a), len(b))
    return (a.iloc[rng.choice(len(a), n, replace=False)], b.iloc[rng.choice(len(b), n, replace=False)])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)
    corp = {k: load(v) for k, v in ALIGNED.items()}
    feats = list(COMPPHISH)
    rows, imps = [], []

    def record(kind, label, cls, a, b):
        X = pd.concat([a, b])[feats].to_numpy(float)
        t = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
        g = pd.concat([a, b]).rdom.astype(str).to_numpy()
        auc, imp = probe(X, t, g, feats, rng)
        rows.append({"probe": kind, "pair": label, "class": cls, "n_each": len(a), "roc_auc": auc})
        imps.extend({"probe": kind, "pair": label, "class": cls, "feature": f, "auc_drop": v}
                    for f, v in imp.items())
        print(f"  {kind:<12} {label:<26} {cls:<9} n={len(a):>6}  AUC={auc:.3f}", flush=True)

    for a, b in itertools.combinations(ALIGNED, 2):
        for cls, name in ((1, "phishing"), (0, "benign")):
            record("corpus_pair", f"{a} vs {b}", name,
                   *balanced(corp[a][corp[a].y == cls], corp[b][corp[b].y == cls], rng))
    for k, d in corp.items():
        record("class", k, "both", d[d.y == 0], d[d.y == 1])

    vn = corp["PhishVN"].merge(
        pd.read_csv(URL_CSV, usecols=["url_norm", "source"], dtype=str)
          .drop_duplicates("url_norm").rename(columns={"url_norm": "url"}), on="url", how="left")
    ben, phi = vn[vn.y == 0], vn[vn.y == 1]
    record("phishvn_feed", "tranco vs tinnhiem", "benign",
           *balanced(ben[ben.source.isin(["tranco", "tranco_vn"])],
                     ben[ben.source.isin(["tinnhiem_org", "tinnhiem_web"])], rng))
    record("phishvn_feed", "chongluadao vs other", "phishing",
           *balanced(phi[phi.source == "chongluadao"],
                     phi[phi.source.isin(["tinnhiemmang", "openphish"])], rng))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "corpus_probe.csv"), index=False)
    imp = pd.DataFrame(imps)
    imp.to_csv(os.path.join(OUT, "corpus_probe_importance.csv"), index=False)
    top = (imp[imp.probe == "corpus_pair"].groupby("feature").auc_drop.mean()
           .sort_values(ascending=False).head(6))
    print("\nmean permutation importance over corpus-pair probes:\n" + top.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
