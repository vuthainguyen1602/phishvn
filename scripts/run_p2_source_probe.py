#!/usr/bin/env python3
"""
run_p2_source_probe.py — does the char-CNN's margin survive a change of benign SOURCE?

WHY. run_p2_charcnn beats all seven tabular families on every metric under both protocols,
42/42 comparisons at q<0.001. Before that reaches a manuscript, one rival explanation has to be
excluded: a model reading the raw string can key on WHICH FEED a row came from rather than on
whether it is phishing. In this corpus source and label are perfectly confounded -- no source
contributes to both classes (benign: tranco, tranco_vn, tinnhiem_org, tinnhiem_web; phishing:
chongluadao, tinnhiemmang, openphish) -- so "predict the source" is literally "predict the label"
and a source classifier proves nothing.

THE STRESS TEST WE CAN RUN is a benign-source swap. Train with one benign FAMILY and test against
the other, holding the phishing side at its phishing-temporal split:

    tranco      -> train on {tranco, tranco_vn},        test against {tinnhiem_org, tinnhiem_web}
    tinnhiem    -> the reverse

A model that learned "not phishing" should transfer better than one that learned "looks like a
Tranco string". Both arms run on the identical phishing rows, so the diagnostic measures relative
sensitivity to the benign family. It does NOT identify provenance separately from label: every
PhishVN source is single-class, the phishing-source mixture stays fixed, and the reverse direction
changes the test population enough to collapse every model.

READ IT AS A SENSITIVITY DIAGNOSTIC, NOT A BENCHMARK OR A PROVENANCE TEST. The swapped design is deliberately unfair to every
model: it trains on one benign population and tests on another, which no deployment does. Its
numbers belong beside the canonical ones as evidence about what the margin is made of, never as
a replacement for them.

RUN:
  python scripts/run_p2_source_probe.py --seeds 3
Design note: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse, os, sys, time

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import COMPPHISH, _metrics
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load, split_phishing, URL_CSV
from run_p2_charcnn import fit_predict

TRANCO = {"tranco", "tranco_vn"}
TINNHIEM = {"tinnhiem_org", "tinnhiem_web"}


def with_source(df):
    """Attach the corpus `source` column, which load() does not carry."""
    src = pd.read_csv(URL_CSV, usecols=["url_norm", "source"], dtype=str)
    return df.merge(src.rename(columns={"url_norm": "url"}), on="url", how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--families", nargs="+", default=["CatBoost", "LogReg"])
    ap.add_argument("--out", default="data/processed/p2/p2_source_probe.csv")
    a = ap.parse_args()
    os.chdir(ROOT)

    df = with_source(load())
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, a.cut)
    print(f"benign by family: tranco {be.source.isin(TRANCO).sum()}  "
          f"tinnhiem {be.source.isin(TINNHIEM).sum()}  (unmapped {be.source.isna().sum()})")

    rows = []
    for train_fam, name in ((TRANCO, "tranco"), (TINNHIEM, "tinnhiem")):
        be_tr = be[be.source.isin(train_fam)]
        be_te = be[~be.source.isin(train_fam) & be.source.notna()]
        for s in range(a.seeds):
            tr = pd.concat([ph_tr, be_tr])
            te = pd.concat([ph_te, be_te])
            yte = te["y"].to_numpy(int)

            t0 = time.time()
            score = fit_predict(tr["url"].tolist(), tr["y"].to_numpy(int),
                                te["url"].tolist(), s, a.epochs)
            met = _metrics(yte, score)
            met.update({"family": "CharCNN", "seed": s, "train_benign": name,
                        "fit_seconds": round(time.time() - t0, 1)})
            rows.append(met)
            print(f"  CharCNN   train_benign={name:<9} seed={s} F1={met['F1']:.3f} "
                  f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")

            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte = te[feats].to_numpy(float)
            for fam in a.families:
                t0 = time.time()
                m = make_any_model(fam, s)
                m.fit(Xtr, ytr)
                sc = m.predict_proba(Xte)[:, 1]
                met = _metrics(yte, sc)
                met.update({"family": fam, "seed": s, "train_benign": name,
                            "fit_seconds": round(time.time() - t0, 1)})
                rows.append(met)
                print(f"  {fam:<9} train_benign={name:<9} seed={s} F1={met['F1']:.3f} "
                      f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[+] {a.out}")


if __name__ == "__main__":
    main()
