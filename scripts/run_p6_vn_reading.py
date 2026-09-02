#!/usr/bin/env python3
"""
run_p6_vn_reading.py — P6 experiment 2: the Vietnamese reading of what the model relies on.

run_p6_protocol_shap found tld_len dominating attribution ~2x under BOTH protocols. This asks what
that feature encodes here and what it costs: a per-TLD SHAP decomposition on the phishing-temporal
test set, and the subgroup error rates that price it. If ".vn" carries a strong benign-direction
contribution while commodity TLDs push phishing-ward, the model's top signal is a ".vn or not"
prior rather than a phishing signal.

RUN:  python scripts/run_p6_vn_reading.py
The decomposition, the subgroup rates and the outputs: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load

TLD_OUT = "data/processed/p6/p6_tld_shap.csv"
BRAND_OUT = "data/processed/p6/p6_brand_hits.csv"
BRANDS = "data/processed/brand_tokens.json"


def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()


def split_fit(df, feats, family: str, seed: int):
    """The phishing-temporal split and fit this experiment's numbers are read from, in one place.

    Extracted 2026-08-17 so run_p6_vn_deficit.py can attribute the misses on EXACTLY the
    population the published $141/156$ counts: a companion script re-deriving the split would
    re-derive the benign mask too, and then be explaining a different set of false negatives."""
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    rng = np.random.RandomState(seed)
    bmask = rng.rand(len(be)) < 0.70
    tr = pd.concat([ph_tr, be[bmask]])
    te = pd.concat([ph_te, be[~bmask]]).reset_index(drop=True)

    m = make_any_model(family, seed)
    m.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))
    return m, tr, te, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    m, tr, te, ph = split_fit(df, feats, args.family, args.seed)

    # ---- (a) per-TLD signed SHAP of tld_len on the full temporal test set
    import shap
    sv = shap.TreeExplainer(m).shap_values(te[feats].to_numpy(float))
    if isinstance(sv, list):
        sv = sv[1]
    i_tld = feats.index("tld_len")
    te["shap_tld_len"] = sv[:, i_tld]
    te["pred"] = m.predict_proba(te[feats].to_numpy(float))[:, 1]
    te["tld_str"] = te["tld"].astype(str).str.lower()
    # group the .vn second-level family with bare .vn: the registry prior is the same
    te["tld_grp"] = np.where(te.tld_str.str.endswith("vn"), ".vn(+2LD)", "." + te.tld_str)

    rows = []
    for tld, g in te.groupby("tld_grp"):
        gp = g[g.y == 1]
        rows.append({
            "tld": tld, "n": len(g), "n_phish": len(gp),
            "phish_rate": round(len(gp) / len(g), 4),
            "mean_shap_tld_len": round(float(g.shap_tld_len.mean()), 5),
            "fnr_phish@0.5": round(float((gp.pred < 0.5).mean()), 4) if len(gp) else np.nan,
        })
    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    tab.to_csv(TLD_OUT, index=False)
    print("=== per-TLD (phishing-temporal test set) ===")
    print(tab.head(15).to_string(index=False))
    vn = te[(te.y == 1) & te.tld_str.str.endswith("vn")]
    other = te[(te.y == 1) & ~te.tld_str.str.endswith("vn")]
    print(f"\n.vn phishing in test: {len(vn)}  FNR@0.5={float((vn.pred < 0.5).mean()):.3f}   "
          f"other-TLD phishing: {len(other)}  FNR@0.5={float((other.pred < 0.5).mean()):.3f}")

    # ---- (c) brand tokens over ALL dated phishing registered domains (train+test: descriptive)
    toks = json.load(open(BRANDS))["tokens"]
    # tokens are already ascii-ish; keep len>=5 to avoid junk substring hits (e.g. 'vien')
    toklist = sorted({t["token"].lower() for t in toks if len(t["token"]) >= 5})
    doms = ph["rdom"].astype(str).str.lower().unique()
    hits = []
    for d in doms:
        flat = strip_diacritics(d)
        squash = re.sub(r"[-\d.]", "", flat)
        for t in toklist:
            if t in flat:
                hits.append({"brand": t, "domain": d, "mangled": 0})
            elif t in squash:  # token only appears once dashes/digits are removed
                hits.append({"brand": t, "domain": d, "mangled": 1})
    hb = pd.DataFrame(hits)
    hb.to_csv(BRAND_OUT, index=False)
    top = (hb.groupby("brand").agg(domains=("domain", "nunique"), mangled=("mangled", "sum"))
             .sort_values("domains", ascending=False))
    print(f"\n=== brand tokens in {len(doms)} dated phishing registrable domains ===")
    print(f"domains with >=1 brand hit: {hb.domain.nunique()} "
          f"({hb.domain.nunique() / len(doms):.1%});  mangled hits: {int(hb.mangled.sum())}")
    print(top.head(15).to_string())
    print(f"[+] -> {TLD_OUT}, {BRAND_OUT}")


if __name__ == "__main__":
    main()
