#!/usr/bin/env python3
"""
p2_dup_leakage.py — how much of P2's full-corpus random split is memorisable, and what
memorisation alone is worth.

Round-2 review, M3. The `random` protocol behind the abstract's 0.922 is a plain train_test_split
with no domain guard and no de-duplication, and the 21 CompPhish features are string statistics
that collapse distinct URLs onto identical vectors in bulk. Measures a census, the twin rate, a
memorisation ORACLE (a lookup table that learns nothing; its F1 is the value of the leak), an
all-positive floor, the ceiling any function of these features can reach, and the twin rates of
the two dated designs Delta_proto rests on.

The oracle scores BELOW the fitted models, so this does not license "the benchmark is only
memorising" — only that this protocol's level is not a clean generalisation estimate.

RUN:  python scripts/p2_dup_leakage.py
Each measurement and what it licenses: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label
from psl import registered_domain

CORPUS = "data/processed/vn_compphish.csv"
OUT = "data/processed/p2/p2_dup_leakage.csv"
OUT_PROTO = "data/processed/p2/p2_dup_leakage_protocols.csv"


def _group_labels(keys, labels):
    acc = collections.defaultdict(list)
    for k, lab in zip(keys, labels):
        acc[k].append(int(lab))
    return acc


def _rates(fv, rdom, itr, ite):
    """(exact-feature-vector twin rate, shared-registrable-domain rate) of a test index set."""
    tr_fv, tr_rd = set(fv[itr]), set(rdom[itr])
    return (float(np.mean([k in tr_fv for k in fv[ite]])),
            float(np.mean([r in tr_rd for r in rdom[ite]])))


def protocol_rates(seeds: int, out_path: str):
    """The same two rates for the dated-row designs, reproducing run_p2_temporal_strict's splits.

    The paper's Delta_proto is defined as "protocol is the only variable" between these two, and
    the guard is applied on only one side of that comparison. Measured rather than asserted.
    """
    try:
        from run_p2_temporal_strict import load as _load
    except Exception as e:                                  # pragma: no cover - optional arm
        print(f"[i] dated-row rates unavailable ({type(e).__name__}: {e})")
        return None
    df = _load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]

    def keys(frame):
        return (np.array([hash(t) for t in map(tuple, frame[feats].to_numpy(float))]),
                frame["rdom"].to_numpy())

    rows = []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        bmask = rng.rand(len(be)) < 0.70
        pool = pd.concat([ph_tr, ph_te])
        pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
        for proto, tr, te in (
                ("temporal_strict", pd.concat([ph_tr, be[bmask]]),
                 pd.concat([ph_te, be[~bmask]])),
                ("random_same_rows", pd.concat([pool[pmask], be[bmask]]),
                 pd.concat([pool[~pmask], be[~bmask]]))):
            fv = np.concatenate([keys(tr)[0], keys(te)[0]])
            rd = np.concatenate([keys(tr)[1], keys(te)[1]])
            itr = np.arange(len(tr))
            ite = np.arange(len(tr), len(tr) + len(te))
            twin, dom = _rates(fv, rd, itr, ite)
            # the phishing side alone is where the guard operates and where drift lives
            pv_tr = keys(tr[tr.y == 1])
            pv_te = keys(te[te.y == 1])
            fvp = np.concatenate([pv_tr[0], pv_te[0]])
            rdp = np.concatenate([pv_tr[1], pv_te[1]])
            twin_p, dom_p = _rates(fvp, rdp, np.arange(len(pv_tr[0])),
                                   np.arange(len(pv_tr[0]), len(fvp)))
            rows.append({"protocol": proto, "seed": s,
                         "test_twin_rate": round(twin, 4),
                         "test_regdom_shared_rate": round(dom, 4),
                         "phish_test_twin_rate": round(twin_p, 4),
                         "phish_test_regdom_shared_rate": round(dom_p, 4)})
            print(f"  {proto:<17} seed={s} twin={twin:.4f} regdom={dom:.4f} "
                  f"(phishing side {twin_p:.4f}/{dom_p:.4f})")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"[+] {len(out)} rows -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=CORPUS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = add_label(pd.read_csv(args.inp)).dropna(subset=COMPPHISH)
    X = df[COMPPHISH].to_numpy(float)
    y = df["y"].to_numpy(int)
    # Exact float tuples, not rounded strings: rounding merges vectors the learner can still
    # tell apart and would overstate the leak.
    fv = np.array([hash(t) for t in map(tuple, X)])
    rdom = df["dom"].astype(str).str.lower().map(registered_domain).to_numpy()

    groups = _group_labels(fv, y)
    mixed = {k for k, s in groups.items() if len(set(s)) > 1}
    mixed_rows = int(np.isin(fv, list(mixed)).sum()) if mixed else 0

    # The schema ceiling: majority label per exact feature vector, fitted and scored on
    # everything. No split, no learner -- the best F1 these 21 features admit on this corpus.
    maj = {k: int(sum(g) * 2 >= len(g)) for k, g in groups.items()}
    ceiling = float(f1_score(y, np.array([maj[k] for k in fv])))

    census = {
        "rows": len(df),
        "unique_regdom": int(len(set(rdom))),
        "unique_featvec": int(len(groups)),
        "mixed_featvec_groups": len(mixed),
        "rows_in_mixed_groups": mixed_rows,
        "positive_rate": float(y.mean()),
        "schema_ceiling_f1": round(ceiling, 4),
    }
    print("census: " + "  ".join(f"{k}={v}" for k, v in census.items()))

    rows = []
    for s in range(args.seeds):
        itr, ite = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y,
                                    random_state=s)
        twin, dom = _rates(fv, rdom, itr, ite)

        acc = _group_labels(fv[itr], y[itr])
        table = {k: int(sum(v) * 2 >= len(v)) for k, v in acc.items()}
        fallback = int(y[itr].mean() >= 0.5)
        pred = np.array([table.get(k, fallback) for k in fv[ite]])

        r = dict(census, seed=s,
                 test_twin_rate=round(twin, 4),
                 test_regdom_shared_rate=round(dom, 4),
                 lookup_oracle_f1=round(float(f1_score(y[ite], pred)), 4),
                 all_positive_f1=round(float(f1_score(y[ite], np.ones(len(ite), int))), 4))
        rows.append(r)
        print(f"  seed={s} twin={twin:.4f} regdom={dom:.4f} "
              f"oracle_F1={r['lookup_oracle_f1']:.4f} allpos_F1={r['all_positive_f1']:.4f}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} rows -> {args.out}")
    print(out[["test_twin_rate", "test_regdom_shared_rate", "lookup_oracle_f1",
               "all_positive_f1"]].mean().round(4).to_string())

    print("\ndated-row designs (run_p2_temporal_strict's own splits):")
    protocol_rates(args.seeds, OUT_PROTO)


if __name__ == "__main__":
    main()
