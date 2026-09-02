#!/usr/bin/env python3
"""
run_p6_suffix_blindspot.py — P6, 2026-08-19 reframing: the blind spot is indexed on PUBLIC-SUFFIX
LENGTH, and `.vn` is a minority of it.

Until then P6 priced every remedy on a `.vn(+2LD)` group pooling two suffix lengths. Decomposed,
the pooled ROC area is a Simpson artefact, `.vn` is about a quarter of the missed phishing against
more than half for other two-letter ccTLDs, and the sweep shows the prior is length-indexed. Also
settles four things the pooled framing left unmeasured: trivial baselines, the leakage bound,
suffix-length-conditional thresholds, and brand locus.

OUTPUTS (all read by scripts/make_p6_xai_assets.py; no number is typed into a .tex):
  data/processed/p6/p6_suffix_strata.csv     per-seed, per-stratum n / FNR / FPR / AUC (+ seed-0 SHAP)
  data/processed/p6/p6_suffix_auc.csv        the decomposition, incl. cross-stratum pair share
  data/processed/p6/p6_suffix_threshold.csv  decision rules and trivial baselines, F1 + MCC + BA
  data/processed/p6/p6_leakage.csv           three leakage granularities and the twin-free AUCs
  data/processed/p6/p6_brand_locus.csv       registry tokens by where in the URL they appear

RUN:  python scripts/run_p6_suffix_blindspot.py --seeds 5
The three measurements and the four additions: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

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
from run_p2_temporal_strict import load
# The split, the fit and the four published decision rules come from the group-threshold script
# rather than being re-implemented: a differently-drawn benign mask would re-price the paper's
# own operating points, and the reframing has to be a re-reading of the SAME experiment.
from run_p6_group_threshold import STRATA, split_and_fit, strata_of, threshold_rules, vn_group
from run_p6_vn_reading import split_fit as reading_split_fit, strip_diacritics

STRATA_OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_suffix_strata.csv")
AUC_OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_suffix_auc.csv")
THR_OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_suffix_threshold.csv")
LEAK_OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_leakage.csv")
LOCUS_OUT = os.path.join(ROOT, "data", "processed", "p6", "p6_brand_locus.csv")
BRANDS = os.path.join(ROOT, "data", "processed", "brand_tokens.json")

def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _metrics(y, pred) -> dict:
    from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
    return {
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4) if len(np.unique(pred)) > 1 else 0.0,
        "bal_acc": round(float(balanced_accuracy_score(y, pred)), 4),
    }


# --------------------------------------------------------------------------- (M1)+(M2)+strata
def strata_rows(te: pd.DataFrame, seed: int) -> list[dict]:
    y = te.y.to_numpy(int)
    pred = (te.pred.to_numpy(float) >= 0.5)
    g = strata_of(te)
    rows = []
    for name in STRATA:
        m = (g == name).to_numpy()
        ph, be = m & (y == 1), m & (y == 0)
        rows.append({
            "seed": seed, "stratum": name, "n": int(m.sum()),
            "n_phish": int(ph.sum()), "n_benign": int(be.sum()),
            "fnr": round(float(1 - pred[ph].mean()), 4) if ph.sum() else np.nan,
            "fpr": round(float(pred[be].mean()), 4) if be.sum() else np.nan,
            "auc": round(_auc(y[m], te.pred.to_numpy(float)[m]), 4),
        })
    return rows


def auc_row(te: pd.DataFrame, seed: int) -> dict:
    """The M1 decomposition. `cross_pair_share` is the share of the POOLED .vn group's
    (phishing, benign) pairs whose two members sit in different suffix-length strata -- the
    pairs a pooled AUC scores and neither stratum's own AUC does."""
    y = te.y.to_numpy(int)
    s = te.pred.to_numpy(float)
    g = strata_of(te).to_numpy()
    vn = vn_group(te["tld"])
    short = (g == "vn_short") | (g == "cc_short")
    pb, nb = int(((g == "vn_short") & (y == 1)).sum()), int(((g == "vn_short") & (y == 0)).sum())
    pc = int(((g == "vn_compound") & (y == 1)).sum())
    nc = int(((g == "vn_compound") & (y == 0)).sum())
    tot = (pb + pc) * (nb + nc)
    return {
        "seed": seed,
        "auc_vn_pooled": round(_auc(y[vn], s[vn]), 4),
        "auc_vn_short": round(_auc(y[g == "vn_short"], s[g == "vn_short"]), 4),
        "auc_vn_compound": round(_auc(y[g == "vn_compound"], s[g == "vn_compound"]), 4),
        "auc_other_pooled": round(_auc(y[~vn], s[~vn]), 4),
        "auc_cc_short": round(_auc(y[g == "cc_short"], s[g == "cc_short"]), 4),
        "auc_long": round(_auc(y[g == "long"], s[g == "long"]), 4),
        "auc_short_all": round(_auc(y[short], s[short]), 4),
        "cross_pair_share": round(float((pb * nc + pc * nb) / tot), 4) if tot else np.nan,
        "n_vn_short_phish": pb, "n_vn_compound_phish": pc,
    }


# ----------------------------------------------------------------------------------- (S)+(B)
def threshold_rows(cal: pd.DataFrame, te: pd.DataFrame, seed: int) -> list[dict]:
    """Every decision rule on one footing, plus the two trivial classifiers that bound what a
    metric may be read to mean. The published `.vn`-vs-other rules come from
    run_p6_group_threshold.threshold_rules so the two tables cannot drift apart."""
    cal_be = cal[cal.y == 0]
    published, alpha = threshold_rules(cal_be)
    q = 1 - alpha

    def bq(scores: pd.Series) -> float:
        return float(scores.quantile(q)) if len(scores) else 0.5

    cal_g, te_g = strata_of(cal), strata_of(te)
    cal_short = (cal_g == "vn_short") | (cal_g == "cc_short")
    te_short = ((te_g == "vn_short") | (te_g == "cc_short")).to_numpy()

    y = te.y.to_numpy(int)
    s = te.pred.to_numpy(float)
    vn = vn_group(te["tld"])

    # Rule set. Each entry maps to a per-row threshold vector on the SAME scores, so the only
    # thing that varies is where the operating point is placed.
    rules: dict[str, np.ndarray] = {}
    for name in ("default", "global", "per-group", "vn-only"):
        t = published[name]
        rules[name] = np.where(vn, t["vn"], t["other"])
    # (S) the group the model is actually blind to: two-character suffixes, .vn or not.
    t_short = bq(cal_be[cal_short[cal_be.index]].pred)
    t_longg = bq(cal_be[~cal_short[cal_be.index]].pred)
    rules["per-suffixlen"] = np.where(te_short, t_short, t_longg)
    # ... and the full four-way split, which separates the two .vn strata the paper pooled.
    thr4 = {k: bq(cal_be[(cal_g[cal_be.index] == k)].pred) for k in STRATA}
    rules["per-stratum"] = te_g.map(thr4).to_numpy(float)

    rows = []
    for name, tvec in rules.items():
        pred = (s >= tvec).astype(int)
        r = {"seed": seed, "rule": name, "alpha_cal": round(alpha, 4),
             "thr_vn_short": round(float(np.median(tvec[(te_g == "vn_short").to_numpy()])), 4)
             if (te_g == "vn_short").any() else np.nan,
             "thr_cc_short": round(float(np.median(tvec[(te_g == "cc_short").to_numpy()])), 4)
             if (te_g == "cc_short").any() else np.nan,
             "fpr": round(float(pred[y == 0].mean()), 4)}
        for k in STRATA:
            m = (te_g == k).to_numpy() & (y == 1)
            r["fnr_" + k] = round(float((pred[m] == 0).mean()), 4) if m.sum() else np.nan
        r["fnr_short_all"] = round(float((pred[te_short & (y == 1)] == 0).mean()), 4)
        # the pooled .vn FNR the published table prints, so the two tables are checkably the
        # same experiment read two ways
        r["fnr_vn_pooled"] = round(float((pred[vn & (y == 1)] == 0).mean()), 4)
        r.update(_metrics(y, pred))
        rows.append(r)
    # (B) the floor. all-positive is the one that matters -- this test set is ~55% phishing, so
    # its F1 sits within 0.01 of the published per-group rule's.
    for name, pred in (("all-positive", np.ones(len(y), int)),
                       ("all-negative", np.zeros(len(y), int))):
        r = {"seed": seed, "rule": name, "alpha_cal": round(alpha, 4),
             "thr_vn_short": np.nan, "thr_cc_short": np.nan,
             "fpr": round(float(pred[y == 0].mean()), 4)}
        for k in STRATA:
            m = (te_g == k).to_numpy() & (y == 1)
            r["fnr_" + k] = round(float((pred[m] == 0).mean()), 4) if m.sum() else np.nan
        r["fnr_short_all"] = round(float((pred[te_short & (y == 1)] == 0).mean()), 4)
        r["fnr_vn_pooled"] = round(float((pred[vn & (y == 1)] == 0).mean()), 4)
        r.update(_metrics(y, pred))
        rows.append(r)
    return rows


# --------------------------------------------------------------------------------------- (L)
def leakage_rows(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], seed: int) -> list[dict]:
    """Limitation (v) bounds benign memorisation with a DOMAIN count. Three granularities, and
    the direction of the correction.

    A domain count answers "how many distinct benign registrable domains recur"; what the ROC
    curve is actually exposed to is how many benign TEST ROWS have a twin in training, and --
    since the model sees only 21 numbers -- how many test rows have a training row with an
    identical FEATURE VECTOR, which needs no shared domain at all. The twin-free re-run deletes
    every such benign test row and recomputes the group AUCs on what is left."""
    tr_be, te_be = tr[tr.y == 0], te[te.y == 0]
    dom_tr = set(tr_be.rdom.astype(str))
    dup_dom_rows = te_be.rdom.astype(str).isin(dom_tr).to_numpy()
    n_dom_te = int(te_be.rdom.nunique())
    n_dom_shared = int(te_be[dup_dom_rows].rdom.nunique())

    key_tr = set(map(tuple, np.round(tr_be[feats].to_numpy(float), 6)))
    dup_vec_rows = np.array([tuple(r) in key_tr
                             for r in np.round(te_be[feats].to_numpy(float), 6)])

    # Two twin-free variants, because the two exclusions do not mean the same thing. Dropping
    # DOMAIN twins is the conservative, defensible one: those rows the model could genuinely have
    # memorised. Dropping FEATURE-VECTOR twins is the aggressive one -- 21 coarse integers
    # collide by coincidence as well as by memorisation, so it deletes far more than it must and
    # its result is an outer bound, not an estimate.
    keep_dom = ~dup_dom_rows
    keep = ~(dup_dom_rows | dup_vec_rows)
    te_domclean = pd.concat([te[te.y == 1], te_be[keep_dom]])
    te_clean = pd.concat([te[te.y == 1], te_be[keep]])
    g_full, g_clean = strata_of(te), strata_of(te_clean)
    g_dom = strata_of(te_domclean)
    vn_full, vn_clean = vn_group(te["tld"]), vn_group(te_clean["tld"])
    vn_dom = vn_group(te_domclean["tld"])
    out = {
        "seed": seed,
        "n_benign_test_domains": n_dom_te,
        "n_benign_test_domains_shared": n_dom_shared,
        "share_domains": round(n_dom_shared / n_dom_te, 4),
        "n_benign_test_rows": int(len(te_be)),
        "n_benign_test_rows_domain_twin": int(dup_dom_rows.sum()),
        "share_rows": round(float(dup_dom_rows.mean()), 4),
        "n_benign_test_rows_vector_twin": int(dup_vec_rows.sum()),
        "share_vectors": round(float(dup_vec_rows.mean()), 4),
        "share_rows_any_twin": round(float((dup_dom_rows | dup_vec_rows).mean()), 4),
        "n_benign_after_domclean": int(keep_dom.sum()),
        "n_benign_after": int(keep.sum()),
    }
    y, s = te.y.to_numpy(int), te.pred.to_numpy(float)
    yc, sc = te_clean.y.to_numpy(int), te_clean.pred.to_numpy(float)
    yd, sd = te_domclean.y.to_numpy(int), te_domclean.pred.to_numpy(float)
    out["auc_vn_pooled"] = round(_auc(y[vn_full], s[vn_full]), 4)
    out["auc_other_pooled"] = round(_auc(y[~vn_full], s[~vn_full]), 4)
    out["auc_vn_pooled_domclean"] = round(_auc(yd[vn_dom], sd[vn_dom]), 4)
    out["auc_other_pooled_domclean"] = round(_auc(yd[~vn_dom], sd[~vn_dom]), 4)
    out["auc_vn_pooled_clean"] = round(_auc(yc[vn_clean], sc[vn_clean]), 4)
    out["auc_other_pooled_clean"] = round(_auc(yc[~vn_clean], sc[~vn_clean]), 4)
    for k in STRATA:
        mf, mc, md = ((g_full == k).to_numpy(), (g_clean == k).to_numpy(),
                      (g_dom == k).to_numpy())
        out["auc_" + k] = round(_auc(y[mf], s[mf]), 4)
        out["auc_" + k + "_domclean"] = round(_auc(yd[md], sd[md]), 4)
        out["auc_" + k + "_clean"] = round(_auc(yc[mc], sc[mc]), 4)
    out["gap_pooled"] = round(out["auc_vn_pooled"] - out["auc_other_pooled"], 4)
    out["gap_pooled_domclean"] = round(
        out["auc_vn_pooled_domclean"] - out["auc_other_pooled_domclean"], 4)
    out["gap_pooled_clean"] = round(out["auc_vn_pooled_clean"] - out["auc_other_pooled_clean"], 4)
    return [out]


# --------------------------------------------------------------------------------------- (K)
def brand_locus(df: pd.DataFrame) -> pd.DataFrame:
    """Where in the URL a registry token appears, on the same corpus and matcher as the paper's
    registrable-domain count. Splits the conclusion's untested triple: `path` and `subdomain`
    are checkable here, `page content` is not (no page was fetched)."""
    toks = json.load(open(BRANDS))["tokens"]
    toklist = sorted({t["token"].lower() for t in toks if len(t["token"]) >= 5})
    ph = df[(df.y == 1) & df.date.notna()].copy()
    ph["rdom_s"] = ph.rdom.astype(str).str.lower()
    ph["dom_s"] = ph.dom.astype(str).str.lower()
    ph["url_s"] = ph.url.astype(str).str.lower()

    def has(hay: str) -> bool:
        flat = strip_diacritics(hay)
        squash = re.sub(r"[-\d.]", "", flat)
        return any((t in flat) or (t in squash) for t in toklist)

    # One row per URL, deduplicated by (registrable domain, path) so a campaign that re-uses one
    # path across many hosts is not counted once per host.
    rows = []
    for _, r in ph.iterrows():
        rd, dom, url = r.rdom_s, r.dom_s, r.url_s
        sub = dom[: -len(rd)] if rd and dom.endswith(rd) else ""
        i = url.find(dom)
        tail = url[i + len(dom):] if i >= 0 else ""
        tail = tail.split("#", 1)[0]
        path = tail.split("?", 1)[0]
        rows.append({"rdom": rd, "path": path,
                     "in_rdom": int(has(rd)), "in_subdomain": int(has(sub)) if sub else 0,
                     "in_path": int(has(path)) if path.strip("/") else 0,
                     "has_path": int(bool(path.strip("/")))})
    out = pd.DataFrame(rows).drop_duplicates(subset=["rdom", "path"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.15)
    ap.add_argument("--skip-shap", action="store_true")
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]

    strata, aucs, thrs, leaks = [], [], [], []
    for seed in range(args.seeds):
        tr, m, cal, te = split_and_fit(df, feats, args.family, seed, args.cal_frac)
        strata += strata_rows(te, seed)
        aucs.append(auc_row(te, seed))
        thrs += threshold_rows(cal, te, seed)
        leaks += leakage_rows(tr, te, feats, seed)
        a = aucs[-1]
        print(f"[i] seed {seed}: AUC .vn pooled {a['auc_vn_pooled']:.3f} = "
              f"short {a['auc_vn_short']:.3f} + compound {a['auc_vn_compound']:.3f} "
              f"(cross-stratum pairs {a['cross_pair_share']:.3f}); "
              f"other {a['auc_other_pooled']:.3f}, cc_short {a['auc_cc_short']:.3f}")

    st = pd.DataFrame(strata)
    if args.family != "CatBoost":
        # Cross-family stress tests get their own compact artefacts. They must not overwrite the
        # canonical CatBoost SHAP, threshold, leakage or brand-locus files, none of which this run
        # is intended to recompute.
        slug = re.sub(r"\W+", "", args.family).lower()
        st_out = STRATA_OUT.replace(".csv", f"_{slug}.csv")
        auc_out = AUC_OUT.replace(".csv", f"_{slug}.csv")
        os.makedirs(os.path.dirname(st_out), exist_ok=True)
        st.to_csv(st_out, index=False)
        pd.DataFrame(aucs).to_csv(auc_out, index=False)
        print(f"[+] {st_out}")
        print(f"[+] {auc_out}")
        print("\n=== strata (mean over seeds) ===")
        print(st.groupby("stratum")[["n", "n_phish", "n_benign", "fnr", "fpr", "auc"]]
              .mean().reindex(list(STRATA)).round(4).to_string())
        return
    # (a) the signed tld_len attribution per stratum, on the SAME single-model split the per-TLD
    # table is read from (run_p6_vn_reading), not on the calibrated split above: mixing the two
    # would put a table's SHAP column and its FNR column on different models.
    if not args.skip_shap:
        import shap
        m0, _tr0, te0, _ph0 = reading_split_fit(df, feats, args.family, 0)
        sv = shap.TreeExplainer(m0).shap_values(te0[feats].to_numpy(float))
        if isinstance(sv, list):
            sv = sv[1]
        te0 = te0.copy()
        te0["shap_tld_len"] = sv[:, feats.index("tld_len")]
        te0["pred"] = m0.predict_proba(te0[feats].to_numpy(float))[:, 1]
        g0 = strata_of(te0)
        srows = []
        for k in STRATA:
            sub = te0[(g0 == k).to_numpy()]
            gp = sub[sub.y == 1]
            srows.append({"stratum": k, "n": len(sub), "n_phish": len(gp),
                          "mean_shap_tld_len": round(float(sub.shap_tld_len.mean()), 5),
                          "fnr_single": round(float((gp.pred < 0.5).mean()), 4) if len(gp) else np.nan})
        sh = pd.DataFrame(srows)
        st = st.merge(sh, on="stratum", how="left", suffixes=("", "_shapsplit"))
        print("\n=== signed SHAP(tld_len) by stratum (single-model split, seed 0) ===")
        print(sh.to_string(index=False))

    os.makedirs(os.path.dirname(STRATA_OUT), exist_ok=True)
    st.to_csv(STRATA_OUT, index=False)
    pd.DataFrame(aucs).to_csv(AUC_OUT, index=False)
    pd.DataFrame(thrs).to_csv(THR_OUT, index=False)
    pd.DataFrame(leaks).to_csv(LEAK_OUT, index=False)
    lo = brand_locus(df)
    lo.to_csv(LOCUS_OUT, index=False)
    for p in (STRATA_OUT, AUC_OUT, THR_OUT, LEAK_OUT, LOCUS_OUT):
        print(f"[+] {p}")

    print("\n=== strata (mean over seeds) ===")
    print(st.groupby("stratum")[["n", "n_phish", "n_benign", "fnr", "fpr", "auc"]]
          .mean().reindex(list(STRATA)).round(4).to_string())
    print("\n=== decision rules (mean over seeds) ===")
    tf = pd.DataFrame(thrs)
    cols = ["fnr_vn_pooled", "fnr_vn_short", "fnr_cc_short", "fnr_vn_compound", "fnr_long",
            "fpr", "f1", "mcc", "bal_acc"]
    print(tf.groupby("rule")[cols].mean().round(4).to_string())
    print("\n=== leakage (mean over seeds) ===")
    lf = pd.DataFrame(leaks)
    print(lf[["share_domains", "share_rows", "share_vectors", "share_rows_any_twin",
              "gap_pooled", "gap_pooled_domclean", "gap_pooled_clean",
              "auc_vn_short", "auc_vn_short_domclean", "auc_long", "auc_long_domclean"]]
          .mean().round(4).to_string())
    print("\n=== brand locus (dated phishing, deduplicated by domain+path) ===")
    print(f"rows {len(lo)}; in registrable domain {int(lo.in_rdom.sum())} "
          f"({lo.in_rdom.mean():.4f}); in subdomain {int(lo.in_subdomain.sum())} "
          f"({lo.in_subdomain.mean():.4f}); with a non-empty path {int(lo.has_path.sum())}; "
          f"in path {int(lo.in_path.sum())} "
          f"({lo[lo.has_path == 1].in_path.mean() if lo.has_path.sum() else float('nan'):.4f} "
          f"of those)")


if __name__ == "__main__":
    main()
