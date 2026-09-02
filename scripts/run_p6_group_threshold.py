#!/usr/bin/env python3
"""
run_p6_group_threshold.py — P6: the cheap ALGORITHMIC remedy for the .vn blind spot, and the
control that any infrastructure-feature remedy must beat.

Before reaching for new data, the standard subgroup-robustness fix is per-group DECISION
THRESHOLDS. FINDING (2026-08-02): thresholding DOES repair the blind spot (.vn FNR 0.177 -> 0.010),
so it is a missing operating point, not a missing representation. What no threshold can fix is the
price: .vn is ~61% of the benign pool, so .vn recall costs either non-.vn recall or a corpus FPR of
0.635. An infrastructure-feature remedy's job is to make .vn recall cheap, not possible.

Four conditions on the untouched test set — default (0.5), global, per-group and vn-only — all
spending the deployed budget of the 0.5 baseline, calibrated on a held-out slice of the train
window so nothing leaks.

OUTPUT: data/processed/p6/p6_group_threshold.csv (per-seed, per-condition)
        papers/P6_xai/sections/tab_group_threshold.tex

RUN:  python scripts/run_p6_group_threshold.py --seeds 5
The split, the budget definition and each condition: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
import math
import os
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
from genfile import write_generated
from train_url_baseline import COMPPHISH
from paired_eval import wilson
from run_p2_benchmark import make_any_model
from run_p2_temporal_strict import load

OUT_CSV = os.path.join(ROOT, "data", "processed", "p6", "p6_group_threshold.csv")
OUT_TEX = os.path.join(ROOT, "papers", "P6_xai", "sections", "tab_group_threshold.tex")
STRATA = ("vn_short", "cc_short", "vn_compound", "long")

def vn_group(tld: pd.Series) -> np.ndarray:
    return tld.astype(str).str.lower().str.endswith("vn").to_numpy()


def strata_of(frame: pd.DataFrame) -> pd.Series:
    """The four suffix strata used throughout P6, defined once so the fitting and audit scripts
    cannot silently balance or score different groups."""
    sl = frame["tld"].astype(str).str.lower().str.lstrip(".").str.len().to_numpy()
    vn = vn_group(frame["tld"])
    out = np.where(sl == 2,
                   np.where(vn, "vn_short", "cc_short"),
                   np.where(vn, "vn_compound", "long"))
    return pd.Series(out, index=frame.index)


def stratum_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal total mass for each observed (suffix stratum, class) cell."""
    groups = strata_of(frame).to_numpy()
    labels = frame["y"].to_numpy(int)
    weights = np.ones(len(frame), dtype=float)
    for group in STRATA:
        for label in (0, 1):
            cell = (groups == group) & (labels == label)
            if cell.sum():
                weights[cell] = len(frame) / (len(STRATA) * 2.0 * cell.sum())
    return weights


def split_and_fit(df, feats, family: str, seed: int, cal_frac: float):
    """The protocol, in one place. Extracted from `one_seed` (2026-08-17) so the budget-frontier
    sweep (run_p6_budget_frontier.py) reproduces this table's split, model and score columns by
    CALLING it rather than by re-implementing it: a frontier drawn on a differently-drawn benign
    mask would put the paper's marked operating points somewhere the frontier never visited.

    Returns (train frame, fitted model, calibration frame, test frame); the two frames carry
    `pred` (P(phish)) and `vn` (the .vn(+2LD) group flag)."""
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr_all, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr_all.rdom))]
    ccut = int(len(ph_tr_all) * (1 - cal_frac))
    ph_fit, ph_cal = ph_tr_all.iloc[:ccut], ph_tr_all.iloc[ccut:]  # calibration = newest train

    rng = np.random.RandomState(seed)
    r = rng.rand(len(be))
    be_fit = be[r < 0.70 * (1 - cal_frac)]
    be_cal = be[(r >= 0.70 * (1 - cal_frac)) & (r < 0.70)]
    be_te = be[r >= 0.70]

    tr = pd.concat([ph_fit, be_fit])
    m = make_any_model(family, seed)
    m.fit(tr[feats].to_numpy(float), tr["y"].to_numpy(int))

    cal = pd.concat([ph_cal, be_cal]).reset_index(drop=True)
    te = pd.concat([ph_te, be_te]).reset_index(drop=True)
    cal["pred"] = m.predict_proba(cal[feats].to_numpy(float))[:, 1]
    te["pred"] = m.predict_proba(te[feats].to_numpy(float))[:, 1]
    cal["vn"] = vn_group(cal["tld"])
    te["vn"] = vn_group(te["tld"])
    return tr, m, cal, te


def threshold_rules(cal_be: pd.DataFrame) -> tuple[dict, float]:
    """The four published decision rules, as (name -> {vn, other} threshold) plus the matched
    false-alarm budget alpha they are all built from. Extracted with `split_and_fit` so the
    frontier figure can MARK the paper's operating points without recomputing them."""
    alpha = float((cal_be.pred >= 0.5).mean())  # the 0.5 baseline's own false-alarm budget
    q = 1 - alpha

    def bq(scores: pd.Series) -> float:
        return float(scores.quantile(q)) if len(scores) else 0.5

    # "vn-only": give .vn pages the same operating point non-.vn pages already get at 0.5
    # (threshold at the quantile matching the OTHER group's realized calibration FPR), leave
    # everything else untouched — total FPR is allowed to rise; that rise is the price tag.
    fpr_other_cal = float((cal_be[~cal_be.vn].pred >= 0.5).mean())
    return {
        "default": {"vn": 0.5, "other": 0.5},
        "global": {"vn": bq(cal_be.pred), "other": bq(cal_be.pred)},
        "per-group": {"vn": bq(cal_be[cal_be.vn].pred), "other": bq(cal_be[~cal_be.vn].pred)},
        "vn-only": {"vn": float(cal_be[cal_be.vn].pred.quantile(1 - fpr_other_cal))
                    if len(cal_be[cal_be.vn]) else 0.5, "other": 0.5},
    }, alpha


def one_seed(df, feats, family: str, seed: int, cal_frac: float,
             return_scored: bool = False):
    tr, m, cal, te = split_and_fit(df, feats, family, seed, cal_frac)

    # STRATUM-BALANCED TRAINING — the only non-post-hoc remedy here. The old arm reweighted four
    # (.vn/non-.vn, class) cells, exactly the pooled grouping Section V-C shows to be invalid.
    # Reweight all eight (suffix stratum, class) cells instead, then threshold at the same matched
    # budget as `global` so the contrast isolates TRAINING from thresholding.
    # PREDICTION, recorded before the run: should NOT beat per-group thresholding at the operating
    # point (\S ssec:groupthr shows the residual is a budget constraint). Open question is the ROC
    # itself (baseline auc_vn 0.855): if it rises, a training-time fix exists; if not, the
    # constraint is intrinsic to the 21-feature space, strengthening the infrastructure-cascade case.
    ytr = tr["y"].to_numpy(int)
    w = stratum_class_weights(tr)
    mb = make_any_model(family, seed)
    mb.fit(tr[feats].to_numpy(float), ytr, sample_weight=w)

    cal["pred_bal"] = mb.predict_proba(cal[feats].to_numpy(float))[:, 1]
    te["pred_bal"] = mb.predict_proba(te[feats].to_numpy(float))[:, 1]

    cal_be = cal[cal.y == 0]
    thr, alpha = threshold_rules(cal_be)
    q = 1 - alpha

    def bq(scores: pd.Series) -> float:
        return float(scores.quantile(q)) if len(scores) else 0.5

    # GROUP-CONDITIONAL BENIGN QUANTILE MAPPING, not per-group calibration: calibration needs
    # positives per group and the .vn calibration slice holds 11 phishing vs 1,033 benign —
    # isotonic/Platt on 11 positives is overfitting. So map each group's BENIGN score
    # distribution onto a common reference, generalising the per-group threshold from one
    # operating point to the whole benign-side curve (what a cascade gate needs to compose).
    # PREDICTION, stated before the number is read: at the matched budget this lands on top of
    # the per-group rule (both place each group at the same benign quantile); it buys
    # composability, not a better operating point.
    def qmap(cal_scores: pd.Series, ref_scores: pd.Series, x: np.ndarray) -> np.ndarray:
        """Map x through the benign quantile function of its own group onto the reference."""
        if not len(cal_scores) or not len(ref_scores):
            return x
        u = np.searchsorted(np.sort(cal_scores.to_numpy()), x, side="right") / len(cal_scores)
        return np.quantile(ref_scores.to_numpy(), np.clip(u, 0.0, 1.0))

    ref_be = cal_be[~cal_be.vn]["pred"]              # non-.vn benign is the reference frame
    te_mapped = np.where(
        te.vn.to_numpy(),
        qmap(cal_be[cal_be.vn]["pred"], ref_be, te.pred.to_numpy()),
        te.pred.to_numpy())

    fpr_other_cal = float((cal_be[~cal_be.vn].pred >= 0.5).mean())
    # The mapped rule scores on te_mapped with ONE threshold for both groups (the property under
    # test). The threshold must come from calibration benign mapped the SAME way, per group: a
    # first version pushed pooled benign through the .vn quantile function alone, putting the
    # threshold far too high (FPR 0.024 against a 0.26 budget).
    cal_be_mapped = np.where(
        cal_be.vn.to_numpy(),
        qmap(cal_be[cal_be.vn]["pred"], ref_be, cal_be["pred"].to_numpy()),
        cal_be["pred"].to_numpy())
    thr_mapped = bq(pd.Series(cal_be_mapped))
    # The balanced model needs its own budget-matched threshold: its scores live on a different
    # scale, so reusing the baseline's would confound training with thresholding.
    alpha_bal = float((cal_be["pred_bal"] >= 0.5).mean())
    thr_bal = float(cal_be["pred_bal"].quantile(1 - alpha)) if len(cal_be) else 0.5

    # Per-group ROC area, carried on every row so the paper's "the .vn group ranks BETTER than
    # the group the model is not blind to" is a checkable number and not a reading of a figure.
    # Threshold-independent by construction, hence identical across the four conditions.
    from sklearn.metrics import roc_auc_score
    auc = {}
    for gkey, gmask in (("vn", te.vn.to_numpy()), ("other", ~te.vn.to_numpy())):
        sub = te[gmask]
        for col, suffix in (("pred", ""), ("pred_bal", "_bal")):
            auc[gkey + suffix] = (
                round(float(roc_auc_score(sub.y.to_numpy(int), sub[col].to_numpy(float))), 4)
                if sub.y.nunique() > 1 else float("nan"))
    te_strata = strata_of(te)
    for skey in STRATA:
        sub = te[(te_strata == skey).to_numpy()]
        for col, suffix in (("pred", ""), ("pred_bal", "_bal")):
            auc[skey + suffix] = (
                round(float(roc_auc_score(sub.y.to_numpy(int), sub[col].to_numpy(float))), 4)
                if sub.y.nunique() > 1 else float("nan"))

    rows = []
    for cond, t in list(thr.items()) + [("mapped", None), ("balanced", None)]:
        if cond == "mapped":
            scores, thr_vec = te_mapped, np.full(len(te), thr_mapped)
            t = {"vn": thr_mapped, "other": thr_mapped}
        elif cond == "balanced":
            scores = te["pred_bal"].to_numpy()
            thr_vec = np.full(len(te), thr_bal)
            t = {"vn": thr_bal, "other": thr_bal}
        else:
            scores, thr_vec = te.pred.to_numpy(), np.where(te.vn, t["vn"], t["other"])
        pred = (scores >= thr_vec).astype(int)
        y = te.y.to_numpy(int)
        vn_ph = te.vn.to_numpy() & (y == 1)
        ot_ph = ~te.vn.to_numpy() & (y == 1)
        vn_be = te.vn.to_numpy() & (y == 0)
        ot_be = ~te.vn.to_numpy() & (y == 0)
        row = {
            "seed": seed, "condition": cond, "alpha_cal": round(alpha, 4),
            "balance_cells": "suffix_stratum_x_class",
            "auc_vn": auc["vn"], "auc_other": auc["other"],
            "auc_vn_bal": auc["vn_bal"], "auc_other_bal": auc["other_bal"],
            "rec_vn": round(float((pred[te.vn.to_numpy() & (te.y.to_numpy() == 1)] == 1).mean()),
                            4),
            "rec_other": round(
                float((pred[~te.vn.to_numpy() & (te.y.to_numpy() == 1)] == 1).mean()), 4),
            "fpr_other_cal@0.5": round(fpr_other_cal, 4),
            "thr_vn": round(t["vn"], 4), "thr_other": round(t["other"], 4),
            "n_vn_phish": int(vn_ph.sum()),
            "fnr_vn": round(float((pred[vn_ph] == 0).mean()), 4),
            "fnr_other": round(float((pred[ot_ph] == 0).mean()), 4),
            "fpr": round(float((pred[y == 0] == 1).mean()), 4),
            "fpr_vn": round(float((pred[vn_be] == 1).mean()), 4),
            "fpr_other": round(float((pred[ot_be] == 1).mean()), 4),
            "f1": round(_f1(y, pred), 4),
        }
        for skey in STRATA:
            sm = (te_strata == skey).to_numpy()
            sph, sbe = sm & (y == 1), sm & (y == 0)
            row[f"auc_{skey}"] = auc[skey]
            row[f"auc_{skey}_bal"] = auc[skey + "_bal"]
            row[f"fnr_{skey}"] = (round(float((pred[sph] == 0).mean()), 4)
                                    if sph.sum() else float("nan"))
            row[f"fpr_{skey}"] = (round(float((pred[sbe] == 1).mean()), 4)
                                    if sbe.sum() else float("nan"))
        rows.append(row)
    # The ROC figure needs the scores the table was computed from, not a re-derivation: a second
    # split would be a second experiment, and the operating points it marked would not be the
    # ones the table reports.
    return (rows, te[["y", "vn", "pred"]].copy(), thr) if return_scored else rows


def _f1(y, pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, zero_division=0))


def make_figure(res, family: str, seeds: int):
    """.vn miss rate vs non-.vn miss rate, one point per decision rule, FPR in the label
    (a rule can look good on both miss rates and be paid for entirely in false alarms — vn-only
    is exactly that). This view surfaced that per-group's residual is a budget constraint, not a
    representational floor: the same scores reach FNR 0.010 once the threshold drops 0.083 -> 0.003."""
    from figstyle import apply, BLUE, ORANGE, TEAL, GRAY, INK
    plt = apply()

    order = ["default", "global", "per-group", "mapped", "vn-only"]
    label = {"default": "global $0.5$", "global": "global, matched budget",
             "per-group": "per-group, equal budgets", "vn-only": ".vn-only",
             "mapped": "benign quantile-mapped, one threshold"}
    # House palette by role (figstyle), not literals: a copied hex cannot follow figstyle.py, and
    # its docstring's rule -- no third saturated hue without re-validating -- only binds importers.
    # Marker shape varies alongside colour, so the conditions separate without it.
    style = {"default": (BLUE, "o"), "global": (BLUE, "s"),
             "per-group": (ORANGE, "^"), "vn-only": (TEAL, "D"),
             "mapped": (ORANGE, "v")}
    g = res.groupby("condition")
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for cond in order:
        if cond not in g.groups:
            continue
        r = g.get_group(cond)
        x, y = r.fnr_other.mean(), r.fnr_vn.mean()
        c, mk = style[cond]
        ax.errorbar(x, y, xerr=r.fnr_other.std(), yerr=r.fnr_vn.std(), fmt=mk, color=c,
                    markersize=7, capsize=2, elinewidth=0.8, zorder=3)
        # Leader line + hand-placed offset per rule: TWO pairs of rules coincide (the global pair
        # by construction; per-group with quantile mapping per Section 5.3), and offsetting only
        # the first pair left two orange labels printed through each other.
        dy = {"default": 6, "global": -22, "per-group": 8, "mapped": -24, "vn-only": 6}[cond]
        ax.annotate(f"{label[cond]}\nFPR {r.fpr.mean():.2f}, thr {r.thr_vn.mean():.3f}",
                    (x, y), textcoords="offset points",
                    xytext=(9, dy), fontsize=7.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.55,
                                    shrinkA=0, shrinkB=4))
    # plain text, not LaTeX — matplotlib renders a backslash literally here
    ax.set_xlabel("miss rate on non-.vn phishing")
    ax.set_ylabel("miss rate on .vn phishing")
    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(-0.06, 1.02)
    ax.axhline(0, color=INK, lw=0.5, ls=":")
    ax.annotate("ideal", (0, 0), textcoords="offset points", xytext=(4, -11),
                fontsize=7.5, style="italic", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{family}, phishing-temporal test, mean$\\pm$sd over {seeds} seeds", fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "papers", "P6_xai", "figures", "fig_group_threshold.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def make_roc_figure(scored: list, family: str):
    """Section 5.3's claim drawn instead of argued: ".vn FNR 0.906 at default, 0.010 once the
    threshold moves" is a statement about the ROC curve, which the paper never showed. One curve
    per TLD group (its phishing vs its own benign), the four decision rules marked where they land.
    Every seed is drawn and markers are seed means — the .vn group has only ~156 test positives,
    so one seed's curve is visibly rougher than the ensemble."""
    from figstyle import apply, BLUE, ORANGE, TEAL, GRAY, INK
    plt = apply()
    from sklearn.metrics import roc_auc_score, roc_curve

    label = {"default": "global $0.5$", "global": "global, matched",
             "per-group": "per-group", "vn-only": ".vn-only"}
    style = {"default": (BLUE, "o"), "global": (BLUE, "s"),
             "per-group": (ORANGE, "^"), "vn-only": (TEAL, "D")}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
    for ax, (name, key, want_vn) in zip(axes, [(".vn (+2LD)", "vn", True),
                                               ("non-.vn", "other", False)]):
        aucs, pts, npos = [], {}, []
        for te, thr in scored:
            mask = te.vn.to_numpy() if want_vn else ~te.vn.to_numpy()
            sub = te[mask]
            y, sc = sub.y.to_numpy(int), sub.pred.to_numpy(float)
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, color=INK, lw=1.0, alpha=0.45, zorder=2)
            aucs.append(roc_auc_score(y, sc))
            npos.append(int((y == 1).sum()))
            for cond, t in thr.items():
                pred = (sc >= t[key]).astype(int)
                pts.setdefault(cond, []).append(
                    (float((pred[y == 0] == 1).mean()), float((pred[y == 1] == 1).mean()),
                     t[key]))
        if not aucs:
            continue
        ax.plot([0, 1], [0, 1], ls=":", color=GRAY, lw=0.8, zorder=1)

        # Coinciding rules share a marker (in the non-.vn panel three do — .vn-only leaves that
        # group at 0.5 by construction); a shared operating point is a finding, not a collision.
        placed: dict = {}
        for cond, vals in pts.items():
            x = float(np.mean([v[0] for v in vals]))
            yv = float(np.mean([v[1] for v in vals]))
            tv = float(np.mean([v[2] for v in vals]))
            # 2 dp, not 3: default and matched-budget differ by 0.0015 in threshold; at 3 dp
            # they stayed separate keys and their labels drew on top of one another.
            placed.setdefault((round(x, 2), round(yv, 2)), []).append((cond, tv))
        for (x, yv), group in sorted(placed.items()):
            c, mk = style[group[0][0]]
            ax.plot(x, yv, mk, color=c, ms=7, mec="white", mew=0.8, zorder=4)
            names = " = ".join(label[c0] for c0, _ in group)
            thrs = ", ".join(sorted({f"{tv:.3f}" for _, tv in group}))
            # Opaque box: the chance diagonal runs behind these operating-point labels.
            ax.annotate(f"{names}\nthr {thrs}", (x, yv), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
                        xytext=(10, 8 if yv < 0.5 else -16), fontsize=7, color=c, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.55,
                                        shrinkA=0, shrinkB=4))
        # ddof=1: the prose quotes the sample sd (pandas .std()).
        ax.set_title(f"{name} — AUC {np.mean(aucs):.3f}$\\pm${np.std(aucs, ddof=1):.3f}  "
                     f"(n$_{{phish}}$={int(np.mean(npos))})", fontsize=8.5)
        ax.set_xlabel("false-positive rate (within group)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("recall on phishing (within group)")
    fig.suptitle(f"{family}, phishing-temporal test, {len(scored)} seeds", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(ROOT, "papers", "P6_xai", "figures", "fig_group_roc.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.15,
                    help="Share of the phishing TRAIN window (newest end) held out to calibrate")
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    rows = []
    scored = []
    for s in range(args.seeds):
        got, te_s, thr_s = one_seed(df, feats, args.family, s, args.cal_frac,
                                    return_scored=True)
        rows += got
        scored.append((te_s, thr_s))
        r = rows[-1]
        print(f"[i] seed {s}: alpha={r['alpha_cal']:.3f} thr_vn={rows[-1]['thr_vn']:.3f} | "
              + " ".join(f"{x['condition']}: fnr_vn={x['fnr_vn']:.3f} fpr={x['fpr']:.3f}"
                         for x in rows[-4:]))
    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if args.family != "CatBoost":
        # A non-canonical family (e.g. Stack[CB+LR]) must not clobber
        # the canonical CatBoost CSV/table/figures: it gets a suffixed CSV plus a generated
        # sentence, and the paper's main assets stay CatBoost's.
        import re as _re
        slug = _re.sub(r"\W+", "", args.family).lower()
        out_csv = OUT_CSV.replace(".csv", f"_{slug}.csv")
        res.to_csv(out_csv, index=False)
        print(f"[+] {out_csv}")
        d = res[res.condition == "default"]
        mp = res[res.condition == "mapped"]
        sent = (f"Under the identical protocol the stacked deployment pick "
                f"(Stack[CB{{+}}LR]) inherits the blind spot: default-threshold "
                f"\\texttt{{.vn}} FNR ${d.fnr_vn.mean():.3f}\\,\\pm\\,{d.fnr_vn.std():.3f}$ "
                f"and ${mp.fnr_vn.mean():.3f}\\,\\pm\\,{mp.fnr_vn.std():.3f}$ under the "
                f"benign-quantile-mapped single threshold (mean over {args.seeds} seeds); "
                f"the constraint sits in the corpus' benign budget, not in the family doing "
                f"the scoring.\n")
        write_generated(os.path.join(os.path.dirname(OUT_TEX), f"gen_groupthr_{slug}.tex"), sent)
        return
    res.to_csv(OUT_CSV, index=False)
    print(f"[+] {OUT_CSV}")
    # make_figure is kept for ad-hoc inspection; the plot was superseded by tab:groupthr.
    if scored:
        make_roc_figure(scored, args.family)

    label = {"default": "Global $0.5$ (baseline)",
             "global": "Global, matched FPR budget",
             "per-group": "Per-group, equal budgets",
             "vn-only": "\\texttt{.vn}-only (others' op.\\ point)",
             "mapped": "Benign quantile-mapped, one threshold"}
    agg = res.groupby("condition")
    n_vn = int(res[res.seed == 0].n_vn_phish.iloc[0])
    body = []
    stats = {}
    for cond in ("default", "global", "per-group", "mapped", "vn-only"):
        if cond not in agg.groups:
            continue
        g = agg.get_group(cond)
        fv, fo, fp, f1 = (g.fnr_vn.mean(), g.fnr_other.mean(), g.fpr.mean(), g.f1.mean())
        lo, hi = wilson(fv * n_vn, n_vn)
        stats[cond] = (fv, fo, fp, f1)
        body.append(f"{label[cond]} & {fv:.3f}{{\\scriptsize\\,$\\pm${g.fnr_vn.std():.3f}}} & "
                    f"{{\\scriptsize[{lo:.2f}, {hi:.2f}]}} & "
                    f"{fo:.3f}{{\\scriptsize\\,$\\pm${g.fnr_other.std():.3f}}} & "
                    f"{fp:.3f}{{\\scriptsize\\,$\\pm${g.fpr.std():.3f}}} & {f1:.3f} \\\\")
    # Caption ABOVE the tabular — the thesis-wide convention (see the caption sweep of 2026-08-17,
    # which fixed every generator it covered; this one sat outside that commit's scope).
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Group-conditional thresholding on the pooled \\texttt{{.vn(+2LD)}} grouping
({args.family}, phishing-temporal split, mean\\,$\\pm$\\,std over {args.seeds} seeds;
$n_{{\\text{{vn}}}}={n_vn}$ test rows).}}
\\label{{tab:groupthr}}
\\small\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l c c c c c}}
\\toprule
Decision rule & FNR .vn & {{\\scriptsize Wilson 95\\%}} & FNR other & FPR & F1 \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    write_generated(OUT_TEX, tex)
    print(f"[+] {OUT_TEX}")
    d, p = stats["default"], stats["per-group"]
    print(f"[+] FNR .vn: default {d[0]:.3f} -> per-group {p[0]:.3f} "
          f"(FPR {d[2]:.3f} -> {p[2]:.3f}, F1 {d[3]:.3f} -> {p[3]:.3f})")


if __name__ == "__main__":
    main()
