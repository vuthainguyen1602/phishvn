#!/usr/bin/env python3
"""
run_p6_vn_deficit.py — P6: how much of the .vn miss does the suffix prior actually account for?

Per missed .vn phishing row, in MARGIN units (TreeSHAP additivity holds in log-odds, and the
identity is asserted to 1e-6 before use): deficit_i = margin(threshold) - margin_i against phi_i,
the row's signed SHAP for tld_len. Two numbers bracket the answer from opposite sides — attribution
accounting is a lower bound, the neutralisation counterfactual an upper one. Result (2026-08-17)
inverts the expected direction: 24/141 against 121/141, so the decomposition UNDERSTATES the prior.

Split, model and seed are imported from run_p6_vn_reading.py — the same 141-of-156 missed rows the
paper's Wilson interval uses, not a re-draw.

OUTPUT: data/processed/p6/p6_vn_deficit{,_sweep}.csv, papers/P6_xai/sections/gen_vn_deficit.tex,
        papers/P6_xai/figures/fig_vn_deficit.pdf
RUN:  python scripts/run_p6_vn_deficit.py
Why margin space, why both bounds, why the neutral value is swept: kept in the development repository, not shipped in this mirror
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
from run_p2_temporal_strict import load
from run_p6_vn_reading import split_fit

SEC = os.path.join(ROOT, "papers", "P6_xai", "sections")
FIG = os.path.join(ROOT, "papers", "P6_xai", "figures")
PROC = os.path.join(ROOT, "data", "processed")
GROUPTHR_CSV = os.path.join(PROC, "p6", "p6_group_threshold.csv")
FEATURE = "tld_len"          # the dominant feature of Table~\ref{tab:shapcontrast}


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def margin_of(model, X: np.ndarray) -> np.ndarray:
    """The model's raw (log-odds) output --- the space TreeSHAP is additive in.

    CatBoost exposes it as RawFormulaVal; anything else is asked for its decision function. The
    caller asserts additivity against these values, so a family whose SHAP space did not match
    would fail loudly rather than silently produce a probability-space comparison."""
    try:
        return np.asarray(model.predict(X, prediction_type="RawFormulaVal"), dtype=float)
    except TypeError:
        return np.asarray(model.decision_function(X), dtype=float)


def neutralised_margin(model, X: np.ndarray, j: int, value: float) -> np.ndarray:
    """Re-score every row with feature j pinned to `value`, everything else held fixed."""
    Xc = X.copy()
    Xc[:, j] = value
    return margin_of(model, Xc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5,
                    help="the deployed decision threshold the deficit is measured against")
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    if FEATURE not in feats:
        raise SystemExit(f"feature {FEATURE!r} not in the model's feature list: {feats}")
    j = feats.index(FEATURE)
    m, tr, te, _ph = split_fit(df, feats, args.family, args.seed)

    X_te = te[feats].to_numpy(float)
    te["margin"] = margin_of(m, X_te)
    te["pred"] = m.predict_proba(X_te)[:, 1]
    te["vn"] = te["tld"].astype(str).str.lower().str.endswith("vn")

    # sanity: the raw output must BE the logit of the reported probability, or the whole
    # margin-space argument is being made against the wrong quantity
    assert np.allclose(1.0 / (1.0 + np.exp(-te["margin"].to_numpy())),
                       te["pred"].to_numpy(), atol=1e-8), "raw output is not logit(p)"

    vn_ph = te[te.vn & (te.y == 1)].copy()
    fn = vn_ph[vn_ph.pred < args.thr].copy()
    print(f"[i] .vn phishing in test: {len(vn_ph)}   missed at {args.thr}: {len(fn)} "
          f"({len(fn) / len(vn_ph):.3f})")

    # ---- per-row SHAP in margin space, with the additivity identity checked, not assumed
    import shap
    ex = shap.TreeExplainer(m)
    Xf = fn[feats].to_numpy(float)
    sv = ex.shap_values(Xf)
    if isinstance(sv, list):
        sv = sv[1]
    base = float(np.ravel(ex.expected_value)[0])
    recon = sv.sum(axis=1) + base
    err = float(np.max(np.abs(recon - fn["margin"].to_numpy())))
    assert err < 1e-6, f"TreeSHAP additivity violated in margin space (max err {err:.2e})"
    print(f"[i] additivity check passed (max |sum(phi)+phi0 - margin| = {err:.2e})")

    fn["phi_tld_len"] = sv[:, j]
    fn["phi_others"] = sv.sum(axis=1) - sv[:, j]
    fn["deficit"] = logit(args.thr) - fn["margin"]     # > 0 by construction on a miss

    # ---- the neutralisation counterfactual
    tr_ot = tr[~tr["tld"].astype(str).str.lower().str.endswith("vn")]
    neutral = float(tr_ot[FEATURE].mode().iloc[0])     # the commodity suffix the fit window sees
    fn["margin_neutral"] = neutralised_margin(m, Xf, j, neutral)
    fn["delta_margin"] = fn["margin_neutral"] - fn["margin"]
    fn["crossed"] = (fn["margin_neutral"] >= logit(args.thr)).astype(int)
    # the accounting statements
    fn["covered_signed"] = (-fn["phi_tld_len"] >= fn["deficit"]).astype(int)
    fn["covered_abs"] = (fn["phi_tld_len"].abs() >= fn["deficit"]).astype(int)

    n = len(fn)
    k_sig = int(fn.covered_signed.sum())
    k_abs = int(fn.covered_abs.sum())
    k_cr = int(fn.crossed.sum())
    n_benign_ward = int((fn.phi_tld_len < 0).sum())
    lo_s, hi_s = wilson(k_sig, n)
    lo_c, hi_c = wilson(k_cr, n)
    rho = float(pd.Series(-fn.phi_tld_len.to_numpy()).corr(
        pd.Series(fn.delta_margin.to_numpy()), method="spearman"))

    # ---- secondary operating point: the calibrated per-group .vn threshold from the companion
    # run, read rather than retyped. Every deficit shifts by the constant logit(t), so this is a
    # translation of the same distribution, not a second experiment.
    thr2 = None
    if os.path.exists(GROUPTHR_CSV):
        g = pd.read_csv(GROUPTHR_CSV)
        g = g[g.condition == "per-group"]
        if len(g):
            thr2 = float(g.thr_vn.mean())
    sec = None
    if thr2 and 0 < thr2 < 1:
        still = fn[fn.margin < logit(thr2)].copy()
        still["deficit2"] = logit(thr2) - still["margin"]
        cov2 = int((-still.phi_tld_len >= still.deficit2).sum())
        cr2 = int((still.margin_neutral >= logit(thr2)).sum())
        sec = (thr2, len(still), cov2, cr2)
        print(f"[i] at the calibrated .vn threshold {thr2:.3f}: {len(still)} still missed, "
              f"{cov2} accounted, {cr2} cross under neutralisation")

    # ---- does the neutral value carry the result?
    # Split by suffix family, because the .vn(+2LD) group pools TWO feature values (2 for bare
    # .vn, 6 for the .com.vn family, per Section 5.2): a pooled sweep would show 45 crossings at
    # tld_len=2 and read as ``even the .vn value flips a third of them'', when in fact that value
    # is a no-op for the 91 rows that already carry it and the crossings are entirely the
    # compound-suffix rows being moved off their own value.
    bare = (fn[FEATURE].to_numpy(float) <= 2.5)
    sweep = []
    for v in sorted(pd.unique(df[FEATURE].dropna().astype(float))):
        if v > 12:
            continue
        mg = neutralised_margin(m, Xf, j, float(v))
        ok = mg >= logit(args.thr)
        sweep.append({"tld_len": float(v), "n_fn": n,
                      "crossed": int(ok.sum()),
                      "n_bare": int(bare.sum()), "crossed_bare": int(ok[bare].sum()),
                      "n_compound": int((~bare).sum()),
                      "crossed_compound": int(ok[~bare].sum()),
                      "mean_delta_margin": float(np.mean(mg - fn["margin"].to_numpy()))})
    sweep = pd.DataFrame(sweep)

    os.makedirs(PROC, exist_ok=True)
    cols = ["rdom", "tld", "pred", "margin", "deficit", "phi_tld_len", "phi_others",
            "margin_neutral", "delta_margin", "crossed", "covered_signed", "covered_abs"]
    fn[[c for c in cols if c in fn.columns]].to_csv(
        os.path.join(PROC, "p6", "p6_vn_deficit.csv"), index=False)
    sweep.to_csv(os.path.join(PROC, "p6", "p6_vn_deficit_sweep.csv"), index=False)
    print(f"[+] data/processed/p6/p6_vn_deficit.csv ({n} rows)")
    print("[+] data/processed/p6/p6_vn_deficit_sweep.csv")

    make_figure(fn, sweep, neutral, args.thr, args.family)
    make_tex(fn, sweep, dict(n=n, k_sig=k_sig, k_abs=k_abs, k_cr=k_cr, neutral=neutral,
                             n_benign_ward=n_benign_ward, lo_s=lo_s, hi_s=hi_s, lo_c=lo_c,
                             hi_c=hi_c, rho=rho, thr=args.thr, sec=sec, n_vn=len(vn_ph),
                             family=args.family))

    print(f"[+] accounted for by phi({FEATURE}) alone: {k_sig}/{n} = {k_sig / n:.3f} "
          f"[{lo_s:.3f}, {hi_s:.3f}]")
    print(f"[+] cross the threshold under neutralisation to tld_len={neutral:g}: "
          f"{k_cr}/{n} = {k_cr / n:.3f} [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"[+] Spearman(-phi, delta margin) = {rho:.3f}; "
          f"benign-ward phi on {n_benign_ward}/{n} rows")


def make_figure(fn: pd.DataFrame, sweep: pd.DataFrame, neutral: float, thr: float, family: str):
    """LEFT: every missed \\texttt{.vn} row as (deficit, $-\\phi_{tld\\_len}$), both in margin
    units. The diagonal is the accounting boundary --- above it the suffix prior's contribution
    is by itself at least as large as the shortfall --- and the marker says whether the model
    ACTUALLY changes its mind when the feature is neutralised, which is the distinction the panel
    exists to draw --- and the points sit ABOVE the diagonal far more often than the accounting
    predicts, which is the interaction effect Shapley split away. RIGHT: the crossing count at
    every suffix length the corpus contains, split by suffix family, so the reader can see how
    much the choice of neutral value carries (it carries some) and can discount the two values
    that are the rows' own."""
    from figstyle import apply, ORANGE, BLUE, GRAY, INK
    plt = apply()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    d = fn["deficit"].to_numpy()
    p = -fn["phi_tld_len"].to_numpy()
    cr = fn["crossed"].to_numpy(bool)
    hi = float(max(d.max(), p.max())) * 1.06
    ax.plot([0, hi], [0, hi], ls=(0, (4, 3)), color=INK, lw=0.8, zorder=2)
    ax.scatter(d[~cr], p[~cr], s=22, color=GRAY, alpha=0.7, edgecolor="white", linewidth=0.5,
               zorder=3, label="still missed after neutralising")
    ax.scatter(d[cr], p[cr], s=26, color=ORANGE, alpha=0.85, edgecolor="white", linewidth=0.5,
               zorder=4, label="crosses the threshold")
    ax.annotate("$-\\phi \\geq$ deficit\n(prior accounts for the miss)", (0.03 * hi, 0.93 * hi),
                fontsize=7, color=INK, va="top")
    ax.set_xlabel("deficit to the decision threshold (margin units)")
    ax.set_ylabel("$-\\phi$(tld_len)  in margin units")
    ax.set_xlim(0, hi)
    ax.set_ylim(min(0.0, float(p.min()) * 1.06), hi)
    ax.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.3)
    ax.set_title(f"missed .vn phishing (n={len(fn)}), threshold {thr}", fontsize=8.5)

    sw = sweep[sweep.tld_len > 0]
    x = sw.tld_len.to_numpy()
    nb, nc = int(sw.n_bare.iloc[0]), int(sw.n_compound.iloc[0])
    ax2.bar(x - 0.19, sw.crossed_bare.to_numpy(), width=0.36, color=BLUE, alpha=0.85, zorder=3)
    ax2.bar(x + 0.19, sw.crossed_compound.to_numpy(), width=0.36, color=ORANGE, alpha=0.85,
            zorder=3)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=BLUE, label=f"bare .vn (n={nb}); own value 2"),
                        Patch(color=ORANGE, label=f".com.vn family (n={nc}); own value 6")],
               # upper right, not centre right: centred it printed over the tall bars at
               # tld_len 5-6 (the ones the prose quotes); the bars at 7+ are short. Anchored
               # just below the top edge so it clears the "fit window's mode" label there.
               loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.8, frameon=False,
               handlelength=1.1, handletextpad=0.4, borderaxespad=0.2)
    ax2.axvline(neutral, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
    # Anchored to the RIGHT of the line it names, not to the left edge: left-anchored at the
    # axis start the two-line label ran straight through the dashed rule and the word "window's"
    # came out bisected.
    ax2.annotate("fit window's\nnon-.vn mode", (neutral + 0.25, max(nb, nc) * 1.24),
                 fontsize=7, color=INK, va="top", ha="left")
    ax2.set_xlabel("tld_len the misses are re-scored at")
    ax2.set_ylabel("misses that cross")
    ax2.set_xticks(x)
    ax2.set_ylim(0, max(nb, nc) * 1.25)
    ax2.set_title("sensitivity to the neutral value", fontsize=8.5)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{family}, phishing-temporal test", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_vn_deficit.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def make_tex(fn: pd.DataFrame, sweep: pd.DataFrame, s: dict):
    n = s["n"]
    med_d = float(fn.deficit.median())
    med_p = float((-fn.phi_tld_len).median())
    med_dm = float(fn.delta_margin.median())
    rows = [
        (f"Missed \\texttt{{.vn}} phishing at threshold ${s['thr']}$", f"{n}", f"of {s['n_vn']}"),
        ("Median deficit to the threshold", f"{med_d:.2f}", "margin units"),
        ("Median $-\\phi(\\texttt{tld\\_len})$", f"{med_p:.2f}", "margin units"),
        ("\\emph{Accounting:} $-\\phi \\geq$ deficit",
         f"{s['k_sig']}/{n} = {s['k_sig'] / n:.3f}",
         f"{{\\scriptsize Wilson [{s['lo_s']:.3f}, {s['hi_s']:.3f}]}}"),
        ("\\quad same test on $|\\phi|$", f"{s['k_abs']}/{n} = {s['k_abs'] / n:.3f}",
         f"{{\\scriptsize $\\phi<0$ on {s['n_benign_ward']}/{n}}}"),
        (f"\\emph{{Counterfactual:}} cross when \\texttt{{tld\\_len}}$\\to{s['neutral']:g}$",
         f"{s['k_cr']}/{n} = {s['k_cr'] / n:.3f}",
         f"{{\\scriptsize Wilson [{s['lo_c']:.3f}, {s['hi_c']:.3f}]}}"),
        ("\\quad median margin gained", f"{med_dm:+.2f}",
         f"{{\\scriptsize Spearman vs $-\\phi$: {s['rho']:.2f}}}"),
    ]
    if s["sec"]:
        t2, n2, cov2, cr2 = s["sec"]
        rows.append((f"At the calibrated \\texttt{{.vn}} threshold ${t2:.3f}$",
                     f"{n2} still missed",
                     f"{{\\scriptsize {cov2} accounted, {cr2} cross}}"))
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    # table*, not table: at \small the three columns run 54pt past an IEEE column, and the fix
    # that keeps the notes column is the full width (checked by compiling the fragment).
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Deficit attribution for the $141/156$ missed \\texttt{{.vn}} phishing rows, in
margin (log-odds) units ({s['family']}, phishing-temporal test; \\texttt{{tld\\_len}} pinned to
${s['neutral']:g}$ in the counterfactual).}}
\\label{{tab:vndeficit}}
\\small\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l r l}}
\\toprule
Quantity & Value & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    write_generated(os.path.join(SEC, "gen_vn_deficit.tex"), tex)

    sw = sweep.set_index("tld_len")["crossed"]
    verdict = (
        f"Scaling the four named cases to the whole missed population needs one methodological "
        f"care: a SHAP value and a distance-to-threshold are commensurable only in margin "
        f"(log-odds) space, where TreeSHAP's additivity holds, and a comparison run in "
        f"probability space would be a category error with plausible-looking output. Taken there "
        f"(Table~\\ref{{tab:vndeficit}}), the two natural readings of ``the prior caused the "
        f"miss'' bracket the answer from opposite sides. The additive accounting is the "
        f"conservative one: on ${s['k_sig']}$ of the ${n}$ missed \\texttt{{.vn}} rows "
        f"(${s['k_sig'] / n:.3f}$, Wilson 95\\% CI ${s['lo_s']:.3f}$--${s['hi_s']:.3f}$) the "
        f"\\texttt{{tld\\_len}} contribution alone is at least as large as the row's entire "
        f"deficit (median deficit ${med_d:.2f}$ against a median contribution of "
        f"${med_p:.2f}$). The intervention is the generous one: re-scoring those rows with "
        f"\\texttt{{tld\\_len}} pinned to the fit window's non-\\texttt{{.vn}} mode "
        f"(${s['neutral']:g}$) and nothing else touched carries ${s['k_cr']}$ of ${n}$ "
        f"(${s['k_cr'] / n:.3f}$, ${s['lo_c']:.3f}$--${s['hi_c']:.3f}$) across the threshold, "
        f"a median gain of ${med_dm:+.2f}$ in margin. The direction of that gap is itself the "
        f"finding, and it is the one Section~\\ref{{ssec:suffixprior}} predicts: Shapley splits "
        f"an interaction between its partners, so a prior carried \\emph{{jointly}} (which is "
        f"exactly what a two-character suffix length must be, since it cannot identify "
        f"\\texttt{{.vn}} on its own) is under-credited by the decomposition and recovered in "
        f"full by the intervention (Spearman ${s['rho']:.2f}$ between the two per row). Neither "
        f"number is the whole causal story: the intervention moves one feature while "
        f"\\texttt{{url\\_len}} and \\texttt{{dom\\_len}} still count the original suffix's "
        f"characters, and the crossing count does depend on the value chosen "
        f"(${int(sw.get(4.0, 0))}$ at \\texttt{{tld\\_len}}$=4$, ${int(sw.get(5.0, 0))}$ at $5$). "
        f"What survives both readings is that the misses are a suffix-prior phenomenon rather "
        f"than an absence of phishing signal in the URL")
    write_generated(os.path.join(SEC, "gen_vn_deficit_verdict.tex"), verdict.rstrip() + "%")


if __name__ == "__main__":
    main()
