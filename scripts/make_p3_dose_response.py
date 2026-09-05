#!/usr/bin/env python3
r"""
make_p3_dose_response.py — P3-A1': the paraphrase attack as a DOSE, not a label.

Estimates miss(J), the probability that a detector misses a held-out phishing message whose
test-role rewrite sits at token Jaccard J from its source, by isotonic regression over the
UNCONTROLLED corpus (the only one with support), with the band study overlaid as an out-of-sample
point. Sources, not rows, are the bootstrap unit. The evaluation loop is the predecessor's,
replicated row-by-row, and the guard in check_paper_claims.py recomputes the A2 cell means from
this CSV to prove it.

Emits data/processed/p3/p3_dose_response.csv,
      papers/P3_multimodal/figures/fig_dose_response.pdf,
      papers/P3_multimodal/sections/gen_dose_response.tex.

RUN:  python scripts/make_p3_dose_response.py
Why isotonic, why that corpus, why cluster bootstrap: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
FIG = os.path.join(ROOT, "papers", "P3_multimodal", "figures", "fig_dose_response.pdf")
DOSE_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_dose_response.csv")
CELLS_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_cells.csv")
PRE_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase.csv")
BAND_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_band.csv")

from make_p3_paraphrase_assets import SEEDS, vec
from p3_jaccard_check import BAND, jaccard, sources
from genfile import write_generated
from train_fusion import perturb

DETECTORS = ("D0", "D1", "D2")
DET_LABEL = {"D0": "D0 naive",
             "D1": "D1 adv. trained on char. obfuscation",
             "D2": "D2 adv. trained on paraphrases"}
BOOT = 600            # cluster-bootstrap replicates; the guard re-runs this, so it is not free
BOOT_SEED = 20260817  # fixed so the printed interval is reproducible by the claims suite
GRID = 201
CI = (2.5, 97.5)


# ---------------------------------------------------------------- evaluation

def per_rewrite(df: pd.DataFrame) -> pd.DataFrame:
    """The predecessor's run() loop with per-rewrite logging on condition A2.

    A0 (clean) and A1 (character-obfuscated) carry no J — the paraphrase is not involved — so
    only A2 is logged. That is also the only condition all three detectors are scored on, which
    is why the figure has exactly three curves.
    """
    txt, y = df["text"].to_numpy(), df["y"].to_numpy()
    pa, pb = df["para_a"].to_numpy(), df["para_b"].to_numpy()
    ids = df["id"].to_numpy()
    rows = []

    for s in range(SEEDS):
        rng = random.Random(s)
        tr, te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=s)
        ytr, yte = y[tr], y[te]
        ph_te = te[yte == 1]

        # Drawn before the D1 augmentation, exactly as in run(): the shared rng makes the
        # consumption ORDER part of the definition of D1.
        clean_ph = txt[ph_te]
        for t in clean_ph:
            perturb(t, rng)
        para_ph = pb[ph_te]                        # variant 'b' — test role only

        v0 = vec()
        clf0 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v0.fit_transform(txt[tr]), ytr)
        tr_ph_txt = txt[tr][ytr == 1]
        aug1 = list(txt[tr]) + [perturb(t, rng) for t in tr_ph_txt for _ in range(2)]
        y1 = list(ytr) + [1] * (2 * len(tr_ph_txt))
        v1 = vec()
        clf1 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v1.fit_transform(aug1), y1)
        aug2 = list(txt[tr]) + [t for t in pa[tr][ytr == 1] if t]
        y2 = list(ytr) + [1] * int(sum(1 for t in pa[tr][ytr == 1] if t))
        v2 = vec()
        clf2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v2.fit_transform(aug2), y2)

        for name, clf, v in (("D0", clf0, v0), ("D1", clf1, v1), ("D2", clf2, v2)):
            pred = clf.predict(v.transform(para_ph))
            for i, src_id in enumerate(ids[ph_te]):
                rows.append({"seed": s, "src_id": src_id, "detector": name,
                             "missed": int(pred[i] != 1)})
    return pd.DataFrame(rows)


def aggregate(log: pd.DataFrame, jmap: dict[str, float]) -> pd.DataFrame:
    """Collapse to the independent unit: one row per (detector, source)."""
    g = (log.groupby(["detector", "src_id"], as_index=False)
            .agg(n_obs=("missed", "size"), n_missed=("missed", "sum")))
    g["miss_rate"] = g["n_missed"] / g["n_obs"]
    g["jaccard"] = g["src_id"].map(jmap)
    return g.dropna(subset=["jaccard"]).sort_values(["detector", "jaccard"]).reset_index(drop=True)


# ---------------------------------------------------------------- the curve

def direction(sub: pd.DataFrame) -> int:
    """+1 if miss rises with J, -1 if it falls. Read off the data, never assumed.

    Spearman over SOURCES (not rows): the question is whether gentler rewrites are caught more
    often, and each source contributes one rewrite and therefore one J.
    """
    from scipy import stats
    if sub["jaccard"].nunique() < 2 or sub["miss_rate"].nunique() < 2:
        return 0
    rho = float(stats.spearmanr(sub["jaccard"], sub["miss_rate"]).statistic)
    return 1 if rho > 0 else (-1 if rho < 0 else 0)


def spearman(sub: pd.DataFrame) -> tuple[float, float]:
    from scipy import stats
    r = stats.spearmanr(sub["jaccard"], sub["miss_rate"])
    return float(r.statistic), float(r.pvalue)


def fit(sub: pd.DataFrame, increasing: bool) -> IsotonicRegression:
    """Weighted isotonic fit. Weights are the observation counts behind each source's miss rate,
    so a source seen in 8 of the 20 splits is not given the same say as one seen in 4."""
    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    iso.fit(sub["jaccard"].to_numpy(float), sub["miss_rate"].to_numpy(float),
            sample_weight=sub["n_obs"].to_numpy(float))
    return iso


def cluster_bootstrap(sub: pd.DataFrame, grid: np.ndarray, increasing: bool,
                      b: int = BOOT, seed: int = BOOT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Resample SOURCES with replacement, refit, and take percentiles of the fitted curve.

    Rows of `sub` are already one-per-source, so a row bootstrap here IS a cluster bootstrap —
    the clustering was done by aggregate(). The distinction still matters and is tested: a
    bootstrap over the un-aggregated (seed, source) log would resample within clusters and shrink
    the interval by roughly sqrt(observations per source).
    """
    rng = np.random.default_rng(seed)
    n = len(sub)
    if n < 2:
        return np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    j = sub["jaccard"].to_numpy(float)
    m = sub["miss_rate"].to_numpy(float)
    w = sub["n_obs"].to_numpy(float)
    draws = np.empty((b, len(grid)))
    for k in range(b):
        idx = rng.integers(0, n, n)
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(j[idx], m[idx], sample_weight=w[idx])
        draws[k] = iso.predict(grid)
    return np.percentile(draws, CI[0], axis=0), np.percentile(draws, CI[1], axis=0)


def band_slope(iso: IsotonicRegression, lo: float, hi: float) -> float:
    """Mean absolute slope of the fitted curve across [lo, hi], in miss-rate units per unit J."""
    a, b = iso.predict(np.array([lo, hi], dtype=float))
    return abs(float(b - a)) / (hi - lo)


def steepness(iso: IsotonicRegression, jmin: float, jmax: float,
              band: tuple[float, float]) -> dict:
    """How much steeper is the curve inside the registered band than across its whole support?

    A ratio above 1 is the figure's claim: the band was placed where the attack's strength
    actually buys something, so the successor study is not measuring a flat stretch of curve.
    Below 1 would be a finding against the band's placement, and the guard checks the printed
    sentence against the sign either way.
    """
    lo, hi = band
    s_band = band_slope(iso, lo, hi)
    s_all = band_slope(iso, jmin, jmax)
    return {"slope_band": s_band, "slope_all": s_all,
            "ratio": (s_band / s_all) if s_all else float("nan"),
            "drop_band": float(np.diff(iso.predict(np.array([lo, hi])))[0]),
            "drop_all": float(np.diff(iso.predict(np.array([jmin, jmax])))[0])}


def band_steeper_fraction(sub: pd.DataFrame, jmin: float, jmax: float,
                          band: tuple[float, float], increasing: bool,
                          b: int = BOOT, seed: int = BOOT_SEED) -> float:
    """Share of cluster-bootstrap replicates in which the band's slope beats the average slope.

    The point-estimate ratio is one number off one isotonic fit, and an isotonic fit on a weak
    monotone signal is a step function whose steps land where they land. Whether the band's
    placement survives resampling the sources is the checkable version of the claim, and it is
    the number the paragraph leans on.
    """
    rng = np.random.default_rng(seed + 1)
    j = sub["jaccard"].to_numpy(float)
    m = sub["miss_rate"].to_numpy(float)
    w = sub["n_obs"].to_numpy(float)
    n, wins = len(sub), 0
    for _ in range(b):
        idx = rng.integers(0, n, n)
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(j[idx], m[idx], sample_weight=w[idx])
        wins += band_slope(iso, *band) > band_slope(iso, jmin, jmax)
    return wins / b


def point_ci(sub: pd.DataFrame, at: float, increasing: bool,
             b: int = BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Fitted miss rate at one J, with its cluster-bootstrap interval."""
    iso = fit(sub, increasing)
    est = float(iso.predict(np.array([at]))[0])
    lo, hi = cluster_bootstrap(sub, np.array([at]), increasing, b, seed)
    return est, float(lo[0]), float(hi[0])


def mean_ci(sub: pd.DataFrame, b: int = BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Weighted mean miss rate over sources, with a cluster bootstrap interval. Used for the
    band corpus, where J has no spread and only a level is estimable."""
    rng = np.random.default_rng(seed)
    m = sub["miss_rate"].to_numpy(float)
    w = sub["n_obs"].to_numpy(float)
    est = float(np.average(m, weights=w))
    n = len(sub)
    draws = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, n, n)
        draws[k] = np.average(m[idx], weights=w[idx])
    return est, float(np.percentile(draws, CI[0])), float(np.percentile(draws, CI[1]))


# ---------------------------------------------------------------- inputs

def predecessor_jaccard() -> dict[str, float]:
    """J between each source lure and its TEST-role rewrite, recomputed from the source texts
    with the registered tokenisation — the same quantity, on the same denominator, as the
    band-control figure and the published lexical-shift numbers."""
    src = sources()
    pre = pd.read_csv(PRE_CSV)
    pre = pre[pre["variant"] == "b"] if "variant" in pre.columns else pre[pre["role"] == "test"]
    return {r.src_id: float(jaccard(src[r.src_id], r.text))
            for r in pre.itertuples() if r.src_id in src}


def strata() -> dict[str, tuple[float, int]]:
    """Mean test-role J for the pilot batch and the 2026-08-07 extension, separately.

    This exists because a review flagged an unexplained gap: the predecessor CSV averages
    J~0.409 over all 386 rewrites, while the commit that introduced the extension reports 0.28
    for the pilot stratum and 0.41 for the extension. Both are true and they are different
    denominators — 0.409 pools the TRAIN-role variant 'a' (a much gentler rewrite) with the
    test-role variant 'b' that the attack is actually scored on. Splitting by stratum here, on
    test-role rows only, reproduces 0.28/0.41 and closes the question; it is also the split the
    dose-response curve turns into a quantity, since the two batches sit at two doses.
    """
    jm = predecessor_jaccard()
    try:
        from p3_paraphrase_corpus import PARA as PILOT_IDS
    except Exception:
        return {}
    out = {}
    for name, keep in (("pilot", lambda i: i in PILOT_IDS),
                       ("extension", lambda i: i not in PILOT_IDS)):
        vals = [v for i, v in jm.items() if keep(i)]
        out[name] = ((float(np.mean(vals)) if vals else float("nan")), len(vals))
    return out


# ---------------------------------------------------------------- figure

def figure(agg: pd.DataFrame, incr: dict[str, bool], jm: dict[str, float],
           band_pt: dict[str, tuple[float, float, float]], band_j: float) -> None:
    from figstyle import apply, ORANGE, BLUE, TEAL, GRAY, INK
    plt = apply()

    lo, hi = BAND
    jmin, jmax = min(jm.values()), max(jm.values())
    grid = np.linspace(jmin, jmax, GRID)
    colour = {"D0": ORANGE, "D1": BLUE, "D2": TEAL}

    # Small multiples, one detector per panel, redrawn 2026-08-29. The three curves used to share
    # one axes: three step functions and three translucent bootstrap bands over the same 3.3in
    # column, which overlapped into a single grey mass and needed a four-entry legend to decode.
    # Stacking them removes the overlap, removes the legend, and shows the claim the paragraph
    # actually makes -- that the shape repeats across detectors -- as a repeated shape.
    #
    # The y axis is SHARED. Per-panel scaling would make D2's curve look like the other two, and
    # it is not: its miss rate is small everywhere, which is the result rather than a nuisance.
    fig, axes = plt.subplots(len(DETECTORS), 1, figsize=(3.3, 3.9), sharex=True, sharey=True)

    ymax = 0.0
    for k, (ax, d) in enumerate(zip(axes, DETECTORS)):
        sub = agg[agg.detector == d]
        iso = fit(sub, incr[d])
        yhat = iso.predict(grid)
        blo, bhi = cluster_bootstrap(sub, grid, incr[d])
        ax.axvspan(lo, hi, color=GRAY, alpha=0.22, zorder=0)
        ax.fill_between(grid, blo * 100, bhi * 100, color=colour[d], alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(grid, yhat * 100, color=colour[d], lw=1.6, zorder=3)
        ymax = max(ymax, float(np.nanmax(bhi)) * 100)
        # The band study's own measurement at its own J, on the curve fitted from the other
        # study's rewrites. A marker and not a curve: that corpus holds J inside the band by
        # construction, so it has no spread to fit against.
        if d in band_pt:
            est, plo, phi = band_pt[d]
            ax.errorbar([band_j], [est * 100],
                        yerr=[[(est - plo) * 100], [(phi - est) * 100]],
                        fmt="D", ms=4.0, color=colour[d], mfc="white", mew=1.2,
                        elinewidth=1.0, capsize=2.2, zorder=4)
            ymax = max(ymax, phi * 100)
        # The panel names itself, which is what the legend used to do from outside the axes.
        ax.annotate(DET_LABEL[d], (0.985, 0.90), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7, color=INK, zorder=5)
        ax.grid(axis="y", zorder=1)
        if k == 0:
            ax.annotate(f"registered band [{lo:.2f}, {hi:.2f}]", ((lo + hi) / 2, 1.04),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=7, color=INK, annotation_clip=False)

    axes[-1].set_xlabel("token Jaccard $J$ (low $J$ = stronger)")
    axes[-1].set_xlim(jmin - 0.01, jmax + 0.01)
    axes[0].set_ylim(0, ymax * 1.08)
    fig.supylabel("miss rate on held-out phishing (%)", fontsize=7.5)

    # subplots_adjust rather than tight_layout: the band annotation sits above the top axes, and
    # tight_layout does not account for artists placed outside them -- that is what cut the axis
    # labels off the saved bbox when the legend hung below the single-panel version.
    fig.subplots_adjust(left=0.19, right=0.98, top=0.91, bottom=0.11, hspace=0.16)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] {FIG}")


# ---------------------------------------------------------------- prose

def tex_int(n: int) -> str:
    """Digit grouping for a single number. Applied to the number, NEVER to a whole sentence:
    a str.replace over prose once turned every comma in a paragraph into a math-mode thin space."""
    return f"{int(n):,}".replace(",", "{,}")


def main() -> None:
    from make_p3_paraphrase_assets import load as load_pre
    df = load_pre()
    jm = predecessor_jaccard()
    missing = set(df.loc[df.y == 1, "id"]) - set(jm)
    if missing:
        raise SystemExit(f"[!] {len(missing)} phishing sources have no recomputed J.")

    log = per_rewrite(df)
    agg = aggregate(log, jm)
    agg.to_csv(DOSE_CSV, index=False)
    print(f"[+] {DOSE_CSV}  ({len(log)} per-rewrite observations -> {len(agg)} source rows)")

    incr, rho = {}, {}
    for d in DETECTORS:
        sub = agg[agg.detector == d]
        incr[d] = direction(sub) > 0
        rho[d] = spearman(sub)

    lo, hi = BAND
    jmin, jmax = min(jm.values()), max(jm.values())
    st = {d: steepness(fit(agg[agg.detector == d], incr[d]), jmin, jmax, BAND)
          for d in DETECTORS}

    # The band corpus: same loop, same seeds, a level rather than a curve.
    band_pt: dict[str, tuple[float, float, float]] = {}
    band_j = float("nan")
    band_agg = None
    shared = 0
    if os.path.exists(BAND_CSV):
        from make_p3_band_assets import load as load_band
        bdf, bpara = load_band()
        bj = {r.src_id: float(r.jaccard) for r in bpara[bpara.variant == "b"].itertuples()}
        band_agg = aggregate(per_rewrite(bdf), bj)
        b0 = band_agg[band_agg.detector == "D0"]
        # The band's achieved J is the number make_p3_band_assets.py prints as "achieved mean":
        # the unweighted mean over every rewrite in the band CSV (both roles). Weighting the
        # test-role J by observation count gave 0.248 here against the band table's 0.247, two
        # roundings of one quantity in one paper; the figure and this note now take the band
        # study's own value rather than recompute a neighbour of it.
        band_j = float(bpara["jaccard"].mean())
        for d in DETECTORS:
            band_pt[d] = mean_ci(band_agg[band_agg.detector == d])
        # How much of the two corpora is the same material. The note used to call the plotted point
        # "independent"; it is independent of the FIT, whose rewrites are all from the other study,
        # but every interval in the figure is a cluster bootstrap over SOURCES and the two studies
        # draw on the same ones. Counted here rather than asserted.
        shared = len(set(b0.src_id) & set(agg[agg.detector == "D0"].src_id))

    figure(agg, incr, jm, band_pt, band_j)

    # --- the numbers the paragraph is built from, all recomputed here -------------------
    d0 = agg[agg.detector == "D0"]
    iso0 = fit(d0, incr["D0"])
    at_lo = float(iso0.predict(np.array([lo]))[0])
    at_hi = float(iso0.predict(np.array([hi]))[0])
    mid = (lo + hi) / 2
    mid_est, mid_lo, mid_hi = point_ci(d0, mid, incr["D0"])
    # Quoted at the J deciles, not at the extremes. Isotonic pins its boundary blocks to whatever
    # single source sits there, and the gentlest rewrite in this corpus happens to give a fitted
    # 0.0% with a degenerate [0, 0] bootstrap interval — a number that reads as a measurement and
    # is an artefact of the estimator's edge.
    q_lo, q_hi = (float(np.quantile(d0["jaccard"], 0.10)),
                  float(np.quantile(d0["jaccard"], 0.90)))
    at_q_lo, at_q_hi = (float(v) for v in iso0.predict(np.array([q_lo, q_hi])))
    steep_frac = band_steeper_fraction(d0, jmin, jmax, BAND, incr["D0"])
    pooled = float(np.average(d0["miss_rate"], weights=d0["n_obs"]))
    n_obs = int(len(log))
    n_src = int(d0["src_id"].nunique())

    # Strata by the batch that wrote them, on test-role rows: the review's open question. The
    # curve-implied level is the fit AVERAGED OVER each stratum's J distribution, not the fit
    # evaluated at the stratum's mean J — the curve is not linear, so those differ, and on this
    # fit the two mean-J values happen to land on the same isotonic block while the two
    # distributions do not.
    strat = strata()
    j_pilot = strat.get("pilot", (float("nan"), 0))[0]
    j_ext = strat.get("extension", (float("nan"), 0))[0]
    stratum: dict[str, dict] = {}
    try:
        from p3_paraphrase_corpus import PARA as PILOT_IDS
        d0s = d0.assign(pilot=d0["src_id"].isin(PILOT_IDS),
                        fitted=iso0.predict(d0["jaccard"].to_numpy(float)))
        for name, flag in (("pilot", True), ("extension", False)):
            t = d0s[d0s["pilot"] == flag]
            stratum[name] = {
                "n": int(len(t)),
                "obs": float(np.average(t["miss_rate"], weights=t["n_obs"])),
                "implied": float(np.average(t["fitted"], weights=t["n_obs"]))}
    except Exception:
        stratum = {}

    verb = "rises" if incr["D0"] else "falls"
    steeper = st["D0"]["ratio"] > 1
    band_word = "steepest" if steeper else "flattest"
    sig = [d for d in DETECTORS if rho[d][1] < 0.05]

    # Trimmed 2026-08-19 for length. The body paragraph keeps the finding, its hedge, the
    # registered band's placement and the width of the interval that stops it being a result;
    # the fitting/bootstrap asides and the batch-level arithmetic that diagnoses the
    # predecessor's null move to a generated appendix note, so nothing is lost and nothing is
    # hand-typed. Both files are written here; sections/10_appendix.tex inputs the second.
    prose = (
        f"\\textbf{{Attack strength behaves like a dose, weakly but consistently, and that is "
        f"what the two studies differ in.}} They report different pooled miss rates; neither "
        f"shows the quantity that connects them, namely how miss rate varies \\emph{{with}} $J$. "
        f"Re-running the uncontrolled study's own splits with per-rewrite logging (same seeds, "
        f"same three detectors, same held-out phishing) gives {tex_int(n_obs)} scored rewrites "
        f"over {n_src} sources, and fitting miss rate on $J$ by isotonic regression, with the "
        f"direction read off the data rather than assumed, gives the same answer for all three "
        f"detectors: miss {verb} as $J$ rises, so a rewrite that keeps more of its source is "
        f"caught more often (Figure~\\ref{{fig:doseresponse}}). The rank correlation behind that "
        f"is modest and only sometimes conventionally significant: Spearman "
        f"$\\rho={rho['D0'][0]:+.2f}$ ($p={rho['D0'][1]:.2f}$) over sources for the naive "
        f"detector, ${rho['D1'][0]:+.2f}$ ($p={rho['D1'][1]:.3f}$) for the char-adversarial one, "
        f"and ${rho['D2'][0]:+.3f}$ ($p={rho['D2'][1]:.3f}$) for the paraphrase-adversarial one"
        f", clearing $p<0.05$ for {len(sig)} of the {len(DETECTORS)} detectors, so this is a "
        f"shape worth reading and not an effect this corpus establishes. The third correlation is "
        f"the reason to put it no higher than that: it is indistinguishable from zero, so the "
        f"agreement of all three directions rests, for that detector, on the sign of a quantity "
        f"this corpus does not resolve. The registered band is "
        f"nonetheless where the curve does its work rather than an arbitrary target: across "
        f"$[{lo:.2f}, {hi:.2f}]$ the fitted curve moves from ${at_lo*100:.1f}\\%$ to "
        f"${at_hi*100:.1f}\\%$, a mean slope ${st['D0']['ratio']:.1f}\\times$ "
        + ("steeper" if steeper else "shallower")
        + f" than the curve's average slope over its whole support, and the band's slope is the "
        f"{band_word[:-3]}er of the two in ${steep_frac*100:.0f}\\%$ of {tex_int(BOOT)} "
        f"cluster-bootstrap resamples of the sources. That agreement is \\emph{{consistency}} and "
        f"not vindication: the band was fixed before any of this was computed, but it was not "
        f"chosen blind to outcome (Section~\\ref{{ssec:paraphraseband}}), and the curve is fitted "
        f"post hoc on the same observations; and the interval at the band's midpoint "
        f"(${mid_est*100:.1f}\\%$, ${mid_lo*100:.1f}$--${mid_hi*100:.1f}\\%$) is wide enough that "
        f"the dose--response is an explanation we can draw, not one this corpus can establish. "
        f"Appendix~\\ref{{sec:appendix}} carries the fit's own caveats and the batch-level "
        f"arithmetic that diagnoses the predecessor's null.\n")

    notes = (
        f"\\subsection*{{Note: how the dose--response curve of Section~\\ref{{ssec:paraphraseband}} "
        f"is fitted, and what it says about the predecessor's null}}\n"
        f"Across the middle of the corpus the naive detector's fitted miss rate goes from "
        f"${at_q_lo*100:.1f}\\%$ at the $10$th percentile of $J$ ($J={q_lo:.2f}$) to "
        f"${at_q_hi*100:.1f}\\%$ at the $90$th ($J={q_hi:.2f}$); the deciles rather than the "
        f"extremes, because isotonic regression pins its end blocks to single sources. "
        f"Confidence bands are a cluster bootstrap over \\emph{{sources}}, not rewrites: each "
        f"source contributes several scored observations that share one $J$, and resampling those "
        f"as if independent would narrow the band by roughly the square root of that "
        f"multiplicity. The strength-controlled study's own miss rate, plotted at its achieved "
        f"$J={band_j:.3f}$, is measured on rewrites the curve was not fitted to, but it is not an "
        f"independent observation of the curve: the two studies paraphrase the same {shared} "
        f"source lures, so the plotted point and the band around it are clustered on the same "
        f"sources that both bootstraps resample. Read the marker against the curve as a "
        f"consistency check, not as external replication.\n\n"
        f"The curve is also what the predecessor's null looks like from the inside. Pooling its "
        f"rewrites gave one number, ${pooled*100:.1f}\\%$ for the naive detector, for a quantity "
        f"that moves by {abs(at_q_hi - at_q_lo)*100:.0f}\\,pp between the corpus's own deciles"
        + (f"; its pilot batch sat at mean $J={j_pilot:.2f}$ and its extension at "
           f"$J={j_ext:.2f}$, and the curve integrated over each batch's $J$ distribution implies "
           f"${stratum['pilot']['implied']*100:.1f}\\%$ against "
           f"${stratum['extension']['implied']*100:.1f}\\%$, against measured "
           f"${stratum['pilot']['obs']*100:.1f}\\%$ and "
           f"${stratum['extension']['obs']*100:.1f}\\%$. The extension added lures and "
           f"moved the corpus up the $J$ axis, so the effect it halved was measured at a gentler "
           f"dose than the effect it was sized to detect"
           if stratum else "")
        + ". This diagnoses the null and does not repair it.\n")
    write_generated(os.path.join(SEC, "gen_dose_response_notes.tex"), notes)
    write_generated(os.path.join(SEC, "gen_dose_response.tex"), prose)

    print(f"    J support [{jmin:.3f}, {jmax:.3f}]; band [{lo:.2f}, {hi:.2f}]")
    for d in DETECTORS:
        s = st[d]
        print(f"    {d}: spearman rho={rho[d][0]:+.3f} (p={rho[d][1]:.3g}) -> "
              f"{'increasing' if incr[d] else 'decreasing'}; "
              f"slope in band {s['slope_band']*100:6.1f} pp/J vs overall "
              f"{s['slope_all']*100:6.1f} pp/J  ratio {s['ratio']:.2f}")
    print(f"    D0 curve: {at_q_lo*100:.1f}% at J10={q_lo:.3f} -> {at_q_hi*100:.1f}% at "
          f"J90={q_hi:.3f}; band edges {at_lo*100:.1f}% -> {at_hi*100:.1f}%; "
          f"midpoint {mid_est*100:.1f}% [{mid_lo*100:.1f}, {mid_hi*100:.1f}]; "
          f"band steeper in {steep_frac*100:.0f}% of {BOOT} resamples")
    for k, (v, n) in strat.items():
        s = stratum.get(k, {})
        print(f"    stratum {k:9s} n={n:3d} mean test-role J={v:.4f}"
              + (f"  D0 miss observed {s['obs']*100:.2f}% / curve-implied "
                 f"{s['implied']*100:.2f}%" if s else ""))
    for d, (e, plo, phi) in band_pt.items():
        print(f"    band corpus {d}: miss {e*100:.1f}% [{plo*100:.1f}, {phi*100:.1f}] "
              f"at J={band_j:.3f}")


if __name__ == "__main__":
    main()
