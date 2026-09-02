#!/usr/bin/env python3
"""
analyze_qr_dfr.py — the registered analysis of the quishing decoder sweep.

The sweep finished on 2026-08-30 at 792,000 renders x 3 decoders = 2,376,000 rows, complete and
balanced (2,000 URLs x 4 EC levels x 3 module sizes x 33 variants; not one URL was dropped for
exceeding a version limit). `PREREG_quishing.md` states what would count as a negative result
before any of it was read; this script computes exactly that and nothing else, so the tests cannot
drift toward whatever the data happens to show.

WHAT IS REGISTERED, and what this script therefore refuses to improvise:

  T1  render scale changes what a decoder survives.  DFR(box 2) - DFR(box 4) over `rotate`,
      pooled across its four strengths and four EC levels, paired PER URL, two-sided Wilcoxon.
  T2  the benchmark is not confounded with the corpus label.  Overall DFR for phishing-labelled
      minus benign-labelled URLs, per decoder, unpaired, Mann-Whitney U on the per-URL DFRs.

Benjamini-Hochberg over m = 2. The diagnostics below the tests are descriptive whatever they show,
which is why they carry no thresholds.

UNIT OF ANALYSIS IS THE URL. The 792,000 rows are 2,000 independent draws crossed with a fixed
design; every quantity here is computed per URL first and aggregated across URLs, so the effective
n is 2,000. Aggregating over renders would report the design matrix as evidence, and would shrink
every confidence interval by a factor of twenty for free.

TWO TRANSFORMS ARE SATURATED AND THE ANALYSIS SAYS SO RATHER THAN AVERAGING THEM AWAY. `blur` and
`motion` fail at ~100% at every strength because their magnitudes are absolute pixels while every
other axis of the design is in modules (see the deviation record in the pre-specification). They are
reported, flagged, and excluded from any statement about a strength gradient; they are NOT dropped
from the overall DFR, because the overall DFR is what a pipeline meets.

RUN
    python3 scripts/analyze_qr_dfr.py
    python3 scripts/analyze_qr_dfr.py --dfr data/processed/qr/qr_dfr.csv --json ...
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd

DFR = os.path.join("data", "processed", "qr", "qr_dfr.csv")
SNAP = os.path.join("data", "processed", "qr", "dfr_snapshot.json")
# Absolute-pixel magnitudes, hence no usable strength axis. Named once, used everywhere, so the
# exclusion is a property of the analysis rather than a habit of whoever reads the table.
SATURATED = ("blur", "motion")


def load(path: str) -> pd.DataFrame:
    """Typed on the way in. The default object dtypes cost about 2 GB on this file and the
    categorical ones cost a tenth of that; more to the point, `strength` read as a float and
    printed as a group key is how 0.25 becomes 0.25000000000000006 in a caption."""
    df = pd.read_csv(
        path,
        dtype={"sample_id": "string", "label": "category", "url_len": "int32",
               "ec_level": "category", "box_size": "int8", "transform": "category",
               "strength": "category", "qr_version": "int16", "modules": "int16",
               "decoder": "category", "decoded": "int8", "correct": "int8"},
        usecols=["sample_id", "label", "url_len", "ec_level", "box_size", "transform",
                 "strength", "qr_version", "modules", "decoder", "decoded", "correct"])
    # The index deliberately does not carry the URL -- only the first 16 hex of its SHA-1, which is
    # the sample_id's prefix. That is the grouping key, and it is also why this file can be
    # published without republishing the corpus.
    df["url"] = df["sample_id"].str[:16].astype("category")
    df.drop(columns=["sample_id"], inplace=True)
    df["fail"] = 1 - df["correct"]
    return df


def bh(pvals: dict, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg over the m registered tests, m fixed at registration and not at the count
    of tests that turned out interesting."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 1.0
    for i in range(m, 0, -1):
        name, p = items[i - 1]
        prev = min(prev, p * m / i)
        adj[name] = prev
    return {k: {"p": pvals[k], "p_bh": adj[k], "reject": adj[k] <= alpha} for k in pvals}


def per_url_dfr(df: pd.DataFrame, by=("url", "decoder")) -> pd.DataFrame:
    return df.groupby(list(by), observed=True)["fail"].mean().rename("dfr").reset_index()


def t1(df: pd.DataFrame) -> dict:
    """Render scale, on `rotate`. Registered success: mean difference >= +10 pp for at least two of
    the three decoders; partial at >= +5 pp; negative below that, or on any reversed sign."""
    from scipy.stats import wilcoxon
    sub = df[(df["transform"] == "rotate") & (df["box_size"].isin([2, 4]))]
    g = (sub.groupby(["url", "decoder", "box_size"], observed=True)["fail"]
            .mean().unstack("box_size"))
    out, ps = {}, {}
    for dec, part in g.groupby(level="decoder", observed=True):
        d = (part[2] - part[4]).dropna().to_numpy()
        # wilcoxon refuses an all-zero vector; that case is itself the negative result, so it is
        # reported rather than raised.
        p = float(wilcoxon(d)[1]) if np.any(d != 0) else 1.0
        out[dec] = {"n_urls": int(d.size), "mean_pp": float(100 * d.mean()),
                    "median_pp": float(100 * np.median(d)),
                    "dfr_box2": float(100 * part[2].mean()),
                    "dfr_box4": float(100 * part[4].mean()),
                    "p": p}
        ps[dec] = p
    means = [v["mean_pp"] for v in out.values()]
    reversed_sign = any(m < 0 for m in means)
    n10 = sum(m >= 10 for m in means)
    n5 = sum(m >= 5 for m in means)
    verdict = ("negative" if reversed_sign or n5 < 2 else
               "success" if n10 >= 2 else "partial")
    return {"per_decoder": out, "p_pooled": float(max(ps.values())), "verdict": verdict}


def t2(df: pd.DataFrame) -> dict:
    """Label confound, over the whole sweep. Registered success (the intended outcome): the
    absolute difference under 2 pp for all three decoders; negative at >= 5 pp for any."""
    from scipy.stats import mannwhitneyu
    lab = df.groupby("url", observed=True)["label"].first()
    d = per_url_dfr(df).merge(lab.rename("label"), left_on="url", right_index=True)
    out, ps = {}, {}
    for dec, part in d.groupby("decoder", observed=True):
        ph = part.loc[part["label"] == "phishing", "dfr"].to_numpy()
        be = part.loc[part["label"] == "benign", "dfr"].to_numpy()
        p = float(mannwhitneyu(ph, be, alternative="two-sided")[1])
        out[dec] = {"n_phishing": int(ph.size), "n_benign": int(be.size),
                    "dfr_phishing": float(100 * ph.mean()), "dfr_benign": float(100 * be.mean()),
                    "diff_pp": float(100 * (ph.mean() - be.mean())), "p": p}
        ps[dec] = p
    diffs = [abs(v["diff_pp"]) for v in out.values()]
    verdict = ("negative" if any(x >= 5 for x in diffs) else
               "success" if all(x < 2 for x in diffs) else "partial")
    return {"per_decoder": out, "p_pooled": float(min(ps.values())), "verdict": verdict}


def crossing(sub: pd.DataFrame) -> str:
    """The strength at which a transform first crosses 50% DFR, linearly interpolated between the
    two bracketing strengths. '<0.25' when it is already over at the weakest setting, 'none' when
    it never gets there -- both are more honest than reporting the nearest grid point as if the
    grid had been dense."""
    s = sub.groupby("strength", observed=True)["fail"].mean()
    s = s.reindex(sorted(s.index, key=float)).dropna()
    xs = [float(i) for i in s.index]
    ys = list(s.to_numpy())
    if not ys:
        return "none"
    if ys[0] >= 0.5:
        return "<%.2f" % xs[0]
    for i in range(1, len(ys)):
        if ys[i] >= 0.5:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return "%.2f" % (x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-9))
    return "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfr", default=DFR)
    ap.add_argument("--json", default=SNAP)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()

    if not os.path.isfile(a.dfr):
        print(f"[!] {a.dfr} not found. It is written on the second Jetson by\n"
              f"    gen_synthetic_qr.py --stream, and pulled by scripts/ops/sync_jetson.sh",
              file=sys.stderr)
        return 1
    df = load(a.dfr)
    n_urls = df["url"].nunique()
    decs = sorted(df["decoder"].cat.categories)
    print(f"[*] {len(df):,} rows, {n_urls:,} URLs, decoders {', '.join(decs)}")

    # Completeness first. A missing cell would not raise anywhere below; it would quietly turn a
    # paired test into an unpaired one and a mean into a mean over whatever survived.
    expect = n_urls * df["ec_level"].nunique() * df["box_size"].nunique() * 33 * len(decs)
    status = "complete" if len(df) == expect else f"INCOMPLETE ({len(df)/expect:.1%})"
    print(f"[*] grid {status}: {len(df):,} of {expect:,} expected rows")

    res = {"rows": int(len(df)), "urls": int(n_urls), "decoders": decs,
           "grid_complete": len(df) == expect}

    print("\n=== T1 — render scale (rotate, box 2 vs box 4, paired per URL) ===")
    r1 = t1(df)
    print(f"  {'decoder':<9} {'DFR@2px':>8} {'DFR@4px':>8} {'mean diff':>10} {'median':>8}  p")
    for dec in decs:
        v = r1["per_decoder"][dec]
        print(f"  {dec:<9} {v['dfr_box2']:7.1f}% {v['dfr_box4']:7.1f}% "
              f"{v['mean_pp']:+9.1f}pp {v['median_pp']:+7.1f}pp  {v['p']:.3g}")
    print(f"  registered verdict: {r1['verdict'].upper()}")

    print("\n=== T2 — label confound (overall DFR, phishing vs benign, unpaired) ===")
    r2 = t2(df)
    print(f"  {'decoder':<9} {'phishing':>9} {'benign':>9} {'diff':>9}  p")
    for dec in decs:
        v = r2["per_decoder"][dec]
        print(f"  {dec:<9} {v['dfr_phishing']:8.1f}% {v['dfr_benign']:8.1f}% "
              f"{v['diff_pp']:+8.1f}pp  {v['p']:.3g}")
    print(f"  ({r2['per_decoder'][decs[0]]['n_phishing']} phishing URLs, "
          f"{r2['per_decoder'][decs[0]]['n_benign']} benign)")
    print(f"  registered verdict: {r2['verdict'].upper()}")

    adj = bh({"T1": r1["p_pooled"], "T2": r2["p_pooled"]}, a.alpha)
    print(f"\n  Benjamini-Hochberg, m=2, alpha={a.alpha}:")
    for k in ("T1", "T2"):
        v = adj[k]
        print(f"    {k}  p={v['p']:.3g}  p_BH={v['p_bh']:.3g}  "
              f"{'reject H0' if v['reject'] else 'retain H0'}")
    # T2 is the one test here whose intended outcome is a NULL, and a null is not something a
    # p-value can establish. The verdict above is the registered effect-size rule; the p-value is
    # printed because it was registered, not because rejecting H0 would be the good news.
    print("    (T2's registered criterion is the effect size, not this p — a null is not proved "
          "by failing to reject)")
    res["T1"], res["T2"], res["bh"] = r1, r2, adj

    print("\n=== Diagnostic: DFR per decoder x transform (all EC, all module sizes) ===")
    tab = df.groupby(["decoder", "transform"], observed=True)["fail"].mean().unstack("decoder")
    tab = tab.reindex(sorted(tab.index, key=lambda t: -tab.loc[t].mean()))
    print(f"  {'transform':<12} " + " ".join(f"{d:>9}" for d in decs) + "   50% crossing")
    for tr in tab.index:
        cross = "saturated" if tr in SATURATED else " / ".join(
            crossing(df[(df["transform"] == tr) & (df["decoder"] == d)]) for d in decs)
        row = " ".join(f"{100*tab.loc[tr, d]:8.1f}%" for d in decs)
        print(f"  {tr:<12} {row}   {cross}")
    res["by_transform"] = {tr: {d: float(100 * tab.loc[tr, d]) for d in decs} for tr in tab.index}
    res["crossing"] = {tr: ("saturated" if tr in SATURATED else
                            {d: crossing(df[(df["transform"] == tr) & (df["decoder"] == d)])
                             for d in decs})
                       for tr in tab.index}

    print("\n=== Diagnostic: does error correction buy what it claims? (DFR by EC level) ===")
    ec = df.groupby(["transform", "ec_level"], observed=True)["fail"].mean().unstack("ec_level")
    ec = ec[[c for c in ("L", "M", "Q", "H") if c in ec.columns]]
    print(f"  {'transform':<12} " + " ".join(f"{c:>7}" for c in ec.columns) + "     L-H")
    for tr in ec.index:
        gain = 100 * (ec.loc[tr, "L"] - ec.loc[tr, "H"])
        print(f"  {tr:<12} " + " ".join(f"{100*ec.loc[tr, c]:6.1f}%" for c in ec.columns)
              + f"   {gain:+6.1f}pp")
    res["by_ec"] = {tr: {c: float(100 * ec.loc[tr, c]) for c in ec.columns} for tr in ec.index}

    print("\n=== Diagnostic: module size, all nine transforms (not only rotate) ===")
    bx = df.groupby(["transform", "box_size"], observed=True)["fail"].mean().unstack("box_size")
    print(f"  {'transform':<12} " + " ".join(f"{b:>6}px" for b in bx.columns) + "     2-4")
    for tr in bx.index:
        print(f"  {tr:<12} " + " ".join(f"{100*bx.loc[tr, b]:6.1f}%" for b in bx.columns)
              + f"   {100*(bx.loc[tr, 2]-bx.loc[tr, 4]):+6.1f}pp")
    res["by_box"] = {tr: {int(b): float(100 * bx.loc[tr, b]) for b in bx.columns}
                     for tr in bx.index}

    print("\n=== Diagnostic: a decoder that returns nothing, against one that lies ===")
    # Registered separately because they are different failures for a defender: silence is a gap
    # the pipeline can detect and retry, a wrong payload is a URL the classifier will score.
    print(f"  {'decoder':<9} {'silent':>9} {'wrong payload':>15} {'correct':>9}")
    for dec in decs:
        s = df[df["decoder"] == dec]
        silent = float(100 * (s["decoded"] == 0).mean())
        lies = float(100 * ((s["decoded"] == 1) & (s["correct"] == 0)).mean())
        print(f"  {dec:<9} {silent:8.1f}% {lies:14.2f}% {100*s['correct'].mean():8.1f}%")
        res.setdefault("failure_kind", {})[dec] = {
            "silent_pct": silent, "wrong_payload_pct": lies,
            "correct_pct": float(100 * s["correct"].mean())}

    print("\n=== Diagnostic: strength gradient (saturated flagged, not averaged away) ===")
    st = df[df["transform"] != "clean"].groupby(
        ["transform", "strength"], observed=True)["fail"].mean().unstack("strength")
    st = st[sorted(st.columns, key=float)]
    print(f"  {'transform':<12} " + " ".join(f"{c:>7}" for c in st.columns))
    for tr in st.index:
        flag = "  <- saturated, no gradient" if tr in SATURATED else ""
        print(f"  {tr:<12} " + " ".join(f"{100*st.loc[tr, c]:6.1f}%" for c in st.columns) + flag)
    res["by_strength"] = {tr: {c: float(100 * st.loc[tr, c]) for c in st.columns}
                          for tr in st.index}
    res["saturated"] = list(SATURATED)

    # ---------------------------------------------------------------- added 2026-08-31, POST-HOC
    # Neither of the two below is in PREREG_quishing.md. They are descriptive, they answer questions
    # a reader asks of the registered diagnostics rather than questions the registration posed, and
    # the paper labels them the same way this comment does.

    print("\n=== Post-hoc: is a cascade worth more than installing the best decoder? ===")
    # "Why not run all three and take whichever reads?" is the first thing a reader asks of the
    # per-decoder table. Answering it costs one groupby and closes the question.
    # load() drops sample_id (it carries the URL's hash, and the url column already does), so a
    # render is identified by the design cell it sits in. modules rides along; it is a function of
    # url and ec_level, so it adds no rows.
    KEY = ["transform", "url", "ec_level", "box_size", "strength", "modules"]
    casc_row = df.groupby(KEY, observed=True)["correct"].max().rename("cascade").reset_index()
    per = casc_row.set_index(KEY)
    for d in decs:
        sub = df[df["decoder"] == d].groupby(KEY, observed=True)["correct"].max().rename(d)
        per = per.join(sub)
    per = per.reset_index()
    best = max(decs, key=lambda d: per[d].mean())
    casc = {}
    print(f"  {'transform':<12} " + " ".join(f"{d:>9}" for d in decs)
          + f" {'cascade':>9} {'gain':>7}")
    for tr in sorted(per["transform"].unique()):
        sub = per[per["transform"] == tr]
        row = {d: float(100 * (1 - sub[d].mean())) for d in decs}
        row["cascade"] = float(100 * (1 - sub["cascade"].mean()))
        row["gain_vs_best_pp"] = row[best] - row["cascade"]
        casc[str(tr)] = row
        print(f"  {tr:<12} " + " ".join(f"{row[d]:8.1f}%" for d in decs)
              + f" {row['cascade']:8.1f}% {row['gain_vs_best_pp']:+6.1f}")
    overall_best = float(100 * (1 - per[best].mean()))
    overall_casc = float(100 * (1 - per["cascade"].mean()))
    res["cascade"] = {"per_transform": casc, "best_decoder": best,
                      "overall_best_dfr": overall_best, "overall_cascade_dfr": overall_casc,
                      "overall_gain_pp": overall_best - overall_casc, "registered": False}
    print(f"  Over the whole grid: {best} {overall_best:.1f}% -> cascade {overall_casc:.1f}%, "
          f"a gain of {overall_best - overall_casc:.2f}pp for two extra libraries.")

    print("\n=== Post-hoc: DFR against module density (the mechanism, not just the effect) ===")
    # Module count is set by URL length, which the draw stratified on, so this is a dose-response
    # the design already supports: occlusion should hurt SPARSE codes (a fixed logo covers more of
    # them) and geometry should hurt DENSE ones (smaller modules lose the sampling grid sooner).
    bins = [0, 25, 33, 41, 49, 10 ** 6]
    names = ["<=25", "26-33", "34-41", "42-49", "50+"]
    per["density"] = pd.cut(per["modules"].astype(int), bins, labels=names)
    dens = {}
    print(f"  {'transform':<12} " + " ".join(f"{n:>8}" for n in names) + "   (DFR%, best decoder)")
    for tr in sorted(per["transform"].unique()):
        sub = per[per["transform"] == tr]
        g = sub.groupby("density", observed=True)[best].mean()
        dens[str(tr)] = {n: (float(100 * (1 - g[n])) if n in g.index else None) for n in names}
        print(f"  {tr:<12} " + " ".join(
            f"{dens[str(tr)][n]:7.1f}%" if dens[str(tr)][n] is not None else f"{'--':>8}"
            for n in names))
    counts = per["density"].value_counts().reindex(names)
    print("  renders per bin: " + ", ".join(f"{n} {int(counts[n]):,}" for n in names)
          + "  <- unbalanced, because URL length is")
    res["by_density"] = {"dfr": dens, "decoder": best,
                         "n_renders": {n: int(counts[n]) for n in names}, "registered": False}

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"\n[+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
