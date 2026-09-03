#!/usr/bin/env python3
"""
make_p3_band_assets.py — the successor study: paraphrase evasion with attack strength CONTROLLED.

The predecessor ran twice and resolved nothing twice, and its post-mortem found the defect in the
design rather than the sample size: the protocol said what a paraphrase must PRESERVE and never
what it must CHANGE, so attack strength drifted between batches (mean token Jaccard 0.28 then
0.41). Here strength is fixed inside a registered band. Protocol committed before recalibration:
papers/P3_multimodal/protocols/PARAPHRASE_BAND_PROTOCOL.md.

Emits tab_paraphrase_band.tex and gen_paraphrase_band_verdict.tex, plus a machine-readable stats CSV.

RUN:  python scripts/p3_paraphrase_band.py && python scripts/make_p3_band_assets.py
The design change, the comparisons and the verdict rule: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

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
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
BAND_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_band.csv")
DRIFT_STATS = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_stats.csv")
from make_p3_paraphrase_assets import (
    run, COMPARISONS, CELLS, SEEDS, strip_url,
    corpus_multiple_for_significance)
from paired_eval import corrected_paired_t, bh_adjust, fmt_p
from p3_jaccard_check import BAND
from genfile import write_generated


def load():
    """Same corpus, same columns as the predecessor — only the rewrites change."""
    frames = [pd.read_csv(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))
              for t in ("sms", "email")]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str).map(strip_url)
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    if not os.path.exists(BAND_CSV):
        raise SystemExit(f"{BAND_CSV} missing — run scripts/p3_paraphrase_band.py first.")
    para = pd.read_csv(BAND_CSV)
    jac = para["jaccard"].to_numpy()
    lo, hi = BAND
    if not ((jac >= lo) & (jac <= hi)).all():
        raise SystemExit("[!] the band CSV contains out-of-band rewrites — refusing to analyse "
                         "an attack set that is not what the protocol registered.")
    para["text"] = para["text"].map(strip_url)
    wide = para.pivot(index="src_id", columns="variant", values="text")
    missing = set(df.loc[df.y == 1, "id"]) - set(wide.index)
    if missing:
        raise SystemExit(f"[!] {len(missing)} phishing messages have no band rewrite.")
    df["para_a"] = df["id"].map(wide["a"]).fillna("")
    df["para_b"] = df["id"].map(wide["b"]).fillna("")
    return df, para


def fig_band_control(src: dict) -> None:
    """The paper's methodological claim, drawn: attack strength was fixed, not labelled.

    Everything the band study concludes rests on one premise --- that the rewrites all sit at a
    comparable distance from their sources --- and until this figure that premise reached the
    reader as three numbers in a sentence. It is also the premise a sceptical reviewer would go
    after first, because the predecessor's null and this study's decisive result differ in
    exactly one respect. Drawn, the contrast is not close: the uncontrolled corpus spreads from
    Jaccard 0.037 to 0.833 with 21% of its rewrites inside the band that was later registered,
    while the recalibrated set has every rewrite inside it by construction.

    Test-role rewrites only, for both sets. Those are the ones the attack is scored on, and they
    are the population the paper's 0.356 and 0.247 refer to --- so the figure and the prose
    cannot be read against different denominators.
    """
    from figstyle import apply, ORANGE, GRAY, INK
    plt = apply()
    import numpy as np
    from p3_jaccard_check import jaccard as jac

    lo, hi = BAND
    pre_p = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase.csv")
    if not (os.path.exists(pre_p) and os.path.exists(BAND_CSV)):
        print("[i] paraphrase corpora incomplete — skipping the band-control figure")
        return
    pre = pd.read_csv(pre_p)
    pre = pre[pre["variant"] == "b"] if "variant" in pre.columns else pre[pre["role"] == "test"]
    j_pre = np.array([jac(src[r.src_id], r.text) for r in pre.itertuples() if r.src_id in src])
    band = pd.read_csv(BAND_CSV)
    j_band = band[band["role"] == "test"]["jaccard"].to_numpy(float)
    if not len(j_pre) or not len(j_band):
        return

    # Sized near the column width it is printed at (3.5 in), so the 7.5 pt labels stay >= 6 pt.
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axvspan(lo, hi, color=GRAY, alpha=0.22, zorder=0)
    # Above the frame, not inside it. The label is wider than the 0.10-wide band it names, so
    # placed inside it straddled the shaded rectangle's edge and printed half on white and half
    # on grey — legible at figure scale, visibly broken once the page scales it up.
    ax.annotate(f"registered band [{lo:.2f}, {hi:.2f}]", ((lo + hi) / 2, 1.015),
                xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7.5,
                color=INK, annotation_clip=False)
    bins = np.linspace(0, 0.9, 46)
    # The band is CLOSED, [lo, hi], and the calibration admits rewrites at exactly J = hi (three
    # sit there). Histogram bins are half-open on the right, so a rewrite at J = hi fell into the
    # bin starting at hi and was drawn just outside the shading while the legend said 100% in
    # band. Edges are aligned with the band (0.02 steps hit 0.20 and 0.30), and a value exactly
    # on the upper edge is counted in the bin that ends there, i.e. the band's last bin is
    # right-closed. The in-band percentages in the legend are computed on the raw values.
    at_hi = lambda j: np.where(np.isclose(j, hi), np.nextafter(hi, 0.0), j)
    # Two-line labels, not one. On one line each label ran ~50 characters and the legend box
    # reached back across the shaded band, printing itself over the band annotation.
    ax.hist(at_hi(j_pre), bins=bins, color=GRAY, alpha=0.85, zorder=2,
            label=f"uncontrolled predecessor\nmean {j_pre.mean():.3f} · "
                  f"{100 * ((j_pre >= lo) & (j_pre <= hi)).mean():.0f}% in band")
    ax.hist(at_hi(j_band), bins=bins, color=ORANGE, alpha=0.9, zorder=3,
            label=f"strength-controlled\nmean {j_band.mean():.3f} · "
                  f"{100 * ((j_band >= lo) & (j_band <= hi)).mean():.0f}% in band")
    ax.set_xlabel("token Jaccard, source lure vs. its test-role rewrite")
    ax.set_ylabel(f"rewrites (n={len(j_band)} each)")
    ax.set_xlim(0, 0.9)
    # The controlled set is a spike, so its tallest bin sets the scale and would otherwise sit
    # flush against the frame with the legend on top of it. Headroom is for the legend only now
    # that the band label has moved outside the axes.
    tallest = max(np.histogram(j_band, bins=bins)[0].max(),
                  np.histogram(j_pre, bins=bins)[0].max())
    ax.set_ylim(0, tallest * 1.34)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.005, 1.0),
              handlelength=1.1, handletextpad=0.6, labelspacing=0.9, borderaxespad=0.2)
    fig.tight_layout()
    out = os.path.join(ROOT, "papers", "P3_multimodal", "figures", "band_control.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def main() -> None:
    df, para = load()
    from p3_jaccard_check import sources as _sources
    fig_band_control(_sources())
    per_seed, fpr = run(df)
    mean = {c: float(np.mean(per_seed[c])) for c in CELLS}
    sd = {c: float(np.std(per_seed[c])) for c in CELLS}

    tests = []
    for key, hi_c, lo_c, label, direction in COMPARISONS:
        d = np.array(per_seed[hi_c]) - np.array(per_seed[lo_c])
        r = corrected_paired_t(d)
        tests.append({"key": key, "hi": hi_c, "lo": lo_c, "label": label,
                      "consistent": int(np.sum(d > 0)) if direction > 0 else int(np.sum(d < 0)),
                      "need": corpus_multiple_for_significance(r.get("t", float("nan"))), **r})
    adj, rej = bh_adjust([t["p"] for t in tests])
    for t, a, rj in zip(tests, adj, rej):
        t["p_adj"], t["reject"] = a, rj
    T = {t["key"]: t for t in tests}

    n_ph = int(df.y.sum())
    n_test_ph = n_ph * 0.30
    j_test = para.loc[para.variant == "b", "jaccard"]
    j_all = para["jaccard"]
    lo_b, hi_b = BAND

    # The predecessor's uncontrolled run on the SAME sources, for the descriptive contrast the
    # protocol permits and explicitly refuses to test.
    prev = {}
    if os.path.exists(DRIFT_STATS):
        prev = {r["key"]: r for _, r in pd.read_csv(DRIFT_STATS).iterrows()}

    def cell(c):
        return f"{mean[c]*100:.1f}\\,$\\pm$\\,{sd[c]*100:.1f}"

    def verdict(key):
        t = T[key]
        return (f"$\\Delta = {t['mean']*100:+.1f}$\\,pp, {t['consistent']}/{SEEDS} splits in the "
                # Both symbols printed in full: stripping the label off the adjusted value
                # produced "BH-adjusted 0.11" beside "BH-adjusted q<0.001", i.e. the same
                # quantity rendered two ways within one sentence.
                f"predicted direction, corrected {fmt_p(t['p'])}, "
                f"BH-adjusted {fmt_p(t['p_adj'], 'q')}")

    survivors = [t for t in tests if t["reject"]]
    naive_hits = [t for t in tests if t["p_naive"] < 0.05]

    tex = rf"""\begin{{table*}}[t]
\caption{{Paraphrase evasion with \textbf{{attack strength controlled}}: all $2\times{n_ph}$
rewrites calibrated into token-Jaccard band $[{lo_b:.2f}, {hi_b:.2f}]$; mean $\pm$ std over {SEEDS} stratified
70/30 splits.}}
\label{{tab:paraphraseband}}
\small
\begin{{tabular}}{{l c c c}}
\toprule
& \multicolumn{{3}}{{c}}{{Miss rate (\%) on held-out phishing}} \\
\cmidrule(lr){{2-4}}
Detector & Clean & Char-obfuscated & Paraphrased \\
\midrule
D0 naive & {cell('D0_A0')} & {cell('D0_A1')} & {cell('D0_A2')} \\
D1 adv.\ trained on char.\ obfuscation & n/a & {cell('D1_A1')} & {cell('D1_A2')} \\
D2 adv.\ trained on paraphrases & n/a & n/a & {cell('D2_A2')} \\
\bottomrule
\end{{tabular}}\end{{table*}}"""
    os.makedirs(SEC, exist_ok=True)
    write_generated(os.path.join(SEC, "tab_paraphrase_band.tex"), tex)

    h1 = T["H1"]
    prev_h1 = float(prev["H1"]["mean_pp"]) if "H1" in prev else float("nan")
    grew = h1["mean"] * 100 > prev_h1
    # H2's clause used to read "still does nothing", hardcoded, while the verdict printed beside it
    # reported +4.8pp at q=0.012 with 19/20 splits in the predicted direction -- the parenthesis
    # refuted the sentence it was inside, and the paper's own abstract said "worse". "still" was
    # wrong too: the predecessor's H2 did not reject. The clause now follows sign and significance.
    h2 = T["H2"]
    prev_h2_rejected = bool(int(prev["H2"]["reject"])) if "H2" in prev else False
    if not h2["reject"]:
        h2_clause = "still does nothing for" if not prev_h2_rejected else "no longer moves"
    elif h2["mean"] > 0:
        h2_clause = ("still worsens" if prev_h2_rejected else "now worsens")
    else:
        h2_clause = ("still repairs" if prev_h2_rejected else "now repairs")
    head = ("Controlling the attack changes the measurement."
            if survivors else
            "Controlling the attack does not rescue the measurement.")
    prose = (
        f"\\textbf{{{head}}} With every rewrite held in $[{lo_b:.2f}, {hi_b:.2f}]$ (achieved mean "
        f"${j_all.mean():.3f}$ over {len(j_all)} rewrites; test-role mean ${j_test.mean():.3f}$) "
        f"the naive detector's miss rate goes from ${mean['D0_A0']*100:.1f}\\%$ on clean held-out "
        f"phishing to ${mean['D0_A2']*100:.1f}\\%$ on their rewrites ({verdict('H1')}), against "
        f"${mean['D0_A1']*100:.1f}\\%$ under the character attack ({verdict('ctrl')}). "
        f"Character-level adversarial training still repairs the character attack "
        f"(${mean['D0_A1']*100:.1f}\\% \\rightarrow {mean['D1_A1']*100:.1f}\\%$) and "
        f"{h2_clause} the paraphrase attack (${mean['D1_A2']*100:.1f}\\%$, {verdict('H2')}); "
        f"training on disjoint rewrites gives ${mean['D2_A2']*100:.1f}\\%$ ({verdict('H3')}). "
        f"{len(survivors)} of the {len(tests)} contrasts survive Benjamini--Hochberg on the "
        f"corrected test, against {len(naive_hits)} that the uncorrected test would license.\n\n"
        f"The time-stamped pre-specified expectation was that H1 would \\emph{{grow}} once strength was fixed, "
        f"because the uncontrolled corpus averaged $0.356$ and this one averages "
        f"${j_all.mean():.3f}$. It "
        + (f"did: ${prev_h1:+.1f}$\\,pp uncontrolled against ${h1['mean']*100:+.1f}$\\,pp here."
           if grew else
           f"did not: ${prev_h1:+.1f}$\\,pp uncontrolled against ${h1['mean']*100:+.1f}$\\,pp "
           f"here. That is evidence against our own post-mortem: if a stronger, fixed attack "
           f"does not move the effect, then attack drift was not what suppressed it, and the "
           f"predecessor's shrinkage needs another explanation.")
        + f" We report that comparison descriptively and attach no test to it: the two runs share "
        f"their sources, so they are not independent, and nothing about the difference was "
        f"randomised.\n\n"
        + (f"What the successor buys, then, is not a $p$-value but a defensible one. The "
           f"predecessor could not distinguish a detector that survives paraphrasing from an "
           f"attack that was too gentle to test it; this design can, because the attack is now a "
           f"fixed, reported quantity rather than a label. "
           if survivors else
           f"The question therefore remains open, but it is now open in a much narrower way. "
           f"The predecessor left two explanations standing: the detector really is robust, or "
           f"the attack was too gentle to tell. With strength pinned to a band at the strong end "
           f"of what the predecessor produced, the second explanation is no longer available at "
           f"this corpus size: what remains unresolved is resolution, not construct validity. ")
        + f"Per the registered stopping rule the corpus is not extended again and the band is not "
        f"adjusted; the residual limits are the ones the design cannot remove: "
        f"{n_test_ph:.0f} held-out phishing messages per split make a single miss worth "
        f"${100.0/n_test_ph:.1f}$\\,pp, and the rewriter is the model that authored the corpus, "
        f"so shared style still places the attack nearer the training distribution than an "
        f"independent attacker's would.\n")
    write_generated(os.path.join(SEC, "gen_paraphrase_band_verdict.tex"), prose)

    pd.DataFrame([{"key": t["key"], "label": t["label"], "mean_pp": t["mean"] * 100,
                   "p": t["p"], "p_naive": t["p_naive"], "p_adj": t["p_adj"],
                   "reject": int(t["reject"]), "consistent": t["consistent"], "k": SEEDS,
                   "corpus_multiple_needed": t["need"]} for t in tests]).to_csv(
        os.path.join(ROOT, "data", "processed", "p3", "p3_band_stats.csv"), index=False)

    print(f"    band achieved: mean J={j_all.mean():.3f} "
          f"[{j_all.min():.3f}, {j_all.max():.3f}] over {len(j_all)} rewrites")
    for c in CELLS:
        print(f"    {c}: {mean[c]*100:5.1f}% ± {sd[c]*100:.1f}")
    for t in tests:
        print(f"    {t['key']:5s} Δ={t['mean']*100:+6.1f}pp  p={t['p']:.4f} "
              f"(naive {t['p_naive']:.4f})  BH={t['p_adj']:.4f} "
              f"{'REJECT-NULL' if t['reject'] else 'no evidence'}  "
              f"{t['consistent']}/{SEEDS}  needs {t['need']:.1f}x")
    print(f"    uncontrolled H1 was {prev_h1:+.1f}pp; controlled H1 is {h1['mean']*100:+.1f}pp")


if __name__ == "__main__":
    main()
