#!/usr/bin/env python3
"""
make_p3_assets.py — Regenerate P1b's content/JS fusion table from the crawled landing pages, so
the manuscript numbers never drift from the code. Produces:

  papers/P3_multimodal/sections/tab_content_fusion.tex — modality ablation on the content subset

Preliminary content experiment: TF-IDF Vietnamese page text + CompPhish URL features + lightweight
JavaScript features, fused by Logistic Regression, evaluated over stratified splits (the crawled
content subset is small and mostly undated, so we use stratified splits and say so). The screenshot
(ResNet18), PhoBERT and CodeBERT encoders are available as options in train_content_fusion.py but
are blocked in this environment by missing torchvision / a torch<2.6 safetensors constraint.

RUN:  python scripts/make_p3_assets.py
Why each table and verdict is shaped the way it is: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import csv
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated

# A caption opens with a count in words, spelled out here because a caption starting with a
# numeral reads as a label. The call was introduced without this helper on 2026-08-21, so
# make_encoder_sweep_table raised NameError at every run until 2026-08-22.
_NUM_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
              "ten", "eleven", "twelve")


def num_word(n: int) -> str:
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else f"{n:,}"
from train_content_fusion import load_pages, evaluate
from paired_eval import bh_adjust, corrected_paired_t, fmt_p as _p

SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
# language-matched, parked/dead-filtered manifest: Vietnamese-content pages ONLY in BOTH classes.
# The raw crawl mixed international phishing with .vn benign, so an unfiltered content
# classifier separates partly on LANGUAGE -- the confound this removes.
MANIFEST = os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv")
SEEDS = 5
# More splits than the cross-dataset study on purpose: on ~520 balanced pages 5 splits move the
# means enough to REORDER the configurations. Differences are never read off these means;
# p3_paired_test.py compares paired on identical splits.
FUSION_SEEDS = 20
CONFIGS = ["url", "content", "js", "content+url", "content+url+js"]
LABELS = {"url": "URL / infrastructure only", "content": "Content (page text) only",
          "js": "JavaScript only", "content+url": "Content $+$ URL (fusion)",
          "content+url+js": "Content $+$ URL $+$ JS"}




# The four-corpus transfer matrix and every intervention run on it moved to
# scripts/make_p2_bench_assets.py on 2026-08-19 with the sections that report them:
# P2 owns the matrix, and Editorial Manager flattens the upload so \input across paper
# directories does not survive submission. No number changed in the move.


def _balanced_rows():
    """The 1:1 ablation subset every content table shares: the manifest is heavily benign-skewed,
    so downsample the majority class (seed 42, matching train_content_fusion.py --balance). Size
    follows the manifest, not pinned. Factored out so per-encoder tables and the sweep provably
    share one subset AND one split sequence (evaluate() derives splits from row list + seed alone)."""
    rows = load_pages(MANIFEST)
    rng = random.Random(42)
    ph = [r for r in rows if r["y"] == 1]
    be = [r for r in rows if r["y"] == 0]
    k = min(len(ph), len(be))
    return rng.sample(ph, k) + rng.sample(be, k)


_PR_PANELS: list = []

# Panels drawn by the PR figure: TF-IDF is the canonical table's encoder, XLM-R one where the
# ranking effect actually separates — either alone would mislead.
_PR_PANEL_TITLES = {"tfidf": "char-$n$-gram TF-IDF (canonical)", "_xlmr": "XLM-R"}


def _fig_pr_curves():
    """P3's content-fusion claim (Section 6: URL-on-content buys ranking quality, only where the
    content channel is weak), drawn as PR curves with the $0.5$ operating point marked. Two panels
    because one would lie either way: on canonical TF-IDF fusion and content are indistinguishable
    (PR-AUC $+0.010$, $p=0.48$); on XLM-R the separation is real ($+0.036$, $p=0.031$). The pair
    says the effect exists and is conditional on the encoder."""
    if not _PR_PANELS:
        return
    import numpy as np
    from figstyle import apply, ORANGE, BLUE, GRAY
    plt = apply()
    from sklearn.metrics import precision_recall_curve

    show = [("content", "content only", BLUE, "-"),
            ("content+url", "content $+$ URL (fusion)", ORANGE, "-"),
            ("url", "URL only", GRAY, "--")]
    grid = np.linspace(0.02, 1.0, 200)
    panels = [pn for pn in _PR_PANELS if pn[0] in _PR_PANEL_TITLES]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    n_splits = 0
    for ax, (tag, curves, agg, n_ph, n_be) in zip(axes, panels):
        for cfg, lab, col, ls in show:
            if not curves.get(cfg):
                continue
            n_splits = len(curves[cfg])
            interp, ops = [], []
            for yte, sc in curves[cfg]:
                prec, rec, _ = precision_recall_curve(yte, sc)
                # precision_recall_curve returns recall DEscending; flip for interpolation
                interp.append(np.interp(grid, rec[::-1], prec[::-1]))
                pred = (sc >= 0.5).astype(int)
                tp = int(((pred == 1) & (yte == 1)).sum())
                ops.append((tp / max(1, int((yte == 1).sum())),
                            tp / max(1, int((pred == 1).sum()))))
            ax.plot(grid, np.mean(interp, axis=0), ls, color=col, lw=1.8, zorder=3,
                    label=f"{lab} — {np.mean(agg[cfg]['PR-AUC']):.3f}")
            ax.plot(np.mean([o[0] for o in ops]), np.mean([o[1] for o in ops]), "o",
                    color=col, ms=7, mec="white", mew=1.0, zorder=4)
        ax.set_xlabel("recall")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="lower left", title="PR-AUC",
                  title_fontsize=8, alignment="left")
        ax.set_title(_PR_PANEL_TITLES[tag], fontsize=9)
    axes[0].set_ylabel("precision")
    fig.suptitle(f"mean over {n_splits} stratified splits; dots mark the $0.5$ operating point",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(ROOT, "papers", "P3_multimodal", "figures", "fig_pr_fusion.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def make_content_fusion_table(text_encoder="tfidf", tag=""):
    """tag='' emits the canonical tab_content_fusion/gen_fusion_verdict pair; a non-empty tag
    (e.g. '_xlmr') emits a sibling table/verdict for that content encoder on the same subset."""
    rows = _balanced_rows()
    n_ph = n_be = sum(r["y"] for r in rows)
    agg, curves = evaluate(rows, CONFIGS, text_encoder, "lightweight", False, FUSION_SEEDS,
                           return_scores=True)
    _PR_PANELS.append((tag or "tfidf", curves, agg, n_ph, n_be))

    def row(cfg):
        m = agg[cfg]
        return " & ".join(f"{np.mean(m[k]):.3f}\\,$\\pm$\\,{np.std(m[k]):.3f}"
                          for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"))
    body = "\n".join(f"{LABELS[c]} & {row(c)} \\\\" +
                     ("\n\\midrule" if c == "js" else "") for c in CONFIGS)

    def f1(cfg):
        return f"{np.mean(agg[cfg]['F1']):.3f}"
    fus, con, url = f1("content+url"), f1("content"), f1("url")
    fpr = f"{np.mean(agg['content+url']['FPR@R0.90']):.3f}"

    # The verdict is DERIVED FROM A PAIRED TEST, never from comparing means -- two earlier caption
    # versions failed that way. Configs share splits, so scores are paired, and the test is the
    # CORRECTED resampled t: the ordinary paired t inflated every t ~3.1x here.
    def paired(a, b, metric):
        d = np.asarray(agg[a][metric], float) - np.asarray(agg[b][metric], float)
        s = corrected_paired_t(d)
        return s["mean"], s["p"], s["wins"], s["k"], s["p_naive"]

    d_f1, p_f1, w_f1, n, p_f1_naive = paired("content+url", "content", "F1")
    d_pr, p_pr, _, _, _ = paired("content+url", "content", "PR-AUC")
    d_ru, p_ru, w_ru, _, _ = paired("content+url", "url", "F1")
    sig = lambda p: p == p and p < 0.05
    f1_claim = (f"the fusion is indistinguishable from content alone on F1 "
                f"(paired difference ${d_f1:+.3f}$ over the same {n} splits, ${_p(p_f1)}$, "
                f"{w_f1}/{n} splits favouring fusion)" if not sig(p_f1) else
                f"the fusion {'beats' if d_f1 > 0 else 'trails'} content alone on F1 "
                f"(${d_f1:+.3f}$ paired, ${_p(p_f1)}$)")
    pr_claim = (f"it does improve ranking quality: ${d_pr:+.3f}$ PR-AUC over content alone "
                f"(${_p(p_pr)}$)" if sig(p_pr) else
                f"nor does it separate on PR-AUC (${d_pr:+.3f}$, ${_p(p_pr)}$)")
    d_ro, p_ro, _, _, _ = paired("content+url", "content", "ROC-AUC")
    ro_claim = (f"The ROC-AUC difference (${d_ro:+.3f}$, ${_p(p_ro)}$) points the same way "
                f"{'and also reaches significance' if sig(p_ro) else 'without reaching significance, so we do not claim it'}."
                )
    # The content-vs-URL p was HARDCODED as "p<0.001" here and survived two regenerations; it is
    # now derived like every other number, so the correction can move it.
    verdict = (f"Measured against the same splits rather than by comparing means, and testing them "
               f"with the corrected resampled $t$ that overlapping splits require, {f1_claim}, but "
               f"{pr_claim}. {ro_claim} What is unambiguous is that both content-bearing "
               f"configurations beat URL features alone (fusion $-$ URL ${d_ru:+.3f}$ F1, "
               f"{w_ru}/{n} splits, ${_p(p_ru)}$); the fusion also holds FPR@90\\%rec.\\ at ${fpr}$")

    # Name the default encoder too: the headline finding is that fusion is encoder-dependent, and
    # a blank note left the claim-carrying table not saying which encoder it used.
    enc_note = (" Content encoder: char-$n$-gram TF-IDF." if text_encoder == "tfidf"
                else f" Content encoder: \\mbox{{{text_encoder}}} (frozen embeddings).")
    tex = f"""\\begin{{table*}}[t]
\\caption{{Content/JS modality ablation on {n_ph} phishing $+$ {n_be} benign
Vietnamese-content pages; mean\\,$\\pm$\\,std over {FUSION_SEEDS} stratified splits.{enc_note}}}
\\label{{tab:content_fusion{tag}}}
\\small
\\begin{{tabular}}{{l c c c c}}
\\toprule
Configuration & F1 & PR-AUC & ROC-AUC & FPR@90\\%rec. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    os.makedirs(SEC, exist_ok=True)
    write_generated(
        os.path.join(SEC, f"tab_content_fusion{tag}.tex"), tex,
        f"(encoder={text_encoder}; balanced {n_ph}:{n_be}; "
        f"fusion F1 {fus}, content {con}, url {url})")
    # The verdict ships as its own \input: prose position, generated content (hand-maintained, it
    # went stale twice). Canonical encoder only -- a sibling-encoder verdict is superseded by the
    # sweep prose, and emitting one produced an asset nothing \input, which no check can guard.
    if not tag:
        write_generated(os.path.join(SEC, "gen_fusion_verdict.tex"),
                        verdict.rstrip(". ") + ".\n")
    return rows, agg


SWEEP_CACHE = os.path.join("data", "processed", "p3", "p3_encoder_sweep.csv")


def _sweep_scores(encoders, refresh=False):
    """Per-split scores for every (encoder, config) in the sweep, cached to CSV. The sweep costs
    ~50 min (five encoders x 20 splits); caching the raw per-split scores makes statistical
    re-analyses (test change, family correction) free and auditable from disk.
    Pass refresh=True (--refresh-sweep) to re-measure from the models."""
    if not refresh and os.path.exists(SWEEP_CACHE):
        agg = {}
        with open(SWEEP_CACHE, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                agg.setdefault(r["encoder"], {}).setdefault(r["config"], {}) \
                   .setdefault(r["metric"], []).append(float(r["value"]))
        present = [e for e in encoders if e in agg]
        if present:
            print(f"[i] sweep scores from cache {SWEEP_CACHE} ({len(present)} encoders); "
                  f"--refresh-sweep to re-measure")
            return {e: agg[e] for e in present}, []
    rows = _balanced_rows()
    cfgs = ["url", "content", "content+url"]
    out, skipped, recs = {}, [], []
    for enc in encoders:
        try:
            a = evaluate(rows, cfgs, enc, "lightweight", False, FUSION_SEEDS)
        except Exception as e:
            skipped.append(f"{enc} ({type(e).__name__})")
            continue
        out[enc] = a
        for cfg, metrics in a.items():
            for metric, vals in metrics.items():
                for i, v in enumerate(vals):
                    recs.append({"encoder": enc, "config": cfg, "metric": metric,
                                 "split": i, "value": f"{float(v):.6f}"})
    if recs:
        os.makedirs(os.path.dirname(SWEEP_CACHE), exist_ok=True)
        with open(SWEEP_CACHE, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["encoder", "config", "metric", "split", "value"])
            w.writeheader(); w.writerows(recs)
        print(f"[+] {SWEEP_CACHE} ({len(recs)} rows)")
    return out, skipped


# --- the four metrics the sweep records, and which direction counts as an improvement. ------
# A claim that the URL channel "adds nothing" is a claim over every metric that could show it
# does, so all four print. F1 and PR-AUC only was a selective-reporting defect: the omitted
# ROC-AUC column carries the one fusion-minus-content separation in the panel.
SWEEP_METRICS = (
    ("F1", "$\\Delta$F1", +1),
    ("PR-AUC", "$\\Delta$PR-AUC", +1),
    ("ROC-AUC", "$\\Delta$ROC-AUC", +1),
    ("FPR@R0.90", "$\\Delta$FPR@90\\%rec.", -1),      # lower is better
)
SWEEP_DIRECTIONS = (("uc", ("content+url", "content")),   # adding URL to content
                    ("cu", ("content+url", "url")))       # adding content to URL
ENC_LABEL = {"tfidf": "char-$n$-gram TF-IDF", "phobert": "PhoBERT",
             "phobert-v2": "PhoBERT-v2", "visobert": "ViSoBERT", "xlm-r": "XLM-R"}


def make_encoder_sweep_table(encoders=("tfidf", "phobert", "phobert-v2", "visobert", "xlm-r"),
                             refresh=False):
    """Cross-encoder summary: does the fusion beat content alone, per content encoder?

    Every encoder runs on the SAME balanced subset (seed 42) and SAME 20 splits, verdicts from the
    corrected resampled t, so the rows are directly comparable. All four cached metrics print in
    both directions -- printing two of them let the paper conclude the URL channel "improves
    neither the operating point nor the ranking for any encoder" while the omitted ROC-AUC column
    falsified it for one encoder. Printed p-values are BH q-values adjusted within each column, so
    adding a column leaves the published ones unchanged; the caption discloses the stricter
    all-columns correction whenever it changes a count."""
    n_ph = sum(r["y"] for r in _balanced_rows() if r["y"])
    scores, skipped = _sweep_scores(encoders, refresh=refresh)
    verdicts = []
    for enc in encoders:
        agg = scores.get(enc)
        if agg is None:
            continue
        rec = {"enc": enc,
               "mean": {cfg: {m: float(np.mean(agg[cfg][m])) for m, _l, _s in SWEEP_METRICS}
                        for cfg in ("url", "content", "content+url")}}
        for direction, (a, b) in SWEEP_DIRECTIONS:
            for metric, _lab, _sign in SWEEP_METRICS:
                rec[(direction, metric)] = corrected_paired_t(
                    np.asarray(agg[a][metric], float) - np.asarray(agg[b][metric], float))
        verdicts.append(rec)
    if not verdicts:
        print("[i] no encoder ran — skipping the sweep table."); return
    k = len(verdicts)
    cols = [(d, m) for d, _ in SWEEP_DIRECTIONS for m, _l, _s in SWEEP_METRICS]

    # BH within each column (family = one question across K encoders), and across all 8K tests
    # for the caption's disclosure.
    for col in cols:
        q, _ = bh_adjust([v[col]["p"] for v in verdicts])
        for v, qi in zip(verdicts, q):
            v[col]["q"] = qi
    q_all, _ = bh_adjust([v[col]["p"] for col in cols for v in verdicts])
    for i, col in enumerate(cols):
        for j, v in enumerate(verdicts):
            v[col]["q_all"] = q_all[i * k + j]

    sign_of = {m: s for m, _l, s in SWEEP_METRICS}

    def win(v, col, field="q"):
        """A win is an improvement IN THE METRIC'S OWN DIRECTION that clears the bar. FPR@90%rec.
        improves by falling, so a sign-blind test would have counted its four largest
        improvements as failures and its one deterioration as a candidate win."""
        return sign_of[col[1]] * v[col]["mean"] > 0 and v[col][field] < 0.05

    def cell(v, col):
        s = v[col]
        body = f"{s['mean']:+.3f}"
        return ("\\textbf{" + body + "}" if win(v, col) else body) + \
               f" & {{\\scriptsize {_p(s['q'], 'q')}}}"

    def n_win(col, field="q"):
        return sum(1 for v in verdicts if win(v, col, field))

    def winners(col):
        return [ENC_LABEL.get(v["enc"], v["enc"]) for v in verdicts if win(v, col)]

    n_tests = len(cols) * k
    n_naive = sum(1 for v in verdicts for col in cols
                  if sign_of[col[1]] * v[col]["mean"] > 0 and v[col]["p_naive"] < 0.05)
    n_raw = sum(1 for v in verdicts for col in cols
                if sign_of[col[1]] * v[col]["mean"] > 0 and v[col]["p"] < 0.05)
    n_adj = sum(n_win(col) for col in cols)
    n_strict = sum(n_win(col, "q_all") for col in cols)
    # The all-columns correction can go EITHER way and the disclosure has to survive both: pooling
    # eight columns of five lends the sparse ones power borrowed from the dense, so the
    # single-family correction is not automatically stricter (the sentence once said "would
    # retain 18 of these 17").
    if n_strict == n_adj:
        strict_note = ""
    elif n_strict < n_adj:
        strict_note = (f" Treating all ${n_tests}$ tests as a single family instead of one family "
                       f"per question would retain ${n_strict}$ of these ${n_adj}$ separations.")
    else:
        strict_note = (f" We correct per question rather than across all ${n_tests}$ tests at "
                       f"once, which is the more conservative choice here and not the more "
                       f"lenient one: a single family would return ${n_strict}$ separations "
                       f"rather than ${n_adj}$, because the columns dense in small $p$-values "
                       f"lend rank to the sparse ones.")
    note = (f" Checkpoints unavailable in this environment: {', '.join(skipped)}." if skipped else "")

    head_metrics = " & ".join("\\multicolumn{2}{c}{" + lab + "}" for _m, lab, _s in SWEEP_METRICS)

    def body_rows(direction, lead):
        return "\n".join(
            f"{ENC_LABEL.get(v['enc'], v['enc'])} & {lead(v)} & "
            + " & ".join(cell(v, (direction, m)) for m, _l, _s in SWEEP_METRICS) + " \\\\"
            for v in verdicts)

    tex_a = f"""\\begin{{table*}}[t]
\\caption{{\\textbf{{Adding the URL channel to content}}: paired (content$+$URL) $-$ (content),
{num_word(k)} encoders, four metrics, same {n_ph}$+${n_ph} subset and {FUSION_SEEDS} splits. Benjamini--Hochberg $q$ per
column; \\textbf{{bold}} marks an improvement at $q<0.05$. Reverse direction:
Table~\\ref{{tab:encsweepcu}}.{note}}}
\\label{{tab:encsweep}}
\\small\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l cc rl rl rl rl}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{F1 (mean)}} & \\multicolumn{{8}}{{c}}{{paired $\\Delta$, content$+$URL minus content}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-11}}
Content encoder & content & fusion & {head_metrics} \\\\
\\midrule
{body_rows("uc", lambda v: f"{v['mean']['content']['F1']:.3f} & {v['mean']['content+url']['F1']:.3f}")}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    write_generated(os.path.join(SEC, "tab_encoder_sweep.tex"), tex_a,
                    "(" + ", ".join(f"{m} {n_win(('uc', m))}/{k}" for m, _l, _s in SWEEP_METRICS) + ")")

    tex_b = f"""\\begin{{table*}}[t]
\\caption{{\\textbf{{Adding the content channel to URL features}}: paired (content$+$URL) $-$ (URL)
against the encoder-independent URL-only row of Table~\\ref{{tab:content_fusion}}; subset, splits,
columns, correction and bolding as in Table~\\ref{{tab:encsweep}}.}}
\\label{{tab:encsweepcu}}
\\small\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l c rl rl rl rl}}
\\toprule
 & F1 (mean) & \\multicolumn{{8}}{{c}}{{paired $\\Delta$, content$+$URL minus URL}} \\\\
\\cmidrule(lr){{2-2}} \\cmidrule(lr){{3-10}}
Content encoder & fusion & {head_metrics} \\\\
\\midrule
{body_rows("cu", lambda v: f"{v['mean']['content+url']['F1']:.3f}")}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    write_generated(os.path.join(SEC, "tab_encoder_sweep_cu.tex"), tex_b,
                    "(" + ", ".join(f"{m} {n_win(('cu', m))}/{k}" for m, _l, _s in SWEEP_METRICS) + ")")

    def phrase(n):
        if n == 0:
            return "none of the encoders"
        return "one of the encoders" if n == 1 else f"{n} of the {k} encoders"

    def detail(col):
        """Name the winners of a column, with the effect and its q, so the sentence cannot claim
        a separation the table does not print."""
        w = [v for v in verdicts if win(v, col)]
        if not w:
            return ""
        return " (" + ", ".join(
            f"{ENC_LABEL.get(v['enc'], v['enc'])} ${v[col]['mean']:+.3f}$, {_p(v[col]['q'], 'q')}"
            for v in w) + ")"

    uc = {m: n_win(("uc", m)) for m, _l, _s in SWEEP_METRICS}
    cu = {m: n_win(("cu", m)) for m, _l, _s in SWEEP_METRICS}
    uc_rank = uc["PR-AUC"] + uc["ROC-AUC"]
    uc_thr = uc["F1"] + uc["FPR@R0.90"]
    if uc_thr == 0 and uc_rank == 0:
        closing = ("on this subset the URL channel improves neither the operating point nor the "
                   "ranking for any encoder, so no measurable contribution of its own survives "
                   "once content is present")
    elif uc_thr == 0:
        closing = ("on this subset the URL channel moves no operating point for any encoder, and "
                   "what survives of its own contribution is a ranking gain confined to "
                   + ", ".join(sorted(set(winners(("uc", "PR-AUC")) + winners(("uc", "ROC-AUC")))))
                   + ", so the fusion premise holds in a narrower form than either the "
                     "threshold or the PR-AUC column alone would have shown")
    else:
        closing = ("the URL channel's own contribution is confined to the "
                   f"{uc_thr} threshold and {uc_rank} ranking separations the table bolds")
    sent = (
        f"Across all {k} content encoders on the identical subset and splits, the ablation is "
        f"asymmetric. Both directions are reported on all four metrics the experiment records "
        f"(F1 and FPR@90\\%rec.\\ at the decision threshold, PR-AUC and ROC-AUC for the ranking) "
        f"because a claim that a channel adds nothing is a claim over every metric that could show "
        f"that it does. Because each question is asked of every encoder, we control the "
        f"false-discovery rate within each column~\\cite{{benjaminihochberg}} and report "
        f"$q$-values: of the {n_tests} comparisons, {n_naive} reach $p<0.05$ under the ordinary "
        f"paired $t$, {n_raw} still do once the variance of overlapping resamples is corrected "
        f"for, and {n_adj} survive false-discovery control on top of that.{strict_note} "
        f"Adding the URL channel to content helps {phrase(uc['F1'])} at the decision threshold "
        f"and {phrase(uc['FPR@R0.90'])} on the false-positive rate at $90\\%$ recall; it improves "
        f"the ranking for {phrase(uc['PR-AUC'])} on PR-AUC and {phrase(uc['ROC-AUC'])} on "
        f"ROC-AUC{detail(('uc', 'ROC-AUC'))}. Adding the content channel to URL features, by "
        f"contrast, helps {phrase(cu['F1'])} on the thresholded decision itself and "
        f"{phrase(cu['FPR@R0.90'])} on the false-positive rate, and improves the ranking for "
        f"{phrase(cu['PR-AUC'])} on PR-AUC and {phrase(cu['ROC-AUC'])} on ROC-AUC. "
        f"Content is doing the work "
        f"the fusion is credited with; " + closing)
    write_generated(os.path.join(SEC, "gen_encoder_sweep.tex"), sent.rstrip(". ") + ".\n")

    # The operating-point ordering the discussion reads off this sweep. It used to be typed, and the
    # sweep does not say it: the best FPR at 90% recall belongs to a MONOLINGUAL backbone, and the
    # claim held only of the configurations that happened to have their own table.
    family = {"tfidf": "encoder-free", "phobert": "monolingual", "phobert-v2": "monolingual",
              "visobert": "monolingual", "xlm-r": "multilingual"}
    fprs = sorted(((v["mean"]["content+url"]["FPR@R0.90"], v["enc"]) for v in verdicts))
    best_v, best_e = fprs[0]
    ml = [(f, e) for f, e in fprs if family.get(e) == "multilingual"]
    ml_txt = (f", against the multilingual {ENC_LABEL.get(ml[0][1], ml[0][1])}'s "
              f"${ml[0][0]:.3f}$" if ml else "")
    write_generated(
        os.path.join(SEC, "gen_encoder_operating.tex"),
        f"the lowest false-positive rate at $90\\%$ recall of any fusion in the sweep belongs to "
        f"the {family.get(best_e, '')} {ENC_LABEL.get(best_e, best_e)} (${best_v:.3f}$){ml_txt}, "
        f"and the five fusions span ${fprs[0][0]:.3f}$--${fprs[-1][0]:.3f}$\n",
        f"(best {best_e} {best_v:.3f})")

    for v in verdicts:
        print(f"[i] {v['enc']:11s} content={v['mean']['content']['F1']:.3f} "
              f"fusion={v['mean']['content+url']['F1']:.3f} | " + " ".join(
                  f"{d}/{m}={v[(d, m)]['mean']:+.3f}(q={float(v[(d, m)]['q']):.3f})"
                  for d, _ in SWEEP_DIRECTIONS for m, _l, _s in SWEEP_METRICS), flush=True)


def make_hybrid_head_table(rows, agg_lin, text_encoder="xlm-r", tag="_xgb"):
    """Hybrid-stack head study: same concatenated blocks, XGBoost head instead of linear — the
    minimal transformer->XGBoost stack the hybrid-DL literature advertises. `rows`/`agg_lin` must
    come from make_content_fusion_table for the same encoder: evaluate() derives splits from
    (rows, seed) alone, so the head effect is a paired difference, not a comparison of means.
    Emits tab_content_fusion_xgb.tex + gen_hybrid_verdict.tex."""
    from train_content_fusion import evaluate as _eval
    agg = _eval(rows, CONFIGS, text_encoder, "lightweight", False, FUSION_SEEDS, head="xgboost")

    def row(cfg):
        m = agg[cfg]
        return " & ".join(f"{np.mean(m[k]):.3f}\\,$\\pm$\\,{np.std(m[k]):.3f}"
                          for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"))
    body = "\n".join(f"{LABELS[c]} & {row(c)} \\\\" +
                     ("\n\\midrule" if c == "js" else "") for c in CONFIGS)
    tex = f"""\\begin{{table*}}[t]
\\caption{{Hybrid transformer$\\to$boosted-trees stack: Table~\\ref{{tab:content_fusion_xlmr}} with the
linear fusion head replaced by XGBoost, same subset, splits and \\mbox{{{text_encoder}}} embeddings;
mean\\,$\\pm$\\,std over {FUSION_SEEDS} stratified splits.}}
\\label{{tab:content_fusion{tag}}}
\\small
\\begin{{tabular}}{{l c c c c}}
\\toprule
Configuration & F1 & PR-AUC & ROC-AUC & FPR@90\\%rec. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    write_generated(
        os.path.join(SEC, f"tab_content_fusion{tag}.tex"), tex,
        f"(head=xgboost; fusion F1 {np.mean(agg['content+url']['F1']):.3f} "
        f"vs linear {np.mean(agg_lin['content+url']['F1']):.3f})")

    # Verdict derived from paired tests only (same discipline as gen_fusion_verdict): the two head
    # runs share every split, and the dilution re-test is paired within the boosted head.
    def paired(a_scores, b_scores):
        s = corrected_paired_t(np.asarray(a_scores, float) - np.asarray(b_scores, float))
        return s["mean"], s["p"], s["wins"], s["k"]

    sig = lambda p: p == p and p < 0.05

    def head_claim(cfg, name):
        d_f1, p_f1, w_f1, n = paired(agg[cfg]["F1"], agg_lin[cfg]["F1"])
        d_pr, p_pr, _, _ = paired(agg[cfg]["PR-AUC"], agg_lin[cfg]["PR-AUC"])
        f1c = (f"{'gains' if d_f1 > 0 else 'loses'} ${d_f1:+.3f}$ F1 "
               f"(${_p(p_f1)}$, {w_f1}/{n} splits)" if sig(p_f1) else
               f"does not move F1 (${d_f1:+.3f}$, ${_p(p_f1)}$)")
        prc = (f"{'gains' if d_pr > 0 else 'loses'} ${d_pr:+.3f}$ PR-AUC (${_p(p_pr)}$)"
               if sig(p_pr) else f"leaves PR-AUC unmoved (${d_pr:+.3f}$, ${_p(p_pr)}$)")
        return f"on {name} the boosted head {f1c} and {prc}"

    d_dil, p_dil, w_dil, n = paired(agg["content+url+js"]["F1"], agg["content+url"]["F1"])
    d_dpr, p_dpr, _, _ = paired(agg["content+url+js"]["PR-AUC"], agg["content+url"]["PR-AUC"])
    dil = (f"the three-channel stack {'now beats' if d_dil > 0 else 'still trails'} "
           f"content$+$URL under it (${d_dil:+.3f}$ F1, ${_p(p_dil)}$, {w_dil}/{n} splits)"
           if sig(p_dil) else
           f"the three-channel stack is no longer separable from content$+$URL under it "
           f"(${d_dil:+.3f}$ F1, ${_p(p_dil)}$)")
    dil += (f"; the PR-AUC verdict {'agrees' if (d_dpr > 0) == (d_dil > 0) or not sig(p_dpr) else 'differs'} "
            f"(${d_dpr:+.3f}$, ${_p(p_dpr)}$)" if sig(p_dpr) else
            f" (PR-AUC ${d_dpr:+.3f}$, ${_p(p_dpr)}$)")
    fpr = f"{np.mean(agg['content+url']['FPR@R0.90']):.3f}"
    verdict = (f"Paired on the same {n} splits under the corrected resampled $t$, "
               f"{head_claim('content+url', 'content$+$URL')}; "
               f"{head_claim('content', 'content alone')}. Within the boosted head, {dil}. "
               f"The hybrid's content$+$URL FPR@90\\%rec.\\ is ${fpr}$ (linear head: "
               f"${np.mean(agg_lin['content+url']['FPR@R0.90']):.3f}$)")
    write_generated(os.path.join(SEC, "gen_hybrid_verdict.tex"), verdict.rstrip(". ") + ".\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    # The sweep used to have no entry point in the repo (least reproducible table); it now always
    # runs (free from the score cache) and re-measures only when asked.
    ap.add_argument("--refresh-sweep", action="store_true",
                    help="re-measure the encoder sweep from the models (~50 min) instead of "
                         f"re-analysing the cached per-split scores in {SWEEP_CACHE}")
    args = ap.parse_args()
    make_content_fusion_table()
    xl_rows, xl_agg = make_content_fusion_table("xlm-r", "_xlmr")
    make_hybrid_head_table(xl_rows, xl_agg)
    make_encoder_sweep_table(refresh=args.refresh_sweep)
    _fig_pr_curves()
    print("Done. Recompile the P1b manuscript to pick up the regenerated table.")


if __name__ == "__main__":
    main()
