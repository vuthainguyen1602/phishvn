#!/usr/bin/env python3
"""Figures and tables for the smishing study, from the ingested corpus.

Every number is read from data/processed/sms/*.csv, which sms_corpus_import.py writes. Nothing is
hand-typed, for the reason figstyle.py records: `make claims` verifies .tex, not compiled PDFs, so
a literal inside a figure is the one place a stale number survives a data change unseen.

RUN:  python3 scripts/make_smishing_assets.py   (after sms_corpus_import.py)
"""
from __future__ import annotations
import collections, csv, datetime as dt, io, json, os, re, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from figstyle import apply, ORANGE, BLUE, FILL_B, FILL_O, INK, GRAY  # noqa: E402
from genfile import write_generated  # noqa: E402

plt = apply()

MSG = os.path.join(ROOT, "data", "processed", "sms", "sms_messages.csv")
URLS = os.path.join(ROOT, "data", "processed", "sms", "sms_urls.csv")
SNAP = os.path.join(ROOT, "data", "processed", "sms", "sms_snapshot.json")
FUSION = os.path.join(ROOT, "data", "processed", "sms", "fusion_results.json")
WHY = os.path.join(ROOT, "data", "processed", "sms", "why_fusion.json")
CUES = os.path.join(ROOT, "data", "processed", "sms", "shallow_cues.json")
ROB = os.path.join(ROOT, "data", "processed", "sms", "split_robustness.json")
FUSION_PSL = os.path.join(ROOT, "data", "processed", "sms", "fusion_results_psl.json")
BATCH = os.path.join(ROOT, "data", "processed", "sms", "batch_audit.json")
EXTRA = os.path.join(ROOT, "data", "processed", "sms", "extra_diagnostics.json")
TEMPORAL = os.path.join(ROOT, "data", "processed", "sms", "temporal_audit.json")
FIG = os.path.join(ROOT, "papers", "future_smishing", "figures")
SEC = os.path.join(ROOT, "papers", "future_smishing", "sections")

# The publisher defines label 1 as spam/scam, not phishing; every legend says so.
LAB = {"0": "ham", "1": "positive (spam/scam)"}
SRC = os.path.join(ROOT, "data", "raw", "sms_hf_full", "full_dataset.csv")
EMB_META = os.path.join(ROOT, "data", "processed", "sms", "phobert_emb.npy.meta.json")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    for f in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime((s or "").strip(), f).date()
        except ValueError:
            pass
    return None


def fig_timeline(msgs, out):
    """The class mix moves across the publisher's date values, which a random split cannot see.
    The card leaves `date` undefined and it takes few distinct values, so the axis is the month of
    that field, not of receipt."""
    by = collections.defaultdict(collections.Counter)
    for m in msgs:
        d = parse_date(m["date"])
        if d:
            by[f"{d:%Y-%m}"][m["label"]] += 1
    months = sorted(by)
    ham = [by[k]["0"] for k in months]
    ph = [by[k]["1"] for k in months]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(months))
    w = 0.38
    ax.bar(x - w / 2, ph, w, color=ORANGE, edgecolor=INK, label=LAB["1"])
    ax.bar(x + w / 2, ham, w, color=FILL_B, edgecolor=INK, hatch="///", label="ham")
    for i, v in enumerate(ph):
        ax.text(i - w / 2, v + 12, str(v), ha="center", fontsize=7, color=INK)
    for i, v in enumerate(ham):
        ax.text(i + w / 2, v + 12, str(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, months)
    ax.set_ylabel("rows")
    ax.set_xlabel("month of the publisher's date field")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(max(ham), max(ph)) * 1.22)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return months, ham, ph


def fig_hosts(urls, out):
    """Concentration: ham reuses a few corporate domains, phishing burns one per message."""
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for lab, colour, style in (("1", ORANGE, "-"), ("0", BLUE, "--")):
        c = collections.Counter(u["apex"] for u in urls if u["label"] == lab)
        counts = np.array(sorted(c.values(), reverse=True), dtype=float)
        if not len(counts):
            continue
        share = np.cumsum(counts) / counts.sum()
        frac = np.arange(1, len(counts) + 1) / len(counts)
        ax.plot(frac * 100, share * 100, style, color=colour, lw=1.8,
                label=f"{LAB[lab]} ({len(counts)} domains)")
    ax.plot([0, 100], [0, 100], ":", color=GRAY, lw=1, label="one URL per domain")
    ax.set_xlabel("registrable domains, ranked by URL count (%)")
    ax.set_ylabel("share of URLs (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_tld(urls, out, top=8):
    """Where the two classes register. The phishing bar chart is the lexical signal."""
    per = {lab: collections.Counter(u["tld"] for u in urls if u["label"] == lab)
           for lab in ("0", "1")}
    names = [t for t, _ in per["1"].most_common(top)]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    y = np.arange(len(names))
    h = 0.38
    tot = {lab: max(sum(per[lab].values()), 1) for lab in per}
    ax.barh(y + h / 2, [100 * per["1"][t] / tot["1"] for t in names], h,
            color=ORANGE, edgecolor=INK, label=LAB["1"])
    ax.barh(y - h / 2, [100 * per["0"][t] / tot["0"] for t in names], h,
            color=FILL_B, edgecolor=INK, hatch="///", label="ham")
    ax.set_yticks(y, [f".{t}" for t in names])
    ax.invert_yaxis()
    ax.set_xlabel("share of that class's URLs (%)")
    # not lower right: .vn is 73% of ham and the legend sat on top of its bar
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_xlim(0, max(100 * per["0"][t] / tot["0"] for t in names) * 1.12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return per


def fig_deltas(r, boot, why, out):
    """The paper's central null, drawn once instead of stated three times in prose.

    Every contrast that asks what the URL channel is worth sits here against zero, with the
    distinct-text cluster bootstrap's interval. Reading three signed numbers and three brackets out
    of two paragraphs cannot separate "no measurable difference" from "interval too wide to say",
    and that distinction is the whole claim. Drawn only for the contrasts whose interval exists;
    an arm whose bootstrap is missing is dropped rather than shown without one.
    """
    rows = []
    d = r.get("posthoc_fusion_minus_text", {})
    c = boot.get("posthoc_fusion_minus_text", {}).get("ci95")
    if d and c:
        rows.append(("fusion $-$ text only", d["mean"], c))
    for key, lab in (("mask_effect", "text with URLs masked\n$-$ text only"),
                     ("url_adds_to_masked", "masked text $+$ 21 URL columns\n$-$ masked text")):
        w = (why or {}).get(key, {})
        if w.get("ci95"):
            rows.append((lab, w["mean"], w["ci95"]))
    if not rows:
        return None

    # single-column width: the panel holds three rows, and at double width most of it was empty
    fig, ax = plt.subplots(figsize=(3.5, 0.62 * len(rows) + 1.25))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0.0, color=INK, lw=1.0, zorder=1)
    for yy, (_lab, m, ci) in zip(y, rows):
        ax.plot([ci[0], ci[1]], [yy, yy], color=BLUE, lw=2.2, solid_capstyle="butt", zorder=2)
        ax.plot([m], [yy], "o", ms=6, color=ORANGE, mec=INK, mew=0.7, zorder=3)
        sg = lambda v: f"{v:+.3f}".replace("-", "\u2212")
        ax.text(ci[1] + 0.0012, yy, f"{sg(m)}\n[{sg(ci[0])}, {sg(ci[1])}]",
                va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r_[0] for r_ in rows], fontsize=7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    lo = min(ci[0] for _l, _m, ci in rows)
    hi = max(ci[1] for _l, _m, ci in rows)
    pad = (hi - lo) * 0.10
    ax.set_xlim(lo - pad, hi + (hi - lo) * 0.62)
    ax.set_xlabel("difference in positive-class F1", fontsize=7.5)
    ax.set_xticks([t for t in (-0.01, 0.0, 0.01, 0.02) if lo - pad <= t <= hi + (hi - lo) * 0.62])
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return rows


def fig_robust(cues, fus, out):
    """The similarity-aware refit beside the registered one, with each split's own floor.

    Deliberately NOT a slope chart. The two splits have different test sets, of different size and
    class mix, so the arms are not paired and a line joining them would invite exactly the reading
    the section warns against -- that an arm improved. What transfers is the ORDERING inside each
    panel, so the panels are drawn separately on one x scale and the reader compares shapes.

    Both floors are drawn because the interesting detail is that one of them moves: the combined
    shallow-cue floor stays above the URL arm on both splits, while the redaction-token floor
    crosses it. Hiding the token floor would hide the one claim in this section that does not
    survive.
    """
    if not os.path.exists(ROB):
        return None
    rb = json.load(open(ROB, encoding="utf-8"))
    panels = [
        ("registered split",
         [fus["arms"]["url"]["f1"], fus["arms"]["text"]["f1"], fus["arms"]["fusion"]["f1"]],
         cues["baselines"]["all_shallow"]["f1"], cues["baselines"]["tokens"]["f1"]),
        ("similarity-grouped split (post-hoc)",
         [rb["arms"]["url"]["f1"], rb["arms"]["text"]["f1"], rb["arms"]["fusion"]["f1"]],
         rb["floors"]["all_shallow"], rb["floors"]["tokens"]),
    ]
    labels = ["URL only", "text only", "fusion"]
    cols = [BLUE, ORANGE, FILL_O]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.72), sharex=True, sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (title, vals, floor, tok) in zip(axes, panels):
        ax.barh(y, vals, 0.58, color=cols, edgecolor=INK, linewidth=0.7, zorder=2)
        # Drawn over the bars only: full-height axvlines ran through the panel title.
        ax.vlines(floor, y.min() - 0.45, y.max() + 0.45, color=GRAY, ls="--", lw=1.0, zorder=3)
        ax.vlines(tok, y.min() - 0.45, y.max() + 0.45, color=GRAY, ls=":", lw=1.2, zorder=3)
        # Values ride at a fixed x, clear of both floor lines: printed at the bar end they
        # collided with whichever line the bar happened to stop near.
        for yy, v in zip(y, vals):
            ax.text(v + 0.015, yy, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
        ax.set_title(title, fontsize=9, color=INK, pad=4)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.75)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("positive-class F1", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=9)
    # The two lines are the whole point of the panel and were unlabelled.
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=GRAY, ls="--", lw=1.0,
                               label="shallow-cue floor (diacritics + length + tokens)"),
                        Line2D([], [], color=GRAY, ls=":", lw=1.2,
                               label="bracketed tokens alone")],
               loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return panels


def fig_shortener(urls, out, operator_links):
    """Shortening per class, with ham split into Viettel's own app deep links and everything else:
    the pooled ham rate is mostly one operator's links, and a single bar hid that."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ph = [u for u in urls if u["label"] == "1"]
    hm = [u for u in urls if u["label"] == "0"]
    v_ph = 100 * sum(int(u["shortener"]) for u in ph) / max(len(ph), 1)
    v_op = 100 * sum(int(u["shortener"]) for u in hm if u["host"] in operator_links) / max(len(hm), 1)
    v_ot = 100 * sum(int(u["shortener"]) for u in hm if u["host"] not in operator_links) / max(len(hm), 1)
    x = [0, 1]
    ax.bar(x[0], v_ph, 0.5, color=ORANGE, edgecolor=INK)
    ax.bar(x[1], v_ot, 0.5, color=FILL_B, edgecolor=INK, hatch="///", label="other shorteners")
    ax.bar(x[1], v_op, 0.5, bottom=v_ot, color=BLUE, edgecolor=INK, hatch="///",
           label="Viettel app deep links")
    ax.text(x[0], v_ph + 0.4, f"{v_ph:.1f}%", ha="center", fontsize=9, color=INK)
    ax.text(x[1], v_op + v_ot + 0.4, f"{v_op + v_ot:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ["positive\n(spam/scam)", "ham"], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("URLs that are shortened (%)", fontsize=9)
    ax.set_ylim(0, (v_op + v_ot) * 1.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return [v_ph, v_op + v_ot]


def emit_results():
    """The registered comparison, if it has been run. Absent until then, and the paper says so."""
    if not os.path.exists(FUSION):
        return None
    r = json.load(open(FUSION, encoding="utf-8"))
    buf = io.StringIO()
    buf.write("\\begin{table}[t]\n\\centering\\small\n")
    pos = int(round(r["confusion_mean"]["text"]["tp"] + r["confusion_mean"]["text"]["fn"]))
    buf.write("\\caption{The two registered arms and their fusion on the registered test split "
              "(%d rows, %d positive), mean over %d seeds. F1, precision and recall are for the "
              "positive class at $\\tau = 0.5$. FPR is the share of ham flagged, and SD is the seed "
              "standard deviation of F1.}\n\\label{tab:sms_fusion}\n"
              % (r["n_test"], pos, r["seeds"]))
    buf.write("\\begin{tabular}{lrrrrr}\n\\toprule\n"
              "Arm & F1 & SD & Precision & Recall & FPR \\\\\n\\midrule\n")
    for k, name in (("url", "URL only (CompPhish-21)"), ("text", "Text only (frozen PhoBERT)"),
                    ("fusion", "Fusion")):
        m = r["arms"][k]
        buf.write(f"{name} & {m['f1']:.3f} & {r['arms_sd'][k]:.3f} & {m['precision']:.3f} & "
                  f"{m['recall']:.3f} & {r['fpr'][k]:.3f} \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_generated(os.path.join(SEC, "tab_sms_fusion.tex"), buf.getvalue())

    t1, t2 = r["T1_fusion_minus_url"], r["T2_text_minus_url"]
    # The seed-level p-values and win counts stay in fusion_results.json as provenance but
    # are NOT exported as macros: ten optimizer seeds on one fixed holdout are not repeated
    # data splits, so a macro in the paper's namespace is an invitation to re-print invalid
    # inference. The paper quotes the distinct-text cluster-bootstrap intervals instead.
    boot = r.get("cluster_bootstrap", {})
    verdict = ("success" if t1["mean"] >= 0.03 else
               "negative" if abs(t1["mean"]) < 0.01 else "inconclusive")
    # SmsScoreUrl, not SmsF1Url: a digit is not a letter in a control sequence name, so
    # \SmsF1Url parses as \SmsF followed by the characters "1Url" and LaTeX reports
    # "You already have nine parameters" from somewhere else entirely.
    mac = ("%% generated by scripts/make_smishing_assets.py; do not edit\n"
           + "".join("\\newcommand{\\SmsScore%s}{%.3f}\n" % (k.capitalize(), r["arms"][k]["f1"])
                     for k in ("url", "text", "fusion"))
           + "\\newcommand{\\SmsTOneDelta}{%+.3f}\n" % t1["mean"]
           + "\\newcommand{\\SmsTOneCI}{[%+.3f,%+.3f]}\n" % tuple(boot.get("T1_fusion_minus_url", {}).get("ci95", [float("nan"), float("nan")]))
           + "\\newcommand{\\SmsTTwoDelta}{%+.3f}\n" % t2["mean"]
           + "\\newcommand{\\SmsTTwoCI}{[%+.3f,%+.3f]}\n" % tuple(boot.get("T2_text_minus_url", {}).get("ci95", [float("nan"), float("nan")]))
           + "\\newcommand{\\SmsSeeds}{%d}\n" % r["seeds"]
           + "\\newcommand{\\SmsTOneVerdict}{%s}\n" % verdict
           + "\\newcommand{\\SmsNoUrlScore}{%.3f}\n"
             % r["diagnostics"].get("url/no_url", {}).get("f1", 0.0)
           + "\\newcommand{\\SmsNoUrlN}{%d}\n"
             % r["diagnostics"].get("url/no_url", {}).get("n", 0)
           + "\\newcommand{\\SmsUrlHasScore}{%.3f}\n"
             % r["diagnostics"].get("url/has_url", {}).get("f1", 0.0)
           + "\\newcommand{\\SmsTextHasScore}{%.3f}\n"
             % r["diagnostics"].get("text/has_url", {}).get("f1", 0.0)
           + "\\newcommand{\\SmsFusHasScore}{%.3f}\n"
             % r["diagnostics"].get("fusion/has_url", {}).get("f1", 0.0)
           + "\\newcommand{\\SmsHasUrlN}{%d}\n"
             % r["diagnostics"].get("url/has_url", {}).get("n", 0)
           + "\\newcommand{\\SmsShortPhishN}{%d}\n"
             % r["diagnostics"].get("url/shortened_phish", {}).get("n", 0)
           + "\\newcommand{\\SmsPostDelta}{%+.3f}\n" % r["posthoc_fusion_minus_text"]["mean"]
           + "\\newcommand{\\SmsPostCI}{[%+.3f,%+.3f]}\n" % tuple(boot.get("posthoc_fusion_minus_text", {}).get("ci95", [float("nan"), float("nan")]))
           + "\\newcommand{\\SmsBootClusters}{%d}\n" % boot.get("T1_fusion_minus_url", {}).get("clusters", 0)
           + "\\newcommand{\\SmsTestN}{%d}\n" % r["n_test"]
           + "".join("\\newcommand{\\SmsFpr%s}{%.1f}\n" % (k.capitalize(), 100 * r["fpr"][k])
                     for k in ("url", "text", "fusion"))
           + "".join("\\newcommand{\\SmsSd%s}{%.3f}\n" % (k.capitalize(), r["arms_sd"][k])
                     for k in ("url", "text", "fusion"))
           # ...Cnt, not ...TP/FP: a macro ending in P reads as a withdrawn p-value to the claims guard
           + "".join("\\newcommand{\\Sms%s%sCnt}{%.1f}\n" % (k.capitalize(), c.upper(),
                                                          r["confusion_mean"][k][c])
                     for k in ("url", "text", "fusion") for c in ("tp", "fp", "fn", "tn"))
           + "".join("\\newcommand{\\SmsRocAuc%s}{%.3f}\n" % (k.capitalize(), r["auc"][k]["roc_auc"])
                     + "\\newcommand{\\SmsPrAuc%s}{%.3f}\n" % (k.capitalize(), r["auc"][k]["pr_auc"])
                     for k in ("url", "text", "fusion"))
           + "\\newcommand{\\SmsTestPos}{%d}\n" % int(round(r["confusion_mean"]["text"]["tp"]
                                                         + r["confusion_mean"]["text"]["fn"]))
           + "\\newcommand{\\SmsTextNoUrlScore}{%.3f}\n" % r["diagnostics"]["text/no_url"]["f1"]
           + "\\newcommand{\\SmsFusNoUrlScore}{%.3f}\n" % r["diagnostics"]["fusion/no_url"]["f1"]
           + "".join("\\newcommand{\\SmsShortDet%s}{%s}\n"
                     % (k.capitalize(), ("%.1f" % r["diagnostics"][f"{k}/shortened_phish"]["detected_mean"]).rstrip("0").rstrip("."))
                     for k in ("url", "text", "fusion")))
    if os.path.exists(EMB_META):
        em = json.load(open(EMB_META, encoding="utf-8"))
        mac += ("\\newcommand{\\SmsEmbMaxLen}{%d}\n" % em["max_length"]
                + "\\newcommand{\\SmsEmbPooling}{%s}\n" % em["pooling"])
    if os.path.exists(WHY):
        w = json.load(open(WHY, encoding="utf-8"))
        mac += ("\\newcommand{\\SmsMaskedScore}{%.3f}\n" % w["text_masked"]
                + "\\newcommand{\\SmsMaskedUrlScore}{%.3f}\n" % w["masked_plus_url"]
                + "\\newcommand{\\SmsMaskDelta}{%+.3f}\n" % w["mask_effect"]["mean"]
                + "\\newcommand{\\SmsMaskCI}{[%+.3f,%+.3f]}\n" % tuple(w["mask_effect"].get("ci95", [float("nan"), float("nan")]))
                + "\\newcommand{\\SmsUrlBackDelta}{%+.3f}\n" % w["url_adds_to_masked"]["mean"]
                + "\\newcommand{\\SmsUrlBackCI}{[%+.3f,%+.3f]}\n" % tuple(w["url_adds_to_masked"].get("ci95", [float("nan"), float("nan")])))
    write_generated(os.path.join(SEC, "gen_sms_fusion.tex"), mac)
    write_generated(os.path.join(SEC, "gen_sms_predecision.tex"), predecision_macros())
    write_generated(os.path.join(SEC, "gen_sms_audit.tex"), audit_macros())
    tab_batches(os.path.join(SEC, "tab_sms_batches.tex"))
    def _ci(k):
        c = boot.get(k, {}).get("ci95")
        return f"[{c[0]:+.3f},{c[1]:+.3f}]" if c else "[no bootstrap in artefact]"
    print(f"  T1 fusion-url {t1['mean']:+.4f} CI {_ci('T1_fusion_minus_url')} -> {verdict}; "
          f"T2 text-url {t2['mean']:+.4f} CI {_ci('T2_text_minus_url')}")

    # The shallow-cue floor and the ladder that puts every arm against it. Written from the audit's
    # snapshot rather than recomputed here, so the paper and `sms_shallow_cues.py` cannot disagree.
    if os.path.exists(CUES):
        cues = json.load(open(CUES, encoding="utf-8"))
        write_generated(os.path.join(SEC, "gen_sms_cues.tex"),
                        cues_macros(cues) + rob_macros())
        fig_ladder(cues, r, os.path.join(FIG, "sms_ladder.pdf"))
        _why = json.load(open(WHY, encoding="utf-8")) if os.path.exists(WHY) else {}
        _d = fig_deltas(r, boot, _why, os.path.join(FIG, "sms_deltas.pdf"))
        print("  delta panel: %s contrast(s) against zero" % (len(_d) if _d else 0))
        if fig_robust(cues, r, os.path.join(FIG, "sms_robust.pdf")):
            print("  robustness panel: both splits, each with its own floor")
        tab_logodds(cues, os.path.join(SEC, "tab_sms_logodds.tex"))
        print(f"  shallow-cue floor: redaction tokens alone reach "
              f"{cues['baselines']['tokens']['f1']:.3f} F1 (post-hoc, unregistered)")
    else:
        print(f"  [!] {CUES} missing — run scripts/sms_shallow_cues.py", file=sys.stderr)
    return r


def _fmt(v: int) -> str:
    return f"{v:,}".replace(",", "{,}")


def audit_macros() -> str:
    """Macros for the corpus checks run after the registered analysis: batch confound, the
    public-suffix sensitivity of the URL arm, the non-linear shallow floor, the URL-as-characters
    baseline, truncation, the error-analysis rows, and the intervals the paper had computed but
    never printed. Each block is emitted only from its JSON."""
    m = ("%% generated by scripts/make_smishing_assets.py from "
         "batch_audit.json, extra_diagnostics.json, shallow_cues.json, fusion_results_psl.json, "
         "phobert_ft.json, split_robustness.json; do not edit\n")
    if os.path.exists(BATCH):
        b = json.load(open(BATCH, encoding="utf-8"))
        pb, a, lb, uc, tw = (b["positive_only_batches"], b["accented_pct"], b["leave_batches_out"],
                             b["url_ceiling"], b["twins"])
        lo, rv, nu = b["leave_one_batch_out"], b["reverse"], b["no_url_positive"]
        bt, cs = lb["batch"], lb["control_span"]
        def rng_(v, d=2):
            # \textup{--}, not a bare --: a range macro used inside $...$ printed as two minus
            # signs (0.85--0.88 -> "0.85 - -0.88") twice before this was fixed at the source
            f = "%%.%df" % d
            return (f + "\\textup{--}" + f) % tuple(v) if round(v[0], d) != round(v[1], d) else f % v[0]
        m += ("\\newcommand{\\SmsBatchN}{%d}\n" % sum(1 for d in b["batches"] if parse_date(d))
              + "\\newcommand{\\SmsPureRows}{%s}\n" % _fmt(b["rows_in_pure_batches"])
              + "\\newcommand{\\SmsPurePct}{%d}\n" % round(100 * b["pure_threshold"])
              + "\\newcommand{\\SmsPosBatchA}{%s}\n" % parse_date(pb["dates"][0]).isoformat()
              + "\\newcommand{\\SmsPosBatchB}{%s}\n" % parse_date(pb["dates"][1]).isoformat()
              + "\\newcommand{\\SmsPosBatchPos}{%d}\n" % pb["positive"]
              + "\\newcommand{\\SmsPosBatchHam}{%d}\n" % pb["ham"]
              + "\\newcommand{\\SmsPosBatchPct}{%.0f}\n" % (100 * pb["positive"] / pb["positive_total"])
              + "\\newcommand{\\SmsAccPosIn}{%.1f}\n" % a["positive_in_batches"]
              + "\\newcommand{\\SmsAccPosOut}{%.1f}\n" % a["positive_outside"]
              + "\\newcommand{\\SmsAccPosOutN}{%d}\n" % a["positive_outside_n"]
              + "\\newcommand{\\SmsLboTexts}{%d}\n" % bt["eval_positive_texts"]
              + "\\newcommand{\\SmsLboEvalHam}{%d}\n" % lb["eval_ham_texts"]
              + "\\newcommand{\\SmsLboTrainPos}{%d}\n" % bt["train_positive_rows"]
              + "".join("\\newcommand{\\SmsLbo%s%s}{%s}\n" % (an, mn, ("%.3f" if mn == "Auc" else "%.2f") % bt[ak][mk])
                        + "\\newcommand{\\SmsLboCtl%s%s}{%s}\n" % (an, mn, rng_(cs[ak][mk], 3 if mn == "Auc" else 2))
                        for ak, an in (("probe", "Probe"), ("text_arm", "Arm"))
                        for mk, mn in (("recall_at_0.5", "Rec"), ("roc_auc", "Auc"),
                                       ("recall_at_2pct_fpr", "RecTwo")))
              + "\\newcommand{\\SmsRevAuc}{%.3f}\n" % rv["probe"]["roc_auc"]
              + "\\newcommand{\\SmsRevRecTwo}{%.2f}\n" % rv["probe"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsRevTexts}{%d}\n" % rv["eval_positive_texts"]
              + "\\newcommand{\\SmsRevCtlAuc}{%s}\n" % rng_(rv["control_span"]["roc_auc"], 3)
              + "\\newcommand{\\SmsRevCtlRecTwo}{%s}\n" % rng_(rv["control_span"]["recall_at_2pct_fpr"])
              + "".join("\\newcommand{\\SmsLo%s%s}{%.3f}\n" % (gn, mn, lo[gk][mk])
                        for gk, gn in (("by_batch", "b"), ("by_component_5fold", "c"))
                        for mk, mn in (("precision_at_0.5", "P"), ("recall_at_0.5", "R"),
                                       ("recall_at_2pct_fpr", "RecTwo"),
                                       ("recall_at_2pct_fpr_two_batches", "RecTwoTwo"),
                                       ("recall_at_2pct_fpr_other", "RecTwoOther")))
              + "\\newcommand{\\SmsLobSpan}{%d}\n" % lo["components_spanning_batches"]
              + "\\newcommand{\\SmsLobBalRecTwo}{%.3f}\n" % lo["by_batch_balanced"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsLobPmRecTwo}{%.3f}\n" % lo["by_batch_per_model"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsLocPmRecTwo}{%.3f}\n" % lo["by_component_5fold_per_model"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsLocBalRecTwo}{%.3f}\n" % lo["by_component_5fold_balanced"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsPseudoRecTwo}{%s}\n" % rng_(lo["pseudo_span"]["recall_at_2pct_fpr"])
              + "\\newcommand{\\SmsPseudoAuc}{%s}\n" % rng_(lo["pseudo_span"]["roc_auc"], 3)
              + "\\newcommand{\\SmsLobArmPool}{%.3f}\n" % lo["text_arm_by_batch"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsLobArmPm}{%.3f}\n" % lo["text_arm_by_batch"]["recall_at_2pct_fpr_per_model"]
              + "\\newcommand{\\SmsLocArmPool}{%.3f}\n" % lo["text_arm_by_component_5fold"]["recall_at_2pct_fpr"]
              + "\\newcommand{\\SmsLocArmPm}{%.3f}\n" % lo["text_arm_by_component_5fold"]["recall_at_2pct_fpr_per_model"]
              + "\\newcommand{\\SmsMixedBatchN}{%d}\n" % sum(1 for v in lo["per_batch_auc_mixed"].values() if min(v["positive_texts"], v["ham_texts"]) >= 5)
              + "\\newcommand{\\SmsMixedBatchAuc}{%s}\n" % rng_([
                  min(v["auc"] for v in lo["per_batch_auc_mixed"].values() if min(v["positive_texts"], v["ham_texts"]) >= 5),
                  max(v["auc"] for v in lo["per_batch_auc_mixed"].values() if min(v["positive_texts"], v["ham_texts"]) >= 5)], 3)
              + "\\newcommand{\\SmsPseudoAllRecTwo}{%s}\n" % rng_([
                  min(lo["pseudo_span"]["recall_at_2pct_fpr"][0], lo["pseudo_balanced_span"][0], lo["pseudo_per_model_span"][0]),
                  max(lo["pseudo_span"]["recall_at_2pct_fpr"][1], lo["pseudo_balanced_span"][1], lo["pseudo_per_model_span"][1])])
              + "\\newcommand{\\SmsPseudoBalRecTwo}{%s}\n" % rng_(lo["pseudo_balanced_span"])
              + "\\newcommand{\\SmsPseudoPmRecTwo}{%s}\n" % rng_(lo["pseudo_per_model_span"])
              + "\\newcommand{\\SmsLobAuc}{%.3f}\n" % lo["by_batch"]["roc_auc"]
              + "\\newcommand{\\SmsLobF}{%.3f}\n" % lo["by_batch"]["f1_at_0.5"]
              + "\\newcommand{\\SmsLocAuc}{%.3f}\n" % lo["by_component_5fold"]["roc_auc"]
              + "\\newcommand{\\SmsLocF}{%.3f}\n" % lo["by_component_5fold"]["f1_at_0.5"]
              + "\\newcommand{\\SmsLobBatches}{%d}\n" % lo["batches"]
              + "\\newcommand{\\SmsNoUrlPos}{%d}\n" % nu["total"]
              + "\\newcommand{\\SmsNoUrlPosBatch}{%d}\n" % nu["in_two_batches"]
              + "\\newcommand{\\SmsShortStratumCorr}{%d}\n" % sum(
                  1 for m in read(MSG) if m["split"] == "test" and m["label"] == "1"
                  and int(m["n_shortened_valid"]) > 0)
              + "\\newcommand{\\SmsUrlCeilNoUrlPct}{%.1f}\n" % uc["test_positive_no_url_pct"]
              + "\\newcommand{\\SmsUrlCeilR}{%.3f}\n" % uc["max_recall"]
              + "\\newcommand{\\SmsUrlCeilF}{%.3f}\n" % uc["max_f1"]
              + "\\newcommand{\\SmsUrlCeilPosUrl}{%d}\n" % uc["test_positive_with_url"]
              + "\\newcommand{\\SmsTwinWs}{%d}\n" % tw["whitespace"]["test_rows"]
              + "\\newcommand{\\SmsTwinLoose}{%d}\n" % tw["case_punct"]["test_rows"]
              + "\\newcommand{\\SmsTwinLoosePos}{%d}\n" % tw["case_punct"]["positive"]
              + "\\newcommand{\\SmsAugTrainRows}{%d}\n" % b["august_rows"]["train"]
              + "".join("\\newcommand{\\SmsIdPrefix%s}{%d of %s}\n" % (k.capitalize(), v["positive"], _fmt(v["ham"] + v["positive"]))
                        for k, v in b["id_prefix"].items()))
    if os.path.exists(EXTRA):
        e = json.load(open(EXTRA, encoding="utf-8"))
        u, t, f = e["url_char"], e["truncation"], e["fp_examples"]
        m += ("\\newcommand{\\SmsUrlCharHasF}{%.3f}\n" % u["has_url_f1"]
              + "\\newcommand{\\SmsUrlCharHasAuc}{%.3f}\n" % u["has_url_roc_auc"]
              + "\\newcommand{\\SmsUrlCharAllF}{%.3f}\n" % u["all_test_f1"]
              + "\\newcommand{\\SmsUrlCharHasN}{%d}\n" % u["test_rows_with_url"]
              + "\\newcommand{\\SmsTruncHamOneTwoEight}{%.1f}\n" % t["ham"]["seg_over_128_pct"]
              + "\\newcommand{\\SmsTruncPosOneTwoEight}{%.1f}\n" % t["positive"]["seg_over_128_pct"]
              + "\\newcommand{\\SmsTruncRawTwoFiveSix}{%.1f}\n" % max(t["ham"]["raw_over_256_pct"], t["positive"]["raw_over_256_pct"])
              + "\\newcommand{\\SmsFpAny}{%d}\n" % f["ham_rows_flagged_by_any_seed"]
              + "\\newcommand{\\SmsFpMajority}{%d}\n" % f["ham_rows_flagged_by_majority"]
              + "\\newcommand{\\SmsFpMajorityAug}{%d}\n" % f["majority_in_august_batch"]
              + "\\newcommand{\\SmsFnMajority}{%d}\n" % e["fn_examples"]["positive_rows_missed_by_majority"]
              + "\\newcommand{\\SmsFnBatch}{%d}\n" % e["fn_examples"]["in_two_positive_batches"]
              + "\\newcommand{\\SmsFnNoUrl}{%d}\n" % e["fn_examples"]["without_url"]
              + "".join("\\newcommand{\\SmsTextMinus%s}{%+.3f}\n" % (n_, e["text_minus_floor"][k]["mean"])
                        + "\\newcommand{\\SmsTextMinus%sCI}{[%+.3f,%+.3f]}\n" % ((n_,) + tuple(e["text_minus_floor"][k]["ci95"]))
                        for k, n_ in (("floor_full", "Floor"), ("floor_strict", "Strict"))))
    if os.path.exists(CUES):
        c = json.load(open(CUES, encoding="utf-8"))
        L = c.get("language") or {}
        if L.get("nonlinear_floor"):
            nf = L["nonlinear_floor"]
            m += ("\\newcommand{\\SmsCueAllGbm}{%.3f}\n" % nf["all_shallow_gbm"]["f1"]
                  + "\\newcommand{\\SmsCueFormatLr}{%.3f}\n" % nf["all_shallow_format_lr"]["f1"]
                  + "\\newcommand{\\SmsCueFormatGbm}{%.3f}\n" % nf["all_shallow_format_gbm"]["f1"]
                  + "\\newcommand{\\SmsCueStrictGbm}{%.3f}\n" % nf["strict_format_gbm"]["f1"]
                  + "\\newcommand{\\SmsTfidfNoBracket}{%.3f}\n" % L["tfidf"]["word_no_brackets"]
                  + "\\newcommand{\\SmsSenderTagPos}{%.1f}\n" % L["sender_tag_pct"]["1"]
                  + "\\newcommand{\\SmsSenderTagHam}{%.1f}\n" % L["sender_tag_pct"]["0"]
                  + "\\newcommand{\\SmsCarrierHam}{%.1f}\n" % L["carrier_mention_pct"]["0"]
                  + "\\newcommand{\\SmsCarrierPos}{%.1f}\n" % L["carrier_mention_pct"]["1"]
                  + "\\newcommand{\\SmsTbPrefixMsgs}{%d}\n" % L["tb_usage"]["0"]["prefix_msgs"]
                  + "\\newcommand{\\SmsTbBareMsgs}{%d}\n" % L["tb_usage"]["0"]["bare_tb_msgs"])
    if os.path.exists(FUSION_PSL):
        q = json.load(open(FUSION_PSL, encoding="utf-8"))
        bq = q["cluster_bootstrap"]
        m += ("\\newcommand{\\SmsPslUrl}{%.3f}\n" % q["arms"]["url"]["f1"]
              + "\\newcommand{\\SmsPslFusion}{%.3f}\n" % q["arms"]["fusion"]["f1"]
              # seed-paired means, the same estimator as the registered \\SmsTOneDelta/\\SmsTTwoDelta
              + "\\newcommand{\\SmsPslTTwo}{%+.3f}\n" % q["T2_text_minus_url"]["mean"]
              + "\\newcommand{\\SmsPslTTwoCI}{[%+.3f,%+.3f]}\n" % tuple(bq["T2_text_minus_url"]["ci95"])
              + "\\newcommand{\\SmsPslTOne}{%+.3f}\n" % q["T1_fusion_minus_url"]["mean"]
              + "\\newcommand{\\SmsPslTOneCI}{[%+.3f,%+.3f]}\n" % tuple(bq["T1_fusion_minus_url"]["ci95"])
              + "\\newcommand{\\SmsPslPost}{%+.3f}\n" % q["posthoc_fusion_minus_text"]["mean"]
              + "\\newcommand{\\SmsPslTextHas}{%.3f}\n" % q["diagnostics"]["text/has_url"]["f1"]
              + "\\newcommand{\\SmsPslPostCI}{[%+.3f,%+.3f]}\n" % tuple(bq["posthoc_fusion_minus_text"]["ci95"])
              + "\\newcommand{\\SmsPslHasUrlN}{%d}\n" % q["diagnostics"]["url/has_url"]["n"]
              + "\\newcommand{\\SmsPslUrlHas}{%.3f}\n" % q["diagnostics"]["url/has_url"]["f1"])
    if os.path.exists(PREDEC["ft"]):
        r = json.load(open(PREDEC["ft"], encoding="utf-8"))
        tm = r["cluster_bootstrap"]["tfidf_minus_text"]
        m += ("\\newcommand{\\SmsFrozenFive}{%.3f}\n" % r["arms"]["text"]["f1"]
              + "\\newcommand{\\SmsTfidfMinusFrozen}{%+.3f}\n" % tm["mean"]
              + "\\newcommand{\\SmsTfidfMinusFrozenCI}{[%+.3f,%+.3f]}\n" % tuple(tm["ci95"]))
    if os.path.exists(ROB):
        rb = json.load(open(ROB, encoding="utf-8"))
        m += ("\\newcommand{\\SmsRobPostCI}{[%+.3f,%+.3f]}\n" % tuple(rb["posthoc_fusion_minus_text"]["ci95"])
              + "\\newcommand{\\SmsRobTTwoCI}{[%+.3f,%+.3f]}\n" % tuple(rb["T2_text_minus_url"]["ci95"])
              + "\\newcommand{\\SmsRobHoldout}{%d}\n" % round(100 * rb["split"]["holdout"]))
    return m


def tab_batches(out):
    """Batch x class, every value of the publisher's date column, three column groups."""
    if not os.path.exists(BATCH):
        return None
    b = json.load(open(BATCH, encoding="utf-8"))["batches"]
    keyf = lambda d: (parse_date(d) or dt.date(2100, 1, 1))
    items = sorted(b.items(), key=lambda kv: keyf(kv[0]))
    k = -(-len(items) // 3)
    cols = [items[i * k:(i + 1) * k] for i in range(3)]
    buf = io.StringIO()
    buf.write("\\begin{table}[htbp]\n\\centering\\footnotesize\n"
              "\\caption{Rows per value of the publisher's \\texttt{date} field (read as a "
              "collection batch) and class. Pos.\\ is the positive (spam/scam) class.}\n"
              "\\label{tab:sms_batches}\n\\setlength{\\tabcolsep}{3pt}\n"
              "\\begin{tabular}{lrr@{\\qquad}lrr@{\\qquad}lrr}\n\\toprule\n"
              "Batch & Ham & Pos. & Batch & Ham & Pos. & Batch & Ham & Pos. \\\\\n\\midrule\n")
    for i in range(k):
        cells = []
        for c in cols:
            if i < len(c):
                d, v = c[i]
                dd = parse_date(d)
                cells.append(f"{dd.isoformat() if dd else 'undated'} & {v['ham']} & {v['positive']}")
            else:
                cells.append(" & & ")
        buf.write(" & ".join(cells) + " \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_generated(out, buf.getvalue())
    return out


def tab_flow(msgs, out):
    """Every evaluation subset in the paper, in one place, with where it comes from."""
    need = (FUSION, ROB, TEMPORAL, BATCH, PREDEC["ft"], FUSION_PSL)
    if not all(os.path.exists(p) for p in need):
        return None
    fr = json.load(open(FUSION, encoding="utf-8"))
    rb = json.load(open(ROB, encoding="utf-8"))["split"]
    tp = json.load(open(TEMPORAL, encoding="utf-8"))["runs"]
    ba = json.load(open(BATCH, encoding="utf-8"))
    ps = json.load(open(FUSION_PSL, encoding="utf-8"))
    rows = collections.Counter((m["split"], m["label"]) for m in msgs)
    texts = {s: len({m["text_sha1"] for m in msgs if m["split"] == s}) for s in ("train", "test")}
    d = fr["diagnostics"]
    lines = [
        ("Corpus", f"{_fmt(len(msgs))} rows, {_fmt(len({m['text_sha1'] for m in msgs}))} distinct texts",
         "\\S\\ref{sec:corpus}"),
        ("Registered split, train", f"{_fmt(rows[('train', '0')] + rows[('train', '1')])} rows "
         f"({rows[('train', '1')]} pos.), {_fmt(texts['train'])} texts", "all registered arms"),
        ("Registered split, test", f"{rows[('test', '0')] + rows[('test', '1')]} rows "
         f"({rows[('test', '1')]} pos.), {texts['test']} texts", "Table~\\ref{tab:sms_fusion}"),
        ("\\quad test rows with a URL", f"{d['url/has_url']['n']} ({d['url/has_url']['n_pos']} pos.), "
         f"{ps['diagnostics']['url/has_url']['n']} under the suffix rule", "has-URL diagnostic"),
        ("\\quad test rows without a URL", f"{d['url/no_url']['n']} ({d['url/no_url']['n_pos']} pos.)",
         "no-URL diagnostic"),
        ("\\quad shortened positive rows", f"{d['url/shortened_phish']['n']} under the registered list, "
         f"{sum(1 for m in msgs if m['split'] == 'test' and m['label'] == '1' and int(m['n_shortened_valid']) > 0)}"
         " under the corrected one", "registered stratum"),
        ("Similarity-grouped split", f"train {_fmt(rb['n_train'])}, test {rb['n_test']} "
         f"({rb['test_phishing']} pos.)", "\\S\\ref{sec:robust}"),
        ("August ham, August in training", f"train {_fmt(tp['registered_august']['n_train'])}, test "
         f"{tp['registered_august']['n_test']} ham", "\\S\\ref{sec:sms_batches}"),
        ("Registered split without August", f"train {_fmt(tp['without_august']['n_train'])}, test "
         f"{tp['without_august']['n_test']} May--July rows of the registered test side",
         "\\S\\ref{sec:sms_batches}"),
        ("August ham, no August in training", f"train {_fmt(tp['forward_august']['n_train'])}, test "
         f"{tp['forward_august']['n_test']} ham", "\\S\\ref{sec:sms_batches}"),
        ("Two-batch hold-out", f"train {_fmt(ba['leave_batches_out']['batch']['train_rows'])}, "
         f"test the batches' {ba['leave_batches_out']['batch']['eval_positive_texts']} pos. texts "
         f"and {ba['leave_batches_out']['eval_ham_texts']} held-out ham texts",
         "\\S\\ref{sec:sms_batches}"),
        ("Leave-one-batch-out", f"all {_fmt(len({m['text_sha1'] for m in msgs}))} texts, "
         f"{ba['leave_one_batch_out']['batches']} folds",
         "\\S\\ref{sec:sms_batches}"),
    ]
    buf = io.StringIO()
    buf.write("\\begin{table*}[t]\n\\centering\\footnotesize\n"
              "\\caption{Every evaluation set in the paper and where it is used. Pos.\\ counts "
              "positive (spam/scam) rows.}\n\\label{tab:sms_flow}\n"
              "\\begin{tabular}{@{}p{42mm}p{95mm}p{30mm}@{}}\n\\toprule\n"
              "Set & Size & Used in \\\\\n\\midrule\n")
    for a_, b_, c_ in lines:
        buf.write(f"{a_} & {b_} & {c_} \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write_generated(out, buf.getvalue())
    return out


PREDEC = {"ft": os.path.join(ROOT, "data", "processed", "sms", "phobert_ft.json"),
          "dia": os.path.join(ROOT, "data", "processed", "sms", "diacritics_ablation.json"),
          "late": os.path.join(ROOT, "data", "processed", "sms", "late_fusion.json")}


def _d(v: float) -> str:
    """A signed 3-decimal delta; a value that rounds to zero prints as 0.000, never -0.000."""
    return "0.000" if abs(v) < 0.0005 else "%+.3f" % v


def _sci(v: float) -> str:
    """A learning rate as math-mode scientific notation: 2e-05 -> $2\\times10^{-5}$."""
    m, e = f"{v:.0e}".split("e")
    return "$%s\\times10^{%d}$" % (m, int(e))


def _ci(d, key):
    c = d.get(key, {}).get("ci95")
    return "[%+.3f,%+.3f]" % tuple(c) if c else "[not run]"


def predecision_macros() -> str:
    """Macros for the three 2026-09-16 pre-decision arms (train_sms_phobert_ft.py,
    run_sms_diacritics.py, run_sms_late_fusion.py). Each block is emitted only from its JSON;
    a missing JSON leaves its macros undefined so a sentence that uses them fails to compile
    rather than printing a placeholder number."""
    m = ("%% generated by scripts/make_smishing_assets.py from "
         "phobert_ft.json, diacritics_ablation.json, late_fusion.json; do not edit\n")
    if os.path.exists(PREDEC["ft"]):
        r = json.load(open(PREDEC["ft"], encoding="utf-8"))
        b = r["cluster_bootstrap"]
        m += ("\\newcommand{\\SmsFtScore}{%.3f}\n" % r["arms"]["text_ft"]["f1"]
              + "\\newcommand{\\SmsSegScore}{%.3f}\n" % r["arms"]["text_seg"]["f1"]
              + "\\newcommand{\\SmsFtSeeds}{%d}\n" % r["seeds"]
              + "\\newcommand{\\SmsFtEpochs}{%d}\n" % r["finetune"]["epochs"]
              + "\\newcommand{\\SmsFtModel}{%s}\n" % r["finetune"]["model"].replace("_", "\\_")
              + "\\newcommand{\\SmsFtLr}{%s}\n" % _sci(r["finetune"]["lr"])
              + "\\newcommand{\\SmsFtBatch}{%d}\n" % r["finetune"]["batch"]
              + "\\newcommand{\\SmsFtMaxLen}{%d}\n" % r["finetune"]["max_len"]
              + "\\newcommand{\\SmsSegDelta}{%s}\n" % _d(b["seg_minus_text"]["mean"])
              + "\\newcommand{\\SmsSegCI}{%s}\n" % _ci(b, "seg_minus_text")
              + "\\newcommand{\\SmsFtDelta}{%+.3f}\n" % b["ft_minus_text"]["mean"]
              + "\\newcommand{\\SmsFtCI}{%s}\n" % _ci(b, "ft_minus_text")
              + "\\newcommand{\\SmsFtTfidfDelta}{%+.3f}\n" % b["ft_minus_tfidf"]["mean"]
              + "\\newcommand{\\SmsFtTfidfCI}{%s}\n" % _ci(b, "ft_minus_tfidf"))
    if os.path.exists(PREDEC["dia"]):
        r = json.load(open(PREDEC["dia"], encoding="utf-8"))
        L, b = r["linear"], r["cluster_bootstrap"]
        m += ("\\newcommand{\\SmsNodiaChanged}{%s}\n" % f"{r['n_changed_by_normalisation']:,}".replace(",", "{,}")
              + "\\newcommand{\\SmsNodiaWord}{%.3f}\n" % L["tfidf_word_nodia"]
              + "\\newcommand{\\SmsNodiaChar}{%.3f}\n" % L["tfidf_char_nodia"]
              + "\\newcommand{\\SmsNodiaWordNoTok}{%.3f}\n" % L["tfidf_word_no_tokens_nodia"]
              + "\\newcommand{\\SmsNodiaText}{%.3f}\n" % r["arms"]["text_nodia"]["f1"]
              + "\\newcommand{\\SmsNodiaTextDelta}{%+.3f}\n" % b["text_nodia_minus_text"]["mean"]
              + "\\newcommand{\\SmsNodiaTextCI}{%s}\n" % _ci(b, "text_nodia_minus_text")
              + "\\newcommand{\\SmsNodiaWordDelta}{%+.3f}\n" % b["tfidf_word_nodia_minus_accented"]["mean"]
              + "\\newcommand{\\SmsNodiaWordCI}{%s}\n" % _ci(b, "tfidf_word_nodia_minus_accented")
              + "\\newcommand{\\SmsNodiaNoTokDelta}{%s}\n" % _d(b["tfidf_word_no_tokens_nodia_minus_accented"]["mean"])
              + "\\newcommand{\\SmsNodiaNoTokCI}{%s}\n" % _ci(b, "tfidf_word_no_tokens_nodia_minus_accented"))
    if os.path.exists(PREDEC["late"]):
        r = json.load(open(PREDEC["late"], encoding="utf-8"))
        b = r["cluster_bootstrap"]
        for k, name in (("early_flag", "EarlyFlag"), ("late_mean", "LateMean"),
                        ("late_gated", "LateGated"), ("stack", "Stack")):
            m += ("\\newcommand{\\Sms%sScore}{%.3f}\n" % (name, r["arms"][k]["f1"])
                  + "\\newcommand{\\Sms%sDelta}{%+.3f}\n" % (name, b[f"{k}_minus_text"]["mean"])
                  + "\\newcommand{\\Sms%sCI}{%s}\n" % (name, _ci(b, f"{k}_minus_text")))
        m += ("\\newcommand{\\SmsLateGatedHasUrl}{%.3f}\n" % r["has_url_only"]["late_gated"]
              + "\\newcommand{\\SmsTextHasUrlLate}{%.3f}\n" % r["has_url_only"]["text"]
              + "\\newcommand{\\SmsLateNoUrlN}{%d}\n" % r["n_test_no_url"])
    return m


def fig_ladder(cues, fus, out):
    """The whole study in one frame: what each channel is worth, against a floor that is not
    language at all. The redaction bar is the publisher's preprocessing fitted alone; if it were
    left out, a reader would take the text arm's 0.93 as evidence about Vietnamese."""
    floor = cues["baselines"]["all_shallow"]["f1"]
    # Colour encodes the CHANNEL, not the class, and nothing else. GRAY is the floor that reads no
    # language, BLUE the URL channel, ORANGE the text channel, pale orange the two combined.
    # Both text arms are therefore the same colour: which of them is taller IS the finding, and an
    # earlier version drew the taller one (word TF-IDF) in pale blue while PhoBERT took full-strength
    # orange, so the picture ranked them opposite to their numbers and put a text model in the URL
    # arm's hue. Fusion stays pale because it is the arm that buys nothing.
    nf = (cues.get("language") or {}).get("nonlinear_floor", {})
    ft = (json.load(open(PREDEC["ft"], encoding="utf-8"))["arms"]["text_ft"]["f1"]
          if os.path.exists(PREDEC["ft"]) else 0.0)
    arms = [("URL only\n(CompPhish-21)", fus["arms"]["url"]["f1"], BLUE),
            ("bracketed-token presence", cues["baselines"]["tokens"]["f1"], GRAY),
            ("diacritics + length\n+ tokens, linear", floor, GRAY),
            ("diacritics + length + format,\nno token names, boosted trees",
             nf.get("strict_format_gbm", {}).get("f1", 0.0), GRAY),
            ("same + token names,\nboosted trees",
             nf.get("all_shallow_format_gbm", {}).get("f1", 0.0), GRAY),
            ("word TF-IDF\n(linear, 1 seed)", cues.get("language", {}).get("tfidf", {}).get("word", 0.0),
             ORANGE),
            ("text only\n(frozen PhoBERT)", fus["arms"]["text"]["f1"], ORANGE),
            ("fine-tuned PhoBERT\n(post-hoc, 5 seeds)", ft, ORANGE),
            ("fusion\n(text + URL)", fus["arms"]["fusion"]["f1"], FILL_O)]
    arms = [a for a in arms if a[1] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    y = np.arange(len(arms))[::-1]
    ax.barh(y, [a[1] for a in arms], 0.62,
            color=[a[2] for a in arms], edgecolor=INK, linewidth=0.7)
    for yy, (_, v, _c) in zip(y, arms):
        # white box: the dashed floor line runs through the labels of the short bars
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", ha="left", fontsize=8, color=INK,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=3)
    ax.axvline(floor, color=GRAY, ls="--", lw=0.9, zorder=0)
    ax.set_yticks(y, [a[0] for a in arms], fontsize=8)
    ax.set_xlabel("positive-class F1")
    ax.set_xlim(0, 1.06)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


VI = "\\vntext{%s}"
ACCENT = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"


def tab_logodds(cues, out):
    """The table the paper needed and did not have: which words mark which class.

    A word carrying diacritics is set through the manuscript's Unicode Vietnamese helper. An
    unaccented one is set in \\texttt, because in this corpus the ASCII spelling IS the finding
    rather than a typesetting accident."""
    lang = cues.get("language") or {}
    if not lang:
        return None
    def cell(w):
        t = w["term"]
        return (VI % t) if any(c in ACCENT for c in t) else ("\\texttt{%s}" % t)
    ph, ha = lang["log_odds"]["phishing"][:10], lang["log_odds"]["ham"][:10]
    buf = io.StringIO()
    buf.write("\\begin{table}[htbp]\n\\centering\\small\n"
              "\\caption{The ten words that most mark each class (log-odds with an informative prior).}\n\\label{tab:sms_logodds}\n"
              "\\begin{tabular}{lrrr@{\\qquad}lrrr}\n\\toprule\n"
              "\\multicolumn{4}{c}{marks the positive class} & \\multicolumn{4}{c}{marks ham} \\\\\n"
              "\\cmidrule(lr){1-4}\\cmidrule(lr){5-8}\n"
              "term & $z$ & pos. & ham & term & $z$ & pos. & ham \\\\\n\\midrule\n")
    for a, b in zip(ph, ha):
        buf.write("%s & %.1f & %d & %d & %s & %.1f & %d & %d \\\\\n"
                  % (cell(a), a["z"], a["phish"], a["ham"],
                     cell(b), b["z"], b["phish"], b["ham"]))
    buf.write("\\bottomrule\n\\end{tabular}\n"
              "\\\\[4pt]\n\\begin{minipage}{0.94\\linewidth}\\footnotesize\n"
              "Log-odds ratio with an informative Dirichlet prior, over the whole corpus. The score $z$ is\n"
              "prior-regularised, and the two count columns are raw occurrences. The positive\n"
              "column is discussed in Section~\\ref{sec:sms_words}. Post-hoc and unregistered.\n"
              "\\end{minipage}\n\\end{table}\n")
    write_generated(out, buf.getvalue())
    return out


def rob_macros() -> str:
    """The similarity-aware split's numbers. Post-hoc, and the macro names say `Rob` so a sentence
    cannot quietly present them as the registered ones."""
    if not os.path.exists(ROB):
        return ""
    r = json.load(open(ROB, encoding="utf-8"))
    m = ""
    for k in ("url", "text", "fusion"):
        m += "\\newcommand{\\SmsRob%s}{%.3f}\n" % (k.capitalize(), r["arms"][k]["f1"])
    m += "\\newcommand{\\SmsRobFloor}{%.3f}\n" % r["floors"]["all_shallow"]
    m += "\\newcommand{\\SmsRobFloorTok}{%.3f}\n" % r["floors"]["tokens"]
    m += "\\newcommand{\\SmsRobPost}{%+.3f}\n" % r["posthoc_fusion_minus_text"]["mean"]
    m += "\\newcommand{\\SmsRobTTwo}{%+.3f}\n" % r["T2_text_minus_url"]["mean"]
    sp = r["split"]
    m += "\\newcommand{\\SmsRobComponents}{%s}\n" % f"{sp['components']:,}"
    m += "\\newcommand{\\SmsRobMoved}{%s}\n" % f"{sp['moved']:,}"
    m += "\\newcommand{\\SmsRobTest}{%s}\n" % f"{sp['n_test']:,}"
    m += "\\newcommand{\\SmsRobSim}{%.2f}\n" % sp["sim"]
    return m


def cues_macros(cues) -> str:
    """The shallow-cue numbers the paper reads. Post-hoc and unregistered, and the prose that uses
    them says so -- the macro names carry `Cue` so that is hard to forget."""
    b = cues["baselines"]
    m = ("%% generated by scripts/make_smishing_assets.py from shallow_cues.json; "
         "do not edit\n")
    m += "\\newcommand{\\SmsCueTokenF}{%.3f}\n" % b["tokens"]["f1"]
    m += "\\newcommand{\\SmsCueAllF}{%.3f}\n" % b["all_shallow"]["f1"]
    m += "\\newcommand{\\SmsCueAccLenF}{%.3f}\n" % b["accent_and_length"]["f1"]
    lang = cues.get("language") or {}
    if lang:
        t = lang["tfidf"]
        m += "\\newcommand{\\SmsTfidfWord}{%.3f}\n" % t["word"]
        m += "\\newcommand{\\SmsTfidfChar}{%.3f}\n" % t["char"]
        m += "\\newcommand{\\SmsTfidfNoTok}{%.3f}\n" % t["word_no_tokens"]
        d90 = lang["near_duplicate_leak"]["0.90"]
        d95 = lang["near_duplicate_leak"]["0.95"]
        m += "\\newcommand{\\SmsDupPhishNinety}{%.1f}\n" % d90["1"]["pct"]
        m += "\\newcommand{\\SmsDupHamNinety}{%.1f}\n" % d90["0"]["pct"]
        m += "\\newcommand{\\SmsDupPhishNfive}{%.1f}\n" % d95["1"]["pct"]
        m += "\\newcommand{\\SmsDupHamNfive}{%.1f}\n" % d95["0"]["pct"]
        for l, tag in (("1", "Phish"), ("0", "Ham")):
            f = lang["families"][l]
            m += "\\newcommand{\\SmsFam%s}{%d}\n" % (tag, f["families"])
            m += "\\newcommand{\\SmsFam%sBiggest}{%d}\n" % (tag, f["largest"][0])
    m += "\\newcommand{\\SmsCueTokenLenF}{%.3f}\n" % b["tokens_and_length"]["f1"]
    m += "\\newcommand{\\SmsCueLenF}{%.3f}\n" % b["length"]["f1"]
    m += "\\newcommand{\\SmsAccentHam}{%.1f}\n" % cues["accented_pct"]["0"]
    m += "\\newcommand{\\SmsAccentPhish}{%.1f}\n" % cues["accented_pct"]["1"]
    m += "\\newcommand{\\SmsTokenHam}{%.1f}\n" % cues["any_token_pct"]["0"]
    m += "\\newcommand{\\SmsTokenPhish}{%.1f}\n" % cues["any_token_pct"]["1"]
    m += "\\newcommand{\\SmsMedCharsHam}{%d}\n" % cues["median_chars"]["0"]
    m += "\\newcommand{\\SmsMedCharsPhish}{%d}\n" % cues["median_chars"]["1"]
    for name, t in (("Qc", "[QC]"), ("Tb", "[TB]"), ("Date", "[DATE]"), ("Money", "[MONEY]")):
        p = cues["token_pct"].get(t, {"0": 0.0, "1": 0.0})
        m += "\\newcommand{\\SmsTok%sHam}{%.1f}\n" % (name, p["0"])
        m += "\\newcommand{\\SmsTok%sPhish}{%.1f}\n" % (name, p["1"])
    return m

WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
REG_CUTOFF = dt.date(2026, 7, 15)   # the temporal-split cutoff PREREG_smishing.md registered
PREFIX_RX = re.compile(r"^\s*\[(TB|QC)\]")
# The Hugging Face revision full_dataset.csv was taken from and the day the repository was created,
# both read from the Hub API (/api/datasets/trannguyenthaituan/vietnamese_sms_dataset) on
# 2026-10-07; the file's SHA-1 then matched the local copy. The card does not define `date`, and
# the last value it takes is this creation day, so the prose reads it as a batch date.
HF_REVISION = "a90e2bd7e8df2376939658075ba8094dcee488ad"
HF_CREATED = dt.date(2026, 8, 3)


def read_raw():
    """The publisher's rows, with their text: the processed CSV carries hashes, not messages."""
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _n(v: int) -> str:
    return f"{v:,}".replace(",", "{,}")


def url_feature_list() -> str:
    """The 21 URL columns the registered arms fit, read from train_sms_fusion.py's own COMPPHISH
    list rather than retyped, so the configuration table cannot name a column the model never saw."""
    import ast
    src = open(os.path.join(_HERE, "train_sms_fusion.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "COMPPHISH" for t in node.targets):
            cols = [e.value for e in node.value.elts]
            break
    else:
        raise SystemExit("[!] COMPPHISH not found in train_sms_fusion.py")
    names = ["\\texttt{" + c.replace("_", "\\_") + "}" for c in cols]
    return (f"\\newcommand{{\\SmsUrlFeatureN}}{{{len(cols)}}}\n"
            f"\\newcommand{{\\SmsUrlFeatureList}}{{{', '.join(names)}}}\n")


def split_and_date_macros(msgs, raw) -> str:
    """Rows and distinct texts per split side and label, undated rows, the registered temporal
    cutoff's class counts, and the regulated [TB]/[QC] prefixes. All counts, no fitting."""
    by_id = {r["message_id"]: r for r in raw}
    rows = collections.Counter((m["split"], m["label"]) for m in msgs)
    texts = {s: len({m["text_sha1"] for m in msgs if m["split"] == s}) for s in ("train", "test")}
    n = {s: sum(v for (ss, _l), v in rows.items() if ss == s) for s in ("train", "test")}
    dup = {s: n[s] - texts[s] for s in n}
    undated = collections.Counter(m["label"] for m in msgs if parse_date(m["date"]) is None)
    dated = [(parse_date(m["date"]), m["label"]) for m in msgs if parse_date(m["date"])]
    after = collections.Counter(l for d, l in dated if d > REG_CUTOFF)
    last_ph = max(d for d, l in dated if l == "1")
    pre = collections.Counter(m["label"] for m in msgs
                              if PREFIX_RX.match(by_id.get(m["message_id"], {}).get("message", "")))
    days = collections.Counter(d for d, _l in dated)
    last_day = max(days)
    last_rows = [l for d, l in dated if d == last_day]
    dates_per_text = collections.defaultdict(set)
    for m in msgs:
        if parse_date(m["date"]):
            dates_per_text[m["text_sha1"]].add(parse_date(m["date"]))
    multi_date = sum(1 for v in dates_per_text.values() if len(v) > 1)
    # tokens the card lists as its redaction vocabulary, counted in the released text
    card_tok = {t: sum(1 for r in raw if f"[{t}]" in r.get("message", "")) for t in ("PHONE", "BANK_ACC")}
    pre_any = sum(1 for m in msgs
                  if re.search(r"\[(TB|QC)\]", by_id.get(m["message_id"], {}).get("message", "")))
    return (f"\\newcommand{{\\SmsTrainRows}}{{{_n(n['train'])}}}\n"
            f"\\newcommand{{\\SmsTestRows}}{{{_n(n['test'])}}}\n"
            f"\\newcommand{{\\SmsTrainHamRows}}{{{_n(rows[('train', '0')])}}}\n"
            f"\\newcommand{{\\SmsTrainPhishRows}}{{{_n(rows[('train', '1')])}}}\n"
            f"\\newcommand{{\\SmsTestHamRows}}{{{_n(rows[('test', '0')])}}}\n"
            f"\\newcommand{{\\SmsTestPhishRows}}{{{_n(rows[('test', '1')])}}}\n"
            f"\\newcommand{{\\SmsTrainTexts}}{{{_n(texts['train'])}}}\n"
            f"\\newcommand{{\\SmsTestTexts}}{{{_n(texts['test'])}}}\n"
            f"\\newcommand{{\\SmsTrainDupRows}}{{{_n(dup['train'])}}}\n"
            f"\\newcommand{{\\SmsTestDupRows}}{{{_n(dup['test'])}}}\n"
            f"\\newcommand{{\\SmsTrainDupPct}}{{{100 * dup['train'] / n['train']:.1f}}}\n"
            f"\\newcommand{{\\SmsTestDupPct}}{{{100 * dup['test'] / n['test']:.1f}}}\n"
            f"\\newcommand{{\\SmsUndated}}{{{_n(sum(undated.values()))}}}\n"
            f"\\newcommand{{\\SmsUndatedHam}}{{{_n(undated['0'])}}}\n"
            f"\\newcommand{{\\SmsUndatedPhish}}{{{_n(undated['1'])}}}\n"
            f"\\newcommand{{\\SmsRegCutoff}}{{{REG_CUTOFF.isoformat()}}}\n"
            f"\\newcommand{{\\SmsPhishAfterCutoff}}{{{_n(after['1'])}}}\n"
            f"\\newcommand{{\\SmsHamAfterCutoff}}{{{_n(after['0'])}}}\n"
            f"\\newcommand{{\\SmsLastPhishDate}}{{{last_ph.isoformat()}}}\n"
            f"\\newcommand{{\\SmsTokPhoneRows}}{{{_n(card_tok['PHONE'])}}}\n"
            f"\\newcommand{{\\SmsTokBankAccRows}}{{{_n(card_tok['BANK_ACC'])}}}\n"
            f"\\newcommand{{\\SmsDistinctDates}}{{{len(days)}}}\n"
            f"\\newcommand{{\\SmsLastDate}}{{{last_day.isoformat()}}}\n"
            f"\\newcommand{{\\SmsLastDateRows}}{{{_n(len(last_rows))}}}\n"
            f"\\newcommand{{\\SmsLastDatePhish}}{{{_n(last_rows.count('1'))}}}\n"
            f"\\newcommand{{\\SmsMultiDateTexts}}{{{_n(multi_date)}}}\n"
            f"\\newcommand{{\\SmsHfCreated}}{{{HF_CREATED.isoformat()}}}\n"
            f"\\newcommand{{\\SmsHfRevision}}{{{HF_REVISION[:8]}}}\n"
            f"\\newcommand{{\\SmsPrefixMsgs}}{{{_n(pre_any)}}}\n"
            f"\\newcommand{{\\SmsPrefixStartMsgs}}{{{_n(sum(pre.values()))}}}\n"
            f"\\newcommand{{\\SmsPrefixStartHam}}{{{_n(pre['0'])}}}\n"
            f"\\newcommand{{\\SmsPrefixStartPhish}}{{{_n(pre['1'])}}}\n")


# Table 1: positive-labelled rows quoted from the corpus, chosen by the host they carry. The
# pretext column is the author's one-line gloss; the message and host are the publisher's.
# None of these rows is in Figure 1, whose panels quote three other positive rows.
EXAMPLE_HOSTS = (("Service charge", "shb.com.vn-zy.top"),
                 ("App reactivation", "vietcombank.vn-nng.top"),
                 ("Points expiry", "vietcompriority.com"),
                 ("Traffic fine", "dichvucong-vn.com"),
                 ("Debt collection", "t.ly"))
QUOTE_CHARS = 60
TEX_SPECIAL = {"&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{",
               "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "\\": "\\textbackslash{}"}


def tex_escape(t: str) -> str:
    return "".join(TEX_SPECIAL.get(c, c) for c in t)


def quote_row(text: str, limit: int = QUOTE_CHARS) -> str:
    """The first `limit` characters, cut back to a word boundary so no token is split, TeX-escaped,
    with an ellipsis appended AFTER escaping so the ellipsis macro survives."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return tex_escape(t)
    cut = t[:limit + 1].rsplit(" ", 1)[0]
    return tex_escape(cut.rstrip(" ,.;:-")) + "\\ldots"


def example_rows(raw, msgs, hosts=EXAMPLE_HOSTS):
    """For each host, the first positive-labelled row in file order that carries it."""
    label = {m["message_id"]: m["label"] for m in msgs}
    out = []
    for gloss, host in hosts:
        hit = next((r for r in raw if host in (r["message"] or "").lower()
                    and label.get(r["message_id"]) == "1"), None)
        if hit is None:
            raise SystemExit(f"[!] no positive-labelled row carries {host}; the table must change")
        out.append((gloss, host, hit))
    return out


def tab_examples(raw, msgs, out):
    buf = io.StringIO()
    buf.write("%% generated by scripts/make_smishing_assets.py from "
              "full_dataset.csv; do not edit\n")
    buf.write("\\begin{table*}[t]\n\\centering\n"
              "\\caption{Five positive-labelled rows quoted from the corpus, chosen by the host they carry.}\n"
              "\\label{tab:sms_examples}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
              "\\begin{tabular}{p{26mm}p{20mm}p{72mm}p{46mm}}\n\\toprule\n"
              "Pretext (gloss) & Row & Message, first %d characters & Embedded host \\\\\n\\midrule\n"
              % QUOTE_CHARS)
    for gloss, host, r in example_rows(raw, msgs):
        buf.write("%s & \\texttt{%s} & \\vntextsf{%s} & \\path{%s} \\\\\n"
                  % (gloss, tex_escape(r["message_id"]), quote_row(r["message"]), host))
    buf.write("\\bottomrule\n\\end{tabular}\n\\\\[4pt]\n"
              "\\begin{minipage}{0.94\\linewidth}\\footnotesize\n"
              "Each row is the first positive-labelled message in the published file that carries\n"
              "the host in the last column, quoted verbatim and cut at a word boundary. Placeholders\n"
              "such as [MONEY] are the publisher's redaction, [TB] and [QC] are sender prefixes, and\n"
              "other brackets are part of the message. The pretext is a gloss added in this study. The table illustrates\n"
              "measured corpus properties and does not estimate pretext frequency.\n"
              "\\end{minipage}\n\\end{table*}\n")
    write_generated(out, buf.getvalue())
    return out


def main() -> int:
    for p in (MSG, URLS, SNAP):
        if not os.path.exists(p):
            print(f"[!] {p} missing — run scripts/sms_corpus_import.py", file=sys.stderr)
            return 1
    msgs = read(MSG)
    # the corpus description uses URLs that end in a real public suffix; the registered
    # extractor's sentence fragments stay in sms_urls.csv with valid = 0
    urls = [u for u in read(URLS) if u.get("valid", "1") == "1"]
    raw = read_raw()
    os.makedirs(FIG, exist_ok=True)

    months, ham, ph = fig_timeline(msgs, os.path.join(FIG, "sms_timeline.pdf"))
    fig_hosts(urls, os.path.join(FIG, "sms_hosts.pdf"))
    TLD_TABLE_N, TLD_FIG_N = 5, 8    # Table 2 ranks five suffixes, Figure 3 draws eight
    per_tld = fig_tld(urls, os.path.join(FIG, "sms_tld.pdf"), top=TLD_FIG_N)
    snap = json.load(open(SNAP, encoding="utf-8"))
    short = fig_shortener(urls, os.path.join(FIG, "sms_shortener.pdf"),
                          set(snap.get("operator_deep_links", [])))

    # the temporal confound, as a number the prose can cite
    dead = [m for m, h_, p_ in zip(months, ham, ph) if p_ == 0]
    top_ph = per_tld["1"].most_common(TLD_TABLE_N)
    tot_ph = max(sum(per_tld["1"].values()), 1)
    tot_ham = max(sum(per_tld["0"].values()), 1)

    buf = io.StringIO()
    buf.write("\\begin{table}[t]\n\\centering\\small\n")
    buf.write("\\caption{Where each class registers, on the %s suffixes the positive class uses "
              "most.}\n\\label{tab:sms_tld}\n" % WORDS[TLD_TABLE_N])
    buf.write("\\begin{tabular}{lrr}\n\\toprule\nSuffix & Positive class & Ham \\\\\n\\midrule\n")
    for t, n in top_ph:
        buf.write(f"\\texttt{{.{t}}} & {100*n/tot_ph:.1f}\\% & "
                  f"{100*per_tld['0'][t]/tot_ham:.1f}\\% \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_generated(os.path.join(SEC, "tab_sms_tld.tex"), buf.getvalue())

    # The suffix list the prose quotes, in the table's own order, so the sentence cannot name
    # suffixes the table does not rank (it once listed .online while .com led the table).
    suffix_list = [f"\\texttt{{.{t}}}" for t, _ in top_ph]
    suffix_words = ", ".join(suffix_list[:-1]) + " and " + suffix_list[-1]
    ham_on_ph = max(100 * per_tld["0"][t] / tot_ham for t, _ in top_ph)
    macros = (
        "%% generated by scripts/make_smishing_assets.py; do not edit\n"
        f"\\newcommand{{\\SmsTldTableN}}{{{WORDS[TLD_TABLE_N]}}}\n"
        f"\\newcommand{{\\SmsTldFigN}}{{{WORDS[TLD_FIG_N]}}}\n"
        f"\\newcommand{{\\SmsPhishTopSuffixes}}{{{suffix_words}}}\n"
        f"\\newcommand{{\\SmsPhishTopSuffix}}{{\\texttt{{.{top_ph[0][0]}}}}}\n"
        f"\\newcommand{{\\SmsPhishTopSuffixPct}}{{{100 * top_ph[0][1] / tot_ph:.1f}}}\n"
        f"\\newcommand{{\\SmsHamMaxOnPhishSuffixPct}}{{{ham_on_ph:.1f}}}\n"
        + split_and_date_macros(msgs, raw)
        + url_feature_list()
        + f"\\newcommand{{\\SmsMonths}}{{{len(months)}}}\n"
        f"\\newcommand{{\\SmsSpanStart}}{{{months[0]}}}\n"
        f"\\newcommand{{\\SmsSpanEnd}}{{{months[-1]}}}\n"
        f"\\newcommand{{\\SmsDeadMonths}}{{{len(dead)}}}\n"
        f"\\newcommand{{\\SmsDeadMonthList}}{{{', '.join(dead)}}}\n"
        f"\\newcommand{{\\SmsDeadMonthHam}}{{{sum(h for h, p in zip(ham, ph) if p == 0):,}}}\n"
        .replace(",", "{,}").replace("{,} ", ", ")
    )
    write_generated(os.path.join(SEC, "gen_sms_figs.tex"), macros)
    tab_examples(raw, msgs, os.path.join(SEC, "tab_sms_examples.tex"))

    emit_results()
    tab_flow(msgs, os.path.join(SEC, "tab_sms_flow.tex"))

    print(f"  timeline {months[0]}..{months[-1]} ({len(months)} months); "
          f"{len(dead)} month(s) with zero phishing: {dead}")
    print(f"  shortened URLs: phishing {short[0]:.1f}%  ham {short[1]:.1f}%")
    print(f"  top phishing suffixes: {top_ph}")
    print(f"  [+] {FIG}/sms_timeline.pdf, sms_hosts.pdf, sms_tld.pdf, sms_shortener.pdf")
    print(f"  [+] {SEC}/tab_sms_tld.tex, gen_sms_figs.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
