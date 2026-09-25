#!/usr/bin/env python3
"""Figures and tables for the smishing study, from the ingested corpus.

Every number is read from data/processed/sms/*.csv, which sms_corpus_import.py writes. Nothing is
hand-typed, for the reason figstyle.py records: `make claims` verifies .tex, not compiled PDFs, so
a literal inside a figure is the one place a stale number survives a data change unseen.

RUN:  python3 scripts/make_smishing_assets.py   (after sms_corpus_import.py)
"""
from __future__ import annotations
import collections, csv, datetime as dt, io, json, os, sys

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
FIG = os.path.join(ROOT, "papers", "future_smishing", "figures")
SEC = os.path.join(ROOT, "papers", "future_smishing", "sections")

LAB = {"0": "ham", "1": "phishing"}


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
    """The class mix is not stationary, and a random split cannot see that."""
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
    ax.bar(x - w / 2, ph, w, color=ORANGE, edgecolor=INK, label="phishing")
    ax.bar(x + w / 2, ham, w, color=FILL_B, edgecolor=INK, hatch="///", label="ham")
    for i, v in enumerate(ph):
        ax.text(i - w / 2, v + 12, str(v), ha="center", fontsize=7, color=INK)
    for i, v in enumerate(ham):
        ax.text(i + w / 2, v + 12, str(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, months)
    ax.set_ylabel("messages")
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
            color=ORANGE, edgecolor=INK, label="phishing")
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

    fig, ax = plt.subplots(figsize=(5.6, 0.62 * len(rows) + 1.15))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0.0, color=INK, lw=1.0, zorder=1)
    for yy, (_lab, m, ci) in zip(y, rows):
        ax.plot([ci[0], ci[1]], [yy, yy], color=BLUE, lw=2.2, solid_capstyle="butt", zorder=2)
        ax.plot([m], [yy], "o", ms=6, color=ORANGE, mec=INK, mew=0.7, zorder=3)
        ax.text(ci[1] + 0.0012, yy, f"{m:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]",
                va="center", ha="left", fontsize=7.5, color=INK)
    ax.set_yticks(y, [r_[0] for r_ in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    lo = min(ci[0] for _l, _m, ci in rows)
    hi = max(ci[1] for _l, _m, ci in rows)
    pad = (hi - lo) * 0.10
    ax.set_xlim(lo - pad, hi + (hi - lo) * 1.15)
    ax.set_xlabel("difference in phishing-class F1")
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
        ax.axvline(floor, color=GRAY, ls="--", lw=1.0, zorder=3)
        ax.axvline(tok, color=GRAY, ls=":", lw=1.2, zorder=3)
        # Values ride at a fixed x, clear of both floor lines: printed at the bar end they
        # collided with whichever line the bar happened to stop near.
        for yy, v in zip(y, vals):
            ax.text(1.03, yy, f"{v:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_title(title, fontsize=8, color=INK, pad=4)
        ax.set_xlim(0, 1.20)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("phishing-class F1", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=8)
    # The two lines are the whole point of the panel and were unlabelled.
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=GRAY, ls="--", lw=1.0,
                               label="shallow-cue floor (diacritics + length + tokens)"),
                        Line2D([], [], color=GRAY, ls=":", lw=1.2,
                               label="redaction tokens alone")],
               loc="lower center", ncol=2, fontsize=7.2, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return panels


def fig_shortener(urls, out):
    """The reversal, drawn: shortening is a ham habit in this corpus."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    vals, labels, colours = [], [], []
    for lab, colour in (("1", ORANGE), ("0", FILL_B)):
        sub = [u for u in urls if u["label"] == lab]
        vals.append(100 * sum(int(u["shortener"]) for u in sub) / max(len(sub), 1))
        labels.append(LAB[lab])
        colours.append(colour)
    # Hatched ham, like every other two-class figure here: this one was the odd one out.
    b = ax.bar(labels, vals, 0.5, color=colours, edgecolor=INK,
               hatch=["", "///"])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center",
                fontsize=9, color=INK)
    ax.set_ylabel("URLs that are shortened (%)")
    ax.set_ylim(0, max(vals) * 1.35)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return vals


def emit_results():
    """The registered comparison, if it has been run. Absent until then, and the paper says so."""
    if not os.path.exists(FUSION):
        return None
    r = json.load(open(FUSION, encoding="utf-8"))
    buf = io.StringIO()
    buf.write("\\begin{table}[t]\n\\centering\\small\n")
    buf.write("\\caption{The registered comparison, %d seeds.}\n\\label{tab:sms_fusion}\n"
              % r["seeds"])
    buf.write("\\begin{tabular}{lrrr}\n\\toprule\n"
              "Arm & F1 & Precision & Recall \\\\\n\\midrule\n")
    for k, name in (("url", "URL only (CompPhish-21)"), ("text", "Text only (PhoBERT)"),
                    ("fusion", "Fusion")):
        m = r["arms"][k]
        buf.write(f"{name} & {m['f1']:.3f} & {m['precision']:.3f} & {m['recall']:.3f} \\\\\n")
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
           + "\\newcommand{\\SmsTestN}{%d}\n" % r["n_test"])
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


PREDEC = {"ft": os.path.join(ROOT, "data", "processed", "sms", "phobert_ft.json"),
          "dia": os.path.join(ROOT, "data", "processed", "sms", "diacritics_ablation.json"),
          "late": os.path.join(ROOT, "data", "processed", "sms", "late_fusion.json")}


def _d(v: float) -> str:
    """A signed 3-decimal delta; a value that rounds to zero prints as 0.000, never -0.000."""
    return "0.000" if abs(v) < 0.0005 else "%+.3f" % v


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
              + "\\newcommand{\\SmsSegDelta}{%+.3f}\n" % b["seg_minus_text"]["mean"]
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
    arms = [("URL only\n(CompPhish-21)", fus["arms"]["url"]["f1"], BLUE),
            ("redaction tokens\n(not language)", cues["baselines"]["tokens"]["f1"], GRAY),
            ("diacritics + length\n+ tokens", floor, GRAY),
            ("word TF-IDF\n(linear)", cues.get("language", {}).get("tfidf", {}).get("word", 0.0),
             ORANGE),
            ("text only\n(PhoBERT)", fus["arms"]["text"]["f1"], ORANGE),
            ("fusion\n(text + URL)", fus["arms"]["fusion"]["f1"], FILL_O)]
    arms = [a for a in arms if a[1] > 0]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    y = np.arange(len(arms))[::-1]
    ax.barh(y, [a[1] for a in arms], 0.62,
            color=[a[2] for a in arms], edgecolor=INK, linewidth=0.7)
    for yy, (_, v, _c) in zip(y, arms):
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(floor, color=GRAY, ls="--", lw=0.9, zorder=0)
    ax.set_yticks(y, [a[0] for a in arms], fontsize=8)
    ax.set_xlabel("phishing-class F1")
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
              "\\caption{What the wording is, per class.}\n\\label{tab:sms_logodds}\n"
              "\\begin{tabular}{lrrr@{\\qquad}lrrr}\n\\toprule\n"
              "\\multicolumn{4}{c}{marks phishing} & \\multicolumn{4}{c}{marks ham} \\\\\n"
              "\\cmidrule(lr){1-4}\\cmidrule(lr){5-8}\n"
              "term & $z$ & ph. & ham & term & $z$ & ph. & ham \\\\\n\\midrule\n")
    for a, b in zip(ph, ha):
        buf.write("%s & %.1f & %d & %d & %s & %.1f & %d & %d \\\\\n"
                  % (cell(a), a["z"], a["phish"], a["ham"],
                     cell(b), b["z"], b["phish"], b["ham"]))
    buf.write("\\bottomrule\n\\end{tabular}\n"
              "\\\\[4pt]\n\\begin{minipage}{0.94\\linewidth}\\footnotesize\n"
              "Log-odds ratio with an informative Dirichlet prior, over the whole corpus; $z$ is the\n"
              "prior-regularised score and the two count columns are raw occurrences. The phishing\n"
              "column is Vietnamese written with diacritics; the ham column is the same language\n"
              "written without them, plus carrier boilerplate (\\texttt{tb}, \\texttt{lh},\n"
              "\\texttt{viettel}) and the redactor's tokens. Post-hoc and unregistered.\n"
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


def main() -> int:
    for p in (MSG, URLS, SNAP):
        if not os.path.exists(p):
            print(f"[!] {p} missing — run scripts/sms_corpus_import.py", file=sys.stderr)
            return 1
    msgs, urls = read(MSG), read(URLS)
    os.makedirs(FIG, exist_ok=True)

    months, ham, ph = fig_timeline(msgs, os.path.join(FIG, "sms_timeline.pdf"))
    fig_hosts(urls, os.path.join(FIG, "sms_hosts.pdf"))
    per_tld = fig_tld(urls, os.path.join(FIG, "sms_tld.pdf"))
    short = fig_shortener(urls, os.path.join(FIG, "sms_shortener.pdf"))

    # the temporal confound, as a number the prose can cite
    dead = [m for m, h_, p_ in zip(months, ham, ph) if p_ == 0]
    top_ph = per_tld["1"].most_common(5)
    tot_ph = max(sum(per_tld["1"].values()), 1)
    tot_ham = max(sum(per_tld["0"].values()), 1)

    buf = io.StringIO()
    buf.write("\\begin{table}[t]\n\\centering\\small\n")
    buf.write("\\caption{Where each class registers, on the five suffixes phishing uses most.}\n"
              "\\label{tab:sms_tld}\n")
    buf.write("\\begin{tabular}{lrr}\n\\toprule\nSuffix & Phishing & Ham \\\\\n\\midrule\n")
    for t, n in top_ph:
        buf.write(f"\\texttt{{.{t}}} & {100*n/tot_ph:.1f}\\% & "
                  f"{100*per_tld['0'][t]/tot_ham:.1f}\\% \\\\\n")
    buf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_generated(os.path.join(SEC, "tab_sms_tld.tex"), buf.getvalue())

    macros = (
        "%% generated by scripts/make_smishing_assets.py; do not edit\n"
        f"\\newcommand{{\\SmsMonths}}{{{len(months)}}}\n"
        f"\\newcommand{{\\SmsSpanStart}}{{{months[0]}}}\n"
        f"\\newcommand{{\\SmsSpanEnd}}{{{months[-1]}}}\n"
        f"\\newcommand{{\\SmsDeadMonths}}{{{len(dead)}}}\n"
        f"\\newcommand{{\\SmsDeadMonthList}}{{{', '.join(dead)}}}\n"
        f"\\newcommand{{\\SmsDeadMonthHam}}{{{sum(h for h, p in zip(ham, ph) if p == 0):,}}}\n"
        .replace(",", "{,}").replace("{,} ", ", ")
    )
    write_generated(os.path.join(SEC, "gen_sms_figs.tex"), macros)

    emit_results()

    print(f"  timeline {months[0]}..{months[-1]} ({len(months)} months); "
          f"{len(dead)} month(s) with zero phishing: {dead}")
    print(f"  shortened URLs: phishing {short[0]:.1f}%  ham {short[1]:.1f}%")
    print(f"  top phishing suffixes: {top_ph}")
    print(f"  [+] {FIG}/sms_timeline.pdf, sms_hosts.pdf, sms_tld.pdf, sms_shortener.pdf")
    print(f"  [+] {SEC}/tab_sms_tld.tex, gen_sms_figs.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
