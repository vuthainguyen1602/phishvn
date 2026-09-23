#!/usr/bin/env python3
"""Generate the quishing paper's collection figures, and pin the snapshot they describe.

WHY THIS EXISTS. The prevalence numbers were typed into the sections by hand. They were wrong
three times -- rows read as pages, a denominator counting scans, a denominator missing an arm --
and then they went stale within the hour, because both collectors are still running and the
denominator grows every tick. A hand-typed number cannot be right for long about a live corpus.

The numbers are now written as macros from one computation, together with the date they describe.
Regenerating is a deliberate act that moves the paper and the snapshot together; the claims suite
compares the paper against the snapshot, not against a moving live count, and separately refuses a
snapshot the ledgers can no longer support.

Since 2026-08-31 it pins a second snapshot as well: the sweep and the restoration arms have both
been run, so the results section reads macros and tables written from `dfr_snapshot.json` and
`restore_snapshot.json` rather than numbers typed out of a terminal. Same reason as the prevalence
macros -- a hand-copied cell is wrong the first time the analysis is re-run, and nothing complains.

RUN:  python3 scripts/make_quishing_assets.py
"""
from __future__ import annotations
import csv, datetime as dt, json, os, sys, time
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated  # noqa: E402

EXAMINED = os.path.join("data", "raw", ".qr_scan_examined.csv")
SUBMIT = os.path.join("data", "raw", "qr_submit_ledger.csv")
HITS = os.path.join("data", "raw", "urlscan_qrs.csv")
SUBMIT_HITS = os.path.join("data", "raw", "qr_submit_qrs.csv")
INDUCED = os.path.join("data", "processed", "infra", "self_induced_hosts.csv")
# The author's reading of every page whose code the automatic triage does not already class as a
# messenger link or a link back to the same site. Tracked with the manuscript, because the count
# of quishing pages is a judgement and a judgement has to be on file to be checked.
ADJUDICATION = os.path.join("papers", "future_quishing", "qr_adjudication.csv")
# verdict -> the bucket the paper reports it in. `quishing` is the narrow sense the paper counts
# by: the code is what carries the victim to the attack.
VERDICT_BUCKET = {"quishing": "quishing", "phishing_inert_qr": "inert_lure",
                  "abuse_app_download": "abuse", "off_target_app_download": "off_target",
                  "benign": "benign_hand", "benign_payment": "benign_hand"}
SNAP = os.path.join("data", "processed", "qr", "prevalence_snapshot.json")
DFR_SNAP = os.path.join("data", "processed", "qr", "dfr_snapshot.json")
RES_SNAP = os.path.join("data", "processed", "qr", "restore_snapshot.json")
V2_SNAP = os.path.join("data", "processed", "qr", "restore_v2_snapshot.json")
V2_ARMS = {
    "conv": os.path.join("data", "processed", "qr", "qr_restore_arms_conv_dfr.csv"),
    "hybrid": os.path.join("data", "processed", "qr", "qr_restore_arms_hybrid_dfr.csv"),
    "restormer": os.path.join("data", "processed", "qr", "qr_restore_arms_restormer_dfr.csv"),
}
SEC = os.path.join("papers", "future_quishing", "sections")
OUT_TEX = os.path.join(SEC, "gen_prevalence.tex")

# The order the tables print transforms in. Worst first per the pooled sweep, so a reader meets the
# saturated pair before the ones with a usable gradient; `clean` last because it is the control.
TR_ORDER = ("motion", "blur", "saltpepper", "invert", "perspective", "logo", "rotate",
            "contrast", "clean")
DEC = ("opencv", "pyzbar", "wechat")
DEC_TEX = {"opencv": r"\texttt{OpenCV}", "pyzbar": r"\texttt{PyZbar}", "wechat": r"\texttt{WeChatQR}"}


def rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


FREEZE = os.path.join("data", "processed", "qr", "prevalence_snapshot.frozen.json")


def frozen_note(live: dict) -> dict | None:
    """The submitted numbers, pinned.

    The landing-page arm keeps collecting, so this paper's denominator moves every tick: it went
    from 1,801 to 2,215 pages examined in the three days around the JCC packet being built, and
    every macro in Section 3.1 followed it. That is right for a living repository and wrong for a
    manuscript under review, where the numbers a referee reads have to be the numbers that were
    submitted. With this file present the macros are written FROM it, the collectors carry on
    untouched, and the drift is printed at every run so nobody forgets the paper is pinned.

    Freeze with `--freeze`, lift with `--thaw`, both deliberate.
    """
    if not os.path.exists(FREEZE):
        return None
    pin = json.load(open(FREEZE, encoding="utf-8"))
    moved = {k: (pin["snapshot"].get(k), live.get(k))
             for k in sorted(set(pin["snapshot"]) | set(live))
             if pin["snapshot"].get(k) != live.get(k)}
    print(f"  [pinned] prevalence frozen {pin['frozen_on']} for {pin['reason']}")
    if moved:
        print("           live data has since moved: "
              + ", ".join(f"{k} {a}->{b}" for k, (a, b) in moved.items()))
    else:
        print("           live data still matches the pin")
    return pin["snapshot"]


def tag_pool(prev: dict) -> dict:
    """The searchable URLScan pool the audit draws from, as counts rather than as prose.

    These three were typed by hand into Section 3.1 -- 32 worldwide `task.tags:quishing`, 1,666
    `task.tags:phishing AND page.domain:vn`, 1,698 together -- because they come from a console
    query rather than from the pipeline. Nothing regenerated them and no guard read them, so if
    the pool moved the paper would have gone on saying otherwise. Verified against the API on
    2026-09-03 (32 and 1,666, unchanged) and captured here.

    The query runs ONLY with --refresh-pool: this paper's numbers are pinned for submission, and
    a generator that silently re-queried a live index would move them underneath the pin.
    """
    if "--refresh-pool" not in sys.argv:
        return {k: prev[k] for k in ("tag_quishing", "tag_vn_phishing") if k in prev}
    import urllib.parse
    import urllib.request
    key = os.environ.get("URLSCAN_API_KEY", "")
    if not key:
        print("[!] --refresh-pool needs URLSCAN_API_KEY (set -a; . scripts/.env)", file=sys.stderr)
        return {k: prev[k] for k in ("tag_quishing", "tag_vn_phishing") if k in prev}
    out = {}
    for field, q in (("tag_quishing", "task.tags:quishing"),
                     ("tag_vn_phishing", "task.tags:phishing AND page.domain:vn")):
        u = "https://urlscan.io/api/v1/search/?" + urllib.parse.urlencode({"q": q, "size": 1})
        req = urllib.request.Request(u, headers={"API-Key": key, "User-Agent": "PhishVN/1.0"})
        out[field] = json.load(urllib.request.urlopen(req, timeout=30))["total"]
        time.sleep(2)
    print(f"  [pool] refreshed from URLScan: {out}")
    return out


def prevalence() -> int:
    ex = rows(EXAMINED)
    if not ex:
        # Names no script on purpose: this file is exported to the public mirror, which
        # does not carry the device-sync wrapper, and the mirror's link gate fails on a
        # path a reader cannot open.
        print(f"[!] {EXAMINED} missing or empty — sync the collector output from the "
              "device before regenerating", file=sys.stderr)
        return 1

    urls = {(r.get("page_url") or "").strip() for r in ex} - {""}
    hosts = {(urlparse(u).hostname or "").lower() for u in urls} - {""}
    crawled, scans = len(urls), len(ex)

    sub = {(r.get("domain") or "").strip().lower()
           for r in rows(SUBMIT) if (r.get("shot_file") or "").strip()}

    # Names that exist only because an earlier submission of ours led to them are not pages
    # anyone deployed: out of the denominator and out of the numerator.
    induced = {(r.get("domain") or "").strip().lower() for r in rows(INDUCED)}
    induced_left_out = len(sub & induced)
    sub -= induced
    overlap = len(hosts & sub)
    examined = crawled + len(sub) - overlap

    # Both arms' findings. Until 2026-09-19 the submitting arm printed its payloads to a cron log
    # and kept them nowhere, so this numerator held the search arm alone while the denominator
    # counted both. One triage function classes both arms.
    from qr_scan import triage
    by_page: dict = {}
    crawled_hit_pages = set()
    for r in rows(HITS):
        pg = (r.get("source_page") or "").strip().rstrip("/")
        if pg:
            crawled_hit_pages.add(pg)
            by_page.setdefault(pg, []).append(triage(pg, r.get("qr_decoded_url", "")))
    for r in rows(SUBMIT_HITS):
        pg = (r.get("source_page") or "").strip().rstrip("/")
        if pg and (urlparse(pg).hostname or "").lower() not in induced:
            by_page.setdefault(pg, []).append(triage(pg, r.get("qr_decoded_url", "")))
    scan_hits = len(rows(HITS))
    hits = len(by_page)
    verdicts = {(r.get("source_page") or "").strip().rstrip("/"): (r.get("verdict") or "").strip()
                for r in rows(ADJUDICATION)}
    buckets = {k: 0 for k in ("benign_auto", "benign_hand", "off_target", "abuse", "inert_lure",
                              "quishing", "unadjudicated")}
    for pg, found in by_page.items():
        if all(t["payload_kind"] == "messenger" or str(t["same_site"]) == "1" for t in found):
            buckets["benign_auto"] += 1
        else:
            buckets[VERDICT_BUCKET.get(verdicts.get(pg, ""), "unadjudicated")] += 1
    if buckets["unadjudicated"]:
        print(f"  [!] {buckets['unadjudicated']} page(s) with a code have no verdict in {ADJUDICATION}: "
              "the paper cannot state a quishing count until they do", file=sys.stderr)
    adjudicated_on = max((r.get("adjudicated_on") or "" for r in rows(ADJUDICATION)), default="")

    rate = 100 * hits / examined if examined else 0.0
    # the search arm's own wrong division (its pages over its scans); the submitting arm has no scans
    scan_rate = 100 * len(crawled_hit_pages) / scans if scans else 0.0
    rescan = scans / crawled if crawled else 0.0
    today = dt.date.today().isoformat()

    snap = {"date": today, "scans": scans, "crawled_pages": crawled,
            "submitted_pages": len(sub), "overlap": overlap, "pages_examined": examined,
            "pages_with_qr": hits, "scans_with_qr": scan_hits,
            "crawled_pages_with_qr": len(crawled_hit_pages), "induced_left_out": induced_left_out,
            "buckets": buckets, "adjudicated_on": adjudicated_on,
            "prevalence_pct": round(rate, 2), "scan_based_pct": round(scan_rate, 2),
            "rescan_factor": round(rescan, 2)}
    prev = json.load(open(SNAP, encoding="utf-8")) if os.path.exists(SNAP) else {}
    snap.update(tag_pool(prev))
    os.makedirs(os.path.dirname(SNAP), exist_ok=True)
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, sort_keys=True)

    # The live snapshot is always written -- collection is not what gets frozen. Only the numbers
    # the PAPER prints are pinned.
    pin = frozen_note(snap)
    if pin:
        snap = dict(pin)
        today = snap["date"]
        scans, crawled, overlap = snap["scans"], snap["crawled_pages"], snap["overlap"]
        examined, hits, scan_hits = (snap["pages_examined"], snap["pages_with_qr"],
                                     snap["scans_with_qr"])
        rate, scan_rate, rescan = (snap["prevalence_pct"], snap["scan_based_pct"],
                                   snap["rescan_factor"])
        sub = [None] * snap["submitted_pages"]
        buckets = snap.get("buckets", buckets)
        adjudicated_on = snap.get("adjudicated_on", adjudicated_on)
        induced_left_out = snap.get("induced_left_out", induced_left_out)
        crawled_hit_pages = [None] * snap.get("crawled_pages_with_qr", len(crawled_hit_pages))

    def m(name, val):
        return "\\newcommand{\\Qr%s}{%s}\n" % (name, val)

    tex = (m("SnapDate", dt.date.fromisoformat(today).strftime("%-d %B %Y"))
           + m("Scans", f"{scans:,}") + m("CrawledPages", f"{crawled:,}")
           + m("SubmittedPages", f"{len(sub):,}") + m("Overlap", f"{overlap:,}")
           + m("PagesExamined", f"{examined:,}") + m("PagesWithQr", f"{hits:,}")
           + m("ScansWithQr", f"{scan_hits:,}") + m("Prevalence", f"{rate:.2f}")
           + m("ScanBased", f"{scan_rate:.2f}") + m("Rescan", f"{rescan:.2f}")
           # The third wrong division the paper reports as a wrong division: QR-bearing ROWS over
           # scans. It was hand-typed as 0.30%, and by 2026-09-03 the live denominator had made it
           # 0.32 -- in a file whose own docstring says a hand-typed number cannot be right for
           # long about a live corpus.
           + m("RowBased", f"{100 * scan_hits / scans:.2f}" if scans else "0.00")
           # What the pages with a code turned out to be. `Quishing` is the narrow count: a code
           # that carries the victim to the attack. `InertLure` is a phishing page that displays
           # a code which delivers nothing.
           + m("CrawledWithQr", len(crawled_hit_pages))
           + m("SubmittedWithQr", hits - len(crawled_hit_pages))
           + m("InducedLeftOut", induced_left_out)
           + m("BenignAuto", buckets["benign_auto"]) + m("BenignHand", buckets["benign_hand"])
           + m("OffTarget", buckets["off_target"]) + m("Abuse", buckets["abuse"])
           + m("InertLure", buckets["inert_lure"]) + m("Quishing", buckets["quishing"])
           + m("Unadjudicated", buckets["unadjudicated"])
           + m("HandRead", hits - buckets["benign_auto"])
           + (m("AdjudicatedOn", dt.date.fromisoformat(adjudicated_on).strftime("%-d %B %Y"))
              if adjudicated_on else "")
           # The searchable pool, no longer typed into the prose by hand.
           + (m("TagQuishing", f"{snap['tag_quishing']:,}")
              + m("TagVnPhishing", f"{snap['tag_vn_phishing']:,}")
              + m("TagPool", f"{snap['tag_quishing'] + snap['tag_vn_phishing']:,}")
              if "tag_quishing" in snap else ""))
    write_generated(OUT_TEX, tex)

    print(f"  snapshot {today}: {hits}/{examined} = {rate:.2f}%  "
          f"({crawled} crawled + {len(sub)} submitted - {overlap} both, from {scans:,} scans)")
    print(f"  [+] {SNAP}\n  [+] {OUT_TEX}")
    return 0


def main() -> int:
    """Both snapshots, in one act. They move together or the paper cites two different studies."""
    return prevalence() or results()


def _p(x: float) -> str:
    """A p-value a caption can carry, WITHOUT math delimiters -- the call site supplies them, so
    that the obvious `$p = \\QrTOneP$` is not a fatal error. Wilcoxon on 2,000 paired URLs
    underflows to 0.0, and printing `p = 0` claims an exactness the test does not have."""
    if x <= 0:
        return r"<10^{-300}"
    if x >= 1e-4:
        return "%.4f" % x
    m, e = ("%.1e" % x).split("e")
    return r"%s{\times}10^{%d}" % (m, int(e))


def _ppm(x: float) -> str:
    """The same number as `_pp`, in math mode, for a table cell: a bare `-1.0` sets a hyphen."""
    return "$%+.1f$" % x


def _pp(x: float) -> str:
    """Bare, no math delimiters. Every call site wraps it, because a macro that carries its own
    `$` is a fatal error the moment someone writes the obvious `$\\QrResTOne$`."""
    return "%+.1f" % x


def fig_strength(d, out):
    """The band the paper keeps describing and never drew.

    A defender's question is not whether a decoder reads a clean code but where it stops reading,
    and that is a curve rather than a number. The two saturated transforms are drawn flat and
    labelled, so the eye does not read their horizontal lines as a gradient that happens to be
    zero."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    from figstyle import apply, ORANGE, BLUE, INK, GRAY, TEAL, PURPLE
    plt = apply()
    st = d["by_strength"]
    xs = [0.25, 0.50, 0.75, 1.00]
    order = ["perspective", "logo", "saltpepper", "rotate", "invert", "contrast"]
    # Paired by hue, separated by DASH -- not by lightness. FILL_O and FILL_B are pale fill tints
    # meant for bar bodies; on a 1.4 pt line they washed out, and `invert` at a flat 69% was nearly
    # invisible in print. Worse, distinguishing a pair by lightness alone contradicts what this
    # paper's own preamble promises -- that meaning also travels through line style, so no
    # conclusion rests on colour. Same hue, same strength, different stroke.
    colors = {"perspective": ORANGE, "logo": BLUE, "saltpepper": ORANGE, "rotate": GRAY,
              "invert": BLUE, "contrast": INK}
    strokes = {"saltpepper": (0, (5, 2)), "invert": (0, (5, 2))}
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = []                       # (y at the right edge, text, colour)
    for tr in order:
        if tr not in st:
            continue
        ys = [st[tr]["%.2f" % x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=colors[tr],
                ls=strokes.get(tr, "-"))
        labels.append([ys[-1], tr, colors[tr]])
    # The saturated pair is labelled at the LEFT, where their lines already start flat. Sending
    # them to the right edge with the others pushed one label past the top of the axes, and a
    # clipped label is worse than no label.
    # ONE label for the pair, not one each. Both lines are grey and dotted, so two stacked labels
    # could not be told apart -- and the reader does not need to: the point is that neither
    # carries a usable gradient. Labelling them separately also invited the reading that both are
    # perfectly flat, which motion is (100 at every strength) and blur is not (97.4 then 100).
    # One colour and one stroke EACH, and the label in the line's own colour. Drawing both in grey
    # and merging the labels was readable but threw away which line was which, and the two are not
    # the same: motion is 100 at every strength, blur climbs from 97.4. TEAL and PURPLE are the
    # palette's categorical slots, CVD-safe against the BLUE already in this panel; the strokes
    # differ as well, so the pair survives greyscale where colour alone would not.
    sat_style = {"blur": (TEAL, ":"), "motion": (PURPLE, (0, (4, 1.6, 1, 1.6)))}
    sat = list(d.get("saturated", []))
    for i, tr in enumerate(sat):
        col, ls = sat_style.get(tr, (GRAY, ":"))
        ys = [st[tr]["%.2f" % x] for x in xs]
        ax.plot(xs, ys, ls=ls, lw=1.3, color=col)
        y0 = min(st[t]["%.2f" % xs[0]] for t in sat)
        ax.annotate(f"{tr} (saturated)", (xs[0] + 0.02, y0 - 6 - 7 * i), fontsize=7,
                    color=col, va="center")
    # Two saturated lines sit on top of each other at 100% and their labels did too. Push the
    # right-edge labels apart by a minimum spacing instead of drawing them where the data lands.
    labels.sort(key=lambda t: t[0])
    gap = 7.0
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < gap:
            labels[i][0] = labels[i - 1][0] + gap
    for y, txt, col in labels:
        ax.annotate(txt, (xs[-1] + 0.015, y), fontsize=7, color=col, va="center")
    ax.set_xlabel("transformation strength")
    ax.set_ylabel("decode failure rate (%)")
    ax.set_xticks(xs)
    ax.set_xlim(0.2, 1.33)
    ax.set_ylim(-4, 108)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _gated_metrics(path: str) -> dict:
    """Score the cascade the manuscript's Algorithm 1 actually deploys.

    Each stage tries WeChatQR, PyZbar and OpenCV in that order.  A decoded-but-wrong payload stops
    the operational cascade just as a correct payload does; ground truth is unavailable at
    deployment.  The current held-out arms contain no such stop, but keeping it in the computation
    prevents a future rerun from silently turning an incorrect payload into a rescue.
    """
    order = ("wechat", "pyzbar", "opencv")
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base, arm = row["sample_id"].rsplit("__", 1)
            cells.setdefault(base, {}).setdefault(arm, {})[row["decoder"]] = (
                int(row["decoded"]), int(row["correct"]))

    raw_ok = polarity_added = restore_added = wrong_stop = 0
    for arms in cells.values():
        stopped = correct = False
        stage_correct = {}
        for arm in ("raw", "control", "restored"):
            stage_correct[arm] = False
            for dec in order:
                decoded, is_correct = arms[arm][dec]
                if not stopped and decoded:
                    stopped = True
                    correct = bool(is_correct)
                    stage_correct[arm] = correct
                    if not is_correct:
                        wrong_stop += 1
                    break
            if stopped:
                break
        if stage_correct["raw"]:
            raw_ok += 1
        elif stage_correct["control"]:
            polarity_added += 1
        elif stage_correct["restored"]:
            restore_added += 1

    n = len(cells)
    raw_dfr = 100 * (1 - raw_ok / n)
    gated_dfr = 100 * (1 - (raw_ok + polarity_added + restore_added) / n)
    return {"n": n, "raw_dfr": raw_dfr, "gated_dfr": gated_dfr,
            "gain_pp": raw_dfr - gated_dfr, "raw_ok": raw_ok,
            "polarity_added": polarity_added, "restore_added": restore_added,
            "wrong_stop": wrong_stop}


def fig_gate(v, gated, out):
    """Deployment result beside the registered ungated safety diagnostic."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
    from figstyle import apply, BLUE, ORANGE, INK, GRAY
    plt = apply()
    names = ["Conv", "Hybrid", "Restormer"]
    keys = ["conv", "hybrid", "restormer"]
    cols = [BLUE, ORANGE, GRAY]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    raw = gated["conv"]["raw_dfr"]
    for i, (key, name, col) in enumerate(zip(keys, names, cols)):
        end = gated[key]["gated_dfr"]
        ax1.plot([0, 1], [raw, end], color=col, marker="o", lw=1.6)
        ax1.annotate(name, (1.03, end), color=col, fontsize=7.5, va="center")
        ax1.annotate(f"{end:.1f}", (1, end), xytext=(-5, 5), textcoords="offset points",
                     color=col, fontsize=7, ha="right")
    ax1.set_xticks([0, 1], ["raw 3-decoder\ncascade", "+ polarity, then\nrestore on failure"])
    ax1.set_ylabel("decode failure rate (%)")
    ax1.set_xlim(-0.08, 1.42)
    ax1.set_ylim(0, 34)
    ax1.set_title("gated deployment path")

    for key, name, col in zip(keys, names, cols):
        g = v["guard"][key]["opencv"]
        ax2.scatter(g["rescued"], g["destroyed_pct_of_correct"], s=36, color=col, zorder=3)
        ax2.annotate(name, (g["rescued"], g["destroyed_pct_of_correct"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=col)
    ax2.axhline(v["guard_pct"], color=INK, ls="--", lw=1.0)
    ax2.annotate(f"registered ungated guard ({v['guard_pct']:.0f}%)",
                 (970, v["guard_pct"]), xytext=(0, 4), textcoords="offset points",
                 fontsize=7, color=INK)
    ax2.set_xlabel("renders rescued by restored OpenCV")
    ax2.set_ylabel("control-readable renders broken (%)")
    ax2.set_title("ungated restoration diagnostic")
    ax2.set_xlim(950, 1550)
    ax2.set_ylim(0, 9)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def v2_assets() -> None:
    """The architecture comparison: macros and one table.

    Every verdict this writes is the analysis script's, including the two that are not wins. The
    prereg puts "choose the better arm afterwards" out of scope, so the table has a row per arm and
    no flag to drop one."""
    if not os.path.isfile(V2_SNAP):
        print(f"  [!] {V2_SNAP} missing — run scripts/analyze_qr_restore_v2.py",
              file=sys.stderr)
        return
    v = json.load(open(V2_SNAP, encoding="utf-8"))
    missing = [p for p in V2_ARMS.values() if not os.path.isfile(p)]
    if missing:
        print(f"  [!] gated-pipeline inputs missing: {', '.join(missing)}", file=sys.stderr)
        return
    gated = {arch: _gated_metrics(path) for arch, path in V2_ARMS.items()}
    NAME = {"hybrid": "Hybrid", "restormer": "Restormer"}
    # No digits anywhere in a macro name: \QrV2 parses as \QrV followed by a literal 2, and T1 as
    # \...T then 1. Hence QrVTwo, TOne, TTwo.
    TNAME = {"T1": "TOne", "T2": "TTwo"}
    m = "%% generated by scripts/make_quishing_assets.py; do not edit\n"
    for arch, tag in NAME.items():
        for t in ("T1", "T2"):
            r = v["registered"][arch][t]
            m += "\\newcommand{\\QrVTwo%s%s}{%+.1f}\n" % (tag, TNAME[t], r["mean_pp"])
            m += "\\newcommand{\\QrVTwo%s%sVerdict}{%s}\n" % (tag, TNAME[t],
                                                                r["verdict"].upper())
        g = v["guard"][arch]["opencv"]
        m += "\\newcommand{\\QrVTwo%sBreak}{%.1f}\n" % (tag, g["destroyed_pct_of_correct"])
        m += "\\newcommand{\\QrVTwo%sRescued}{%s}\n" % (tag, f"{g['rescued']:,}")
        m += "\\newcommand{\\QrVTwo%sDestroyed}{%s}\n" % (tag, f"{g['destroyed']:,}")
        b = v.get("budget", {}).get(arch, {})
        m += "\\newcommand{\\QrVTwo%sParams}{%s}\n" % (tag, f"{b.get('params', 0):,}")
        if b.get("train_seconds"):
            m += "\\newcommand{\\QrVTwo%sMinutes}{%.0f}\n" % (tag, b["train_seconds"] / 60)
        by = v["by_transform"][arch]
        for tr, key in (("logo", "Logo"), ("contrast", "Contrast"), ("motion", "Motion")):
            k = "opencv/" + tr
            if k in by:
                m += "\\newcommand{\\QrVTwo%s%s}{%+.1f}\n" % (
                    tag, key, by[k]["control"] - by[k]["restored"])
    cg = v["guard"]["conv"]["opencv"]
    m += "\\newcommand{\\QrVTwoConvBreak}{%.1f}\n" % cg["destroyed_pct_of_correct"]
    m += "\\newcommand{\\QrVTwoConvLogo}{%+.1f}\n" % (
        v["by_transform"]["conv"]["opencv/logo"]["control"]
        - v["by_transform"]["conv"]["opencv/logo"]["restored"])
    m += "\\newcommand{\\QrVTwoGuard}{%.0f}\n" % v["guard_pct"]
    m += "\\newcommand{\\QrVTwoBarOne}{%.0f}\n" % v["bars"]["T1"]
    m += "\\newcommand{\\QrVTwoBarTwo}{%.0f}\n" % v["bars"]["T2"]
    m += "\\newcommand{\\QrVTwoUrls}{%d}\n" % v["registered"]["hybrid"]["T1"]["n_urls"]
    m += "\\newcommand{\\QrGateRawDfr}{%.1f}\n" % gated["conv"]["raw_dfr"]
    for arch, tag in (("conv", "Conv"), ("hybrid", "Hybrid"), ("restormer", "Restormer")):
        g = gated[arch]
        m += "\\newcommand{\\QrGate%sDfr}{%.1f}\n" % (tag, g["gated_dfr"])
        m += "\\newcommand{\\QrGate%sGain}{%.1f}\n" % (tag, g["gain_pp"])
        m += "\\newcommand{\\QrGate%sAdded}{%s}\n" % (tag, f"{g['restore_added']:,}")
    m += "\\newcommand{\\QrGatePolarityAdded}{%s}\n" % f"{gated['conv']['polarity_added']:,}"
    m += "\\newcommand{\\QrGateRenders}{%s}\n" % f"{gated['conv']['n']:,}"
    write_generated(os.path.join(SEC, "gen_qr_v2.tex"), m)

    body = ""
    for arch, tag in (("conv", "A: conv $3{\\times}3$"), ("hybrid", "B: hybrid"),
                      ("restormer", "C: Restormer")):
        r = v["registered"].get(arch)
        g = v["guard"][arch]["opencv"]
        b = v.get("budget", {}).get(arch, {})
        t1 = ("---" if not r else "$%+.1f$ (%s)" % (r["T1"]["mean_pp"], r["T1"]["verdict"][:4]))
        t2 = ("$+3.5$ (part)" if not r
              else "$%+.1f$ (%s)" % (r["T2"]["mean_pp"], r["T2"]["verdict"][:4]))
        flag = "" if g["destroyed_pct_of_correct"] <= v["guard_pct"] else r"$^{\ddagger}$"
        body += ("%s & %s & %s & %s & %s & %s%s \\\\\n"
                 % (tag, f"{b.get('params', 0):,}", t1, t2,
                    f"{g['rescued']:,}", f"{g['destroyed_pct_of_correct']:.1f}\\%", flag))
    write_generated(os.path.join(SEC, "tab_qr_v2.tex"), r"""\begin{table}[htbp]
\centering\small
\caption{The architecture comparison, registered in \texttt{PREREG\_restore\_v2.md}.}
\label{tab:qr_v2}
\setlength{\tabcolsep}{5pt}\footnotesize
\begin{tabular}{lrrrrr}
\toprule
Arm & params & T1: \texttt{logo} vs A & T2: vs raw+WeChat & resc. & broke \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\\[4pt]
\begin{minipage}{0.94\linewidth}\footnotesize
Percentage points of OpenCV DFR on \QrVTwoUrls{} held-out URLs, positive when the arm is ahead;
Benjamini--Hochberg over $m=4$. T1's bar is $+\QrVTwoBarOne$\,pp and T2's is $+\QrVTwoBarTwo$\,pp,
both fixed at registration. Arm A's T2 figure is the first study's and is shown for comparison, not
re-tested. \emph{broke} is the share of renders the control arm decoded and the restored arm did
not: $^{\ddagger}$~exceeds the registered $\QrVTwoGuard\%$ guard, which is reported whatever T1 and
T2 say.
\end{minipage}
\end{table}
""")
    fig_gate(v, gated, os.path.join(os.path.dirname(SEC), "figures", "fig_qr_gate.pdf"))


def results() -> int:
    """Write the results macros and tables from the two analysis snapshots.

    Nothing here recomputes anything: `analyze_qr_dfr.py` and `train_qr_restore.py --from-dfr` are
    the registered analyses and they write the snapshots. This only formats them, so a number in
    the paper and a number in the snapshot cannot disagree without one of the two files being
    older than the other -- which the claims suite is what checks.
    """
    if not (os.path.isfile(DFR_SNAP) and os.path.isfile(RES_SNAP)):
        print(f"[!] {DFR_SNAP} or {RES_SNAP} missing — run scripts/analyze_qr_dfr.py and "
              f"scripts/train_qr_restore.py --from-dfr", file=sys.stderr)
        return 1
    d = json.load(open(DFR_SNAP, encoding="utf-8"))
    r = json.load(open(RES_SNAP, encoding="utf-8"))

    def m(name, val):
        return "\\newcommand{\\Qr%s}{%s}\n" % (name, val)

    # ---- macros the prose reads -------------------------------------------------------------
    t1, t2 = d["T1"], d["T2"]
    worst_label = max(t2["per_decoder"].values(), key=lambda v: abs(v["diff_pp"]))
    rt1, rt2 = r["registered"]["T1"], r["registered"]["T2"]
    ec_logo = d["by_ec"]["logo"]
    ec_gain = {k: v["L"] - v["H"] for k, v in d["by_ec"].items()}
    wrong = max(v["wrong_payload_pct"] for v in d["failure_kind"].values())

    tex = (m("Rows", f"{d['rows']:,}") + m("Urls", f"{d['urls']:,}")
           + m("Renders", f"{d['rows'] // len(d['decoders']):,}")
           + m("GridComplete", "complete" if d["grid_complete"] else "INCOMPLETE")
           # T1 — render scale
           + m("TOneOpencv", _pp(t1["per_decoder"]["opencv"]["mean_pp"]))
           + m("TOnePyzbar", _pp(t1["per_decoder"]["pyzbar"]["mean_pp"]))
           + m("TOneWechat", _pp(t1["per_decoder"]["wechat"]["mean_pp"]))
           + m("TOneP", _p(t1["p_pooled"]))
           + m("TOneVerdict", t1["verdict"].upper())
           # T2 — label confound
           + m("TTwoMax", _pp(worst_label["diff_pp"]))
           + m("TTwoP", _p(t2["p_pooled"]))
           + m("TTwoVerdict", t2["verdict"].upper())
           + m("TTwoNPhish", f"{worst_label['n_phishing']:,}")
           + m("TTwoNBenign", f"{worst_label['n_benign']:,}")
           # what error correction buys, and what it does not
           + m("EcLogoL", "%.1f" % ec_logo["L"]) + m("EcLogoH", "%.1f" % ec_logo["H"])
           + m("EcLogoGain", "%.1f" % ec_gain["logo"])
           + m("EcNextGain", "%.1f" % sorted(ec_gain.values())[-2])
           + m("EcNextName", sorted(ec_gain, key=ec_gain.get)[-2])
           # the failure a pipeline meets
           + m("WrongPayload", "%.4f" % wrong)
           # post-hoc: the cascade question, and the density mechanism
           + m("CascadeBest", "%.1f" % d["cascade"]["overall_best_dfr"])
           + m("CascadeAll", "%.1f" % d["cascade"]["overall_cascade_dfr"])
           + m("CascadeGain", "%.2f" % d["cascade"]["overall_gain_pp"])
           + m("CascadePersp", "%.1f" % d["cascade"]["per_transform"]["perspective"]["gain_vs_best_pp"])
           + m("DensLogoSparse", "%.1f" % d["by_density"]["dfr"]["logo"]["<=25"])
           + m("DensLogoDense", "%.1f" % d["by_density"]["dfr"]["logo"]["50+"])
           + m("DensPerspSparse", "%.1f" % d["by_density"]["dfr"]["perspective"]["<=25"])
           + m("DensPerspDense", "%.1f" % d["by_density"]["dfr"]["perspective"]["50+"])
           + m("DensNSparse", "{:,}".format(d["by_density"]["n_renders"]["<=25"]))
           + m("DensNDense", "{:,}".format(d["by_density"]["n_renders"]["50+"]))
           + m("SilentOpencv", "%.1f" % d["failure_kind"]["opencv"]["silent_pct"])
           + m("SilentWechat", "%.1f" % d["failure_kind"]["wechat"]["silent_pct"])
           # restoration, registered
           + m("ResUrls", f"{rt1['n_urls']:,}")
           + m("ResTOne", _pp(rt1["mean_pp"])) + m("ResTOneMed", _pp(rt1["median_pp"]))
           + m("ResTOneP", _p(rt1["p_bh"])) + m("ResTOneVerdict", rt1["verdict"].upper())
           + m("ResTTwo", _pp(rt2["mean_pp"])) + m("ResTTwoMed", _pp(rt2["median_pp"]))
           + m("ResTTwoP", _p(rt2["p_bh"])) + m("ResTTwoVerdict", rt2["verdict"].upper())
           + m("ResRawWechat", "%.1f" % r["by_arm"]["wechat"]["raw"])
           + m("ResRestoredOpencv", "%.1f" % r["by_arm"]["opencv"]["restored"])
           + m("ResRestoredWechat", "%.1f" % r["by_arm"]["wechat"]["restored"])
           + m("ResPolarityOpencv",
               "%.1f" % (r["by_arm"]["opencv"]["raw"] - r["by_arm"]["opencv"]["control"]))
           + m("ResPolarityWechat",
               "%.1f" % (r["by_arm"]["wechat"]["raw"] - r["by_arm"]["wechat"]["control"]))
           + m("ResNetworkOpencv",
               "%.1f" % (r["by_arm"]["opencv"]["control"] - r["by_arm"]["opencv"]["restored"]))
           + m("ResNetworkWechat",
               "%.1f" % (r["by_arm"]["wechat"]["control"] - r["by_arm"]["wechat"]["restored"]))
           + m("ResRawOpencv", "%.1f" % r["by_arm"]["opencv"]["raw"])
           + m("ResMotionWechat", "%.1f" % r["by_transform"]["wechat"]["motion"]["control"])
           + m("ResMotionWechatOut", "%.1f" % r["by_transform"]["wechat"]["motion"]["restored"])
           + m("ResRescued", f"{r['transitions']['opencv']['rescued']:,}")
           + m("ResDestroyed", f"{r['transitions']['opencv']['destroyed']:,}")
           + m("ResRenders", f"{r['transitions']['opencv']['n']:,}"))
    write_generated(os.path.join(SEC, "gen_qr_results.tex"), tex)

    # ---- table: DFR per decoder x transform ---------------------------------------------------
    def cross(t, dec):
        # `<0.25` in text mode is an inverted question mark in OT1, silently. Math mode or nothing.
        c = d["crossing"][t]
        if isinstance(c, str):
            return "sat."
        v = c[dec]
        return "--" if v == "none" else ("$%s$" % v)

    body = ""
    for t in TR_ORDER:
        row = d["by_transform"][t]
        flag = r"$^{\dagger}$" if t in d["saturated"] else ""
        casc = d["cascade"]["per_transform"][t]["cascade"]
        body += ("\\texttt{%s}%s & %.1f & %.1f & %.1f & %.1f & %s & %s & %s \\\\\n"
                 % (t, flag, row["opencv"], row["pyzbar"], row["wechat"], casc,
                    cross(t, "opencv"), cross(t, "pyzbar"), cross(t, "wechat")))
    write_generated(os.path.join(SEC, "tab_qr_dfr.tex"), r"""\begin{table}[htbp]
\centering\small
\caption{Decode failure rate per decoder and transformation.}
\label{tab:qr_dfr}
\setlength{\tabcolsep}{4pt}\footnotesize
\begin{tabular}{lrrrrccc}
\toprule
& \multicolumn{4}{c}{DFR (\%)} & \multicolumn{3}{c}{50\% crossing} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Transformation & OpenCV & PyZbar & WeChat & \emph{all 3} & OpenCV & PyZbar & WeChat \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\\[4pt]
\begin{minipage}{0.94\linewidth}\footnotesize
Pooled over all four error-correction levels, all three module sizes and all four strengths
(\QrRenders{} renders, each decoded by all three libraries). The right-hand block gives the
strength at which DFR crosses 50\%, or \texttt{--} where it never does. \emph{all three} is
a cascade: the render counts as read if any library reads it, and it is post-hoc rather than
registered.
$^{\dagger}$~saturated: the magnitude is in absolute pixels rather than modules, so these two carry
no usable strength gradient and are excluded from every statement about one.
\end{minipage}
\end{table}
""")

    # ---- table: module size, replacing the pilot's 2px-vs-4px table ---------------------------
    body = ""
    for t in TR_ORDER:
        b = d["by_box"][t]
        body += ("\\texttt{%s} & %.1f & %.1f & %.1f & %s \\\\\n"
                 % (t, b["2"], b["3"], b["4"], _ppm(b["2"] - b["4"])))
    write_generated(os.path.join(SEC, "tab_qr_scale.tex"), r"""\begin{table}[htbp]
\centering\small
\caption{Decode failure rate by module size.}
\label{tab:qr_scale}
\begin{tabular}{lrrrr}
\toprule
Transformation & $2$\,px & $3$\,px & $4$\,px & $2-4$ (pp) \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\\[4pt]
\begin{minipage}{0.82\linewidth}\footnotesize
Pooled over decoders, strengths and error-correction levels; the last column is the $2$\,px penalty
in percentage points. $2$\,px is the scale measured in the wild.
\end{minipage}
\end{table}
""")

    # ---- table: error correction ---------------------------------------------------------------
    body = ""
    for t in TR_ORDER:
        e = d["by_ec"][t]
        body += ("\\texttt{%s} & %.1f & %.1f & %.1f & %.1f & %s \\\\\n"
                 % (t, e["L"], e["M"], e["Q"], e["H"],
                    _ppm(e["L"] - e["H"])))
    write_generated(os.path.join(SEC, "tab_qr_ec.tex"), r"""\begin{table}[htbp]
\centering\small
\caption{What error correction buys, per controlled perturbation.}
\label{tab:qr_ec}
\begin{tabular}{lrrrrr}
\toprule
Transformation & L & M & Q & H & L$-$H (pp) \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\\[4pt]
\begin{minipage}{0.86\linewidth}\footnotesize
DFR pooled over decoders, module sizes and strengths at each error-correction level; the last
column is what moving from L to H buys. Redundancy helps most under occlusion and is not a
universal defence.
\end{minipage}
\end{table}
""")

    # ---- table: the restoration arms ------------------------------------------------------------
    body = ""
    for dec in DEC:
        a = r["by_arm"][dec]
        p = r["paired"][dec]
        body += ("%s & %.1f & %.1f & %.1f & %s & %s & %s \\\\\n"
                 % (DEC_TEX[dec], a["raw"], a["control"], a["restored"],
                    _ppm(a["raw"] - a["control"]),
                    _ppm(p["mean_pp"]), "$%s$" % _p(p["p"])))
    fig_strength(d, os.path.join(os.path.dirname(SEC), "figures", "fig_qr_strength.pdf"))
    v2_assets()
    write_generated(os.path.join(SEC, "tab_qr_restore.tex"), r"""\begin{table}[htbp]
\centering\small
\caption{The restoration arms, on the held-out URLs.}
\label{tab:qr_restore}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{DFR (\%)} & polarity & network & \\
\cmidrule(lr){2-4}
Decoder & raw & control & restored & (pp) & (pp) & $p$ \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}
\\[4pt]
\begin{minipage}{0.94\linewidth}\footnotesize
\QrResUrls{} URLs, \QrResRenders{} renders per arm, all three decoders decoded in one invocation on
one machine. \emph{control} is the raw image with the polarity line alone; \emph{restored} adds the
network on top of that control, so the two right-hand columns separate one line of preprocessing
from the network that follows it. Paired per URL, two-sided Wilcoxon.
\end{minipage}
\end{table}
""")
    print(f"  sweep {d['rows']:,} rows, T1 {d['T1']['verdict']}, T2 {d['T2']['verdict']}; "
          f"restoration T1 {rt1['verdict']} ({rt1['mean_pp']:+.1f}pp), "
          f"T2 {rt2['verdict']} ({rt2['mean_pp']:+.1f}pp)")
    return 0


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        live = json.load(open(SNAP, encoding="utf-8"))
        reason = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--reason=")),
                      "submission")
        json.dump({"frozen_on": dt.date.today().isoformat(), "reason": reason,
                   "snapshot": live}, open(FREEZE, "w", encoding="utf-8"),
                  indent=2, sort_keys=True)
        print(f"[+] pinned {FREEZE} ({reason})")
        raise SystemExit(main())
    if "--thaw" in sys.argv:
        if os.path.exists(FREEZE):
            os.remove(FREEZE)
            print(f"[-] removed {FREEZE}; the paper follows live collection again")
        raise SystemExit(main())
    raise SystemExit(main())
