#!/usr/bin/env python3
"""
make_p7_wildcard.py — how much of this corpus could be names nobody ever registered.

Some registries answer for every name under their TLD, so a feed fed by automated submission can
accumulate "indicators" that never existed. The same screen the infrastructure study registered
(`audit_capture_labels.wildcard_ips`), applied here, finds almost nothing — and that NEGATIVE result is
the point: the artefact is a property of how a feed is fed, not of the TLD landscape. Registry and
platform suffixes are split by whether the suffix is itself an ICANN public suffix. It measures
EXPOSURE, not contamination.

    python scripts/make_p7_wildcard.py           # probe (about 30s of DNS)
    python scripts/make_p7_wildcard.py --cached  # reuse the recorded probe

Writes data/processed/p7/p7_wildcard_probe.csv, papers/P7_cti/figures/fig_wildcard.pdf and
papers/P7_cti/sections/gen_wildcard.tex.
The hazard, the registry/platform split and the method's limits: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

from genfile import write_generated
# The probe is the one the infrastructure study registered, imported rather than reimplemented:
# two papers reporting the same screen must not be running two versions of it.
from audit_capture_labels import _EXTRACT, wildcard_ips
from psl import apex

SEC = os.path.join(ROOT, "papers", "P7_cti", "sections")
FIG = os.path.join(ROOT, "papers", "P7_cti", "figures")
PROC = os.path.join(ROOT, "data", "processed")
URL_CSV = os.path.join(PROC, "dataset_url.csv")
PROBE_CSV = os.path.join(PROC, "p7", "p7_wildcard_probe.csv")
# The companion study's funnel, for the contrast. Read rather than retyped.
P4_FUNNEL = os.path.join(PROC, "infra", "funnel.csv")
# Second-tier probe: the platforms the PSL cannot see (reviewer #1, M9). See platform_probe().
PLATFORM_CSV = os.path.join(PROC, "p7", "p7_platform_probe.csv")
PLATFORM_MIN_HOSTS = 5


def _icann():
    """A second extractor with the PSL private section OFF, so a suffix can be asked what kind of
    thing it is: still a suffix here = a registry; a registrable domain = a platform."""
    import tldextract
    return tldextract.TLDExtract(include_psl_private_domains=False)


def suffix_counts() -> collections.Counter:
    import pandas as pd
    df = pd.read_csv(URL_CSV, low_memory=False)
    ph = df[df["label"] == "phishing"]
    c: collections.Counter = collections.Counter()
    for u in ph["url"].astype(str):
        s = _EXTRACT(u).suffix
        if s:
            c[s] += 1
    return c


def probe(counts: collections.Counter, probe_date: str) -> list[dict]:
    icann = _icann()
    rows = []
    for sfx, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        try:
            ips = sorted(wildcard_ips(sfx))
        except Exception:            # a suffix that will not resolve is not a wildcard
            ips = []
        kind = "platform" if icann(sfx).domain else "registry"
        rows.append({"suffix": sfx, "indicators": n, "answers": int(bool(ips)),
                     "kind": kind if ips else "", "ips": ";".join(ips),
                     "probed_on": probe_date})
    return rows


def load_probe() -> list[dict]:
    return list(csv.DictReader(open(PROBE_CSV, newline="", encoding="utf-8")))


# --------------------------------------------------------------- the screen's blind spot
def platform_candidates() -> list[tuple[str, int]]:
    """Registrable domains carrying many distinct phishing hosts.

    WHY A SECOND TIER EXISTS. The screen above is keyed on the PUBLIC SUFFIX, so it can only see
    a hosting platform that got itself into the PSL's private section. `weebly.com` did not, and
    it carries more phishing hosts than any other name in this corpus -- all of them counted as
    plain `.com`, all of them invisible to a suffix-keyed probe. That is not a bug in the screen;
    it is the screen's boundary, and a paper that reports the screen has to report the boundary.

    Candidates are registrable domains (PSL private section ON, so anything the first tier
    already resolved is not re-counted here) under which at least PLATFORM_MIN_HOSTS distinct
    phishing hosts sit. That threshold selects on FAN-OUT, not on a curated platform list, so it
    catches attacker-owned subdomain fans too -- and separating those from real shared hosting is
    exactly what the probe then does."""
    import pandas as pd
    df = pd.read_csv(URL_CSV, low_memory=False)
    ph = df[df["label"] == "phishing"].copy()
    ph["h"] = (ph["domain"].astype(str).str.strip().str.lower()
               .str.replace(r"^www\.", "", regex=True))
    u = ph.drop_duplicates("h")
    reg = u["h"].map(lambda x: apex(_EXTRACT(x)).lower())
    vc = reg.groupby(reg).size()
    out = [(d, int(n)) for d, n in vc.items()
           if n >= PLATFORM_MIN_HOSTS and d not in ("", "nan")]
    return sorted(out, key=lambda kv: (-kv[1], kv[0]))


def platform_probe(cands: list[tuple[str, int]], probe_date: str) -> list[dict]:
    rows = []
    for dom, n in cands:
        try:
            ips = sorted(wildcard_ips(dom))
        except Exception:
            ips = []
        rows.append({"registrable_domain": dom, "phishing_hosts": n,
                     "answers": int(bool(ips)), "ips": ";".join(ips),
                     "probed_on": probe_date})
    return rows


def load_platform_probe() -> list[dict]:
    if not os.path.exists(PLATFORM_CSV):
        return []
    return list(csv.DictReader(open(PLATFORM_CSV, newline="", encoding="utf-8")))


def make_hosting_tex(rows: list[dict], total_hosts: int, tier1_platform_hosts: int) -> None:
    """One generated paragraph: how much of the corpus sits on shared hosting the suffix-keyed
    screen cannot see, and the explicit statement that the screen bounds rather than filters.

    The probe answers one question and only one: does a name nobody registered resolve under
    this domain? That is enough to tell a delegating operator from N independent registrations.
    It is NOT enough to say who the operator is, so the paragraph does not claim to."""
    if not rows:
        return
    ans = sorted([r for r in rows if str(r["answers"]) == "1"],
                 key=lambda r: -int(r["phishing_hosts"]))
    own = sorted([r for r in rows if str(r["answers"]) != "1"],
                 key=lambda r: -int(r["phishing_hosts"]))
    ans_hosts = sum(int(r["phishing_hosts"]) for r in ans)
    top = ans[0] if ans else None
    names = ", ".join(f"\\texttt{{{r['registrable_domain']}}} ({int(r['phishing_hosts'])})"
                      for r in ans)
    own_names = ", ".join(f"\\texttt{{{r['registrable_domain']}}} ({int(r['phishing_hosts'])})"
                          for r in own[:4])
    body = (
        f"The screen is keyed on the public suffix, which makes it a bound rather than a filter: "
        f"a hosting platform that never entered the PSL's private section is invisible to it, "
        f"and its subdomains are counted as ordinary registrations under whatever TLD it sits "
        f"in. Asking the corpus directly which registrable domains carry many distinct phishing "
        f"hosts finds {len(rows)} with at least {PLATFORM_MIN_HOSTS} of them "
        f"(covering {sum(int(r['phishing_hosts']) for r in rows):,} of the "
        f"{total_hosts:,} phishing hosts), and the same unlikely label partitions their probe responses. "
        f"{len(ans)} answer the wildcard label, so their subdomains are delegated under one "
        f"registration rather than registered one by one ({ans_hosts} hosts under {len(ans)} "
        f"registrable domains, not {ans_hosts} registrations): {names}. Answering says only that "
        f"the domain carries a wildcard record; it does not say the registrant is a hosting "
        f"platform, and an attacker-owned domain with a wildcard record answers the same way"
        + (f". \\texttt{{{top['registrable_domain']}}} alone carries "
           f"{100 * int(top['phishing_hosts']) / total_hosts:.1f}\\% of every phishing host in "
           f"the corpus, against {tier1_platform_hosts} hosts on all the platform suffixes the "
           f"first tier could see, and each of them is counted as a \\texttt{{.com}} "
           f"registration in Table~\\ref{{tab:infra}}" if top else "")
        + f". The other {len(own)} do not answer and are subdomain fans under a single "
        f"registration ({own_names}{', among others' if len(own) > 4 else ''}). Neither group is "
        f"proof of individual host existence or abuse: an arbitrary child of a wildcard can resolve. "
        f"Both illustrate that a TLD count is a "
        f"count of \\emph{{names}}, not of registrations or of operators, and the "
        f"platform-hosted share the suffix screen reports is a floor rather than a total"
    )
    write_generated(os.path.join(SEC, "gen_hosting.tex"), body.rstrip() + "%")


def p4_contrast() -> tuple[int, int] | None:
    """(wildcard candidates removed, candidates screened) from the companion funnel."""
    if not os.path.exists(P4_FUNNEL):
        return None
    rows = list(csv.DictReader(open(P4_FUNNEL, newline="", encoding="utf-8")))
    try:
        i = next(i for i, r in enumerate(rows) if r["stage"].startswith("less registry"))
        wild = int(rows[i]["removed"])
        # The population the screen actually operates on, which is the stage BEFORE it -- not
        # rows[0]. Reading the first row credited the screen with a denominator including the 63
        # hosted subdomains removed a stage earlier and never shown to it, understating the share
        # (1,139/1,525 = 74.7% instead of 1,139/1,462 = 77.9%).
        screened = int(rows[i - 1]["surviving"]) if i > 0 else int(rows[0]["surviving"])
    except (StopIteration, ValueError, KeyError, IndexError):
        return None
    return wild, screened


def phishing_total() -> int:
    """The headline phishing count (36,706). Two rows carry no public suffix at all, so the
    per-suffix probe table sums to two fewer; they cannot answer a wildcard probe either, so the
    denominator the figure and prose quote is the whole phishing arm, not the probe-table sum."""
    import pandas as pd
    df = pd.read_csv(URL_CSV, low_memory=False, usecols=["label"])
    return int((df["label"] == "phishing").sum())


def summarise(rows: list[dict], denominator: int | None = None) -> dict:
    """`indicators` is what these rows carry; `denominator` is what a share is taken against.
    They differ by the two phishing rows with no public suffix, which can answer no probe but
    are still part of the arm the figure and prose quote, so the caller passes the headline
    count. Keeping them separate leaves this function a pure summary of its argument."""
    hit = [r for r in rows if r["answers"] == "1" or r["answers"] == 1]
    by = {k: [r for r in hit if r["kind"] == k] for k in ("registry", "platform")}
    tot = sum(int(r["indicators"]) for r in rows)
    return {
        "suffixes": len(rows), "indicators": tot,
        "denominator": denominator if denominator is not None else tot,
        "answering": len(hit),
        "registry_n": sum(int(r["indicators"]) for r in by["registry"]),
        "platform_n": sum(int(r["indicators"]) for r in by["platform"]),
        "registry_s": len(by["registry"]), "platform_s": len(by["platform"]),
        "hits": hit, "by": by,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    write_generated(path, buf.getvalue())


def make_figure(s: dict, p4: tuple[int, int] | None) -> str:
    from figstyle import apply, BLUE, ORANGE, TEAL, PURPLE, INK
    plt = apply()

    # acmsmall's text block is 5.5 in, so this 7.5 in canvas is scaled to 0.73 and every size
    # below prints at three quarters of its nominal value: 7.5 pt type came out at 5.5 pt. The
    # canvas stays (narrowing it collides the two panels' labels); the type is raised instead.
    # HEIGHT, arithmetically rather than by eye. The canvas is scaled to 0.73 (above), so a 3.2 in
    # canvas printed ~1.7 in of axes for the left panel's 21 suffix rows -- 5.8 pt per row against
    # type printing at 6.9 pt, which overlapped every label with its neighbours (2026-09-02). One
    # row needs the printed type plus a gap, ~9 pt, so 21 rows need ~2.6 in printed and therefore
    # ~3.6 in of canvas axes; with the two-line title and the x label that is a 5.4 in canvas.
    # The right panel holds two bars and must not stretch to match, so its box aspect is pinned.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.5, 5.4),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    # --- LEFT: which suffixes answer for names nobody registered, and how much they cover.
    hits = sorted(s["hits"], key=lambda r: int(r["indicators"]))
    ypos = range(len(hits))
    for y, r in zip(ypos, hits):
        reg = r["kind"] == "registry"
        ax.barh(y, int(r["indicators"]), height=0.6, zorder=3,
                color=ORANGE if reg else BLUE, alpha=0.95 if reg else 0.5)
        ax.annotate(f"{int(r['indicators'])}", (int(r["indicators"]), y),
                    textcoords="offset points", xytext=(4, -3), fontsize=9.5, color=INK)
    ax.set_yticks(list(ypos), [f".{r['suffix']}" for r in hits], fontsize=9.5)
    ax.set_xlim(0, max(int(r["indicators"]) for r in hits) * 1.28)
    ax.set_xlabel("phishing indicators in this corpus")
    ax.set_title(f"{s['answering']} of {s['suffixes']} suffixes answer for names\n"
                 "that were never registered", fontsize=10)
    ax.grid(axis="x", alpha=0.6)
    ax.annotate("registry", (0.97, 0.12), xycoords="axes fraction", ha="right",
                fontsize=9.5, color=ORANGE)
    ax.annotate("hosting platform", (0.97, 0.05), xycoords="axes fraction", ha="right",
                fontsize=9.5, color=BLUE)

    # --- RIGHT: the claim. Same screen, two collection methods, four orders of magnitude apart.
    # TEAL and PURPLE, not the left panel's BLUE and ORANGE. Those two already mean
    # "hosting platform" and "registry" four centimetres away, and a reader who has just learned
    # that pairing meets orange again here standing for a collection method.
    bars = [("curated\n(this corpus)", s["registry_n"], s["denominator"], TEAL)]
    if p4:
        bars.append(("auto-submitted\n(brand-token feed)", p4[0], p4[1], PURPLE))
    for i, (name, k, n, colour) in enumerate(bars):
        share = 100.0 * k / n if n else 0.0
        ax2.bar(i, max(share, 0.004), width=0.5, color=colour, alpha=0.9, zorder=3)
        # decimals follow the prose: gen_wildcard.tex prints the registry share with three
        # (0.025%, not a 0.02% that rounds 9/36,706 away) and the companion share with one
        ax2.annotate(f"{share:.3f}%\n{k:,} of {n:,}" if share < 1 else f"{share:.1f}%\n{k:,} of {n:,}",
                     (i, max(share, 0.004)),
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9.5,
                     color=INK)
    ax2.set_yscale("log")
    ax2.set_ylim(0.002, 400)
    # Tighter than the bars need, so the two tick labels sit further apart in points: at
    # (-0.6, 1.6) and 9 pt "this corpus" and "brand-token feed" ran together into one word.
    ax2.set_xlim(-0.45, len(bars) - 0.55)
    ax2.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    # Shorter than the panel is tall, or it overflows the pinned box and runs into the title
    # (which is what pinning the aspect first did). "candidates" is carried by the bar
    # annotations, which read "9 of 36,706".
    ax2.set_ylabel("never-registered (%, log)")
    ax2.set_title("the artefact is a property of the method", fontsize=10)
    ax2.grid(axis="y", alpha=0.6)

    ax2.set_box_aspect(1.25)          # two bars; without this it becomes a sliver 5 in tall
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_wildcard.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def tex_int(n: int) -> str:
    """Digit grouping that survives math mode. Applied to the number, never to a sentence."""
    return f"{n:,}".replace(",", "{,}")


def make_tex(s: dict, p4: tuple[int, int] | None, probe_date: str) -> None:
    reg_share = 100.0 * s["registry_n"] / s["denominator"]
    plat_share = 100.0 * s["platform_n"] / s["denominator"]
    reg_names = ", ".join(f"\\texttt{{.{r['suffix']}}}" for r in s["by"]["registry"])
    p4_txt = ""
    if p4:
        p4_txt = (
            f" The comparison that gives the number meaning is a separate collection "
            f"that compares candidate addresses with probe answers in a brand-token "
            f"submission and removes ${tex_int(p4[0])}$ of ${tex_int(p4[1])}$ candidates "
            f"(${100.0 * p4[0] / p4[1]:.1f}\\%$) as registry answers. "
            # Computed, not asserted. This read "Four orders of magnitude" as a literal while the
            # ratio was 3,046x -- 3.5 decades. Four orders would need the companion to remove more
            # than 100% of its candidates, so the claim was not merely stale but unreachable.
            f"That is a factor of ${tex_int(round(100.0 * p4[0] / p4[1] / max(reg_share, 1e-9)))}$, "
            f"or ${math.log10((100.0 * p4[0] / p4[1]) / max(reg_share, 1e-9)):.1f}$ orders of magnitude, "
            "between these descriptive percentages. The denominators and selection differ: this study counts "
            "historical indicators under answering suffixes, whereas the companion removes "
            "current candidates whose addresses match a probe. This ratio is not a relative "
            "contamination rate and does not isolate feed-generation practices as its cause."
        )
    body = (
        "Registries that answer for every name under their TLD can manufacture indicators out of "
        "nothing, so before reading anything into this corpus's suffix distribution we asked how "
        f"exposed it is (Figure~\\ref{{fig:wildcard}}). Probing each of the ${s['suffixes']}$ "
        "distinct public suffixes its phishing indicators occupy with a fixed unlikely "
        f"label (probe of {probe_date}), ${s['answering']}$ answer. Almost all of them are "
        f"hosting platforms rather than registries: ${s['platform_s']}$ suffixes covering "
        f"${tex_int(s['platform_n'])}$ indicators (${plat_share:.2f}\\%$), where a resolving "
        "answer does not establish abuse or individual tenant existence. Registry-level probe "
        f"wildcarding covers ${tex_int(s['registry_n'])}$ indicators (${reg_share:.3f}\\%$), on "
        f"{reg_names}. This is low observed suffix exposure, not proof of a contamination-free corpus."
        + p4_txt +
        " The measurement bounds exposure rather than contamination: these indicators are "
        "historical and mostly no longer resolve, so no individual entry can be re-tested now, "
        "and wildcard status is a property of a registry on the day it is probed"
    )
    write_generated(os.path.join(SEC, "gen_wildcard.tex"), body.rstrip() + "%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached", action="store_true",
                    help="reuse the recorded probe instead of querying DNS")
    ap.add_argument("--date", default="", help="probe date to record (ISO); required when "
                                               "probing, since the script cannot read a clock "
                                               "it can also reproduce")
    args = ap.parse_args()

    if args.cached:
        if not os.path.exists(PROBE_CSV):
            print(f"[i] {PROBE_CSV} absent — run without --cached to probe.")
            return 0
        rows = load_probe()
        probe_date = rows[0].get("probed_on", "") if rows else ""
        prows = load_platform_probe()
    else:
        import datetime as dt
        probe_date = args.date or dt.date.today().isoformat()
        rows = probe(suffix_counts(), probe_date)
        os.makedirs(PROC, exist_ok=True)
        write_csv(PROBE_CSV, rows)
        prows = platform_probe(platform_candidates(), probe_date)
        write_csv(PLATFORM_CSV, prows)

    s = summarise(rows, denominator=phishing_total())
    p4 = p4_contrast()
    make_figure(s, p4)
    make_tex(s, p4, probe_date)
    if prows:
        import pandas as pd
        _d = pd.read_csv(URL_CSV, low_memory=False)
        _h = (_d[_d.label == "phishing"]["domain"].astype(str).str.strip().str.lower()
              .str.replace(r"^www\.", "", regex=True)).drop_duplicates()
        _t = int(_h.nunique())
        # tier-1 platform mass in HOSTS, so the two tiers are compared in the same unit
        _sfx = {r["suffix"] for r in s["by"]["platform"]}
        _t1 = int(_h.map(lambda x: _EXTRACT(x).suffix in _sfx).sum())
        make_hosting_tex(prows, _t, _t1)
        print(f"[i] platform tier: {len(prows)} high-fan registrable domains, "
              f"{sum(1 for r in prows if str(r['answers']) == '1')} answer")
    print(f"[i] {s['suffixes']} suffixes, {s['answering']} answer; registry "
          f"{s['registry_n']}/{s['indicators']} = {100 * s['registry_n'] / s['indicators']:.3f}%, "
          f"platform {s['platform_n']} = {100 * s['platform_n'] / s['indicators']:.3f}%"
          + (f"; companion study {p4[0]}/{p4[1]}" if p4 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
