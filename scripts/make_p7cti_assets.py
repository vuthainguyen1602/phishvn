#!/usr/bin/env python3
"""
make_p7cti_assets.py — regenerate P7's (CTI) figures and tables from data/processed/dataset_url.csv,
so no manuscript number is hand-entered.

Writes into papers/P7_cti/: figures/fig_{feed_composition,sector,tld,urlfeat,timeline,bulkdays}.pdf
(the first two only with --all-figures) and sections/{tab_feed_composition,tab_feed_overlap,
tab_brand_dist,tab_infra,gen_dedup,gen_hoststruct,gen_vnframe,gen_sector_ci,gen_bulk,gen_dating}.tex.
Reads dataset_url.csv, first_seen_validation.csv and first_seen_validation_summary.json.

THE HOST KEY IS NOT COSMETIC: the phishing arm carries `foo.tld` and `www.foo.tld` as two rows for
16,799 hosts while the benign arm carries almost none, so `host_key()` is the unit for every claim
a duplicate would move. Row-level values are printed alongside — the difference is itself a result.
Deterministic; the one RNG is the seeded cluster bootstrap BOOT_SEED.

RUN:  python scripts/make_p7cti_assets.py
The host-key finding and the palette rules: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType so PDFs are portable
    "figure.dpi": 150,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated

SEC = os.path.join(ROOT, "papers", "P7_cti", "sections")
FIG = os.path.join(ROOT, "papers", "P7_cti", "figures")
PROC = os.path.join(ROOT, "data", "processed")
DATASET = os.path.join(PROC, "dataset_url.csv")
FS_VALID = os.path.join(PROC, "first_seen_validation.csv")
FS_SUMMARY = os.path.join(PROC, "first_seen_validation_summary.json")
P7_PROC = os.path.join(PROC, "p7")

# Seeded so the bootstrap intervals are a property of the data, not of the run. Changing this
# changes every printed CI by a little; it is fixed deliberately.
BOOT_SEED = 20260819
BOOT_B = 5000
# A "bulk-ingest day" is a calendar day carrying at least this multiple of the recovering
# feed's median daily volume -- a stated rule, so the flagged set is recomputed, not remembered.
BULK_MULTIPLE = 20

# Okabe--Ito colorblind-safe palette (validated: normal-vision worst adjacent dE 20.0).
C_PHISH = "#D55E00"   # vermillion
C_BENIGN = "#0072B2"  # blue
C_ACCENT = "#009E73"  # green
C_GREY = "#7f7f7f"
INK = "#2b2b2b"     # all label text; a mark never names itself in its own colour

# feeds carrying each class (single-class by construction: see paper's methodology caveat)
PHISH_FEEDS = ["chongluadao", "openphish", "tinnhiemmang"]
BENIGN_FEEDS = ["tranco", "tinnhiem_web", "tranco_vn", "tinnhiem_org"]
# impersonated-sector taxonomy, ordered for display
SECTORS = ["bank", "ecommerce", "gaming", "social", "delivery", "gov", "telecom", "tax"]
SECTOR_LABEL = {"bank": "Banking", "ecommerce": "E-commerce", "gaming": "Gaming",
                "social": "Social", "delivery": "Delivery", "gov": "Government",
                "telecom": "Telecom", "tax": "Tax"}

# accumulate headline numbers so main() can print a spot-check summary
STATS: dict[str, object] = {}


def host_key(s: pd.Series) -> pd.Series:
    """The canonical host: lower-cased and stripped of ONE leading `www.` label.

    `www.foo.tld` and `foo.tld` are the same site by every question this paper asks. The corpus
    deduplicates exact URLs, which does not collapse the pair, and the phishing arm carries the pair
    for 16,799 hosts against 58 in the benign arm -- so leaving it uncollapsed is not a wash, it is
    a one-sided inflation of the phishing arm."""
    return s.astype(str).str.strip().str.lower().str.replace(r"^www\.", "", regex=True)


def load() -> pd.DataFrame:
    df = pd.read_csv(DATASET, low_memory=False)
    df["host"] = host_key(df["domain"])
    df["is_www"] = df["domain"].astype(str).str.strip().str.lower().str.startswith("www.")
    STATS["n_total"] = len(df)
    STATS["n_phish"] = int((df.label == "phishing").sum())
    STATS["n_benign"] = int((df.label == "benign").sum())
    STATS["n_host_total"] = int(df.host.nunique())
    STATS["n_host_phish"] = int(df.loc[df.label == "phishing", "host"].nunique())
    STATS["n_host_benign"] = int(df.loc[df.label == "benign", "host"].nunique())
    return df


def hosts(df: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    """One row per distinct host: the FIRST row for that host in corpus order.

    The corpus is stored in a fixed order, so this is deterministic, and exactly one phishing
    host carries two different `scenario` values across its rows -- the tie-break therefore moves
    one indicator, and check_paper_claims mirrors this rule rather than guessing it."""
    d = df if label is None else df[df.label == label]
    return d.drop_duplicates("host")


def boot_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_bootstrap(counts: np.ndarray, seed: int = BOOT_SEED,
                      B: int = BOOT_B) -> np.ndarray:
    """Resample CLUSTERS (hosts) with replacement and return the per-replicate column sums.

    `counts` is (K clusters x C columns) of per-cluster totals; column 0 must be the cluster's row
    count, so a share is column_j / column_0 within each replicate. The duplicate rows a naive
    bootstrap would treat as independent draws are exactly the `www.` twins, so an unclustered
    interval on the row-level share is too narrow by construction. Chunked to bound memory."""
    rng = np.random.default_rng(seed)
    K = counts.shape[0]
    out = np.empty((B, counts.shape[1]))
    step = max(1, int(5e6 // max(K, 1)))
    for lo in range(0, B, step):
        hi = min(B, lo + step)
        idx = rng.integers(0, K, size=(hi - lo, K))
        out[lo:hi] = counts[idx].sum(axis=1)
    return out


def parse_first_seen(s: pd.Series) -> pd.Series:
    """Robustly parse the mixed-format `collected_at` first-seen field: DD/MM/YYYY and ISO both
    appear. Rows that parse under neither stay NaT (uncovered)."""
    s = s.astype(str).str.strip().str.slice(0, 10)
    dt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    dt = dt.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))
    return dt


# Inline generated fragments end with a comment marker: `\input{f}.` would otherwise render the
# file's trailing newline as a space and print " ." before the sentence's full stop.
def _esc(x: str) -> str:
    return str(x).replace("_", r"\_").replace("&", r"\&")


# ------------------------------------------------------------------ figures
def fig_feed_composition(df: pd.DataFrame):
    """Horizontal bars per source, coloured by the class it contributes. Sources are single-class by
    construction (a phishing feed contributes only phishing rows), which the paper states plainly;
    the figure therefore reads as the volume each feed brings and which side of the corpus it fills."""
    g = df.groupby("source").agg(n=("label", "size"),
                                 phish=("label", lambda s: (s == "phishing").sum())).reset_index()
    g["cls"] = np.where(g["phish"] > 0, "phishing", "benign")
    g = g.sort_values("n", ascending=True)
    colors = [C_PHISH if c == "phishing" else C_BENIGN for c in g["cls"]]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(g))
    ax.barh(y, g["n"], color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(g["source"])
    ax.set_xscale("log")
    ax.set_xlabel("indicators (log scale)")
    ax.set_xlim(right=g["n"].max() * 2.2)
    for yi, v in zip(y, g["n"]):
        ax.text(v * 1.12, yi, f"{v:,}", va="center", ha="left", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_PHISH),
               plt.Rectangle((0, 0), 1, 1, color=C_BENIGN)]
    ax.legend(handles, ["phishing feed", "benign reference"], frameon=False,
              loc="lower right", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_feed_composition.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def fig_sector(df: pd.DataFrame):
    """Impersonated-sector breakdown of the phishing indicators (excluding the untagged `other`
    majority, whose size is annotated). Log-scaled horizontal bars."""
    ph = df[df.label == "phishing"]
    counts = ph.scenario.value_counts()
    n_other = int(counts.get("other", 0))
    sect = [(s, int(counts.get(s, 0))) for s in SECTORS if counts.get(s, 0) > 0]
    sect.sort(key=lambda kv: kv[1])
    labels = [SECTOR_LABEL[s] for s, _ in sect]
    vals = [v for _, v in sect]
    STATS["sector_counts"] = dict(sect)
    STATS["n_other_phish"] = n_other
    STATS["n_tagged_phish"] = int(sum(vals))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=C_PHISH, edgecolor="white", height=0.72)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("phishing indicators (log scale)")
    ax.set_xlim(right=max(vals) * 2.0)
    for yi, v in zip(y, vals):
        ax.text(v * 1.1, yi, f"{v:,}", va="center", ha="left", fontsize=8.5)
    ax.text(0.98, 0.06,
            f"untagged (\"other\"): {n_other:,} of {len(ph):,} phishing",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C_GREY)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_sector.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def fig_tld(df: pd.DataFrame):
    """Two panels: (a) share of .vn vs non-.vn by class; (b) top-N TLDs, phishing vs benign counts."""
    ph = df[df.label == "phishing"]
    be = df[df.label == "benign"]
    vn_ph = 100 * (ph.tld == "vn").mean()
    vn_be = 100 * (be.tld == "vn").mean()
    STATS["vn_share_phish"] = round(float(vn_ph), 1)
    STATS["vn_share_benign"] = round(float(vn_be), 1)
    STATS["n_tld"] = int(df.tld.nunique())

    topN = 10
    top = df.tld.value_counts().head(topN).index.tolist()
    ph_c = ph.tld.value_counts()
    be_c = be.tld.value_counts()
    STATS["top_tlds"] = top

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4),
                                   gridspec_kw={"width_ratios": [1, 2.1]})
    # Panel a: one 100%-stacked bar per class (the grouped version drew four bars for two
    # numbers); stacking puts the contrast (2.6 vs 61.6) on a shared 0-100 axis.
    x = np.arange(2)
    vn_vals, non_vals = [vn_ph, vn_be], [100 - vn_ph, 100 - vn_be]
    ax1.bar(x, vn_vals, width=0.55, color=[C_PHISH, C_BENIGN], edgecolor="white")
    ax1.bar(x, non_vals, width=0.55, bottom=vn_vals, edgecolor="white",
            color=["#f2c9a8", "#a9cfe6"])
    ax1.set_xticks(x); ax1.set_xticklabels(["phishing", "benign"])
    ax1.set_ylabel("% of class")
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    for xi, val in zip(x, vn_vals):
        # The .vn figure goes inside its own segment when that segment is tall enough to hold it.
        # When it is not, the label used to sit just above the segment -- which put it in the
        # MIDDLE of the non-.vn segment with nothing tying the two together, so a 2.6% band was
        # labelled by text floating inside the 97.4% one (2026-09-02). Outside now means outside
        # with a leader line down into the band it belongs to.
        if val > 12:
            ax1.text(xi, val - 3, f"{val:.1f}% .vn", ha="center", va="top",
                     fontsize=7.5, color="white")
        else:
            ax1.annotate(f"{val:.1f}% .vn", (xi, max(val / 2, 0.6)),
                         textcoords="offset points", xytext=(0, 20), ha="center", va="bottom",
                         fontsize=7.5, color="#333333",
                         arrowprops=dict(arrowstyle="-", lw=0.6, color="#666666",
                                         shrinkA=1, shrinkB=0))
    for xi, val in zip(x, vn_vals):
        ax1.text(xi, val + (100 - val) / 2, "non-.vn", ha="center", va="center",
                 fontsize=7.5, color="#333333")
    ax1.set_title("(a) .vn share by class", fontsize=9.5)

    # panel b: top-N TLDs grouped
    y = np.arange(len(top))[::-1]
    pv = [int(ph_c.get(t, 0)) for t in top]
    bv = [int(be_c.get(t, 0)) for t in top]
    ax2.barh(y + 0.19, pv, height=0.36, color=C_PHISH, edgecolor="white", label="phishing")
    ax2.barh(y - 0.19, bv, height=0.36, color=C_BENIGN, edgecolor="white", label="benign")
    ax2.set_yticks(y); ax2.set_yticklabels(["." + t for t in top])
    ax2.set_xscale("log"); ax2.set_xlabel("indicators (log scale)")
    ax2.set_xlim(right=max(max(pv), max(bv)) * 2.4)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax2.set_title("(b) top TLDs, phishing vs benign", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_tld.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def fig_urlfeat(df: pd.DataFrame):
    """Box plots of two lexical features by class, AT HOST GRANULARITY.

    Drawn per row, this figure produced the paper's combosquatting inference: 46.3% of phishing rows
    are `www.` twins carrying num_subdomains = 1, which lifted the phishing median above benign and
    was read as attackers prepending a brand label. Collapse the twins and the ordering reverses.
    Row-level values stay in STATS so both can be printed rather than one quietly replacing the
    other. `url_len` is the URL STRING's length -- the host for 95.6% of the corpus, with a path for
    the rest -- and the axis says so."""
    feats = [("url_len", "URL-string length (chars)"),
             ("num_subdomains", "number of subdomains")]
    ph_r, be_r = df[df.label == "phishing"], df[df.label == "benign"]
    ph, be = hosts(df, "phishing"), hosts(df, "benign")
    STATS["url_is_host_pct"] = round(
        float(100 * (df.url.astype(str).str.lower() == df.domain.astype(str).str.lower()).mean()), 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, (col, lab) in zip(axes, feats):
        data = [ph[col].dropna().values, be[col].dropna().values]
        bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], [C_PHISH, C_BENIGN]):
            patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("white")
        ax.set_xticks([1, 2]); ax.set_xticklabels(["phishing", "benign"])
        ax.set_ylabel(lab)
        for tag, dfr in (("row", (ph_r, be_r)), ("host", (ph, be))):
            for cls, sub in zip(("phish", "benign"), dfr):
                v = sub[col].dropna()
                STATS[f"{col}_{tag}_med_{cls}"] = float(v.median())
                STATS[f"{col}_{tag}_mean_{cls}"] = round(float(v.mean()), 3)
                STATS[f"{col}_{tag}_iqr_{cls}"] = [float(v.quantile(.25)), float(v.quantile(.75))]
    axes[0].set_title("(a) URL-string length", fontsize=9.5)
    axes[1].set_title("(b) subdomain depth", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_urlfeat.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def gen_hoststruct(df: pd.DataFrame):
    """The corrected host-structure sentence (M5/C11): what the two medians say once the `www.`
    twins are collapsed, and what they said before."""
    def g(col, tag, cls, stat="med"):
        return STATS[f"{col}_{tag}_{stat}_{cls}"]
    sub_ph_r, sub_be_r = g("num_subdomains", "row", "phish"), g("num_subdomains", "row", "benign")
    sub_ph = g("num_subdomains", "host", "phish")
    m_ph = g("num_subdomains", "host", "phish", "mean")
    m_be = g("num_subdomains", "host", "benign", "mean")
    mr_ph = g("num_subdomains", "row", "phish", "mean")
    len_ph, len_be = g("url_len", "host", "phish"), g("url_len", "host", "benign")
    iq_ph = STATS["url_len_host_iqr_phish"]; iq_be = STATS["url_len_host_iqr_benign"]
    tex = (
        f"median URL-string length is \\textbf{{{len_ph:.0f}}} characters for phishing against "
        f"\\textbf{{{len_be:.0f}}} for benign (the benign arm is the more dispersed of the two, "
        f"IQR {iq_be[0]:.0f}--{iq_be[1]:.0f} against {iq_ph[0]:.0f}--{iq_ph[1]:.0f}). Subdomain "
        f"depth is a median of \\textbf{{{sub_ph:.0f}}} on both sides, and what separates them "
        f"runs the \\emph{{opposite}} way to the intuition: mean depth is {m_ph:.2f} for phishing "
        f"against {m_be:.2f} for benign, so it is the legitimate Vietnamese sites that carry the "
        f"deeper subdomain stacks. Per row the ordering inverts (median {sub_ph_r:.0f} "
        f"phishing against "
        f"{sub_be_r:.0f} benign, mean {mr_ph:.2f}), but that inversion is entirely the "
        f"\\texttt{{www.}} twins of Section~\\ref{{sec:method}}, each of which contributes one "
        f"subdomain that the bare host does not")
    write_generated(os.path.join(SEC, "gen_hoststruct.tex"), tex.rstrip() + "%")


def gen_dedup(df: pd.DataFrame):
    """The `www.` duplication (M1): how large it is, how one-sided it is, and the three
    conclusions it moves --- feed share, dated coverage, and subdomain depth."""
    ph, be = df[df.label == "phishing"], df[df.label == "benign"]
    raw_ph = int(ph.domain.astype(str).str.lower().nunique())
    n_ph_host = int(ph.host.nunique())
    pairs_ph = len(set(ph.loc[~ph.is_www, "host"]) & set(ph.loc[ph.is_www, "host"]))
    pairs_be = len(set(be.loc[~be.is_www, "host"]) & set(be.loc[be.is_www, "host"]))
    www_ph, www_be = int(ph.is_www.sum()), int(be.is_www.sum())
    STATS["www_rows_phish"], STATS["www_rows_benign"] = www_ph, www_be
    STATS["www_pairs_phish"], STATS["www_pairs_benign"] = pairs_ph, pairs_be
    STATS["raw_hosts_phish"] = raw_ph
    shrink = 100 * (1 - n_ph_host / len(ph))
    shrink_be = 100 * (1 - int(be.host.nunique()) / len(be))
    tex = (
        f"The phishing arm's \\textbf{{{len(ph):,}}} rows occupy {raw_ph:,} distinct host strings "
        f"but only \\textbf{{{n_ph_host:,}}} distinct hosts, because \\textbf{{{pairs_ph:,}}} of "
        f"them are listed twice: once as \\texttt{{foo.tld}} and once as "
        f"\\texttt{{www.foo.tld}}. Exact-URL deduplication does not collapse that pair, and "
        f"the two arms are not symmetric in it: {www_ph:,} phishing rows carry a leading "
        f"\\texttt{{www.}} label ({100 * www_ph / len(ph):.1f}\\% of the arm) against {www_be:,} "
        f"benign rows ({100 * www_be / len(be):.1f}\\%), and the duplication is concentrated in "
        f"the dominant community feed. Per-row phishing statistics are therefore inflated by an "
        f"asymmetric, feed-specific factor: collapsing the pair removes {shrink:.1f}\\% of the "
        f"phishing rows and {shrink_be:.1f}\\% of the benign ones")
    write_generated(os.path.join(SEC, "gen_dedup.tex"), tex.rstrip() + "%")


def tab_feed_overlap(df: pd.DataFrame):
    """The phishing feeds at host granularity: what each contributes, what it shares, and what
    only it has (M1 + M6). The 'marginal value of fusion' the introduction claims is exactly the
    unique-host column, which no earlier version of this paper measured."""
    ph = df[df.label == "phishing"]
    sets = {s: set(g.host) for s, g in ph.groupby("source")}
    union = set().union(*sets.values())
    order = sorted(sets, key=lambda s: -len(sets[s]))
    rows = []
    for s in order:
        others = set().union(*[sets[o] for o in sets if o != s]) if len(sets) > 1 else set()
        uniq = sets[s] - others
        nrows = int((ph.source == s).sum())
        rows.append((s, nrows, 100 * nrows / len(ph), len(sets[s]),
                     100 * len(sets[s]) / len(union), len(uniq), 100 * len(uniq) / len(sets[s])))
    STATS["feed_rows_vs_hosts"] = {r[0]: (r[2], r[4]) for r in rows}
    STATS["feed_unique_hosts"] = {r[0]: (r[5], r[6]) for r in rows}
    pairs = [(a, b, len(sets[a] & sets[b]))
             for i, a in enumerate(order) for b in order[i + 1:]]
    STATS["feed_pair_overlap"] = {f"{a}|{b}": n for a, b, n in pairs}
    body = "\n".join(
        f"\\texttt{{{_esc(s)}}} & {nr:,} & {pr:.2f}\\% & {nh:,} & {ph_:.2f}\\% & "
        f"{nu:,} & {pu:.1f}\\% \\\\" for (s, nr, pr, nh, ph_, nu, pu) in rows)
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{The three phishing feeds by rows, distinct hosts and hosts unique to each feed.
Host shares are of the {len(union):,}-host union and exceed $100\\%$ by the overlap.}}
\\label{{tab:feed_overlap}}
\\small
\\begin{{tabular}}{{l r r r r r r}}
\\toprule
& \\multicolumn{{2}}{{c}}{{rows}} & \\multicolumn{{2}}{{c}}{{distinct hosts}}
& \\multicolumn{{2}}{{c}}{{hosts only this feed has}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
Phishing feed & $n$ & \\% of rows & $n$ & \\% of union & $n$ & \\% of feed \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_feed_overlap.tex"), tex)


def common_window_sensitivity(df: pd.DataFrame):
    """Calendar-align the two phishing archives that actually carry per-indicator dates.

    This is not silently extended to OpenPhish: its 296 rows have no first-seen value.  Assigning
    their July-2026 scrape time as an event time would turn one snapshot into a longitudinal feed.
    The common window is therefore the intersection of the dated spans of ChongLuaDao and
    TinNhiemMang, and the manuscript calls it date-aligned archive sensitivity rather than a live
    common-window recall study.
    """
    ph = df[df.label == "phishing"].copy()
    ph["date"] = parse_first_seen(ph["collected_at"])
    sources = ("chongluadao", "tinnhiemmang")
    dated = ph[ph.source.isin(sources) & ph.date.notna()]
    spans = dated.groupby("source").date.agg(["min", "max"])
    start, end = spans["min"].max(), spans["max"].min()
    if start > end:
        raise ValueError("dated phishing feeds have no common calendar window")

    def row(condition: str, d: pd.DataFrame, lo, hi):
        sets = {s: set(d[d.source == s].host) for s in sources}
        inter = sets[sources[0]] & sets[sources[1]]
        union = sets[sources[0]] | sets[sources[1]]
        smaller = min(len(sets[sources[0]]), len(sets[sources[1]]))
        return {
            "condition": condition, "start": lo, "end": hi,
            "chongluadao_hosts": len(sets["chongluadao"]),
            "tinnhiemmang_hosts": len(sets["tinnhiemmang"]),
            "overlap_hosts": len(inter), "union_hosts": len(union),
            "jaccard": len(inter) / len(union) if union else np.nan,
            "share_of_smaller": len(inter) / smaller if smaller else np.nan,
            "excluded_source": "openphish",
            "excluded_reason": "no per-indicator first-seen dates (0/296 dated)",
        }

    all_two = ph[ph.source.isin(sources)]
    common = dated[dated.date.between(start, end)]
    rows = pd.DataFrame([
        row("all_retrievable", all_two, "source-specific", "source-specific"),
        row("common_dated_window", common, start.date(), end.date()),
    ])
    os.makedirs(P7_PROC, exist_ok=True)
    rows.to_csv(os.path.join(P7_PROC, "p7_common_window.csv"), index=False,
                float_format="%.6f")
    a, c = rows.iloc[0], rows.iloc[1]
    body = "\n".join([
        (f"All retrievable rows & {int(a.chongluadao_hosts):,} & "
         f"{int(a.tinnhiemmang_hosts):,} & {int(a.overlap_hosts):,} & "
         f"{100*a.jaccard:.2f}\\% & {100*a.share_of_smaller:.2f}\\% " + r"\\"),
        (f"Common dated window & {int(c.chongluadao_hosts):,} & "
         f"{int(c.tinnhiemmang_hosts):,} & {int(c.overlap_hosts):,} & "
         f"{100*c.jaccard:.2f}\\% & {100*c.share_of_smaller:.2f}\\% " + r"\\"),
    ])
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Feed-overlap sensitivity on the two archives with per-indicator dates. The common
window is {start.date()}--{end.date()}; OpenPhish is ineligible because none of its 296 snapshot
rows has a first-seen date. `smaller' is TinNhiemMang in both rows.}}
\\label{{tab:common_window}}
\\small
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Condition & CLD hosts & TNM hosts & shared & Jaccard & \\% smaller \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_common_window.tex"), tex)
    gen = (
        f"Restricting both dated archives to their common {start.date()}--{end.date()} calendar "
        f"support leaves {int(c.chongluadao_hosts):,} ChongLuaDao hosts and "
        f"{int(c.tinnhiemmang_hosts):,} TinNhiemMang hosts, of which only "
        f"{int(c.overlap_hosts):,} are shared (Jaccard {100*c.jaccard:.2f}\\%; "
        f"{100*c.share_of_smaller:.2f}\\% of the smaller archive), against "
        f"{int(a.overlap_hosts):,} shared hosts in the unaligned archive totals. The low-overlap "
        f"direction therefore survives calendar alignment, but OpenPhish remains outside the "
        f"test and recovered ingestion dates do not make this a contemporaneous recall study"
    )
    write_generated(os.path.join(SEC, "gen_common_window.tex"), gen.rstrip() + "%")
    STATS["common_window"] = rows.to_dict("records")


def tier_sensitivity(df: pd.DataFrame):
    """Recompute the key infrastructure and tagging quantities per evidence tier, at host level.

    Tier and source are nearly aliased in this corpus.  The table is intentionally descriptive:
    it shows which directions survive restricting evidence, and which conclusions move with the
    source/tier mixture; it cannot estimate an independent causal effect of corroboration.
    """
    ph = df[df.label == "phishing"]
    per_tier = ph.drop_duplicates(["tier", "host"])
    groups = [
        ("bronze", per_tier[per_tier.tier == "bronze"]),
        ("silver", per_tier[per_tier.tier == "silver"]),
        ("gold", per_tier[per_tier.tier == "gold"]),
        ("verified", ph[ph.tier.isin(["silver", "gold"])].drop_duplicates("host")),
    ]
    rows = []
    for name, g in groups:
        rows.append({
            "tier": name, "n_hosts": len(g),
            "vn_pct": 100 * g.tld.astype(str).eq("vn").mean(),
            "suspicious_tld_pct": 100 * pd.to_numeric(g.suspicious_tld).mean(),
            "sector_tagged_pct": 100 * g.scenario.ne("other").mean(),
            "bank_pct": 100 * g.scenario.eq("bank").mean(),
            "ecommerce_pct": 100 * g.scenario.eq("ecommerce").mean(),
            "sources": ";".join(sorted(g.source.astype(str).unique())),
        })
    out = pd.DataFrame(rows)
    os.makedirs(P7_PROC, exist_ok=True)
    out.to_csv(os.path.join(P7_PROC, "p7_tier_sensitivity.csv"), index=False,
               float_format="%.6f")
    labels = {"bronze": "Bronze", "silver": "Silver", "gold": "Gold",
              "verified": "Silver+gold"}
    body = "\n".join(
        f"{labels[r.tier]} & {int(r.n_hosts):,} & {r.vn_pct:.1f}\\% & "
        f"{r.suspicious_tld_pct:.1f}\\% & {r.sector_tagged_pct:.1f}\\% & "
        f"{r.bank_pct:.1f}\\% & {r.ecommerce_pct:.1f}\\% \\\\"
        for r in out.itertuples())
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Tier sensitivity at distinct-host granularity. Silver+gold is deduplicated across
the two tiers. Shares use all phishing hosts in that row as denominator.}}
\\label{{tab:tier_sensitivity}}
\\small
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Evidence slice & Hosts & \\texttt{{.vn}} & watchlist TLD & tagged & bank & e-commerce \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_tier_sensitivity.tex"), tex)
    b = out.set_index("tier").loc["bronze"]
    v = out.set_index("tier").loc["verified"]
    gen = (
        f"the namespace and abused-TLD directions survive the evidence restriction: "
        f"\\texttt{{.vn}} remains below 5\\% in every tier ({out.vn_pct.min():.1f}--"
        f"{out.vn_pct.max():.1f}\\%), and watchlist TLDs range from "
        f"{out.suspicious_tld_pct.min():.1f}\\% to {out.suspicious_tld_pct.max():.1f}\\%, "
        f"still far above the benign-host reference. The sector reading does not: only "
        f"{b.sector_tagged_pct:.1f}\\% of bronze hosts are tagged against "
        f"{v.sector_tagged_pct:.1f}\\% of silver+gold, and e-commerce exceeds banking in bronze "
        f"({b.ecommerce_pct:.1f}\\% vs. {b.bank_pct:.1f}\\%) while banking dominates the "
        f"verified slice ({v.bank_pct:.1f}\\% vs. {v.ecommerce_pct:.1f}\\%)"
    )
    write_generated(os.path.join(SEC, "gen_tier_sensitivity.tex"), gen.rstrip() + "%")
    STATS["tier_sensitivity"] = out.to_dict("records")


def gen_vnframe(df: pd.DataFrame):
    """The benign .vn rate against the frame that sets it (M4). 61.6% is not a population
    statistic about legitimate Vietnamese sites; it is a weighted average of four lists whose
    own rates run from 0.3% to 100%, so it moves with the row counts drawn from each."""
    be = df[df.label == "benign"]
    per = (be.groupby("source")
             .apply(lambda g: pd.Series({"n": len(g), "vn": 100 * (g.tld == "vn").mean()}),
                    include_groups=False)
             .sort_values("n", ascending=False))
    beh, phh = hosts(df, "benign"), hosts(df, "phishing")
    STATS["vn_share_benign_host"] = round(float(100 * (beh.tld == "vn").mean()), 1)
    STATS["vn_share_phish_host"] = round(float(100 * (phh.tld == "vn").mean()), 1)
    STATS["vn_per_source"] = {s: round(float(r.vn), 1) for s, r in per.iterrows()}
    STATS["susp_tld_phish_host"] = round(float(100 * phh.suspicious_tld.mean()), 1)
    STATS["susp_tld_benign_host"] = round(float(100 * beh.suspicious_tld.mean()), 1)
    lst = ", ".join(f"\\texttt{{{_esc(s)}}} {r.vn:.1f}\\% ($n={int(r.n):,}$)"
                    for s, r in per.iterrows())
    tex = (
        f"The \\textbf{{{STATS['vn_share_benign']:.1f}\\%}} figure is a property of how the "
        f"benign reference was assembled, not a measurement of the Vietnamese web: its four "
        f"constituent lists carry \\texttt{{.vn}} at wholly different rates ({lst}), so the "
        f"pooled number is fixed by the row counts drawn from each. It should be read as "
        f"\\emph{{the \\texttt{{.vn}} rate of this popularity- and trust-anchored reference}}, "
        f"and the contrast it supports is with the phishing arm measured on the same key: "
        f"\\textbf{{{STATS['vn_share_phish_host']:.1f}\\%}} of distinct phishing hosts against "
        f"\\textbf{{{STATS['vn_share_benign_host']:.1f}\\%}} of distinct benign hosts")
    write_generated(os.path.join(SEC, "gen_vnframe.tex"), tex.rstrip() + "%")


def gen_sector_ci(df: pd.DataFrame):
    """Cluster-bootstrap intervals on the sector shares and on the one ordinal claim the paper
    leans on (M7). Clusters are hosts, so the `www.` twins cannot masquerade as independent
    evidence. Row-level and host-level are both reported: they disagree, and which one disagrees
    is the finding."""
    ph = df[df.label == "phishing"]
    g = ph.groupby("host").scenario
    cnt = pd.DataFrame({
        "n": g.size(),
        "bank": g.apply(lambda s: int((s == "bank").sum())),
        "ecom": g.apply(lambda s: int((s == "ecommerce").sum())),
    })
    draws = cluster_bootstrap(cnt[["n", "bank", "ecom"]].to_numpy(float))
    d_row = 100 * (draws[:, 1] - draws[:, 2]) / draws[:, 0]
    lo_r, hi_r = boot_ci(d_row)
    obs_r = 100 * ((ph.scenario == "bank").mean() - (ph.scenario == "ecommerce").mean())

    u = hosts(df, "phishing")
    one = pd.DataFrame({"n": 1.0,
                        "bank": (u.scenario == "bank").astype(float),
                        "ecom": (u.scenario == "ecommerce").astype(float)})
    dh = cluster_bootstrap(one.to_numpy(float))
    d_host = 100 * (dh[:, 1] - dh[:, 2]) / dh[:, 0]
    lo_h, hi_h = boot_ci(d_host)
    obs_h = 100 * ((u.scenario == "bank").mean() - (u.scenario == "ecommerce").mean())
    STATS["bank_minus_ecom_row"] = (round(obs_r, 3), round(lo_r, 3), round(hi_r, 3))
    STATS["bank_minus_ecom_host"] = (round(obs_h, 3), round(lo_h, 3), round(hi_h, 3))

    # per-sector host-level share intervals, reused by tab_brand_dist
    ci = {}
    for s in SECTORS:
        col = pd.DataFrame({"n": 1.0, "s": (u.scenario == s).astype(float)}).to_numpy(float)
        dr = cluster_bootstrap(col)
        v = 100 * dr[:, 1] / dr[:, 0]
        ci[s] = (100 * float((u.scenario == s).mean()), *boot_ci(v))
    STATS["sector_host_ci"] = {k: tuple(round(x, 2) for x in v) for k, v in ci.items()}

    tex = (
        f"Per row the banking--e-commerce gap is \\textbf{{{obs_r:+.2f}}}~pp with a "
        f"host-clustered bootstrap interval of $[{lo_r:+.2f}, {hi_r:+.2f}]$, which contains "
        f"zero: at row granularity the corpus does not establish that banking outranks "
        f"e-commerce. Per distinct host it does (\\textbf{{{obs_h:+.2f}}}~pp, "
        f"$[{lo_h:+.2f}, {hi_h:+.2f}]$), and the host key is the granularity at which the "
        f"two are separable, because the duplicated rows of Section~\\ref{{sec:method}} are not "
        f"independent evidence about a sector")
    write_generated(os.path.join(SEC, "gen_sector_ci.tex"), tex.rstrip() + "%")


# Public-service brands whose names a Vietnamese reader recognises on sight. Fixed here rather
# than mined from the corpus, so the search cannot be tuned until it finds something.
GOV_TOKENS = {"dichvucong": "the national public-service portal",
              "vssid": "social insurance",
              "gplx": "driving licences"}


def gen_govtokens(df: pd.DataFrame):
    """Does the corpus contain public-service impersonation? (M8/C14.)

    The Ethics section asserted it does not. It does. The counts are small, which is the honest
    finding, and the tags on them are the second one: a host named for a government service that the
    tagger files under `other` is a recall failure, not an absence."""
    from watch_urlscan_brands import token_at_boundary
    u = hosts(df, "phishing")
    out = {}
    for tok in GOV_TOKENS:
        m = u[u.host.map(lambda h, t=tok: token_at_boundary(h, t))]
        out[tok] = (len(m), int((m.scenario == "gov").sum()),
                    sorted(m.host.tolist())[:3])
    STATS["gov_token_hosts"] = {k: (v[0], v[1]) for k, v in out.items()}
    n_tot = sum(v[0] for v in out.values())
    n_gov = sum(v[1] for v in out.values())
    lst = "; ".join(f"\\texttt{{{t}}} ({GOV_TOKENS[t]}) matches {out[t][0]} host"
                    f"{'s' if out[t][0] != 1 else ''}, {out[t][1]} of them tagged \\texttt{{gov}}"
                    for t in GOV_TOKENS)
    ex = ", ".join(f"\\texttt{{{h}}}" for h in
                   [out[t][2][0] for t in GOV_TOKENS if out[t][2]])
    tex = (
        f"{lst}: {n_tot} hosts in total, {n_gov} of them reaching the "
        f"\\texttt{{gov}} bucket; {ex} are among the names. So the shortfall against the survey is "
        f"two failures compounded, not one: the feeds capture little of this category, "
        f"\\emph{{and}} the sector tagger files part of what they do capture as "
        f"\\texttt{{other}}. Only the first is a statement about the Vietnamese threat "
        f"landscape; the second is a statement about our own annotation")
    write_generated(os.path.join(SEC, "gen_govtokens.tex"), tex.rstrip() + "%")


def _bulk_days(df: pd.DataFrame) -> tuple[pd.Series, set, float]:
    """Daily first-seen volume of the recovering feed, and the days carrying at least
    BULK_MULTIPLE times its median daily volume."""
    d = df[(df.label == "phishing") & (df.source == "chongluadao")].copy()
    d["t"] = parse_first_seen(d["collected_at"])
    daily = d.dropna(subset=["t"]).t.dt.floor("D").value_counts().sort_index()
    thr = BULK_MULTIPLE * float(daily.median())
    return daily, set(daily[daily >= thr].index), thr


def fig_bulkdays(df: pd.DataFrame):
    """Day-level first-seen volume, with the bulk-ingest days flagged (M2). The monthly curve
    cannot show this: a month that is one 680-record insertion looks exactly like a month of
    steady arrival, and the recovery method (a database identifier) times INGESTION, not
    detection."""
    daily, bulk, thr = _bulk_days(df)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ax.vlines(daily.index, 0, daily.values, color=C_GREY, lw=0.7, alpha=0.8)
    b = daily[daily.index.isin(bulk)]
    ax.vlines(b.index, 0, b.values, color=C_PHISH, lw=1.8)
    ax.axhline(thr, color=C_ACCENT, lw=0.9, ls="--")
    ax.text(daily.index.min(), thr * 1.12,
            f"  bulk threshold: {BULK_MULTIPLE}$\\times$ the median day ({daily.median():.0f})",
            fontsize=7.5, color=C_ACCENT, va="bottom")
    # Two bulk days sit 18 days apart at nearly the same height, so labels placed straight above
    # their spikes overprint. Alternate the vertical offset along the sorted bulk days and draw a
    # leader line; the earliest day's label also has to clear the 10^3 tick.
    for k, (t, v) in enumerate(sorted(b.items())):
        dy = 22 if k % 2 == 0 else 8
        # the earliest bulk day hugs the left spine: hang its label to the right of the spike
        # so it stays clear of the y-axis tick labels
        ha = "left" if k == 0 else "center"
        # Leader and text are SEPARATE artists, and the split is the point. Drawn as one
        # annotation, the arrow inherits its own label's z-order, so a label drawn later put its
        # leader straight over the white box of the label drawn before it: 2022-11-04's leader
        # ran through the digits of "2022-10-17: 680" (2026-09-02). Split, every text box outranks
        # every leader, whatever order the days come in.
        off = (3 if k == 0 else 0, dy)
        ax.annotate("", (t, v), xytext=off, textcoords="offset points", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6, shrinkA=0, shrinkB=1))
        ax.annotate(f"{t:%Y-%m-%d}: {int(v):,}", (t, v), xytext=off,
                    textcoords="offset points", fontsize=7, ha=ha, va="bottom", color=INK,
                    zorder=10,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))
    ax.set_yscale("log")
    ax.set_ylabel("new phishing indicators / day (log)", fontsize=8)
    ax.set_xlabel("recovered first-seen day (ChongLuaDao stratum)")
    ax.set_ylim(bottom=0.7, top=daily.max() * 6)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_bulkdays.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


def gen_bulk(df: pd.DataFrame):
    """What the bulk days do to the series and to the dates themselves (M2/M3).

    The second half of this is the load-bearing part: on bulk days the recovered date disagrees
    with the national feed's attested date by an order of magnitude more than elsewhere, which is
    what a bulk import of a backlog looks like and what a burst of real detections does not."""
    daily, bulk, thr = _bulk_days(df)
    tot = int(daily.sum())
    bulk_mass = int(daily[daily.index.isin(bulk)].sum())
    top20 = int(daily.sort_values(ascending=False).head(20).sum())
    peak_day = daily.idxmax()
    STATS["bulk_days"] = [f"{d:%Y-%m-%d}" for d in sorted(bulk)]
    STATS["bulk_mass_pct"] = round(100 * bulk_mass / tot, 1)
    STATS["top20_day_mass_pct"] = round(100 * top20 / tot, 1)
    STATS["n_dated_days"] = int(len(daily))

    # the peak month, and how much of it is one day
    pm = daily[(daily.index >= peak_day.replace(day=1)) &
               (daily.index < peak_day.replace(day=1) + pd.offsets.MonthBegin(1))]
    STATS["peak_day"] = (f"{peak_day:%Y-%m-%d}", int(daily.max()))

    v = pd.read_csv(FS_VALID)
    o = v[v.check == "ncsc_overlap"].copy()
    o["cld"] = pd.to_datetime(o.cld_date)
    o["bulk"] = o.cld.isin(bulk)
    a, e = o.loc[o.bulk, "delta_days"].abs(), o.loc[~o.bulk, "delta_days"].abs()
    STATS["ncsc_delta_bulk"] = (int(len(a)), float(a.median()))
    STATS["ncsc_delta_other"] = (int(len(e)), float(e.median()))

    oct_line = ""
    if len(pm):
        oct_line = (f" The single largest, {peak_day:%d~%B~%Y}, carries {int(daily.max()):,} of "
                    f"the {int(pm.sum()):,} indicators its feed contributes that month "
                    f"({100 * daily.max() / pm.sum():.0f}\\%).")
    # the month the manuscript quotes as the peak deserves its own sentence: the reader is
    # entitled to know how much of THAT number is one insertion.
    pk = pd.Period(STATS.get("peak_month", ""), "M") if STATS.get("peak_month") else None
    if pk is not None:
        w = daily[(daily.index >= pk.start_time) & (daily.index <= pk.end_time)]
        if len(w):
            oct_line += (f" In the month the timeline peaks, {pk.start_time:%B~%Y}, one day "
                         f"({w.idxmax():%d~%B}) supplies {int(w.max()):,} of that stratum's "
                         f"{int(w.sum()):,} ({100 * w.max() / w.sum():.0f}\\%).")
    tex = (
        f"Recovered dates are ingestion times, and ingestion is lumpy. Across the "
        f"{len(daily):,} days on which the community feed's recovered dates fall, "
        f"\\textbf{{{len(bulk)}}} carry at least {BULK_MULTIPLE} times the median daily volume "
        f"({daily.median():.0f} indicators); together they account for "
        f"\\textbf{{{100 * bulk_mass / tot:.1f}\\%}} of that stratum's dated mass, and the "
        f"twenty largest days for {100 * top20 / tot:.1f}\\%.{oct_line} These days behave like "
        f"imports rather than detections, and the corpus's own date validation says so: on the "
        f"{len(a)} bulk-day indicators that the national feed also lists, the recovered date "
        f"differs from the national feed's attested date by a median of "
        f"\\textbf{{{a.median():.0f}~days}}, against \\textbf{{{e.median():.0f}~days}} for the "
        f"{len(e)} indicators dated on any other day")
    write_generated(os.path.join(SEC, "gen_bulk.tex"), tex.rstrip() + "%")


def gen_dating(df: pd.DataFrame):
    """The recovered dates' measured accuracy (M3), imported from the validation the repository
    already runs rather than asserted. Also the dated coverage at host granularity, which the
    `www.` collapse moves further than any other number in the paper."""
    s = json.load(open(FS_SUMMARY, encoding="utf-8"))
    ov, mc = s["ncsc_overlap"], s["mirror_consistency"]
    ph = df[df.label == "phishing"]
    dated_rows = int(parse_first_seen(ph["collected_at"]).notna().sum())
    dph = ph.assign(d=parse_first_seen(ph["collected_at"]).notna()).groupby("host").d.any()
    STATS["dated_hosts_phish"] = (int(dph.sum()), round(float(100 * dph.mean()), 1))
    d_all = df.assign(d=parse_first_seen(df["collected_at"]).notna()).groupby("host").d.any()
    STATS["dated_hosts_corpus"] = (int(d_all.sum()), round(float(100 * d_all.mean()), 1))
    tex = (
        f"Two checks answer the first question, neither consulting the recovery's own inputs. "
        f"Against the national feed's attested dates for the {ov['n']:,} indicators both list, "
        f"the recovered date sits within 30~days of it for "
        f"\\textbf{{{100 * ov['share_abs_within_30d']:.1f}\\%}} of them and within 90~days for "
        f"{100 * ov['share_abs_within_90d']:.1f}\\%, with a median difference of "
        f"{ov['median_delta_days']:.0f}~days (IQR {ov['iqr_days'][0]:.0f}--"
        f"{ov['iqr_days'][1]:.0f}); that spread bounds the recovery error from above, since it "
        f"also contains genuine inter-feed detection lag. Against a downstream mirror that can "
        f"only lag the database it copies, \\textbf{{{100 * mc['violation_rate']:.1f}\\%}} of "
        f"{mc['n']:,} comparable entries violate the ordering and are therefore definite "
        f"recovery errors, {100 * mc['violation_share_within_30d']:.1f}\\% of them by less than "
        f"30~days (maximum {mc['violation_max_days']:.0f}). We use the recovered dates as a "
        f"month-scale first-seen estimate with that error attached, never as an exact insertion "
        f"time. Coverage, finally, depends entirely on the unit: {dated_rows:,} phishing "
        f"\\emph{{rows}} carry a date ({100 * dated_rows / len(ph):.0f}\\%), but the undated "
        f"rows are almost exactly the \\texttt{{www.}} twins, so "
        f"\\textbf{{{STATS['dated_hosts_phish'][0]:,}}} of the "
        f"{int(ph.host.nunique()):,} distinct phishing \\emph{{hosts}} "
        f"(\\textbf{{{STATS['dated_hosts_phish'][1]:.1f}\\%}}) are dated")
    write_generated(os.path.join(SEC, "gen_dating.tex"), tex.rstrip() + "%")


def dated_span(df: pd.DataFrame):
    """The [min, max] of every dated row (both classes), padded by 2% on each side. fig_timeline
    and fig_feed_lifespan both set their x-limits from this, so the two charts share one time
    scale and a reader can align the lanes with the curve (Section 7 says so)."""
    dt = parse_first_seen(df["collected_at"]).dropna()
    lo, hi = dt.min(), dt.max()
    pad = (hi - lo) * 0.02
    return lo - pad, hi + pad


def fig_feed_lifespan(df: pd.DataFrame):
    """The timeline's right tail, explained: Section~\\ref{sec:drift} claims the post-2023
    thinning is feed mortality (both phishing feeds stopped publishing), a claim about WHEN each
    source stops that the aggregate curve cannot show — so one lifeline per source, on the same
    time axis as fig_timeline. The benign lane running to 2026 rules out 'the collection stopped'; its dates are
    registry certifications, hence the benign colour and label."""
    dt = parse_first_seen(df["collected_at"])
    d = pd.DataFrame({"t": dt, "source": df["source"].astype(str),
                      "label": df["label"].astype(str)}).dropna(subset=["t"])
    lanes = (d.groupby(["source", "label"])
               .agg(n=("t", "size"), first=("t", "min"), last=("t", "max"))
               .reset_index().sort_values("last"))

    # Lane chart only: the monthly curve that used to sit above the lanes duplicated fig_timeline,
    # so the lanes stand alone over the timeline's full dated span.
    fig, ax1 = plt.subplots(1, 1, figsize=(8.4, 2.4))

    # A lane ending near the right edge would push its label off the axes, so flip that one to
    # sit inside the bar instead of after it.
    span_end = lanes["last"].max()
    span_start = lanes["first"].min()
    flip_after = span_start + (span_end - span_start) * 0.88
    for i, r in enumerate(lanes.itertuples()):
        c = C_PHISH if r.label == "phishing" else C_BENIGN
        ax1.hlines(i, r.first, r.last, color=c, lw=5, alpha=0.85)
        ax1.plot([r.last], [i], marker="|", color=c, markersize=11, mew=2)
        right = r.last < flip_after
        # Labels near the right edge flip inward and ride ABOVE the lane (over the bar they'd
        # collide with); all labels wear ink, not the series colour, so none blends into its mark.
        ax1.annotate(f"  last {r.last:%Y-%m-%d}  ", (r.last, i), fontsize=7.5, color=INK,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=6,
                     va="center" if right else "bottom",
                     xytext=(0, 0 if right else 7), textcoords="offset points",
                     ha="left" if right else "right")
    ax1.set_yticks(range(len(lanes)))
    ax1.set_yticklabels([f"{r.source} ({r.label}, {r.n:,})" for r in lanes.itertuples()],
                        fontsize=7.5)
    ax1.set_ylim(-0.7, len(lanes) - 0.3)
    ax1.set_xlim(*dated_span(df))
    ax1.set_xlabel("first-seen date")
    ax1.spines[["top", "right"]].set_visible(False)

    # the two phishing terminations, dashed so they can be matched against fig_timeline
    for r in lanes[lanes.label == "phishing"].itertuples():
        ax1.axvline(r.last, color=C_GREY, lw=0.9, ls="--", zorder=0)
    STATS["feed_last_phishing"] = {r.source: str(r.last.date())
                                   for r in lanes[lanes.label == "phishing"].itertuples()}
    STATS["feed_last_benign"] = {r.source: str(r.last.date())
                                 for r in lanes[lanes.label == "benign"].itertuples()}
    fig.tight_layout()
    out = os.path.join(FIG, "fig_feed_lifespan.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def fig_timeline(df: pd.DataFrame):
    """Monthly first-seen volume over the dated subset only. Coverage fraction is annotated and
    restated in the caption so the reader never mistakes this for the full corpus."""
    dt = parse_first_seen(df["collected_at"])
    dated = dt.dropna()
    cov = 100 * len(dated) / len(df)
    STATS["n_dated"] = int(len(dated))
    STATS["dated_coverage_pct"] = round(float(cov), 1)
    STATS["date_min"] = str(dated.min().date())
    STATS["date_max"] = str(dated.max().date())

    # PHISHING ONLY: `dated` also holds 2,026 benign rows whose date is a registry CERTIFICATION,
    # not a detection -- including them put the annotated peak at 1,788 against 1,761 in the prose.
    ph_dated = dt[df["label"].to_numpy() == "phishing"].dropna()
    STATS["n_dated_phishing"] = int(len(ph_dated))
    ph_cov = 100 * len(ph_dated) / int((df["label"] == "phishing").sum())
    monthly = ph_dated.dt.to_period("M").value_counts().sort_index()
    # Plot the FULL dated span (February 2020 onward): the 2020 tail is sparse, but the prose
    # describes a low-but-nonzero 2020 baseline and the lifespan lanes start there. Empty months
    # are filled as zero so the curve does not bridge gaps.
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(),
                                              freq="M"), fill_value=0)
    x = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.fill_between(x, monthly.values, color=C_PHISH, alpha=0.25)
    ax.plot(x, monthly.values, color=C_PHISH, lw=1.6)
    ax.set_ylabel("newly first-seen indicators / month")
    ax.set_xlabel("first-seen month")
    # same x-limits as fig_feed_lifespan, so the two charts share one time scale
    ax.set_xlim(*dated_span(df))
    peak = monthly.idxmax()
    ax.annotate(f"peak {peak}: {int(monthly.max()):,}",
                xy=(peak.to_timestamp(), monthly.max()),
                xytext=(0, 10), textcoords="offset points", fontsize=8, ha="center")
    # the caveat must describe what is PLOTTED, not the corpus-wide dated subset
    ax.text(0.99, 0.95, f"dated phishing only: {len(ph_dated):,}/"
                        f"{int((df['label'] == 'phishing').sum()):,} ({ph_cov:.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C_GREY)
    STATS["peak_month"] = str(peak)
    STATS["peak_month_n"] = int(monthly.max())
    fig.tight_layout()
    out = os.path.join(FIG, "fig_timeline.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[+] {out}")


# ------------------------------------------------------------------ tables
def tab_feed_composition(df: pd.DataFrame):
    """Per-source: n, share of corpus, %phishing, and tier mix (bronze/silver/gold counts)."""
    rows = []
    order = df.source.value_counts().index.tolist()
    for s in order:
        sub = df[df.source == s]
        n = len(sub)
        pct = 100 * n / len(df)
        pph = 100 * (sub.label == "phishing").mean()
        tm = sub.tier.value_counts()
        rows.append((s, n, int(sub.host.nunique()), pct, pph, int(tm.get("bronze", 0)),
                     int(tm.get("silver", 0)), int(tm.get("gold", 0))))
    body = "\n".join(
        f"\\texttt{{{_esc(s)}}} & {n:,} & {nh:,} & {pct:.1f}\\% & {pph:.0f}\\% & "
        f"{b:,} & {si:,} & {g:,} \\\\"
        for (s, n, nh, pct, pph, b, si, g) in rows)
    tot = len(df)
    tb = int((df.tier == "bronze").sum()); ts = int((df.tier == "silver").sum())
    tg = int((df.tier == "gold").sum())
    pph_all = 100 * (df.label == "phishing").mean()
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Per-source composition of the fused corpus: indicators, \\texttt{{www.}}-collapsed
hosts, corpus share, phishing fraction and confidence-tier mix. The Hosts column is not additive
across sources.}}
\\label{{tab:feed_composition}}
\\small
\\begin{{tabular}}{{l r r r r r r r}}
\\toprule
Source feed & Indicators & Hosts & \\% corpus & \\% phish & Bronze & Silver & Gold \\\\
\\midrule
{body}
\\midrule
\\textbf{{All}} & {tot:,} & {int(df.host.nunique()):,} & 100\\% & {pph_all:.0f}\\% & {tb:,} & {ts:,} & {tg:,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_feed_composition.tex"), tex)


def tab_brand_dist(df: pd.DataFrame):
    """Impersonated-sector distribution among phishing indicators: count and % of phishing."""
    ph = df[df.label == "phishing"]
    u = hosts(df, "phishing")
    n_ph, n_h = len(ph), len(u)
    counts = ph.scenario.value_counts()
    hcounts = u.scenario.value_counts()
    ci = STATS["sector_host_ci"]
    rows = []
    for s in SECTORS:
        c = int(counts.get(s, 0))
        if c == 0:
            continue
        hc = int(hcounts.get(s, 0))
        rows.append((SECTOR_LABEL[s], c, 100 * c / n_ph, hc, 100 * hc / n_h, ci[s][1], ci[s][2]))
    rows.sort(key=lambda r: -r[3])
    body = "\n".join(f"{lab} & {c:,} & {p:.2f}\\% & {hc:,} & {hp:.2f}\\% & "
                     f"[{lo:.2f}, {hi:.2f}] \\\\"
                     for (lab, c, p, hc, hp, lo, hi) in rows)
    n_other = int(counts.get("other", 0))
    n_tagged = n_ph - n_other
    h_other = int(hcounts.get("other", 0))
    h_tagged = n_h - h_other
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Phishing by impersonated sector, per row and per distinct host; shares are of all
phishing. The CI is a 95\\% host-clustered bootstrap ({BOOT_B:,} replicates, seed {BOOT_SEED}).}}
\\label{{tab:brand_dist}}
\\small
\\begin{{tabular}}{{l r r r r c}}
\\toprule
& \\multicolumn{{2}}{{c}}{{rows}} & \\multicolumn{{3}}{{c}}{{distinct hosts}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-6}}
Impersonated sector & $n$ & \\% & $n$ & \\% & 95\\% CI \\\\
\\midrule
{body}
\\midrule
\\textbf{{Sector-tagged}} & {n_tagged:,} & {100 * n_tagged / n_ph:.2f}\\% & {h_tagged:,} & {100 * h_tagged / n_h:.2f}\\% & \\\\
Untagged (\\emph{{other}}) & {n_other:,} & {100 * n_other / n_ph:.2f}\\% & {h_other:,} & {100 * h_other / n_h:.2f}\\% & \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_brand_dist.tex"), tex)
    STATS["sector_host_counts"] = {s: int(hcounts.get(s, 0)) for s in SECTORS}
    STATS["n_tagged_hosts"] = h_tagged
    STATS["h_tagged_pct"] = round(100 * h_tagged / n_h, 2)
    STATS["h_other_pct"] = round(100 * h_other / n_h, 2)


def tab_infra(df: pd.DataFrame):
    """Top TLDs with phishing/benign split and .vn share; suspicious-TLD rate by class. Registrar,
    ASN, certificate and lifecycle columns are deferred to the planned WHOIS/DNS enrichment."""
    ph = df[df.label == "phishing"]
    be = df[df.label == "benign"]
    uph, ube = hosts(df, "phishing"), hosts(df, "benign")
    topN = 12
    top = hosts(df).tld.value_counts().head(topN).index.tolist()
    ph_c = ph.tld.value_counts()
    uph_c = uph.tld.value_counts(); ube_c = ube.tld.value_counts()
    rows = []
    for t in top:
        pc = int(ph_c.get(t, 0))
        up = int(uph_c.get(t, 0)); ub = int(ube_c.get(t, 0))
        tot = up + ub
        rows.append((t, pc, up, ub, 100 * up / tot if tot else 0))
    body = "\n".join(
        f"\\texttt{{.{_esc(t)}}} & {pc:,} & {up:,} & {ub:,} & {ppct:.0f}\\% \\\\"
        for (t, pc, up, ub, ppct) in rows)
    susp_ph = 100 * ph.suspicious_tld.mean()
    susp_be = 100 * be.suspicious_tld.mean()
    STATS["susp_tld_phish"] = round(float(susp_ph), 1)
    STATS["susp_tld_benign"] = round(float(susp_be), 1)
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Top {topN} of the {int(df.tld.nunique())} observed TLDs by host volume. Phishing is
given as rows and as hosts; the phishing share is computed on hosts.}}
\\label{{tab:infra}}
\\small
\\begin{{tabular}}{{l r r r r}}
\\toprule
& \\multicolumn{{2}}{{c}}{{phishing}} & benign & \\\\
\\cmidrule(lr){{2-3}}
TLD & rows & hosts & hosts & \\% phishing (hosts) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_infra.tex"), tex)


def main():
    os.makedirs(SEC, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df = load()
    # The feed-composition and sector bar charts duplicate their tables and are no longer placed in
    # the manuscript; the functions stay, the PDFs are produced only on --all-figures.
    if "--all-figures" in sys.argv:
        fig_feed_composition(df)
        fig_sector(df)
    fig_tld(df)
    fig_urlfeat(df)
    fig_timeline(df)
    fig_feed_lifespan(df)
    fig_bulkdays(df)
    tab_feed_composition(df)
    tab_feed_overlap(df)
    common_window_sensitivity(df)
    tier_sensitivity(df)
    gen_dedup(df)
    gen_hoststruct(df)
    gen_vnframe(df)
    gen_sector_ci(df)      # must precede tab_brand_dist: it computes the intervals that table prints
    tab_brand_dist(df)
    tab_infra(df)
    gen_bulk(df)
    gen_dating(df)
    gen_govtokens(df)
    print("\n===== HEADLINE NUMBERS (spot-check against the manuscript) =====")
    for k, v in STATS.items():
        print(f"  {k:24s} = {v}")
    print("Done. Recompile the P7 (CTI) manuscript to pick up the regenerated assets.")


if __name__ == "__main__":
    main()
