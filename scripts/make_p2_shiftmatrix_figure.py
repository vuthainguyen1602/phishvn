#!/usr/bin/env python3
"""
make_p2_shiftmatrix_figure.py — the shift matrix drawn against time, where its point lives.

`tab_shiftmatrix` orders ten pairwise separabilities by WINDOW; the claim made about them is about
TIME. Plotted against distance the shape is immediate, and it is the shape the companion papers
lean on: drift here is episodic, so a model of it as a smooth function of elapsed time has nothing
to fit. The rank correlation is reported on the panel with its p-value and deliberately not leaned
on — ten pairs from five blocks are not ten independent observations.

    python scripts/make_p2_shiftmatrix_figure.py

Writes papers/P2_url_benchmark/figures/fig_shiftmatrix.pdf and
papers/P2_url_benchmark/sections/gen_shiftmatrix.tex. Reads the CSV the table reads.
What each panel says, and the dependence caveat: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import csv
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

SEC = os.path.join(ROOT, "papers", "P2_url_benchmark", "sections")
FIG = os.path.join(ROOT, "papers", "P2_url_benchmark", "figures")
CSV_PATH = os.path.join(ROOT, "data", "processed", "p2", "p2_forecastability_shiftmatrix.csv")

BLOCKS = ("W1", "W2", "W3", "W4", "TEST")
# Pairs one block apart in the training sequence. Named rather than derived so that a reordered
# CSV cannot silently change what "adjacent" means.
ADJACENT = (("W1", "W2"), ("W2", "W3"), ("W3", "W4"))


def load() -> list[dict]:
    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    for r in rows:
        r["auc"] = float(r["auc"])
        r["delta_days"] = float(r["delta_days"])
        r["boundary"] = r["b"] == "TEST"
    return rows


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    from scipy.stats import spearmanr
    rho, p = spearmanr(xs, ys)
    return float(rho), float(p)


def summarise(rows: list[dict]) -> dict:
    adj = [r for r in rows if (r["a"], r["b"]) in ADJACENT]
    bnd = [r for r in rows if r["boundary"]]
    rho, p = spearman([r["delta_days"] for r in rows], [r["auc"] for r in rows])
    strongest = max(rows, key=lambda r: r["auc"])
    furthest = max(rows, key=lambda r: r["delta_days"])
    nearest = min(rows, key=lambda r: r["delta_days"])
    return {
        "n": len(rows), "rho": rho, "p": p,
        "adj_mean": sum(r["auc"] for r in adj) / len(adj),
        "adj_lo": min(r["auc"] for r in adj), "adj_hi": max(r["auc"] for r in adj),
        "bnd_mean": sum(r["auc"] for r in bnd) / len(bnd),
        "strongest": strongest, "furthest": furthest, "nearest": nearest,
        "adj": adj, "bnd": bnd,
    }


# (a, b) -> (dx, dy, ha) in points, for the labels the default placement collides at one column
LABEL_NUDGE = {("W1", "W2"): (0, 8, "center"), ("W2", "W3"): (0, -14, "center"),
               ("W1", "TEST"): (-7, 2, "right")}


def figure(rows: list[dict], s: dict) -> str:
    from figstyle import apply, BLUE, ORANGE, GRAY, INK
    plt = apply()

    # One panel, one column. The right panel used to reprint Table~\ref{tab:shiftmatrix} as a
    # heat map: the same ten numbers, in the same layout, half a page from the table itself,
    # which also carries the block distances the heat map had no room for. The table was
    # strictly the more informative of the two. What only a figure can do is put separability
    # against elapsed time, and that is the panel that stayed.
    fig, ax = plt.subplots(figsize=(3.9, 3.1))

    # --- LEFT: separability against elapsed time.
    ax.axhline(0.5, color=GRAY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", (0.015, 0.5), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK)
    for r in rows:
        colour = ORANGE if r["boundary"] else BLUE
        ax.scatter(r["delta_days"], r["auc"], s=44, color=colour, zorder=3,
                   marker="D" if r["boundary"] else "o", edgecolor="white", linewidth=0.6)
        # Default is above and centred. Three pairs need something else at one column: W1-W2
        # and W2-W3 are 40 days and 0.013 apart, and W1-TEST is the rightmost point, whose
        # centred label would run off the axis.
        dx, dy, ha = LABEL_NUDGE.get((r["a"], r["b"]), (0, 8, "center"))
        ax.annotate(f"{r['a']}–{r['b']}", (r["delta_days"], r["auc"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.8,
                    color=colour)
    # The two points that break the ordering by distance. The label used to read "further apart
    # in time, less separable", which is a directional law read off two hand-picked points, and
    # the panel's own rho is +0.33: the weak trend across all ten runs the OTHER way. The two
    # also differ in whether they cross the boundary, so elapsed time is not even the only thing
    # separating them. What they show is the ordering failing, which is what the label now says
    # and what the prose beside the table claims.
    st, fu = s["strongest"], s["furthest"]
    ax.annotate("", xy=(fu["delta_days"], fu["auc"]), xytext=(st["delta_days"], st["auc"]),
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9,
                            "shrinkA": 7, "shrinkB": 7,
                            "connectionstyle": "arc3,rad=-0.3"})
    # Above the arc rather than on it: the old placement put the text across the arrow and needed
    # an opaque box to hide the collision, which broke the arrow into two pieces.
    # No text on the arrow. At one column every placement collided with something: above ran
    # through the Spearman line, below ran through W2-W4. The caption says what the arrow joins
    # and the prose beside it gives both pairs with their numbers, so the panel does not need to
    # repeat either.
    ax.annotate(f"Spearman $\\rho={s['rho']:+.2f}$ ($p={s['p']:.2f}$, $n={s['n']}$ pairs)",
                (0.015, 0.965), xycoords="axes fraction", fontsize=7.5, color=INK, va="top")
    ax.set_xlabel("distance between block median dates (days)")
    ax.set_ylabel("discriminator ROC-AUC")
    ax.set_ylim(0.45, max(r["auc"] for r in rows) + 0.10)
    span = max(r["delta_days"] for r in rows) - min(r["delta_days"] for r in rows)
    ax.set_xlim(min(r["delta_days"] for r in rows) - 0.09 * span,
                max(r["delta_days"] for r in rows) + 0.09 * span)
    ax.set_title("elapsed time does not order distributional distance", fontsize=8.5)
    ax.grid(axis="y", alpha=0.6)
    ax.scatter([], [], s=44, color=BLUE, marker="o", edgecolor="white", linewidth=0.6,
               label="interior pair")
    ax.scatter([], [], s=44, color=ORANGE, marker="D", edgecolor="white", linewidth=0.6,
               label="crosses the boundary")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7,
                    handletextpad=0.4, borderpad=0.0, labelspacing=0.3)
    for txt, col in zip(leg.get_texts(), (BLUE, ORANGE)):
        txt.set_color(col)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_shiftmatrix.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def make_tex(rows: list[dict], s: dict) -> None:
    st, fu, ne = s["strongest"], s["furthest"], s["nearest"]
    body = (
        f"Plotted against elapsed time rather than laid out by window "
        f"(Figure~\\ref{{fig:shiftmatrix}}), the matrix makes the negative result legible: "
        "separability does not order by distance at all. The strongest separation in the table "
        f"is an \\emph{{interior}} pair, {st['a']}--{st['b']} at ${st['auc']:.3f}$ over "
        f"${st['delta_days']:.0f}$ days, while the widest gap in the table, "
        f"{fu['a']}--{fu['b']} at ${fu['delta_days']:.0f}$ days, separates less "
        f"(${fu['auc']:.3f}$); and the closest pair, {ne['a']}--{ne['b']} at "
        f"${ne['delta_days']:.0f}$ days, separates more (${ne['auc']:.3f}$) than the adjacent "
        f"interior pairs average (${s['adj_mean']:.3f}$, range ${s['adj_lo']:.3f}$--"
        f"${s['adj_hi']:.3f}$). Over all ${s['n']}$ pairs the rank correlation between distance "
        f"and separability is $\\rho = {s['rho']:+.2f}$ ($p = {s['p']:.2f}$). We report that "
        "correlation rather than resting on it: ten pairs drawn from five blocks are not ten "
        "independent observations, since each block appears in three or four of them, so the "
        "defensible reading is that this design cannot resolve a monotone trend, not that "
        "elapsed time is irrelevant. That is enough for the use the benchmark makes of it. The "
        "decay-extrapolation methods of Section~\\ref{sec:related} need distance in time to stand "
        "in for distance in distribution, and here it does not, which is why forecasting the "
        "random-minus-temporal gap from a family's own decay slope fails"
    )
    write_generated(os.path.join(SEC, "gen_shiftmatrix.tex"), body.rstrip() + "%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    if not os.path.exists(CSV_PATH):
        print(f"[i] {CSV_PATH} absent — run the forecastability study first.")
        return 0
    rows = load()
    s = summarise(rows)
    figure(rows, s)
    make_tex(rows, s)
    print(f"[i] {s['n']} pairs; rho={s['rho']:+.3f} (p={s['p']:.3f}); "
          f"adjacent interior mean {s['adj_mean']:.3f} "
          f"({s['adj_lo']:.3f}-{s['adj_hi']:.3f}); "
          f"strongest {s['strongest']['a']}-{s['strongest']['b']} "
          f"{s['strongest']['auc']:.3f} at {s['strongest']['delta_days']:.0f}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
