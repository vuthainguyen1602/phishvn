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


def figure(rows: list[dict], s: dict) -> str:
    from figstyle import apply, BLUE, ORANGE, GRAY, INK
    plt = apply()
    import numpy as np

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                  gridspec_kw={"width_ratios": [1.45, 1]})

    # --- LEFT: separability against elapsed time.
    ax.axhline(0.5, color=GRAY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", (0.015, 0.5), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 3), fontsize=7, color=INK)
    for r in rows:
        colour = ORANGE if r["boundary"] else BLUE
        ax.scatter(r["delta_days"], r["auc"], s=44, color=colour, zorder=3,
                   marker="D" if r["boundary"] else "o", edgecolor="white", linewidth=0.6)
        ax.annotate(f"{r['a']}–{r['b']}", (r["delta_days"], r["auc"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=6.8,
                    color=colour)
    # The two points that carry the non-monotonicity: further apart, less separable.
    st, fu = s["strongest"], s["furthest"]
    ax.annotate("", xy=(fu["delta_days"], fu["auc"]), xytext=(st["delta_days"], st["auc"]),
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9,
                            "shrinkA": 6, "shrinkB": 6,
                            "connectionstyle": "arc3,rad=0.25"})
    # Boxed: the annotation arrow drawn just above runs diagonally through this text.
    ax.annotate("further apart in time,\nless separable", (fu["delta_days"], fu["auc"]),
                textcoords="offset points", xytext=(-4, 14), ha="right", va="bottom",
                fontsize=7, color=INK, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=6)
    ax.annotate(f"Spearman $\\rho={s['rho']:+.2f}$ ($p={s['p']:.2f}$, $n={s['n']}$ pairs)",
                (0.015, 0.965), xycoords="axes fraction", fontsize=7.5, color=INK, va="top")
    ax.set_xlabel("distance between block median dates (days)")
    ax.set_ylabel("discriminator ROC-AUC")
    ax.set_ylim(0.45, max(r["auc"] for r in rows) + 0.10)
    ax.set_title("elapsed time does not order distributional distance", fontsize=8.5)
    ax.grid(axis="y", alpha=0.6)
    ax.annotate("interior pair", (0.015, 0.875), xycoords="axes fraction",
                fontsize=7.5, color=BLUE)
    ax.annotate("crosses the deployment boundary", (0.015, 0.795), xycoords="axes fraction",
                fontsize=7.5, color=ORANGE)

    # --- RIGHT: the same ten numbers, indexed the way a reader looks a pair up.
    idx = {b: i for i, b in enumerate(BLOCKS)}
    grid = np.full((len(BLOCKS) - 1, len(BLOCKS) - 1), np.nan)
    for r in rows:
        grid[idx[r["a"]], idx[r["b"]] - 1] = r["auc"]
    im = ax2.imshow(grid, cmap="BuPu", vmin=0.5, vmax=max(r["auc"] for r in rows),
                    aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                # White on the dark end of the ramp, ink on the light end: one fixed colour
                # would be unreadable at whichever end it did not suit.
                dark = grid[i, j] > 0.5 + 0.62 * (max(r["auc"] for r in rows) - 0.5)
                ax2.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                         color="white" if dark else INK)
    ax2.set_xticks(range(len(BLOCKS) - 1), BLOCKS[1:], fontsize=8)
    ax2.set_yticks(range(len(BLOCKS) - 1), BLOCKS[:-1], fontsize=8)
    ax2.set_title("the matrix itself", fontsize=8.5)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).outline.set_visible(False)

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
