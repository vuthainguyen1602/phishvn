#!/usr/bin/env python3
"""
make_p5_budget_figure.py — the labelling-budget frontier as a curve, not three columns.

At `tab_budget`'s resolution the two things worth knowing are invisible: the drift-triggered
policy's crossover with the periodic one, and that uncertainty sampling loses to random selection
at every budget. The x axis is labels SPENT, not the per-retrain budget B. The second panel plots
each policy against periodic retraining at the same cost, which is the contrast the design
isolates. Preliminary throughout: 20 windows on a 41%-dated stream, one seed per policy — the
curve's shape is the claim, its exact crossover point is not.

    python scripts/make_p5_budget_figure.py            # the dense grid
    python scripts/make_p5_budget_figure.py --quick    # the table's three budgets only

Writes data/processed/p5/p5_budget_frontier.csv,
papers/P5_temporal_drift/figures/fig_budget_frontier.pdf and
papers/P5_temporal_drift/sections/gen_budget_frontier.tex.
Why the curve, the axis and the second panel: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import argparse
import csv
import io
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
# The stream, the window count and the trigger constants are imported from the asset script that
# builds the tables, so the figure and `tab_budget` cannot end up describing different runs.
from make_p5_assets import (BUDGETS, DRIFT_CSV, F1_DROP, PERIOD,
                            PSI_TAU, WINDOWS, build_input)
from drift_detectors import load, run_budget

SEC = os.path.join(ROOT, "papers", "P5_temporal_drift", "sections")
FIG = os.path.join(ROOT, "papers", "P5_temporal_drift", "figures")
PROC = os.path.join(ROOT, "data", "processed")

# Denser than the table's three, and geometric because the interesting behaviour is at the cheap
# end: the crossover the table straddles sits between its first two columns.
DENSE = (25, 50, 100, 200, 400, 800, 1600)
POLICIES = ("static", "periodic", "drift", "active")
PRETTY = {"static": "static", "periodic": "periodic \u00b7 random",
          "drift": "drift-triggered \u00b7 random", "active": "active \u00b7 uncertainty"}


def frontier(budgets) -> list[dict]:
    build_input()
    df, feats = load(DRIFT_CSV, "collected_at", spread_undated=True)
    out = run_budget(df, feats, WINDOWS, list(budgets), period=PERIOD, psi_tau=PSI_TAU,
                     f1_drop=F1_DROP)
    rows = []
    for b in sorted(out):
        for pol in POLICIES:
            autc, labels = out[b][pol]
            rows.append({"budget": b, "policy": pol, "autc": autc, "labels": labels,
                         "registered": int(b in BUDGETS)})
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    write_generated(path, buf.getvalue())


def tex_int(n: int) -> str:
    """Digit grouping that survives math mode. Applied to the NUMBER, never to the sentence: a
    str.replace on the whole body once turned every prose comma into a LaTeX group."""
    return f"{n:,}".replace(",", "{,}")


def crossover(rows: list[dict]) -> tuple[int | None, float, float]:
    """The budget at which drift-triggered overtakes periodic, and the gap at each end. Reported
    as the first grid point where the sign flips -- interpolating a crossing from one seed per
    point would dress a coarse experiment up as a precise one."""
    per = {r["budget"]: r["autc"] for r in rows if r["policy"] == "periodic"}
    dri = {r["budget"]: r["autc"] for r in rows if r["policy"] == "drift"}
    bs = sorted(per)
    gaps = [(b, dri[b] - per[b]) for b in bs]
    flip = next((b for (b, g), (_, gp) in zip(gaps[1:], gaps) if g > 0 >= gp), None)
    return flip, gaps[0][1], gaps[-1][1]


def make_figure(rows: list[dict]) -> str:
    from figstyle import apply, BLUE, ORANGE, TEAL, GRAY, INK
    plt = apply()

    colour = {"periodic": BLUE, "drift": ORANGE, "active": TEAL, "static": GRAY}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # --- LEFT: AUTC against what it cost.
    static = [r for r in rows if r["policy"] == "static"]
    if static:
        ax.axhline(static[0]["autc"], color=GRAY, lw=1.0, ls=(0, (4, 3)), zorder=2)
        # Parked mid-axis just above its own line: the far right collides with the tick
        # labels and the far left with the cheapest active point.
        ax.annotate("static: no labels, no recovery", (0.62, static[0]["autc"]),
                    xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(0, 5), fontsize=7.5, color=INK, ha="center")
    for pol in ("periodic", "drift", "active"):
        pts = sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["labels"])
        ax.plot([r["labels"] for r in pts], [r["autc"] for r in pts], "-o", ms=3.2, lw=1.4,
                color=colour[pol], zorder=3, label=PRETTY[pol])
        reg = [r for r in pts if r["registered"]]
        ax.scatter([r["labels"] for r in reg], [r["autc"] for r in reg], s=42,
                   facecolor="white", edgecolor=colour[pol], linewidth=1.2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("labels bought over the stream")
    ax.set_ylabel("AUTC (mean per-window F1)")
    ax.set_title("what a label buys", fontsize=8.5)
    ax.grid(axis="y", alpha=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # --- RIGHT: each policy against periodic at the same cost. Zero is the periodic baseline.
    ax2.axhline(0, color=INK, lw=0.9, zorder=3)
    per = {r["budget"]: r["autc"] for r in rows if r["policy"] == "periodic"}
    for pol in ("drift", "active"):
        pts = sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["budget"])
        xs = [r["labels"] for r in pts]
        ys = [r["autc"] - per[r["budget"]] for r in pts]
        ax2.plot(xs, ys, "-o", ms=3.2, lw=1.4, color=colour[pol], zorder=4)
        # Anchor index chosen per series: the orange curve is alone on the left of the
        # zero line, the teal one at its own trough.
        i = 1 if pol == "drift" else min(range(len(ys)), key=lambda j: ys[j])
        ax2.annotate({"drift": "trigger", "active": "selection"}[pol],
                     (xs[i], ys[i]), textcoords="offset points",
                     xytext=(8, 7) if pol == "drift" else (10, 0),
                     ha="left", va="bottom" if pol == "drift" else "center",
                     fontsize=8, color=colour[pol])
    flip, _, _ = crossover(rows)
    if flip is not None:
        xf = next(r["labels"] for r in rows if r["policy"] == "drift" and r["budget"] == flip)
        ax2.axvline(xf, color=ORANGE, lw=0.8, ls=(0, (2, 3)), alpha=0.8, zorder=2)
        ax2.annotate("crossover", (xf, 0.0), textcoords="offset points",
                     xytext=(6, -13), fontsize=7, color=ORANGE, ha="left")
    ax2.set_xscale("log")
    ax2.set_xlabel("labels bought over the stream")
    ax2.set_ylabel("AUTC $-$ periodic AUTC")
    ax2.set_title("changing the trigger, changing the selection", fontsize=8.5)
    ax2.grid(axis="y", alpha=0.6)

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_budget_frontier.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def make_tex(rows: list[dict]) -> None:
    flip, gap_lo, gap_hi = crossover(rows)
    bs = sorted({r["budget"] for r in rows})
    stat = next(r["autc"] for r in rows if r["policy"] == "static")
    best = max((r for r in rows if r["policy"] != "static"), key=lambda r: r["autc"])
    act = {r["budget"]: r["autc"] for r in rows if r["policy"] == "active"}
    per = {r["budget"]: r["autc"] for r in rows if r["policy"] == "periodic"}
    act_loses = sum(1 for b in bs if act[b] < per[b])
    act_gap_lo, act_gap_hi = act[bs[0]] - per[bs[0]], act[bs[-1]] - per[bs[-1]]
    # Whether the penalty actually shrinks at every step, rather than only between the endpoints.
    # It does not on the current grid -- it widens 54% from B=25 to B=50 before closing -- and the
    # sentence below used to assert monotonicity while quoting exactly the two budgets that hide
    # the excursion.
    act_gaps = [act[b] - per[b] for b in bs]
    act_monotone = all(act_gaps[i] <= act_gaps[i + 1] for i in range(len(act_gaps) - 1))
    act_worst_b = bs[min(range(len(act_gaps)), key=lambda i: act_gaps[i])]
    act_worst = min(act_gaps)
    act_small = sum(1 for g in act_gaps if -0.01 < g < 0)

    flip_txt = (f"between $B={bs[bs.index(flip) - 1]}$ and $B={flip}$"
                if flip is not None else "nowhere on this grid")
    dri = {r["budget"]: r["autc"] for r in rows if r["policy"] == "drift"}
    lab = {(r["policy"], r["budget"]): r["labels"] for r in rows}
    # equal label spend: the trigger at B against the schedule at the budget that buys as many labels
    eq = [(b, h, dri[b] - per[h]) for b in bs for h in bs
          if lab[("drift", b)] == lab[("periodic", h)] and lab[("drift", b)] > 0]
    ratio = lab[("drift", bs[-1])] / lab[("periodic", bs[-1])]
    if flip is None and all(dri[b] < per[b] for b in bs):
        trig_txt = (f"The label-free trigger retrains less often than the schedule (it buys "
                    f"{100 * ratio:.0f}\\% of the schedule's labels at every budget) and is behind "
                    f"it at every budget, by ${gap_lo:+.3f}$ AUTC at the cheapest and ${gap_hi:+.3f}$ "
                    f"at the dearest. Matched on labels bought instead (the trigger at $B$ against "
                    f"the schedule at the budget that buys as many), it is still behind on every "
                    f"pair, by ${max(e[2] for e in eq):+.3f}$ to ${min(e[2] for e in eq):+.3f}$. ")
    else:
        trig_txt = (f"The drift trigger is not uniformly better than a fixed schedule: it is worth "
                    f"${gap_lo:+.3f}$ AUTC against periodic retraining at the cheapest budget and "
                    f"${gap_hi:+.3f}$ at the dearest, crossing over {flip_txt}. ")
    body = (
        f"Figure~\\ref{{fig:budgetfrontier}} runs the same experiment across {len(bs)} budgets "
        f"rather than the three Table~\\ref{{tab:budget}} prices, and two things the table cannot "
        "resolve become visible. " + trig_txt + "Uncertainty sampling is the second negative "
        f"result: it is worse than random selection at {act_loses} of the {len(bs)} budgets "
        f"({act_small} of them by less than $0.01$ AUTC, a margin the seed repeat below shows "
        "is not seed noise), so on this stream choosing \\emph{which} examples "
        "to label costs accuracy rather than saving labels. On the dense grid the "
        + (f"penalty shrinks monotonically from ${act_gap_lo:+.3f}$ at $B={bs[0]}$ to "
           f"${act_gap_hi:+.3f}$ at $B={bs[-1]}$, which is what convergence of the two selections "
           "under a growing budget looks like and not what a constant handicap would. "
           if act_monotone else
           f"penalty closes from ${act_gap_lo:+.3f}$ at $B={bs[0]}$ to ${act_gap_hi:+.3f}$ at "
           f"$B={bs[-1]}$, though not at every step: it widens to ${act_worst:+.3f}$ at "
           f"$B={act_worst_b}$ first. The closing is what convergence of the two selections under "
           "a growing budget looks like and not what a constant handicap would. ")
        + "Every policy beats "
        f"the static baseline's ${stat:.3f}$ from the first budget onward, and the best point "
        f"anywhere on the grid is ${best['autc']:.3f}$ at "
        f"${tex_int(best['labels'])}$ labels, so the question is never whether to maintain the "
        "model but how to spend the labels"
    )
    write_generated(os.path.join(SEC, "gen_budget_frontier.tex"), body.rstrip() + "%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="use the table's three budgets instead of the dense grid")
    ap.add_argument("--cached", action="store_true",
                    help="re-emit the figure and prose from data/processed/p5/p5_budget_frontier.csv "
                         "without re-running the stream (prose edits only; no training)")
    args = ap.parse_args()

    os.makedirs(PROC, exist_ok=True)
    csv_path = os.path.join(PROC, "p5", "p5_budget_frontier.csv")
    if args.cached:
        rows = [{"budget": int(r["budget"]), "policy": r["policy"], "autc": float(r["autc"]),
                 "labels": int(r["labels"]), "registered": int(r["registered"])}
                for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8"))]
        grid = tuple(sorted({r["budget"] for r in rows}))
    else:
        grid = tuple(BUDGETS) if args.quick else tuple(sorted(set(DENSE) | set(BUDGETS)))
        rows = frontier(grid)
        write_csv(csv_path, rows)
    make_figure(rows)
    make_tex(rows)
    flip, lo, hi = crossover(rows)
    print(f"[i] {len(grid)} budgets; drift-vs-periodic {lo:+.3f} -> {hi:+.3f}, crossover at "
          f"B={flip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
