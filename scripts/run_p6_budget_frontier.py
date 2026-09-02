#!/usr/bin/env python3
"""
run_p6_budget_frontier.py — P6: the budget-allocation frontier behind "budget constraint, not
representational".

run_p6_group_threshold.py prices four named rules; a reviewer's next question is whether a fifth,
cleverer allocation of the same false-alarm budget is cheap on both groups. A per-group rule is a
pair (t_vn, t_other), so the reachable set is computable in closed form from the two ROC curves —
no search, no grid resolution to argue about. Two frontiers are drawn: an ORACLE (thresholds read
off the TEST curves, the most generous baseline an impossibility claim can face) and the DEPLOYABLE
curve, which must sit on or inside it.

PROTOCOL: split, model and seeds imported from run_p6_group_threshold.py (split_and_fit /
threshold_rules), so the published rules are marked at the coordinates the published table reports.

OUTPUT: data/processed/p6/p6_budget_frontier.csv        (oracle frontier, per seed)
        data/processed/p6/p6_budget_frontier_cal.csv    (deployable frontier, per seed)
        data/processed/p6/p6_budget_frontier_points.csv (the four published rules, per seed)
        data/processed/p6/p6_budget_parity.csv          (min budget for a common miss ceiling)
        papers/P6_xai/figures/fig_budget_frontier.pdf
        papers/P6_xai/sections/gen_budget_frontier.tex, gen_budget_verdict.tex

RUN:  python scripts/run_p6_budget_frontier.py --seeds 5
The closed form and why the budget is the corpus FPR: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
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
from genfile import write_generated
from train_url_baseline import COMPPHISH
from run_p2_temporal_strict import load
from run_p6_group_threshold import split_and_fit, threshold_rules

SEC = os.path.join(ROOT, "papers", "P6_xai", "sections")
FIG = os.path.join(ROOT, "papers", "P6_xai", "figures")
PROC = os.path.join(ROOT, "data", "processed")

# The .vn miss ceilings the table reports. 0.906 is the deployed rule's own .vn miss rate, kept as
# the last row so the table contains the status quo as a frontier point rather than only as prose.
TAUS_VN = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.906]
# Common ceilings for the parity question ("both groups below tau, what does it cost?").
TAUS_BOTH = [0.05, 0.10, 0.20, 0.30]
# The .vn-miss axis the per-seed frontiers are averaged on (a shared grid; averaging two staircases
# sampled at different places would smear the step structure into a curve that is nobody's).
TAU_GRID = np.unique(np.round(np.concatenate([np.linspace(0.0, 1.0, 101), TAUS_VN]), 3))


# --------------------------------------------------------------------------------------------
# Pure ROC arithmetic. Kept free of frames and models: these four functions carry the whole
# optimality argument, and tests/test_p6_frontier.py pins them against hand-built staircases.
# --------------------------------------------------------------------------------------------
def roc_arrays(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) of the empirical ROC staircase, both non-decreasing."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def min_fpr_for_miss(fpr: np.ndarray, tpr: np.ndarray, tau: float) -> float:
    """Cheapest false-alarm rate in a group that still misses at most `tau` of its phishing.

    A miss rate of `tau` is a recall of `1 - tau`; the ROC staircase is non-decreasing in both
    coordinates, so the first point clearing that recall is the cheapest one that does."""
    i = int(np.searchsorted(tpr, 1.0 - tau - 1e-12, side="left"))
    if i >= len(tpr):                      # unreachable even at FPR 1 (cannot happen here)
        return 1.0
    return float(fpr[i])


def min_miss_for_fpr(fpr: np.ndarray, tpr: np.ndarray, budget: float) -> float:
    """Lowest miss rate a group can reach without spending more than `budget` false alarms."""
    if budget < 0:
        return 1.0                          # the other group already overspent the whole budget
    i = int(np.searchsorted(fpr, budget + 1e-12, side="right")) - 1
    if i < 0:
        return 1.0
    return float(1.0 - tpr[i])


def frontier_curve(roc_vn, roc_ot, pi_vn: float, b: float,
                   taus: np.ndarray) -> np.ndarray:
    """miss_other at each .vn miss ceiling, under the budget b. The optimal allocation is forced:
    buy the .vn ceiling at the cheapest FPR the .vn curve offers, then spend the entire remainder
    on the other group (spending less is dominated, spending more breaks the constraint)."""
    pi_ot = 1.0 - pi_vn
    out = np.empty(len(taus))
    for k, tau in enumerate(taus):
        f_vn = min_fpr_for_miss(roc_vn[0], roc_vn[1], float(tau))
        rem = (b - pi_vn * f_vn) / pi_ot if pi_ot > 0 else -1.0
        out[k] = min_miss_for_fpr(roc_ot[0], roc_ot[1], rem)
    return out


def min_budget_for_both(roc_vn, roc_ot, pi_vn: float, tau: float) -> float:
    """The smallest corpus-level FPR at which BOTH groups miss at most `tau` --- the price of
    parity, and the quantity that turns ``budget constraint'' from a reading into a number."""
    f_vn = min_fpr_for_miss(roc_vn[0], roc_vn[1], tau)
    f_ot = min_fpr_for_miss(roc_ot[0], roc_ot[1], tau)
    return float(pi_vn * f_vn + (1.0 - pi_vn) * f_ot)


def is_pareto_frontier(miss_vn: np.ndarray, miss_ot: np.ndarray) -> bool:
    """No point is dominated by another (dominated = at least as good on both, better on one)."""
    for i in range(len(miss_vn)):
        for j in range(len(miss_vn)):
            if i == j:
                continue
            if (miss_vn[j] <= miss_vn[i] + 1e-12 and miss_ot[j] <= miss_ot[i] + 1e-12
                    and (miss_vn[j] < miss_vn[i] - 1e-12 or miss_ot[j] < miss_ot[i] - 1e-12)):
                return False
    return True


# --------------------------------------------------------------------------------------------
def one_seed(df, feats, family: str, seed: int, cal_frac: float):
    _tr, _m, cal, te = split_and_fit(df, feats, family, seed, cal_frac)
    cal_be = cal[cal.y == 0]
    thr, alpha = threshold_rules(cal_be)

    y = te.y.to_numpy(int)
    s = te.pred.to_numpy(float)
    g = te.vn.to_numpy(bool)
    be = y == 0
    pi_vn = float((g & be).sum() / be.sum())        # the group's share of the BENIGN pool

    roc_vn = roc_arrays(y[g], s[g])
    roc_ot = roc_arrays(y[~g], s[~g])

    # the deployed budget: the 0.5 rule's own realised corpus FPR on this seed's test set
    b0 = float((s[be] >= 0.5).mean())

    # ---- the four published rules, at the coordinates the published table reports
    pts = []
    for cond, t in thr.items():
        pred = s >= np.where(g, t["vn"], t["other"])
        pts.append({
            "seed": seed, "condition": cond,
            "miss_vn": float((pred[g & (y == 1)] == 0).mean()),
            "miss_other": float((pred[~g & (y == 1)] == 0).mean()),
            "fpr_vn": float((pred[g & be] == 1).mean()),
            "fpr_other": float((pred[~g & be] == 1).mean()),
            "budget": float((pred[be] == 1).mean()),
            "thr_vn": t["vn"], "thr_other": t["other"],
        })
    ppts = pd.DataFrame(pts)
    b_vnonly = float(ppts[ppts.condition == "vn-only"].budget.iloc[0])

    # ---- ORACLE frontier at each budget level (thresholds read off the test curves: an upper
    # bound on what any allocation of that budget can achieve, deployable or not)
    budgets = {"half": 0.5 * b0, "deployed": b0, "double": 2.0 * b0, "vn-only": b_vnonly}
    fr = []
    for bname, b in budgets.items():
        miss_ot = frontier_curve(roc_vn, roc_ot, pi_vn, b, TAU_GRID)
        for tau, mo in zip(TAU_GRID, miss_ot):
            fr.append({"seed": seed, "budget_name": bname, "budget": b,
                       "tau_vn": float(tau), "miss_other": float(mo), "pi_vn": pi_vn})
    frontier = pd.DataFrame(fr)

    # ---- DEPLOYABLE frontier at the deployed budget: thresholds are calibration-benign
    # quantiles, the same instrument the published rules use, so this curve is reachable. The
    # budget is booked in CALIBRATION units (all a deployer can observe); the realised test
    # budget is recorded next to it so the drift is visible rather than assumed away.
    cal_vn = cal_be[cal_be.vn].pred.to_numpy(float)
    cal_ot = cal_be[~cal_be.vn].pred.to_numpy(float)
    pi_vn_cal = float(len(cal_vn) / len(cal_be))
    cal_rows = []
    for a_vn in np.round(np.linspace(0.0, 1.0, 101), 3):
        a_ot = (b0 - pi_vn_cal * a_vn) / (1.0 - pi_vn_cal)
        if a_ot < 0:
            continue
        t_vn = float(np.quantile(cal_vn, 1.0 - a_vn)) if len(cal_vn) else 0.5
        t_ot = float(np.quantile(cal_ot, 1.0 - min(a_ot, 1.0))) if len(cal_ot) else 0.5
        pred = s >= np.where(g, t_vn, t_ot)
        cal_rows.append({
            "seed": seed, "alpha_vn": float(a_vn), "alpha_other": float(a_ot),
            "thr_vn": t_vn, "thr_other": t_ot,
            "miss_vn": float((pred[g & (y == 1)] == 0).mean()),
            "miss_other": float((pred[~g & (y == 1)] == 0).mean()),
            "budget_test": float((pred[be] == 1).mean()),
        })
    caldf = pd.DataFrame(cal_rows)

    # ---- price of parity
    par = pd.DataFrame([{"seed": seed, "tau": t,
                         "b_star": min_budget_for_both(roc_vn, roc_ot, pi_vn, t),
                         "b0": b0}
                        for t in TAUS_BOTH])
    return frontier, caldf, ppts, par, b0, pi_vn


def make_figure(fr: pd.DataFrame, cal: pd.DataFrame, pts: pd.DataFrame, par: pd.DataFrame,
                family: str, seeds: int):
    """Two panels, one argument. LEFT: the reachable set. Every (t_vn, t_other) pair at the
    deployed budget lands on or above the solid curve, the four published rules included, and the
    curve never approaches the origin --- that shape IS ``no allocation is cheap on both groups''.
    RIGHT: what parity would cost, as a function of how good you insist on being; the deployed
    budget is a horizontal line the requirement crosses almost immediately."""
    from figstyle import apply, BLUE, ORANGE, TEAL, GRAY, INK
    plt = apply()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5))

    # --- left panel: frontiers
    style = {"half": (GRAY, (0, (1, 2)), 1.0), "deployed": (INK, "-", 1.9),
             "double": (GRAY, (0, (5, 2)), 1.1), "vn-only": (GRAY, (0, (3, 1, 1, 1)), 1.1)}
    name = {"half": "half budget", "deployed": "deployed budget",
            "double": "double budget", "vn-only": ".vn-only's budget"}
    # Four nested frontiers all crowd the top-left corner, so every automatic placement collides:
    # each label is PLACED in data coordinates in the empty lower-right and led back to a named
    # point on its own curve. ("x", v) anchors where the curve runs flat, ("y", v) where it runs
    # steep. What was tried and why it failed: kept in the development repository, not shipped in this mirror
    anchor = {"half": (0.62, (0.66, 0.46)), "deployed": (0.44, (0.48, 0.28)),
              "double": (0.42, (0.53, 0.17)), "vn-only": (0.72, (0.62, 0.04))}
    for bname in ("half", "deployed", "double", "vn-only"):
        sub = fr[fr.budget_name == bname]
        m = sub.groupby("tau_vn").miss_other.mean()
        bmean = sub.budget.mean()
        c, ls, lw = style[bname]
        xs, ys = m.to_numpy(), m.index.to_numpy()
        ax.plot(xs, ys, color=c, ls=ls, lw=lw, zorder=3)
        if bname == "deployed":
            # everything to the LEFT of the deployed frontier is unbuyable at that budget --- the
            # ideal corner included, which is the claim the panel exists to make
            ax.fill_betweenx(ys, 0, xs, color=GRAY, alpha=0.13, lw=0, zorder=1)
        xat, xytext = anchor[bname]
        j = int(np.argmin(np.abs(xs - xat)))
        # Four nested curves in one panel means a label placed in a gap for one budget lies on
        # another budget's line. A white patch behind the text is the only placement-independent
        # fix; zorder above the curves so it actually clears them.
        ax.annotate(f"{name[bname]} (FPR {bmean:.2f})", (xs[j], ys[j]),
                    textcoords="data", xytext=xytext, fontsize=7.5, color=c, va="center",
                    zorder=6,
                    # Fully opaque, not 0.82: at that alpha a curve passing behind the label
                    # still showed through the text as a grey stroke.
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.55,
                                    shrinkA=0, shrinkB=3))

    # Dashed over the solid oracle curve, because the two coincide almost everywhere: the
    # selection optimism in reading thresholds off the test curves turns out to be worth nearly
    # nothing here, and a solid line would simply hide the bound it is being compared against.
    cg = cal.groupby("alpha_vn")[["miss_vn", "miss_other"]].mean()
    ax.plot(cg.miss_other.to_numpy(), cg.miss_vn.to_numpy(), color=ORANGE, lw=1.4,
            ls=(0, (4, 2)), alpha=0.95, zorder=5)
    ax.annotate("reachable by calibration\n(deployed budget)", (0.43, 0.62),
                fontsize=7, color=ORANGE)

    mstyle = {"default": (BLUE, "o"), "global": (BLUE, "s"),
              "per-group": (ORANGE, "^"), "vn-only": (TEAL, "D")}
    mlabel = {"default": "global $0.5$", "global": "global, matched",
              "per-group": "per-group", "vn-only": ".vn-only"}
    # Coinciding rules share a marker (default and matched-budget global do, by construction) ---
    # a shared operating point is a finding, not a collision. Same convention as fig_group_roc.
    placed: dict = {}
    for cond, r in pts.groupby("condition"):
        placed.setdefault((round(r.miss_other.mean(), 2), round(r.miss_vn.mean(), 2)),
                          []).append(cond)
    # per-group has empty space only OUTSIDE its own frontier: down-left lands on the
    # .vn-only label, up-left on the curve itself.
    off = {"default": (10, -14), "per-group": (14, 4), "vn-only": (14, 10)}
    for (x, yv), conds in placed.items():
        c, mk = mstyle[conds[0]]
        ax.plot(x, yv, mk, color=c, ms=7, mec="white", mew=0.8, zorder=6)
        dx, dy = off.get(conds[0], (10, 4))
        # Boxed opaquely: the dotted budget contours pass behind these operating-point labels.
        ax.annotate(" = ".join(mlabel[c0] for c0 in conds), (x, yv),
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="right" if dx < 0 else "left",
                    fontsize=7, color=c, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.55,
                                    shrinkA=0, shrinkB=4))
    ax.set_xlabel("miss rate on non-.vn phishing")
    ax.set_ylabel("miss rate on .vn phishing")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    # Raised above the .vn-only marker's label, which sits at +10 pt of the diamond near the
    # origin; at +16 pt the two printed through each other once the corner was spelt out.
    ax.annotate("ideal\n(no misses)", (0, 0), textcoords="offset points", xytext=(4, 18),
                fontsize=7.5, va="bottom",
                style="italic", color=INK)
    ax.set_title("reachable (miss$_{.vn}$, miss$_{other}$) by budget", fontsize=8.5)

    # --- right panel: price of parity
    pg = par.groupby("tau").b_star
    x = np.array(sorted(par.tau.unique()))
    ymean, ysd = pg.mean().to_numpy(), pg.std().to_numpy()
    b0 = par.b0.mean()
    ax2.errorbar(x, ymean, yerr=ysd, fmt="o-", color=ORANGE, lw=1.6, ms=6, capsize=3,
                 mec="white", mew=0.8, zorder=4)
    ax2.axhline(b0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax2.annotate(f"deployed budget (FPR {b0:.2f})", (x[0], b0), zorder=6,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
                 textcoords="offset points",
                 xytext=(0, -12), fontsize=7, color=INK, ha="left")
    for xi, yi in zip(x, ymean):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                     fontsize=7, color=ORANGE, ha="center")
    ax2.set_xlabel("common miss ceiling $\\tau$ (both groups)")
    ax2.set_ylabel("minimum corpus false-alarm rate")
    ax2.set_ylim(0, max(1.0, float(np.nanmax(ymean + np.nan_to_num(ysd))) * 1.15))
    ax2.set_title("price of parity", fontsize=8.5)

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{family}, phishing-temporal test, {seeds} seeds", fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_budget_frontier.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[+] {out}")


def make_tex(fr: pd.DataFrame, cal: pd.DataFrame, pts: pd.DataFrame, par: pd.DataFrame,
             family: str, seeds: int):
    dep = fr[fr.budget_name == "deployed"]
    b0 = dep.budget.mean()
    pi_vn = dep.pi_vn.mean()

    def at(tau: float) -> tuple[float, float, float, float]:
        sub = dep[np.isclose(dep.tau_vn, round(tau, 3))]
        if not len(sub):                      # ceiling off the shared grid: nearest grid point
            k = (dep.tau_vn - tau).abs().idxmin()
            sub = dep[np.isclose(dep.tau_vn, dep.loc[k, "tau_vn"])]
        cs = cal.groupby("alpha_vn")[["miss_vn", "miss_other"]].mean()
        reach = cs[cs.miss_vn <= tau + 1e-9].miss_other
        return (sub.miss_other.mean(), sub.miss_other.std(),
                float(reach.min()) if len(reach) else float("nan"), float(sub.tau_vn.iloc[0]))

    body = []
    for tau in TAUS_VN:
        mo, sd, cal_mo, used = at(tau)
        cal_txt = "n/a" if np.isnan(cal_mo) else f"{cal_mo:.3f}"
        body.append(f"$\\leq {used:.3f}$ & {mo:.3f}{{\\scriptsize\\,$\\pm${sd:.3f}}} & "
                    f"{cal_txt} \\\\")
    pbody = []
    for tau in TAUS_BOTH:
        g = par[np.isclose(par.tau, tau)]
        pbody.append(f"both $\\leq {tau:.2f}$ & {g.b_star.mean():.3f}"
                     f"{{\\scriptsize\\,$\\pm${g.b_star.std():.3f}}} & "
                     f"$\\times{g.b_star.mean() / b0:.1f}$ \\\\")

    # the best BALANCED point the deployed budget allows: the frontier's own minimax
    dg = dep.groupby("tau_vn").miss_other.mean()
    mm = float(np.min(np.maximum(dg.index.to_numpy(), dg.to_numpy())))
    dflt = pts[pts.condition == "default"]

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Budget-allocation frontier over \\emph{{all}} per-group threshold pairs
({family}, phishing-temporal test, {seeds} seeds). Top: lowest miss elsewhere under each
\\texttt{{.vn}} ceiling at the deployed FPR ${b0:.3f}$; best balanced point ${mm:.3f}$.
Bottom: price of parity.}}
\\label{{tab:budgetfrontier}}
\\small\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l c c}}
\\toprule
\\multicolumn{{3}}{{l}}{{\\emph{{At the deployed budget, FPR $={b0:.3f}$}}}} \\\\
\\texttt{{.vn}} miss ceiling & best miss other (oracle) & (calibrated) \\\\
\\midrule
{chr(10).join(body)}
\\midrule
\\multicolumn{{3}}{{l}}{{\\emph{{Price of parity}}}} \\\\
Requirement & min.\\ corpus FPR & vs deployed \\\\
\\midrule
{chr(10).join(pbody)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "gen_budget_frontier.tex"), tex)

    tau10 = par[np.isclose(par.tau, 0.10)]
    mo10, _, cal10, _ = at(0.10)
    verdict = (
        f"Exhausting the space the four rules sample from (every pair of per-group thresholds, "
        f"scored against the budget identity $\\mathrm{{FPR}} = \\pi_{{vn}}\\mathrm{{FPR}}_{{vn}} "
        f"+ \\pi_{{other}}\\mathrm{{FPR}}_{{other}}$ with $\\pi_{{vn}} = {pi_vn:.3f}$) leaves "
        f"the conclusion intact and makes it exhaustive rather than illustrative "
        f"(Table~\\ref{{tab:budgetfrontier}}, Figure~\\ref{{fig:budgetfrontier}}). At the deployed "
        f"false-alarm budget of ${b0:.3f}$ the best allocation that holds \\texttt{{.vn}} misses "
        f"to $10\\%$ concedes ${mo10:.3f}$ of non-\\texttt{{.vn}} phishing, and the most balanced "
        f"point the budget admits still misses ${mm:.3f}$ of \\emph{{both}} groups; bringing both "
        f"below $10\\%$ requires a corpus false-alarm rate of "
        f"${tau10.b_star.mean():.3f}\\,\\pm\\,{tau10.b_star.std():.3f}$, "
        f"$\\times{tau10.b_star.mean() / b0:.1f}$ the deployed budget. These frontiers are "
        f"\\emph{{oracle}} bounds (their thresholds are read off the test curves, which no "
        f"deployment can do), so the impossibility they state is the generous form of it: what "
        f"a calibrated rule reaches (the second column of Table~\\ref{{tab:budgetfrontier}}) is "
        f"weaker still, and the published rules of Table~\\ref{{tab:groupthr}} sit on that "
        f"reachable curve rather than inside it. The blind spot is thus not one badly-chosen "
        f"operating point among good ones: at this budget there is no good one")
    write_generated(os.path.join(SEC, "gen_budget_verdict.tex"), verdict.rstrip() + "%")
    return b0, pi_vn, mm, float(tau10.b_star.mean()), mo10, cal10, float(dflt.miss_vn.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="CatBoost")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    FR, CAL, PTS, PAR = [], [], [], []
    for s in range(args.seeds):
        fr, cal, pts, par, b0, pi = one_seed(df, feats, args.family, s, args.cal_frac)
        FR.append(fr)
        CAL.append(cal)
        PTS.append(pts)
        PAR.append(par)
        print(f"[i] seed {s}: budget(deployed FPR)={b0:.3f} pi_vn={pi:.3f} "
              f"b*(both<=0.10)={par[np.isclose(par.tau, 0.10)].b_star.iloc[0]:.3f}")
    fr = pd.concat(FR, ignore_index=True)
    cal = pd.concat(CAL, ignore_index=True)
    pts = pd.concat(PTS, ignore_index=True)
    par = pd.concat(PAR, ignore_index=True)
    os.makedirs(PROC, exist_ok=True)
    for name, d in (("p6_budget_frontier", fr), ("p6_budget_frontier_cal", cal),
                    ("p6_budget_frontier_points", pts), ("p6_budget_parity", par)):
        d.to_csv(os.path.join(PROC, name + ".csv"), index=False)
        print(f"[+] data/processed/{name}.csv")

    make_figure(fr, cal, pts, par, args.family, args.seeds)
    b0, pi_vn, mm, bstar10, mo10, cal10, miss_default = make_tex(
        fr, cal, pts, par, args.family, args.seeds)
    print(f"[+] deployed budget FPR={b0:.3f}  pi_vn={pi_vn:.3f}")
    print(f"[+] best balanced point at that budget: both groups miss {mm:.3f}")
    print(f"[+] miss_vn<=0.10 costs miss_other={mo10:.3f} (oracle) / {cal10:.3f} (calibrated)")
    print(f"[+] both<=0.10 needs corpus FPR {bstar10:.3f} = x{bstar10 / b0:.1f} the budget")


if __name__ == "__main__":
    main()
