#!/usr/bin/env python3
"""
analyze_p3.py — Statistical analysis for the P3 study (phishing susceptibility & training effect).

Expects a long-format CSV (one row per participant per wave), columns (see data/docs codebook):
  pid, group (0 control/1 treatment), wave (T0/T1), clicked (0/1), reported (0/1)
  optional covariates: sex, year, major, dig_skill, channel

Runs:
  H1/H3 (between groups at T1): two-proportion chi-square + risk ratio + Cohen's h
  H2 (within group, pre vs post): McNemar
  H4/H5: mixed-effects logistic regression (clicked ~ group*wave + covariates, random intercept per pid/class)
  Effect sizes + 95% CIs where available.

INSTALL: pip install pandas scipy statsmodels
RUN:  python analyze_p3.py --in p3_results.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from math import asin, sqrt
from scipy.stats import chi2_contingency


def cohens_h(p1, p2):
    return 2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2))


def two_prop(df, outcome):
    """Between-group comparison at T1."""
    d = df[df.wave.astype(str).str.upper() == "T1"]
    tab = pd.crosstab(d.group, d[outcome])
    if tab.shape != (2, 2):
        print(f"  [{outcome}@T1] need 2x2 table; got {tab.shape}"); return
    chi2, p, _, _ = chi2_contingency(tab, correction=False)
    p_t = d[d.group == 1][outcome].mean()
    p_c = d[d.group == 0][outcome].mean()
    rr = (p_t / p_c) if p_c else float("nan")
    print(f"  [{outcome} @T1] treat={p_t:.3f} control={p_c:.3f}  chi2={chi2:.3f} p={p:.4f} "
          f"RR={rr:.3f} h={cohens_h(p_t,p_c):+.3f}")


def mcnemar_within(df, outcome, group):
    """Pre/post within a group (paired by pid)."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except Exception:
        print("  (statsmodels needed for McNemar)"); return
    d = df[df.group == group]
    piv = d.pivot_table(index="pid", columns=d.wave.astype(str).str.upper(),
                        values=outcome, aggfunc="first")
    if not {"T0", "T1"}.issubset(piv.columns):
        print("  (need T0 & T1 per pid)"); return
    piv = piv.dropna(subset=["T0", "T1"])
    b = int(((piv.T0 == 1) & (piv.T1 == 0)).sum())   # improved
    c = int(((piv.T0 == 0) & (piv.T1 == 1)).sum())   # worsened
    res = mcnemar([[0, b], [c, 0]], exact=True)
    grp = "treatment" if group == 1 else "control"
    print(f"  [{outcome} pre/post, {grp}] improved={b} worsened={c} p={res.pvalue:.4f}")


def mixed_logit(df, outcome):
    """GEE logistic (Binomial) with exchangeable correlation clustered by participant —
    the correct model for repeated BINARY outcomes; reports odds ratios."""
    try:
        import numpy as _np
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception:
        print("  (statsmodels needed)"); return
    d = df.copy()
    d["wave_post"] = (d.wave.astype(str).str.upper() == "T1").astype(int)
    covs = [c for c in ("sex", "year", "dig_skill", "channel") if c in d.columns]
    formula = f"{outcome} ~ group * wave_post" + ("".join(f" + C({c})" for c in covs))
    try:
        m = smf.gee(formula, groups="pid", data=d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        print(f"  GEE logit ({outcome}); key terms (OR = exp(beta)):")
        for name in ("group", "wave_post", "group:wave_post"):
            if name in m.params.index:
                print(f"    {name}: beta={m.params[name]:+.3f} OR={_np.exp(m.params[name]):.3f} "
                      f"p={m.pvalues[name]:.4f}")
    except Exception as e:
        print(f"  (GEE failed: {type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    for c in ("group", "clicked", "reported"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"n_rows={len(df)} participants={df.pid.nunique()} "
          f"(treat={int((df.group==1).sum())}, control={int((df.group==0).sum())})")
    print("== H1/H3: between-group at T1 ==")
    two_prop(df, "clicked"); two_prop(df, "reported")
    print("== H2: within-group pre/post (McNemar) ==")
    mcnemar_within(df, "clicked", 1); mcnemar_within(df, "clicked", 0)
    print("== H4/H5: mixed-effects (group*wave + covariates) ==")
    mixed_logit(df, "clicked")


if __name__ == "__main__":
    main()
