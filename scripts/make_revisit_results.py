#!/usr/bin/env python3
"""
make_revisit_results.py — the registered revisit run of the subdomain study, as LaTeX macros.

WHY. Section "Results of the prospective cohort" reports one run of the bound analysis
(PREREG_subdomain_evasion.md, run record of 2026-11-08). Typed into the prose, each number would
be a second copy of the outputs that nothing checks; here the prose holds macros and this script
is the only reader. Retention is written as a percentage with one decimal, differences and
intervals in percentage points, p-values to three places.

It reads outputs and computes no contrast. Inputs: revisit_analysis.json and
revisit_sensitivity_network.json (committed), the brand-owned, stale-feed and withdrawn
sensitivity outputs (local: they list hostnames), and revisit_descriptive.json (the post hoc
descriptive script). Without any of them it stops rather than write a partial file.

RUN: python scripts/make_revisit_results.py
"""
from __future__ import annotations

import json
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

from genfile import write_generated  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed", "subdomain")
OUT = os.path.join(ROOT, "papers", "future_subdomain_evasion", "sections", "gen_revisit.tex")
INPUTS = ("revisit_analysis.json", "revisit_sensitivity_network.json",
          "revisit_sensitivity_brandowned.json", "revisit_sensitivity_stalefeed.json",
          "revisit_sensitivity_withdrawn.json", "revisit_descriptive.json")


def _n(x: int) -> str:
    return f"{x:,}"


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}" if n else "--"


def _pp(x: float) -> str:
    return f"{100 * x:+.1f}".replace("-", "$-$")


def _ci(ci) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]".replace("-", "$-$")


def _p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _two_by_two(prefix: str, d: dict) -> dict:
    e, u = d["exposed"], d["unexposed"]
    return {f"{prefix}MultiK": _n(e["retained"]), f"{prefix}MultiN": _n(e["n"]),
            f"{prefix}SoloK": _n(u["retained"]), f"{prefix}SoloN": _n(u["n"]),
            f"{prefix}MultiPct": _pct(e["retained"], e["n"]), f"{prefix}SoloPct": _pct(u["retained"], u["n"]),
            f"{prefix}Diff": _pp(d["difference"]), f"{prefix}CI": _ci(d["newcombe95"]),
            f"{prefix}P": _p(d["fisher_p"]), f"{prefix}Missing": _n(d["missing_observation"])}


def _rq3(prefix: str, d: dict) -> dict:
    return {f"{prefix}Hosts": _n(d["hosts"]), f"{prefix}MultiN": _n(d["multi"]["n"]),
            f"{prefix}MultiExact": _n(d["multi"]["exact"]), f"{prefix}MultiExpansion": _n(d["multi"]["expansion"]),
            f"{prefix}SoloN": _n(d["solo"]["n"]), f"{prefix}SoloExact": _n(d["solo"]["exact"]),
            f"{prefix}Gap": f"{d['difference_in_mean_gap']:.4f}", f"{prefix}P": _p(d["permutation_p"]),
            f"{prefix}Missing": _n(d["missing_observation"])}


def macros(a: dict, net: dict, bo: dict, sf: dict, wd: dict, ds: dict) -> dict:
    st, c, sens = a["status"], a["confirmatory"], a["sensitivity"]
    m = {"RvCohort": _n(st["cohort"]), "RvPhish": _n(st["by_label"]["phish"]),
         "RvBenign": _n(st["by_label"]["benign"]), "RvConfirmatory": _n(st["confirmatory_phish"]),
         "RvSelfInduced": _n(st["self_induced_excluded"]), "RvObservations": _n(st["observations"]),
         "RvLate": _n(st["late"]), "RvLatePct": f"{100 * st['late_rate']:.2f}",
         "RvCTBrands": _n(st["by_source"].get("ct_brands", 0)),
         "RvUrlscan": _n(st["by_source"].get("urlscan_brands", 0)),
         "RvFeed": _n(st["by_source"].get("vn_phishing_live", 0)),
         "RvChongluadao": _n(st["by_source"].get("chongluadao_live", 0)),
         "RvCompleted": _n(a["stopping_rule"]["completed_512h"]),
         "RvFirstDay": min(st["admissions_by_day"]), "RvLastDay": max(st["admissions_by_day"])}
    weeks = st["control_128h_by_week"]
    rets = [w["retention"] for w in weeks.values()]
    m.update({"RvControlWeeks": len(weeks), "RvControlMin": f"{100 * min(rets):.1f}",
              "RvControlMax": f"{100 * max(rets):.1f}",
              "RvSuspectWeeks": len(sens["vantage_suspect_weeks"]),
              "RvFirstWeek": min(weeks), "RvLastWeek": max(weeks)})

    r1 = c["RQ1"]
    m["RvQOnePop"] = _n(r1["population"])
    m.update(_two_by_two("RvQOne", r1["primary_128h"]))
    m.update(_two_by_two("RvQOneTwoFiveSix", r1["descriptive"][0]))
    m.update(_two_by_two("RvQOneFiveTwelve", r1["descriptive"][1]))
    r2 = c["RQ2"]
    m.update({"RvQTwoPop": _n(r2["population"]), "RvQTwoHostedN": _n(r2["primary_256h"]["exposed"]["n"]),
              "RvQTwoNotHostedK": _n(r2["primary_256h"]["unexposed"]["retained"]),
              "RvQTwoNotHostedN": _n(r2["primary_256h"]["unexposed"]["n"]),
              "RvQTwoMissing": _n(r2["primary_256h"]["missing_observation"])})
    m.update(_rq3("RvQThree", c["RQ3"]))
    exact_all = c["RQ3"]["multi"]["exact"] + c["RQ3"]["solo"]["exact"]
    m.update({"RvQThreeExactAll": _n(exact_all), "RvQThreeExactPct": _pct(exact_all, c["RQ3"]["hosts"]),
              "RvQThreeExtra": _n(c["RQ3"]["multi"]["expansion"] - c["RQ3"]["multi"]["exact"])})
    bh = c["bh_adjusted"]
    m.update({"RvQOnePBH": _p(bh.get("RQ1")), "RvQThreePBH": _p(bh.get("RQ3"))})

    # registered sensitivities and the wildcard-free reading
    wf = sens["wildcard_free"]
    m["RvWfPop"] = _n(wf["RQ1"]["population"])
    m.update(_two_by_two("RvWf", wf["RQ1"]["primary_128h"]))
    m.update(_two_by_two("RvWfTwoFiveSix", wf["RQ1"]["descriptive"][0]))
    m.update(_two_by_two("RvWfFiveTwelve", wf["RQ1"]["descriptive"][1]))
    m.update({"RvMultiUnderWildcard": _n(wf["multi_hosts_under_wildcard"]), "RvMultiHosts": _n(wf["multi_hosts"]),
              "RvMultiUnderWildcardPct": _pct(wf["multi_hosts_under_wildcard"], wf["multi_hosts"])})
    m.update(_two_by_two("RvLedger", sens["multi_by_whole_ledger"]["RQ1"]["primary_128h"]))
    m.update(_two_by_two("RvInduced", sens["self_induced_included"]["RQ1"]["primary_128h"]))
    m.update(_rq3("RvInducedQThree", sens["self_induced_included"]["RQ3"]))
    late_same = sens["late_excluded"]["RQ1"]["primary_128h"] == r1["primary_128h"] and \
        sens["late_excluded"]["RQ3"]["multi"] == c["RQ3"]["multi"]
    m["RvLateSame"] = "identical" if late_same else "different"
    for d, name in ((bo, "BrandOwned"), (sf, "StaleFeed"), (wd, "Withdrawn")):
        m.update(_two_by_two(f"Rv{name}", d["confirmatory"]["RQ1"]["primary_128h"]))
        m.update(_rq3(f"Rv{name}QThree", d["confirmatory"]["RQ3"]))
        m[f"Rv{name}Removed"] = _n(len(d["removed_hosts"]))
    for cohort, name in (("old_network", "Old"), ("new_network", "New")):
        m.update(_two_by_two(f"RvNet{name}", net[cohort]["confirmatory"]["RQ1"]["primary_128h"]))

    # per source
    for src, name in (("urlscan_brands", "Urlscan"), ("vn_phishing_live", "Feed")):
        m.update(_two_by_two(f"RvSrc{name}", a["per_source"][src]["RQ1"]["primary_128h"]))

    # ct_brands stratum
    ct = a["strata"]["ct_brands"]
    m.update({"RvCTPublic": _n(ct["public_a_at_admission"]),
              "RvCTOneTwoEightK": _n(ct["by_lag"]["128"]["public_a"]),
              "RvCTOneTwoEightN": _n(ct["by_lag"]["128"]["observed"]),
              "RvCTOneTwoEightPct": _pct(ct["by_lag"]["128"]["public_a"], ct["by_lag"]["128"]["observed"]),
              "RvCTFiveTwelveK": _n(ct["by_lag"]["512"]["public_a"]),
              "RvCTFiveTwelveN": _n(ct["by_lag"]["512"]["observed"]),
              "RvCTFiveTwelvePct": _pct(ct["by_lag"]["512"]["public_a"], ct["by_lag"]["512"]["observed"])})
    si = a["strata"]["self_induced"]
    m.update({"RvSIPublic": _n(si["public_a_at_admission"]),
              "RvSIFiveTwelveK": _n(si["by_lag"]["512"]["public_a"]),
              "RvSIDays": len(ds["self_induced_by_admission_day"])})

    # post hoc descriptive
    dp = ds["displacement"]
    m.update({"RvDispCT": _n(dp["ct_brands_phish_admitted"]), "RvDispLater": _n(dp["later_in_a_registered_source"])})
    pool = ds["four_source_pooled"]
    m.update(_two_by_two("RvPool", pool["RQ1"]["primary_128h"]))
    m.update(_rq3("RvPoolQThree", pool["RQ3"]))
    m.update({"RvPoolQTwoHostedN": _n(pool["RQ2"]["primary_256h"]["exposed"]["n"])})
    h = ds["hosted"]
    m.update({"RvHostedBase": _n(h["public_a_at_admission"]), "RvHostedKey": _n(h["hosted_key"]),
              "RvHostedMulti": _n(h["hosted_key_and_multi"])})
    dec = ds["wildcard_free_decomposition"]
    for arm, name in (("multi", "Multi"), ("solo", "Solo")):
        for stt, sname in (("wildcard", "Wild"), ("undetermined", "Undet"), ("no_wildcard", "Clean")):
            cell = dec[arm][stt]["at_128h"]
            m[f"RvDec{name}{sname}N"] = _n(cell["n"])
            m[f"RvDec{name}{sname}K"] = _n(cell["retained"])
            m[f"RvDec{name}{sname}Pct"] = _pct(cell["retained"], cell["n"])
    return m


def load() -> tuple:
    missing = [f for f in INPUTS if not os.path.exists(os.path.join(PROC, f))]
    if missing:
        raise SystemExit(f"missing outputs of the registered run: {', '.join(missing)}")
    return tuple(json.loads(open(os.path.join(PROC, f), encoding="utf-8").read().replace("NaN", "null"))
                 for f in INPUTS)


def render(m: dict) -> str:
    lines = ["% GENERATED by scripts/make_revisit_results.py from the",
             "% registered run's outputs. Do not edit: rerun the generator."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    write_generated(OUT, render(macros(*load())))
    print(f"[+] {os.path.relpath(OUT, ROOT)}")
