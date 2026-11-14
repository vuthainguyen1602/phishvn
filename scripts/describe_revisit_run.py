#!/usr/bin/env python3
"""
describe_revisit_run.py — the descriptive quantities the registration asks for and the bound
analysis script does not compute, plus two decompositions of its results. Post hoc: written on
2026-11-08 after `analyze_revisit.py --run` had been read. Outside the Benjamini-Hochberg family;
no conclusion of the paper rests on anything here.

Registered but missing from the bound code (PREREG_subdomain_evasion.md):
  * displacement (amendment of 2026-09-19 00:20): admitted `ct_brands` hosts that later appear in
    a registered source's detections.csv, and RQ1'-RQ3' repeated with all four sources pooled;
  * the number of self-induced hosts by admission day (amendment of 2026-09-19 23:41, item 4).
Decompositions, not registered:
  * the wildcard-free reading of RQ1' split into what it removes: wildcard-resolving and
    `undetermined` domains, per arm, with their 128 h retention;
  * how many confirmatory hosts have a hosted tenant key at all (RQ2' has an empty hosted arm).

It imports analyze_revisit.py and calls its functions unchanged, so every unit, key and
statistic is the bound file's. Writes data/processed/subdomain/revisit_descriptive.json
(aggregates only, no hostname).

RUN: python scripts/describe_revisit_run.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import analyze_revisit as ar  # noqa: E402

OUT = os.path.join(ar.ROOT, "data", "processed", "subdomain", "revisit_descriptive.json")
NAMED_DETECTIONS = {
    "urlscan_brands": ar.BRAND_DET,
    "vn_phishing_live": os.path.join(ar.RAW, "vn_phishing_live", "detections.csv"),
    "chongluadao_live": os.path.join(ar.RAW, "chongluadao_live", "detections.csv"),
}


def _named_hosts() -> set[str]:
    out: set[str] = set()
    for path in NAMED_DETECTIONS.values():
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                out |= {ar._nowww((r.get("domain") or "").strip().lower()) for r in csv.DictReader(f)}
    return out


def _retained(rows, obs, lag=128):
    seen = [obs[r["hostname"]][lag] for r in rows if lag in obs.get(r["hostname"], {})]
    return {"n": len(seen), "retained": sum(o["state"] == "public_a" for o in seen)}


def main() -> int:
    cohort, obs_rows = ar._read(ar.COHORT), ar._read(ar.OBS)
    induced = ar.self_induced_hosts()
    obs = ar.observations_by_host(obs_rows)
    tenant, icann, is_hosted = ar.key_functions()
    pop = ar.confirmatory(cohort, induced)
    out: dict = {"post_hoc": True, "outside_registered_family": True}

    # displacement and the four-source pooled reading
    named = _named_hosts()
    ct = [r for r in cohort if r["label"] == "phish" and r["source"] == "ct_brands"]
    out["displacement"] = {"ct_brands_phish_admitted": len(ct),
                           "later_in_a_registered_source": sum(ar._nowww(r["hostname"]) in named for r in ct)}
    pooled = [r for r in cohort if r["label"] == "phish" and r["source"] in ar.REGISTERED_SOURCES + ("ct_brands",)
              and r["hostname"] not in induced]
    pm = ar.landmark_multi(pooled, tenant)
    out["four_source_pooled"] = {"RQ1": ar.rq1(pooled, pm, obs, tenant, is_hosted),
                                 "RQ2": ar.rq2(pooled, pm, obs, tenant, is_hosted),
                                 "RQ3": ar.rq3(pooled, pm, obs, tenant)}

    # self-induced hosts by admission day
    si = ar.self_induced_rows(cohort, induced)
    out["self_induced_by_admission_day"] = dict(sorted(Counter(r["admitted_at"][:10] for r in si).items()))

    # hosted keys in the confirmatory population
    multi = ar.landmark_multi(pop, tenant)
    base = [r for r in pop if r["admission_state"] == "public_a"]
    out["hosted"] = {"public_a_at_admission": len(base),
                     "hosted_key": sum(is_hosted(tenant(r["hostname"])) for r in base),
                     "hosted_key_and_multi": sum(is_hosted(tenant(r["hostname"])) and multi[r["hostname"]] for r in base)}

    # what the wildcard-free reading removes, per arm of RQ1'
    last = {r["registered_domain"]: r for r in ar._read(ar.PROBE)}
    status = {d: r["status"] for d, r in last.items()}
    rq1_rows = [r for r in base if not is_hosted(tenant(r["hostname"]))]
    split: dict = {}
    for arm, flag in (("multi", True), ("solo", False)):
        rows = [r for r in rq1_rows if bool(multi[r["hostname"]]) is flag]
        for st in ("wildcard", "undetermined", "no_wildcard", "platform", "unprobed"):
            sel = [r for r in rows if status.get(icann(r["hostname"]), "unprobed") == st]
            split.setdefault(arm, {})[st] = {"hosts": len(sel), "at_128h": _retained(sel, obs)}
    out["wildcard_free_decomposition"] = split
    out["probe_status_counts"] = dict(Counter(status.values()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[+] {os.path.relpath(OUT, ar.ROOT)}")
    print(json.dumps({k: out[k] for k in ("displacement", "hosted", "wildcard_free_decomposition")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
