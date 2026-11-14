#!/usr/bin/env python3
"""
sensitivity_revisit_brandowned.py — the registered revisit analysis without brand-owned hosts.

Registered by the amendment of 2026-10-02 (brand-owned hosts) in PREREG_subdomain_evasion.md. On
2026-10-02 at least twelve hosts in the confirmatory phishing arm were shown, from registrar
records, to belong to the brand they carry (bidvinfo.com.vn, media-vnpt.vnptvas.vn and ten
viettelpost.vn hosts). The bound analysis keeps them, as admitted. This script reruns
`analyze_revisit.analyse`, unchanged, with them removed, and reports the result outside the
registered family.

WHICH HOSTS. A phishing-labelled host is removed when it equals, or is a subdomain of, a name in
the dated table of kept in the development repository, not shipped in this mirror whose date is ON OR BEFORE the day
the registered stopping rule is met. A name decided later does not enter, so the set cannot be
chosen after the result is known. The stopping day is the day the STOP_N-th confirmatory host
with a public A record at admission completed its 512 h observation, or STOP_DATE, whichever is
earlier.

Gate: it refuses until the registered stopping rule holds on the full data.

RUN: python scripts/sensitivity_revisit_brandowned.py [--run]
"""
import argparse
import json
import os
import re
import sys
from datetime import date, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analyze_revisit as ar  # noqa: E402

DECISIONS = os.path.join(ar.ROOT, "docs", "decisions", "urlscan-brands-official.md")
OUT = os.path.join(ar.ROOT, "data", "processed", "subdomain", "revisit_sensitivity_brandowned.json")
_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([^`]+)`\s*\|", re.M)


def dated_names(text: str, cutoff: date) -> set[str]:
    """Names from the decision table dated on or before the cutoff."""
    return {n.lower() for d, n in _ROW.findall(text) if date.fromisoformat(d) <= cutoff}


def owned(host: str, names: set[str]) -> bool:
    h = host.lower().rstrip(".")
    return any(h == n or h.endswith("." + n) for n in names)


def stop_day(cohort, obs_rows, induced) -> date:
    """The day the registered rule was met: the STOP_N-th 512 h completion, or STOP_DATE."""
    pop = {r["hostname"] for r in ar.confirmatory(cohort, induced) if r["admission_state"] == "public_a"}
    days = sorted(datetime.fromisoformat(o["observed_at"][:19]).date() for o in obs_rows
                  if o["hostname"] in pop and str(o.get("lag_planned_h")) == "512")
    nth = days[ar.STOP_N - 1] if len(days) >= ar.STOP_N else None
    return min(nth, ar.STOP_DATE) if nth else ar.STOP_DATE


def run(cohort, obs_rows, induced, today: date, decisions_text: str) -> dict:
    full = ar.analyse(cohort, obs_rows, induced, today, enforce_stopping_rule=True)
    if "refused" in full:
        return {"refused": "the registered stopping rule does not hold on the full data: "
                           "no brand-owned sensitivity is computed",
                "stopping_rule": full["stopping_rule"]}
    cutoff = stop_day(cohort, obs_rows, induced)
    names = dated_names(decisions_text, cutoff)
    removed = sorted(r["hostname"] for r in cohort if r["label"] == "phish" and owned(r["hostname"], names))
    kept = [r for r in cohort if not (r["label"] == "phish" and r["hostname"] in set(removed))]
    res = ar.analyse(kept, obs_rows, induced, today, enforce_stopping_rule=False)
    return {"outside_registered_family": True, "cutoff": cutoff.isoformat(),
            "names_in_force": sorted(names), "removed_hosts": removed,
            "status": res["status"], "confirmatory": res.get("confirmatory")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    cohort, obs_rows = ar._read(ar.COHORT), ar._read(ar.OBS)
    text = open(DECISIONS, encoding="utf-8").read() if os.path.exists(DECISIONS) else ""
    res = run(cohort, obs_rows, ar.self_induced_hosts(), date.today(), text)
    if "refused" in res:
        print(json.dumps(res, indent=2, default=str))
        if a.run:
            raise SystemExit(res["refused"])
        return 0
    if a.run:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"[+] {os.path.relpath(OUT, ar.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
