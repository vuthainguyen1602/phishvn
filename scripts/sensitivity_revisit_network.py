#!/usr/bin/env python3
"""
sensitivity_revisit_network.py — the registered revisit analysis, run once per access network.

Registered by the amendment of 2026-10-02 in PREREG_subdomain_evasion.md. On 2026-10-01 the
collection device came back from a 25-hour outage on a different access network: a different
provider answers the positive DNS queries and may block or sinkhole a different set of names,
and the new path returns NXDOMAIN where the old one never did. The registered outcome (a public
A record or not) is defined the same on both, but the vantage is not the same vantage.

This script does not change the registered analysis and is not part of its BH family. It reruns
`analyze_revisit.analyse` unchanged on two single-network cohorts:

  old_network  hosts admitted before the switch, with only the observations taken before it;
  new_network  hosts admitted at or after the switch (their whole life is on the new network).

A host admitted before the switch and observed across it belongs to neither: that is the point
of a single-network cohort. Each cohort's p-values are reported raw and BH-adjusted within that
cohort only, labelled as outside the registered family.

Gate: it refuses until the registered stopping rule holds on the full data, exactly as
`analyze_revisit.py --run` does, so it cannot be used to look early.

RUN: python scripts/sensitivity_revisit_network.py [--run]
"""
import argparse
import json
import os
import sys
from datetime import date, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analyze_revisit as ar  # noqa: E402

# Device-local naive stamps, as cohort.csv and observations.csv store them: the device booted on
# the new network at 2026-10-01 22:57:39 +07. The last observation on the old network is
# 2026-09-30 22:15:51 and the first on the new one 2026-10-01 23:15:12, so any instant in the gap
# separates them; the boot time is used because it is when the network changed.
SWITCH_LOCAL = datetime(2026, 10, 1, 22, 57, 39)
OUT = os.path.join(ar.ROOT, "data", "processed", "subdomain", "revisit_sensitivity_network.json")


def _t(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "")[:19])


def split(cohort: list[dict], obs_rows: list[dict]) -> dict:
    """The two single-network cohorts, as (cohort rows, observation rows)."""
    old_hosts = {r["hostname"] for r in cohort if _t(r["admitted_at"]) < SWITCH_LOCAL}
    return {
        "old_network": ([r for r in cohort if r["hostname"] in old_hosts],
                        [o for o in obs_rows if o["hostname"] in old_hosts
                         and _t(o["observed_at"]) < SWITCH_LOCAL]),
        "new_network": ([r for r in cohort if r["hostname"] not in old_hosts],
                        [o for o in obs_rows if o["hostname"] not in old_hosts]),
    }


def run(cohort, obs_rows, induced, today: date) -> dict:
    full = ar.analyse(cohort, obs_rows, induced, today, enforce_stopping_rule=True)
    if "refused" in full:
        return {"refused": "the registered stopping rule does not hold on the full data: "
                           "no single-network contrast is computed",
                "stopping_rule": full["stopping_rule"]}
    out = {"switch_local": SWITCH_LOCAL.isoformat(), "outside_registered_family": True}
    for name, (c, o) in split(cohort, obs_rows).items():
        res = ar.analyse(c, o, induced, today, enforce_stopping_rule=False)
        out[name] = {"status": res["status"], "confirmatory": res.get("confirmatory")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    cohort, obs_rows = ar._read(ar.COHORT), ar._read(ar.OBS)
    res = run(cohort, obs_rows, ar.self_induced_hosts(), date.today())
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
