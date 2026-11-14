#!/usr/bin/env python3
"""sensitivity_revisit_stalefeed.py — the registered revisit analysis rerun without the hosts that a
stale feed entry admitted (amendment of 2026-10-06), reported OUTSIDE the registered family.

WHY. On 2026-10-06 the PhishTank feed joined `vn_phishing_live` with a `.vn`-only gate but, for
its first tick, no freshness gate. Nine of the thirteen names it admitted had no PhishTank
submission newer than 180 days (one from 2023), their captures are the home pages of compromised
but legitimate Vietnamese sites, and all thirteen reached `host_infra.csv` at 20:05 and the
revisit cohort before the ledger rows were withdrawn. The cohort is append-only, so the bound
analysis keeps them, as admitted, and that stays the registered result. This script reruns
`analyze_revisit.analyse`, unchanged, with those nine removed, so a reader can see whether they
move it.

WHICH HOSTS. A phishing-labelled cohort host is removed when its hostname equals a `domain` in
`data/raw/vn_phishing_live/quarantine_phishtank_stale_20261006.csv`, the dated file the
withdrawal wrote. The file is append-only and names nine hosts; a host quarantined later for the
same reason enters only if its `quarantined_at` is on or before the registered stopping day.
Benign controls are never removed.

Gate: it refuses until the registered stopping rule holds on the full data.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import date, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analyze_revisit as ar  # noqa: E402
from sensitivity_revisit_brandowned import stop_day  # noqa: E402

QUARANTINE = os.path.join(ar.ROOT, "data", "raw", "vn_phishing_live",
                          "quarantine_phishtank_stale_20261006.csv")
OUT = os.path.join(ar.ROOT, "data", "processed", "subdomain", "revisit_sensitivity_stalefeed.json")


def stale_names(rows: list[dict], cutoff: date) -> set[str]:
    """Quarantined names whose withdrawal is dated on or before the cutoff."""
    out = set()
    for r in rows:
        try:
            when = datetime.fromisoformat((r.get("quarantined_at") or "")[:19]).date()
        except ValueError:
            continue
        if when <= cutoff and r.get("domain"):
            out.add(r["domain"].lower().rstrip("."))
    return out


def run(cohort, obs_rows, induced, today: date, quarantine_rows: list[dict]) -> dict:
    full = ar.analyse(cohort, obs_rows, induced, today, enforce_stopping_rule=True)
    if "refused" in full:
        return {"refused": "the registered stopping rule does not hold on the full data: "
                           "no stale-feed sensitivity is computed",
                "stopping_rule": full["stopping_rule"]}
    cutoff = stop_day(cohort, obs_rows, induced)
    names = stale_names(quarantine_rows, cutoff)
    removed = sorted(r["hostname"] for r in cohort
                     if r["label"] == "phish" and r["hostname"].lower().rstrip(".") in names)
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
    rows = list(csv.DictReader(open(QUARANTINE, newline="", encoding="utf-8"))) \
        if os.path.exists(QUARANTINE) else []
    res = run(cohort, obs_rows, ar.self_induced_hosts(), date.today(), rows)
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
