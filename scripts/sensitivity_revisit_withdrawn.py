#!/usr/bin/env python3
"""sensitivity_revisit_withdrawn.py — the registered revisit analysis rerun without every host the
phishing ledger has withdrawn (amendment of 2026-10-07), reported OUTSIDE the registered family.

WHY. Two withdrawals have left the ledger after their hosts reached the append-only cohort: the
nine stale PhishTank names of 2026-10-06 (quarantine_phishtank_stale_20261006.csv) and the four
names a by-hand review of 2026-10-07 found to be compromised legitimate sites or off-target lures
(quarantine_review_20261007.csv). The bound analysis keeps all of them, as admitted, and that
stays the registered result. sensitivity_revisit_stalefeed.py removes the first set only; this
script reruns `analyze_revisit.analyse`, unchanged, with both sets removed, so a reader can see
what the ledger's withdrawals do together.

WHICH HOSTS. A phishing-labelled cohort host is removed when its hostname equals a `domain` in
either quarantine file and that row's `quarantined_at` is on or before the registered stopping
day. Benign controls are never removed.

Gate: it refuses until the registered stopping rule holds on the full data.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analyze_revisit as ar  # noqa: E402
from sensitivity_revisit_brandowned import stop_day  # noqa: E402
from sensitivity_revisit_stalefeed import stale_names as withdrawn_names  # noqa: E402

LEDGER = os.path.join(ar.ROOT, "data", "raw", "vn_phishing_live")
QUARANTINES = ("quarantine_phishtank_stale_20261006.csv", "quarantine_review_20261007.csv")
OUT = os.path.join(ar.ROOT, "data", "processed", "subdomain", "revisit_sensitivity_withdrawn.json")


def run(cohort, obs_rows, induced, today: date, quarantine_rows: list[dict]) -> dict:
    full = ar.analyse(cohort, obs_rows, induced, today, enforce_stopping_rule=True)
    if "refused" in full:
        return {"refused": "the registered stopping rule does not hold on the full data: "
                           "no withdrawn-host sensitivity is computed",
                "stopping_rule": full["stopping_rule"]}
    cutoff = stop_day(cohort, obs_rows, induced)
    names = withdrawn_names(quarantine_rows, cutoff)
    removed = sorted(r["hostname"] for r in cohort
                     if r["label"] == "phish" and r["hostname"].lower().rstrip(".") in names)
    gone = set(removed)
    kept = [r for r in cohort if not (r["label"] == "phish" and r["hostname"] in gone)]
    res = ar.analyse(kept, obs_rows, induced, today, enforce_stopping_rule=False)
    return {"outside_registered_family": True, "cutoff": cutoff.isoformat(),
            "names_in_force": sorted(names), "removed_hosts": removed,
            "status": res["status"], "confirmatory": res.get("confirmatory")}


def read_quarantines() -> list[dict]:
    rows = []
    for name in QUARANTINES:
        path = os.path.join(LEDGER, name)
        if os.path.exists(path):
            rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    cohort, obs_rows = ar._read(ar.COHORT), ar._read(ar.OBS)
    res = run(cohort, obs_rows, ar.self_induced_hosts(), date.today(), read_quarantines())
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
