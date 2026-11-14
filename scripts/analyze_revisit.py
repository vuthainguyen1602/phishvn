#!/usr/bin/env python3
"""
analyze_revisit.py — the confirmatory analysis of the prospective revisit cohort, as registered.

Implements, and only implements, `papers/future_subdomain_evasion/PREREG_subdomain_evasion.md`:
the prospective registration of 2026-09-16 and its amendments of 2026-09-19 00:20 (the `ct_brands`
stratum) and 2026-09-19 23:41 (self-induced hosts, the wildcard probe). Section names below are
the registration's.

FAILS CLOSED. No contrast is computed until the registered stopping rule holds: at least 300
phishing hosts of the confirmatory population with `public_a` at admission have a 512 h
observation, or the date is 2026-12-01 or later. Before that, `--status` prints what the
registration allows to be read while data accrue (row counts, the late rate, the control's
retention) and nothing else, and `--run` refuses.

CHOICES THE REGISTRATION LEFT TO THE CODE, fixed here before any blocklist match or contrast was
looked at, and listed so that a reader can disagree with them:
  * a blocklist URL's hostname and a cohort hostname are compared lower-cased with one leading
    `www.` removed from each (the collectors store hostnames without it);
  * the snapshot a host is judged against is, per feed, the latest successful fetch strictly
    before that host's 128 h observation, both clocks converted to UTC (the device writes naive
    Asia/Ho_Chi_Minh stamps, the fetcher writes UTC); the two feeds' hostnames are pooled;
  * a host with no 128 h observation is reported as missing and enters no 2x2 and no gap;
  * the Newcombe interval is the hybrid-score one (his method 10), from two Wilson intervals;
  * the permutation test of RQ3' relabels *multi* within source strata 10,000 times with
    seed 20260916 and reports the two-sided p of the difference in mean gap;
  * Benjamini-Hochberg is applied to the three primary p-values only.

RUN:
  python scripts/analyze_revisit.py --status
  python scripts/analyze_revisit.py --run       # only after the stopping rule
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from urllib.parse import urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

REGISTERED_SOURCES = ("urlscan_brands", "vn_phishing_live", "chongluadao_live")
LAGS = (8, 32, 64, 128, 256, 512)
DEVICE_UTC_OFFSET_H = 7                 # cohort.csv / observations.csv carry naive device-local stamps
STOP_N, STOP_DATE = 300, date(2026, 12, 1)
MIN_ARM = 100                           # RQ1': below this per arm at 128 h, no test is claimed
LANDMARK_H = 32
PERMUTATIONS, SEED = 10_000, 20260916
BH_Q = 0.05
CONTROL_FLOOR = 0.90

RAW = os.path.join(ROOT, "data", "raw")
COHORT = os.path.join(RAW, "revisit", "cohort.csv")
OBS = os.path.join(RAW, "revisit", "observations.csv")
PROBE = os.path.join(RAW, "revisit", "wildcard_probe.csv")
BLOCK_MANIFEST = os.path.join(RAW, "blocklists", "manifest.csv")
INFRA = os.path.join(RAW, "host_infra", "host_infra.csv")
BRAND_DET = os.path.join(RAW, "urlscan_brands", "detections.csv")
CT_DET = os.path.join(RAW, "ct_brands", "detections.csv")
OWN_LEDGERS = os.path.join(ROOT, "scripts", "core", "collect", "own_scan_ledgers.local")
KEYS_MANIFEST = os.path.join(ROOT, "data", "processed", "subdomain", "input_manifest.json")
OUT = os.path.join(ROOT, "data", "processed", "subdomain", "revisit_analysis.json")


# ------------------------------------------------------------------ small statistics

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """(difference p1-p2, lower, upper): Newcombe's hybrid score interval."""
    if not n1 or not n2:
        return (float("nan"),) * 3
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]], by summing tables no likelier than the
    observed one. Written out so the analysis has no dependency that could move under it."""
    n1, n2, k = a + b, c + d, a + c
    lg = math.lgamma

    def logp(x: int) -> float:
        return (lg(n1 + 1) - lg(x + 1) - lg(n1 - x + 1) + lg(n2 + 1) - lg(k - x + 1)
                - lg(n2 - k + x + 1) - (lg(n1 + n2 + 1) - lg(k + 1) - lg(n1 + n2 - k + 1)))
    lo, hi = max(0, k - n2), min(k, n1)
    obs = logp(a)
    return min(1.0, sum(math.exp(logp(x)) for x in range(lo, hi + 1) if logp(x) <= obs + 1e-9))


def benjamini_hochberg(ps: dict[str, float]) -> dict[str, float]:
    """Adjusted p-values (step-up). The family size is the number of keys given, also when a test
    is not claimed (NaN): an unclaimed test does not make the other two cheaper."""
    items = sorted(((p, k) for k, p in ps.items() if p == p), reverse=True)
    m, out, running = len(ps), {}, 1.0
    for rank_from_top, (p, k) in enumerate(items):
        i = len(items) - rank_from_top
        running = min(running, p * m / i)
        out[k] = running
    return out


# ------------------------------------------------------------------ inputs

def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _local(ts: str) -> datetime:
    return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")


def _nowww(h: str) -> str:
    return (h or "").strip().lower().rstrip(".").removeprefix("www.")


def self_induced_hosts() -> set[str]:
    """Amendment of 2026-09-19 23:41, item 2: decided by scan identifier, from these files only."""
    own: set[str] = set()
    with open(OWN_LEDGERS, encoding="utf-8") as f:
        ledgers = [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
    for rel in ledgers:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            own |= {(r.get("scan_uuid") or "").strip() for r in _read(p)}
    own.discard("")
    return {(r.get("domain") or "").strip().lower() for r in _read(BRAND_DET)
            if (r.get("scan_uuid") or "").strip() in own}


def key_functions():
    """tenant key and hosted test, from the frozen manifest and the audit's own code."""
    import make_subdomain_assets as msa
    hosted = json.load(open(KEYS_MANIFEST, encoding="utf-8"))["hosted_suffixes"]

    def tenant(h: str) -> str:
        return msa.tenant_key(msa.normalize(h), hosted)

    def icann(h: str) -> str:
        return msa.boundary(msa.normalize(h), msa.ICANN)

    def is_hosted(key: str) -> bool:
        return any(key == s or key.endswith("." + s) for s in hosted)
    return tenant, icann, is_hosted


def observations_by_host(obs: list[dict]) -> dict:
    out: dict = defaultdict(dict)
    for r in obs:
        out[r["hostname"]][int(float(r["lag_planned_h"]))] = r
    return out


# ------------------------------------------------------------------ populations and exposure

def confirmatory(cohort: list[dict], induced: set[str]) -> list[dict]:
    return [r for r in cohort if r["label"] == "phish" and r["source"] in REGISTERED_SOURCES
            and r["hostname"] not in induced]


def self_induced_rows(cohort: list[dict], induced: set[str]) -> list[dict]:
    """The descriptive stratum of the 23:41 amendment: what the criterion removed from the
    confirmatory population. A `ct_brands` host whose own capture later echoed back is already
    outside that population and stays in the `ct_brands` stratum."""
    return [r for r in cohort if r["label"] == "phish" and r["source"] in REGISTERED_SOURCES
            and r["hostname"] in induced]


def landmark_multi(pop: list[dict], tenant) -> dict[str, bool]:
    """h is multi if another host OF THIS POPULATION shares its tenant key and was admitted no
    later than admitted(h) + 32 h."""
    by_key = defaultdict(list)
    for r in pop:
        by_key[tenant(r["hostname"])].append((_local(r["admitted_at"]), r["hostname"]))
    out = {}
    for r in pop:
        limit = _local(r["admitted_at"]) + timedelta(hours=LANDMARK_H)
        out[r["hostname"]] = any(h != r["hostname"] and t <= limit for t, h in by_key[tenant(r["hostname"])])
    return out


def ledger_multi(pop: list[dict], tenant) -> dict[str, bool]:
    """The registered sensitivity definition: any other hostname in host_infra.csv shares the key."""
    seen = defaultdict(set)
    with open(INFRA, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            h = (r.get("domain") or "").strip().lower()
            if h:
                seen[tenant(h)].add(h)
    return {r["hostname"]: len(seen[tenant(r["hostname"])] - {r["hostname"]}) > 0 for r in pop}


# ------------------------------------------------------------------ the three contrasts

def two_by_two(rows: list[dict], exposure: dict, obs: dict, lag: int, drop_late: bool = False) -> dict:
    cells, missing = Counter(), 0
    for r in rows:
        o = obs.get(r["hostname"], {}).get(lag)
        if o is None or (drop_late and o.get("late") == "1"):
            missing += 1
            continue
        cells[(bool(exposure[r["hostname"]]), o["state"] == "public_a")] += 1
    a, b = cells[(True, True)], cells[(True, False)]
    c, d = cells[(False, True)], cells[(False, False)]
    diff, lo, hi = newcombe(a, a + b, c, c + d)
    return {"lag_h": lag, "exposed": {"retained": a, "n": a + b}, "unexposed": {"retained": c, "n": c + d},
            "missing_observation": missing, "difference": diff, "newcombe95": [lo, hi],
            "fisher_p": fisher_two_sided(a, b, c, d) if (a + b) and (c + d) else float("nan")}


def rq1(pop, multi, obs, tenant, is_hosted, drop_late=False) -> dict:
    rows = [r for r in pop if r["admission_state"] == "public_a" and not is_hosted(tenant(r["hostname"]))]
    primary = two_by_two(rows, multi, obs, 128, drop_late)
    primary["underpowered"] = min(primary["exposed"]["n"], primary["unexposed"]["n"]) < MIN_ARM
    if primary["underpowered"]:
        primary["note"] = f"an arm holds fewer than {MIN_ARM} hosts at 128 h: reported with its CI, no test claimed"
    return {"population": len(rows), "primary_128h": primary,
            "descriptive": [two_by_two(rows, multi, obs, lag, drop_late) for lag in (256, 512)]}


def rq2(pop, multi, obs, tenant, is_hosted, drop_late=False) -> dict:
    rows = [r for r in pop if multi[r["hostname"]] and r["admission_state"] == "public_a"]
    hosted = {r["hostname"]: is_hosted(tenant(r["hostname"])) for r in rows}
    return {"population": len(rows), "primary_256h": two_by_two(rows, hosted, obs, 256, drop_late)}


def _blocklist_hosts(path: str, feed: str) -> set[str]:
    hosts: set[str] = set()
    with gzip.open(os.path.join(ROOT, path), "rt", encoding="utf-8", errors="replace") as f:
        if feed == "phishtank":
            for r in csv.DictReader(f):
                hosts.add(_nowww(urlparse(r.get("url", "")).hostname or ""))
        else:
            for line in f:
                hosts.add(_nowww(urlparse(line.strip()).hostname or ""))
    hosts.discard("")
    return hosts


def rq3(pop, multi, obs, tenant, drop_late=False, permutations=PERMUTATIONS, seed=SEED) -> dict:
    fetches = sorted(((datetime.fromisoformat(r["fetched_at_utc"]).replace(tzinfo=None), r["feed"], r["file"])
                      for r in _read(BLOCK_MANIFEST) if r.get("status") == "200" and r.get("file")))
    cache: dict[str, set[str]] = {}
    by_key = defaultdict(set)
    for r in pop:
        by_key[tenant(r["hostname"])].add(_nowww(r["hostname"]))
    gaps, missing, no_snapshot = [], 0, 0
    for r in pop:
        o = obs.get(r["hostname"], {}).get(128)
        if o is None or (drop_late and o.get("late") == "1"):
            missing += 1
            continue
        when = _local(o["observed_at"]) - timedelta(hours=DEVICE_UTC_OFFSET_H)
        listed: set[str] = set()
        for feed in {f for _, f, _ in fetches}:
            prior = [p for t, f, p in fetches if f == feed and t < when]
            if prior:
                listed |= cache.setdefault(prior[-1], _blocklist_hosts(prior[-1], feed))
        if not listed:
            no_snapshot += 1
            continue
        exact = int(_nowww(r["hostname"]) in listed)
        expansion = int(bool(by_key[tenant(r["hostname"])] & listed))
        gaps.append({"host": r["hostname"], "source": r["source"], "multi": bool(multi[r["hostname"]]),
                     "exact": exact, "expansion": expansion, "gap": expansion - exact})

    def diff(rows):
        m = [g["gap"] for g in rows if g["multi"]]
        s = [g["gap"] for g in rows if not g["multi"]]
        return (sum(m) / len(m) if m else float("nan")) - (sum(s) / len(s) if s else 0.0)
    observed = diff(gaps)
    rng, extreme = random.Random(seed), 0
    strata = defaultdict(list)
    for i, g in enumerate(gaps):
        strata[g["source"]].append(i)
    if observed == observed:
        for _ in range(permutations):
            shuffled = [dict(g) for g in gaps]
            for idx in strata.values():
                labels = [gaps[i]["multi"] for i in idx]
                rng.shuffle(labels)
                for i, lab in zip(idx, labels):
                    shuffled[i]["multi"] = lab
            d = diff(shuffled)
            if d == d and abs(d) >= abs(observed) - 1e-12:
                extreme += 1
    n_m, n_s = sum(g["multi"] for g in gaps), sum(not g["multi"] for g in gaps)
    return {"hosts": len(gaps), "missing_observation": missing, "no_snapshot_before_observation": no_snapshot,
            "multi": {"n": n_m, "exact": sum(g["exact"] for g in gaps if g["multi"]),
                      "expansion": sum(g["expansion"] for g in gaps if g["multi"])},
            "solo": {"n": n_s, "exact": sum(g["exact"] for g in gaps if not g["multi"])},
            "difference_in_mean_gap": observed,
            "permutation_p": (extreme + 1) / (permutations + 1) if observed == observed else float("nan"),
            "permutations": permutations, "seed": seed}


# ------------------------------------------------------------------ what may be read while data accrue

def retention_table(rows: list[dict], obs: dict) -> dict:
    base = [r for r in rows if r["admission_state"] == "public_a"]
    out = {}
    for lag in LAGS:
        seen = [obs[r["hostname"]][lag] for r in base if lag in obs.get(r["hostname"], {})]
        k = sum(o["state"] == "public_a" for o in seen)
        out[str(lag)] = {"observed": len(seen), "public_a": k, "wilson95": list(wilson(k, len(seen)))}
    return {"public_a_at_admission": len(base), "by_lag": out}


def control_weeks(cohort: list[dict], obs: dict) -> dict:
    weeks = defaultdict(lambda: [0, 0])
    for r in cohort:
        o = obs.get(r["hostname"], {}).get(128)
        if r["label"] == "benign" and r["admission_state"] == "public_a" and o is not None:
            y, w, _ = _local(o["observed_at"]).isocalendar()
            weeks[f"{y}-W{w:02d}"][0] += o["state"] == "public_a"
            weeks[f"{y}-W{w:02d}"][1] += 1
    return {w: {"public_a": k, "n": n, "retention": k / n, "vantage_suspect": k / n < CONTROL_FLOOR}
            for w, (k, n) in sorted(weeks.items())}


def stopping_rule(pop: list[dict], obs: dict, today: date) -> dict:
    done = sum(1 for r in pop if r["admission_state"] == "public_a" and 512 in obs.get(r["hostname"], {}))
    return {"completed_512h": done, "needed": STOP_N, "date_limit": STOP_DATE.isoformat(),
            "met": done >= STOP_N or today >= STOP_DATE}


def status(cohort, obs_rows, obs, pop, induced) -> dict:
    late = sum(1 for o in obs_rows if o.get("late") == "1")
    return {"cohort": len(cohort), "by_label": dict(Counter(r["label"] for r in cohort)),
            "by_source": dict(Counter(r["source"] for r in cohort)),
            "admissions_by_day": dict(sorted(Counter(r["admitted_at"][:10] for r in cohort).items())),
            "confirmatory_phish": len(pop), "self_induced_excluded": len(self_induced_rows(cohort, induced)),
            "observations": len(obs_rows), "late": late, "late_rate": late / len(obs_rows) if obs_rows else 0.0,
            "control_128h_by_week": control_weeks(cohort, obs)}


# ------------------------------------------------------------------ main

def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyse(cohort, obs_rows, induced, today: date, enforce_stopping_rule: bool = True) -> dict:
    tenant, icann, is_hosted = key_functions()
    obs = observations_by_host(obs_rows)
    pop = confirmatory(cohort, induced)
    rule = stopping_rule(pop, obs, today)
    out = {"status": status(cohort, obs_rows, obs, pop, induced), "stopping_rule": rule}
    if enforce_stopping_rule and not rule["met"]:
        out["refused"] = ("the registered stopping rule does not hold: no contrast is computed. "
                          f"{rule['completed_512h']} of {STOP_N} hosts have completed 512 h and it is before {STOP_DATE}.")
        return out
    multi = landmark_multi(pop, tenant)
    r1, r2, r3 = rq1(pop, multi, obs, tenant, is_hosted), rq2(pop, multi, obs, tenant, is_hosted), rq3(pop, multi, obs, tenant)
    ps = {"RQ1": float("nan") if r1["primary_128h"]["underpowered"] else r1["primary_128h"]["fisher_p"],
          "RQ2": r2["primary_256h"]["fisher_p"], "RQ3": r3["permutation_p"]}
    out["confirmatory"] = {"RQ1": r1, "RQ2": r2, "RQ3": r3, "p_values": ps,
                           "bh_adjusted": benjamini_hochberg(ps), "q": BH_Q, "family_size": 3}
    out["per_source"] = {}
    for s in REGISTERED_SOURCES:
        sub = [r for r in pop if r["source"] == s]
        out["per_source"][s] = {"RQ1": rq1(sub, multi, obs, tenant, is_hosted),
                                "RQ2": rq2(sub, multi, obs, tenant, is_hosted)}
    # sensitivity analyses, all outside the family
    sens = {"late_excluded": {"RQ1": rq1(pop, multi, obs, tenant, is_hosted, True),
                              "RQ2": rq2(pop, multi, obs, tenant, is_hosted, True),
                              "RQ3": rq3(pop, multi, obs, tenant, True)}}
    lm = ledger_multi(pop, tenant)
    sens["multi_by_whole_ledger"] = {"RQ1": rq1(pop, lm, obs, tenant, is_hosted)}
    with_induced = [r for r in cohort if r["label"] == "phish" and r["source"] in REGISTERED_SOURCES]
    mi = landmark_multi(with_induced, tenant)
    sens["self_induced_included"] = {"RQ1": rq1(with_induced, mi, obs, tenant, is_hosted),
                                     "RQ2": rq2(with_induced, mi, obs, tenant, is_hosted),
                                     "RQ3": rq3(with_induced, mi, obs, tenant)}
    if os.path.exists(PROBE):
        last = {r["registered_domain"]: r for r in _read(PROBE)}      # append-only file: the latest row counts
        bad = {d for d, r in last.items() if r["status"] in ("wildcard", "undetermined")}
        free = [r for r in pop if icann(r["hostname"]) not in bad]
        sens["wildcard_free"] = {"RQ1": rq1(free, landmark_multi(free, tenant), obs, tenant, is_hosted),
                                 "multi_hosts_under_wildcard": sum(1 for r in pop if multi[r["hostname"]] and icann(r["hostname"]) in bad),
                                 "multi_hosts": sum(multi.values())}
    flagged = {w for w, v in out["status"]["control_128h_by_week"].items() if v["vantage_suspect"]}
    sens["vantage_suspect_weeks"] = sorted(flagged)
    out["sensitivity"] = sens
    # descriptive strata
    out["strata"] = {"self_induced": retention_table(self_induced_rows(cohort, induced), obs),
                     "ct_brands": retention_table([r for r in cohort if r["source"] == "ct_brands"], obs)}
    if os.path.exists(CT_DET):
        from watch_ct_brands import CLOUD_INTERNAL_SUFFIXES, CT_BRAND_OWNED, short_token_closed
        from watch_urlscan_brands import is_official
        brand = {r["domain"]: r.get("brand", "") for r in _read(CT_DET)}
        keep = [r for r in cohort if r["source"] == "ct_brands"
                and not is_official(r["hostname"], set(CT_BRAND_OWNED))
                and not r["hostname"].endswith(CLOUD_INTERNAL_SUFFIXES)
                and short_token_closed(r["hostname"], brand.get(r["hostname"], "x" * 9))]
        out["strata"]["ct_brands_without_brand_owned_and_filtered"] = retention_table(keep, obs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="what the registration allows to be read while data accrue")
    ap.add_argument("--run", action="store_true", help="the registered analysis; refuses until the stopping rule holds")
    a = ap.parse_args()
    cohort, obs_rows = _read(COHORT), _read(OBS)
    induced = self_induced_hosts()
    res = analyse(cohort, obs_rows, induced, date.today(), enforce_stopping_rule=True)
    if a.run and "refused" not in res:
        res["inputs_sha256"] = {os.path.relpath(p, ROOT): _sha(p) for p in (COHORT, OBS, BLOCK_MANIFEST, BRAND_DET, KEYS_MANIFEST)
                                if os.path.exists(p)}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"[+] {os.path.relpath(OUT, ROOT)}")
        return 0
    print(json.dumps({k: res[k] for k in ("status", "stopping_rule")}, indent=2, default=str))
    if a.run:
        raise SystemExit(res["refused"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
