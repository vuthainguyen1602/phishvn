#!/usr/bin/env python3
"""
watch_revisit.py — outcome-independent DNS revisits of every newly admitted candidate hostname.

WHY. The infrastructure watcher (watch_host_infra.py) re-attempts a hostname only while it has
never returned an A record, so its ledger holds no later observation of any host that resolved
at first capture. The subdomain-evasion audit (papers/future_subdomain_evasion) showed that this
makes every survival question unidentifiable from that ledger: 5,572 hosts with a public A at
first capture, none revisited. This collector is the prospective design that audit specified.

WHAT IT DOES. Every tick it (1) admits hostnames whose FIRST capture in host_infra.csv falls on
or after the registered admission start, recording the admission time and the DNS state seen at
admission; (2) for every admitted host, resolves A and AAAA at each fixed lag after admission
(8, 32, 64, 128, 256, 512 hours) that has come due and has not yet been observed, regardless of
what any earlier observation returned. The schedule is the same for every host; nothing about an
earlier answer changes whether a later query is asked. Each observation records the resolver's
outcome as one of a fixed set of states, the raw answers, the planned and actual lag, and
whether the observation was late (an outage shifts observations later, never drops them, and
the analysis can see that it did).

WHAT IT DOES NOT DO. No HTTP request, no page fetch, no WHOIS: the outcome is DNS availability
of a name, which the audit's Section 5 says is the right first outcome and explicitly not a
persistence-of-abuse outcome. Group keys (ICANN / tenant) are NOT computed here: the analysis
computes them from the admitted hostnames with a frozen suffix snapshot, so the covariate is
defined at a stated landmark rather than by whatever parser the device happens to run.

Benign control. A daily-capped random sample of benign-labelled first captures is admitted on
the same schedule, so that a resolver or vantage failure shows up as a drop in a population that
should not perish.

Files (all append-only under data/raw/revisit/):
  cohort.csv        one row per admitted hostname
  observations.csv  one row per (hostname, planned lag) observation
RUN (cron, hourly):
  python3 scripts/watch_revisit.py
  python3 scripts/watch_revisit.py --dry-run --max-queries 5
"""
from __future__ import annotations
import argparse, csv, hashlib, ipaddress, os, socket, sys, time
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

INFRA = os.path.join(ROOT, "data", "raw", "host_infra", "host_infra.csv")
OUTDIR = os.path.join(ROOT, "data", "raw", "revisit")
COHORT = os.path.join(OUTDIR, "cohort.csv")
OBS = os.path.join(OUTDIR, "observations.csv")

# Registered 2026-09-16 (PREREG_subdomain_evasion.md, prospective amendment). Hosts first captured
# before this instant are never admitted: they already sit in the outcome-conditioned ledger.
ADMISSION_START = "2026-09-17T00:00:00"
LAGS_H = (8, 32, 64, 128, 256, 512)
BENIGN_PER_DAY = 25          # control arm target per day ...
BENIGN_FIRST_CAPTURES_PER_DAY = 400   # ... at roughly this many benign first captures a day
LATE_FACTOR = 1.5            # an observation later than planned x this is flagged, not dropped

COHORT_FIELDS = ["hostname", "label", "source", "first_detected", "admitted_at",
                 "admission_state", "admission_a_records", "admission_captured_at"]
OBS_FIELDS = ["hostname", "label", "lag_planned_h", "lag_actual_h", "late", "observed_at",
              "state", "rcode", "a_records", "aaaa_records", "cname", "a_ttl",
              "resolver", "query_ms"]

STATES = ("public_a", "private_only", "empty", "nxdomain", "servfail", "timeout", "error")


def _resolver():
    try:
        import dns.resolver
    except ImportError:
        raise SystemExit("dnspython is not installed; refusing to record null observations.\n"
                         "    pip install dnspython")
    r = dns.resolver.Resolver()
    r.lifetime = 6
    r.timeout = 3
    return r


def classify(answers: list[str]) -> str:
    """public_a if any answer is a global IPv4, private_only if answers exist but none is."""
    if not answers:
        return "empty"
    for a in answers:
        try:
            ip = ipaddress.ip_address(a)
            if ip.version == 4 and ip.is_global:
                return "public_a"
        except ValueError:
            pass
    return "private_only"


def observe(host: str) -> dict:
    """One outcome-independent DNS observation of `host`: A, AAAA and CNAME."""
    import dns.exception, dns.resolver
    r = _resolver()
    out = {"state": "", "rcode": "", "a_records": "", "aaaa_records": "", "cname": "",
           "a_ttl": "", "resolver": ";".join(r.nameservers), "query_ms": ""}
    t0 = time.time()
    try:
        ans = r.resolve(host, "A")
        addrs = sorted(a.address for a in ans)
        out.update(a_records=";".join(addrs), a_ttl=ans.rrset.ttl, rcode="NOERROR",
                   state=classify(addrs))
    except dns.resolver.NXDOMAIN:
        out.update(state="nxdomain", rcode="NXDOMAIN")
    except dns.resolver.NoAnswer:
        out.update(state="empty", rcode="NOERROR")
    except dns.resolver.NoNameservers:
        out.update(state="servfail", rcode="SERVFAIL")
    except dns.exception.Timeout:
        out.update(state="timeout", rcode="")
    except Exception as e:  # noqa: BLE001 — recorded, never silently dropped
        out.update(state="error", rcode=type(e).__name__)
    out["query_ms"] = int((time.time() - t0) * 1000)
    for rtype, key in (("AAAA", "aaaa_records"), ("CNAME", "cname")):
        try:
            ans = r.resolve(host, rtype)
            out[key] = ";".join(sorted(str(a.address if rtype == "AAAA" else a.target).rstrip(".")
                                       for a in ans))
        except Exception:
            pass
    return out


def load_first_captures() -> dict:
    """hostname -> its FIRST capture row in host_infra.csv (by captured_at, then attempt)."""
    first: dict = {}
    if not os.path.exists(INFRA):
        return first
    with open(INFRA, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            h = (r.get("domain") or "").strip().lower().rstrip(".")
            if not h:
                continue
            key = (r.get("captured_at") or "", int(r.get("attempt") or 1))
            if h not in first or key < first[h][0]:
                first[h] = (key, r)
    return {h: v[1] for h, v in first.items()}


def load_cohort() -> dict:
    coh = {}
    if os.path.exists(COHORT):
        with open(COHORT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                coh[r["hostname"]] = r
    return coh


def load_done() -> set:
    done = set()
    if os.path.exists(OBS):
        with open(OBS, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["hostname"], int(r["lag_planned_h"])))
    return done


def benign_admitted(host: str, day: str, cap: int) -> bool:
    """A deterministic sample of about `cap` benign hosts a day: the hash of (day, host) decides,
    so a rerun admits the same set and no earlier outcome can influence membership."""
    per_mille = cap * 1000 // BENIGN_FIRST_CAPTURES_PER_DAY
    return int(hashlib.sha1(f"{day}|{host}".encode()).hexdigest(), 16) % 1000 < per_mille


def append(path: str, fields: list, rows: list) -> None:
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
            f.flush()


def main() -> int:
    global OUTDIR, COHORT, OBS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-queries", type=int, default=400, help="observations per tick")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true", help="admit and plan, query nothing, write nothing")
    ap.add_argument("--now", default=None, help="override the clock (tests)")
    ap.add_argument("--admission-start", default=ADMISSION_START, help="override (tests only)")
    ap.add_argument("--outdir", default=OUTDIR, help="override the output directory (tests only)")
    a = ap.parse_args()
    OUTDIR = a.outdir
    COHORT, OBS = os.path.join(OUTDIR, "cohort.csv"), os.path.join(OUTDIR, "observations.csv")
    if not a.dry_run:
        _resolver()          # die here, before admission is written, if dnspython is absent
    socket.setdefaulttimeout(10)
    os.makedirs(OUTDIR, exist_ok=True)
    now = datetime.fromisoformat(a.now) if a.now else datetime.now()
    start = datetime.fromisoformat(a.admission_start)

    # 1. admission
    cohort = load_cohort()
    first = load_first_captures()
    admitted = []
    for h, r in first.items():
        if h in cohort:
            continue
        try:
            cap_at = datetime.fromisoformat((r.get("captured_at") or "")[:19])
        except ValueError:
            continue
        if cap_at < start or cap_at > now:
            continue
        label = (r.get("label") or "").strip()
        if label == "benign" and not benign_admitted(h, cap_at.date().isoformat(), BENIGN_PER_DAY):
            continue
        if label not in ("phish", "benign"):
            continue
        addrs = [x for x in (r.get("a_records") or "").split(";") if x]
        row = {"hostname": h, "label": label, "source": r.get("source", ""),
               "first_detected": r.get("first_detected", ""),
               "admitted_at": cap_at.isoformat(timespec="seconds"),
               "admission_state": classify(addrs), "admission_a_records": ";".join(addrs),
               "admission_captured_at": r.get("captured_at", "")}
        admitted.append(row)
        cohort[h] = row
    if admitted and not a.dry_run:
        append(COHORT, COHORT_FIELDS, admitted)

    # 2. due observations, oldest planned first so an outage is caught up in order
    done = load_done()
    due = []
    for h, c in cohort.items():
        adm = datetime.fromisoformat(c["admitted_at"])
        for lag in LAGS_H:
            if (h, lag) in done:
                continue
            planned = adm + timedelta(hours=lag)
            if planned <= now:
                due.append((planned, h, lag, c))
    due.sort()
    todo = due[:a.max_queries]
    print(f"[{now.isoformat(timespec='seconds')}] cohort {len(cohort)} (+{len(admitted)} admitted), "
          f"{len(due)} observations due, {len(todo)} this tick"
          + (" [dry run]" if a.dry_run else ""))
    if a.dry_run or not todo:
        return 0

    rows = []
    for i, (planned, h, lag, c) in enumerate(todo, 1):
        obs = observe(h)
        t = datetime.now() if not a.now else now
        actual = (t - datetime.fromisoformat(c["admitted_at"])).total_seconds() / 3600
        rows.append({"hostname": h, "label": c["label"], "lag_planned_h": lag,
                     "lag_actual_h": round(actual, 2), "late": int(actual > lag * LATE_FACTOR),
                     "observed_at": t.isoformat(timespec="seconds"), **obs})
        print(f"[{i}/{len(todo)}] {h} lag={lag}h state={obs['state']} a={obs['a_records'] or '-'}")
        time.sleep(a.delay)
    append(OBS, OBS_FIELDS, rows)
    print(f"[+] {len(rows)} observations -> {OBS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
