#!/usr/bin/env python3
"""
probe_wildcard.py — does a cohort host's registered domain answer for a name nobody created?

WHY THIS EXISTS. The registered exposure of this study is multiplicity: how many hostnames share a
key. On 2026-09-19 the second-largest family in the phishing arm turned out to be 84 hostnames
under one registered domain that resolved EVERY label (wildcard DNS), the 84 being a subdomain
wordlist some third party had run through urlscan. Nobody created them, they count as *multi*, and
a wildcard name resolves for as long as its parent does, which is the hypothesised direction. The
draft amendment of that day leaves the registered definition alone and adds this measurement, so
that the contrast can be reported with and without such domains.

WHAT IT DOES. For every ICANN registered domain of a host in cohort.csv that has no row yet, one
lookup of a fixed unlikely label under it, appended to data/raw/revisit/wildcard_probe.csv with the
time, the resolver and the answer:

  wildcard      the unlikely label returned an address
  no_wildcard   it did not, and the domain itself still has name servers
  undetermined  the domain no longer has name servers, or the lookup failed: NOT a negative.
                A row that failed on the transport (probe TIMEOUT) is asked again on a later run;
                rows are never rewritten, so a domain can have several and the latest one counts
  platform      the host sits on a hosting platform's suffix; the platform answers for every
                label by design and the question is not about the tenant, so nothing is asked

It reads the domain's answer for a name nobody uses, once, and never a cohort host's own state, so
it cannot be tuned on an outcome. It does not touch watch_revisit.py, cohort.csv or
observations.csv.

FAILS CLOSED. Against the cohort it runs only once the registration carries the amendment that
names its output file; until then only explicit domains on the command line are probed, and
nothing is written (a smoke test, not a measurement).

RUN:
  python scripts/probe_wildcard.py example.com   # smoke test, prints only
  python scripts/probe_wildcard.py --cohort      # after adoption; append-only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

OUTDIR = os.path.join(ROOT, "data", "raw", "revisit")
COHORT = os.path.join(OUTDIR, "cohort.csv")
OUT = os.path.join(OUTDIR, "wildcard_probe.csv")
PREREG = os.path.join(ROOT, "papers", "future_subdomain_evasion", "PREREG_subdomain_evasion.md")
# the label the infrastructure gate's registry-wildcard probe already uses: one convention
PROBE_LABEL = "phishvn-wildcard-probe-x7q9z3"
FIELDS = ["registered_domain", "probed_at_utc", "probe_name", "status", "answers", "detail",
          "resolver"]


def adopted() -> bool:
    """The registration names this script's output only once the amendment is appended to it."""
    return os.path.exists(PREREG) and "wildcard_probe.csv" in open(PREREG, encoding="utf-8").read()


# asked directly, over TCP, when the system resolver times out: the stub on the device forwards
# over UDP whatever transport it was asked on, so a TCP question has to bypass it
TCP_RESOLVERS = ("8.8.8.8", "1.1.1.1")


def _resolver(tcp: bool = False):
    import dns.resolver
    r = dns.resolver.Resolver(configure=not tcp)
    if tcp:
        r.nameservers = list(TCP_RESOLVERS)
    r.lifetime = r.timeout = 6
    return r


def ask(name: str, rtype: str, tcp: bool = False) -> tuple[str, list[str]]:
    """(rcode-ish, answers). NOERROR with no records is 'NODATA'.

    A UDP timeout is asked again over TCP of TCP_RESOLVERS, and the code then carries ' tcp'. On the collection
    device's network a negative answer sent over UDP never arrives (measured 2026-09-20: every
    nonexistent name times out, also when 8.8.8.8 is asked directly, while the same question over
    TCP returns NXDOMAIN in 0.1 s), and a negative answer is exactly what this probe is for."""
    import dns.exception
    import dns.resolver
    via = " tcp" if tcp else ""
    try:
        ans = _resolver(tcp).resolve(name, rtype, tcp=tcp)
        return "NOERROR" + via, sorted(str(x).rstrip(".") for x in ans)
    except dns.resolver.NXDOMAIN:
        return "NXDOMAIN" + via, []
    except dns.resolver.NoAnswer:
        return "NODATA" + via, []
    except dns.resolver.NoNameservers:
        return "SERVFAIL" + via, []
    except dns.exception.Timeout:
        return ("TIMEOUT tcp", []) if tcp else ask(name, rtype, tcp=True)
    except dns.exception.DNSException as e:
        return type(e).__name__ + via, []


def probe(domain: str) -> dict:
    from hosting import classify
    row = {"registered_domain": domain, "probe_name": f"{PROBE_LABEL}.{domain}",
           "probed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "answers": "", "detail": "", "resolver": ";".join(_resolver().nameservers)}
    if classify(f"x.{domain}")[0] == "platform_hosted":
        return {**row, "status": "platform", "detail": "hosting-platform suffix; not asked"}
    code, answers = ask(row["probe_name"], "A")
    if answers:
        return {**row, "status": "wildcard", "answers": ";".join(answers), "detail": code}
    ns_code, ns = ask(domain, "NS")
    if " tcp" in code + ns_code:
        row["resolver"] += ";tcp:" + ";".join(TCP_RESOLVERS)
    if code.split()[0] in ("NXDOMAIN", "NODATA") and ns:
        return {**row, "status": "no_wildcard", "detail": f"{code}; domain has name servers"}
    return {**row, "status": "undetermined", "detail": f"probe {code}; NS {ns_code}"}


def transport_failure(row: dict) -> bool:
    """An `undetermined` row whose probe lookup timed out says something about the path to the
    resolver and nothing about the domain. Such a domain is asked again; the old row stays, and a
    reader takes each domain's LATEST row (`latest()`)."""
    return row.get("status") == "undetermined" and "probe TIMEOUT" in (row.get("detail") or "")


def latest(path: str = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path or OUT, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["registered_domain"]] = r
    return out


def cohort_domains(path: str = None) -> list[str]:
    import tldextract
    icann = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)
    out: list[str] = []
    with open(path or COHORT, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            e = icann((r.get("hostname") or "").strip().lower())
            d = f"{e.domain}.{e.suffix}" if e.domain and e.suffix else ""
            if d and d not in out:
                out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="*", help="smoke test: probe these, print, write nothing")
    ap.add_argument("--cohort", action="store_true", help="probe every registered domain of cohort.csv not yet on file")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if not a.cohort:
        for d in a.domains:
            r = probe(d.strip().lower())
            print(f"{r['registered_domain']}: {r['status']} ({r['detail']}) {r['answers']}")
        return 0
    if not adopted():
        raise SystemExit("the registration does not name wildcard_probe.csv yet: the amendment is still a "
                         "draft. Append it to PREREG_subdomain_evasion.md, dated, and commit; until then "
                         "this probes only domains given on the command line and writes nothing.")
    have = set()
    if os.path.exists(a.out):
        have = {d for d, r in latest(a.out).items() if not transport_failure(r)}
    todo = [d for d in cohort_domains() if d not in have]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fresh = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for d in todo:
            w.writerow(probe(d))
            f.flush()
    print(f"[+] {len(todo)} registered domain(s) probed, {len(have)} already on file -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
