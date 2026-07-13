#!/usr/bin/env python3
"""
whois_dns_enrich.py — Host/third-party features for LIVE domains (WHOIS/DNS/SSL). Optional P1 enrichment.

Adds: domain_age_days (WHOIS creation), has_dns_a, has_mx, ns_count, ssl_valid.
Best-effort: dead phishing domains often have no DNS/SSL; WHOIS may still give age. Failures -> blank.

INSTALL: pip install python-whois dnspython
RUN:
  python whois_dns_enrich.py --in data/processed/dataset_url.csv --col domain \
         --out data/interim/host_features.csv --limit 500
"""
from __future__ import annotations
import argparse, csv, os, socket, ssl, time
from datetime import datetime, timezone


def whois_age(domain):
    try:
        import whois
        w = whois.whois(domain)
        c = w.creation_date
        if isinstance(c, list):
            c = c[0]
        if c:
            if c.tzinfo is None:
                c = c.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - c).days
    except Exception:
        pass
    return ""


def dns_feats(domain):
    a = mx = ns = ""
    try:
        import dns.resolver
        r = dns.resolver.Resolver(); r.lifetime = 5
        try:
            a = 1 if r.resolve(domain, "A") else 0
        except Exception:
            a = 0
        try:
            mx = 1 if r.resolve(domain, "MX") else 0
        except Exception:
            mx = 0
        try:
            ns = len(list(r.resolve(domain, "NS")))
        except Exception:
            ns = 0
    except Exception:
        pass
    return a, mx, ns


def ssl_valid(domain):
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(5); s.connect((domain, 443))
        return 1
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--col", default="domain")
    ap.add_argument("--out", default="data/interim/host_features.csv")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.inp, newline="", encoding="utf-8") as f:
        doms, seen = [], set()
        for r in csv.DictReader(f):
            d = (r.get(args.col) or "").strip().lower()
            if d and d not in seen:
                seen.add(d); doms.append(d)
    doms = doms[:args.limit]

    fields = ["domain", "domain_age_days", "has_dns_a", "has_mx", "ns_count", "ssl_valid"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, d in enumerate(doms, 1):
            a, mx, ns = dns_feats(d)
            w.writerow({"domain": d, "domain_age_days": whois_age(d), "has_dns_a": a,
                        "has_mx": mx, "ns_count": ns, "ssl_valid": ssl_valid(d) if a else 0})
            f.flush()
            print(f"[{i}/{len(doms)}] {d}")
            time.sleep(args.delay)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
