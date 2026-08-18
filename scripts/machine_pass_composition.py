#!/usr/bin/env python3
"""
machine_pass_composition.py — a first, machine estimate of what the positive class contains.

WHAT THIS IS NOT. It is not the label audit and cannot stand in for it. The audit's product is
Cohen's kappa between two INDEPENDENT annotators; one pass by one process produces no such thing,
and a machine that has very likely seen the public ChongLuaDao lists cannot honestly claim to have
judged a domain without consulting the feed being audited. This writes to its own file and never
touches annotator_A.csv / annotator_B.csv, so the blinded instrument stays untouched and the human
result can later be compared against this rather than contaminated by it.

WHAT IT DOES. It judges ARCHIVED PAGE CONTENT, not the domain string. For each sampled row with a
web-archive snapshot near its first-seen date it fetches the archived HTML and extracts objective
signals -- a password or OTP input, gambling vocabulary in visible text, a Vietnamese business
registration marker, adult or investment vocabulary -- then classifies by rule. The classification
is therefore reproducible and auditable rather than a judgement call, which is the only form of
machine estimate worth putting next to a human audit.

WHY `unsure` IS LARGE AND SHOULD BE. Most of these pages are JavaScript shells whose archived body
is a loader, and a clone inherits its victim's address, hotline and registration text verbatim --
so `legitimate` is reachable only from the trusted-organisation registry, or from a printed
registration NUMBER on a domain carrying no Vietnamese brand token. Rows with no snapshot, a dead
snapshot, or no rule firing stay `unsure`. That share is the finding, not a tuning failure: it
measures how much of this corpus can be adjudicated from archives at all.

WHAT THE FIRST FULL RUN ACTUALLY SHOWED (200 rows, 2026-08-11) — READ THIS BEFORE RE-RUNNING.
The pass does not work, and the sample's benign arm is what proves it. The sample is stratified over
(label x tier) across the WHOLE dataset, so 56 of the 200 rows are known-legitimate sites: an
unplanned but decisive control group. The credential tier fired on 20/144 (13.9%) of listed domains
and 7/56 (12.5%) of known-legitimate ones — Fisher p = 1.000. A password field means "this site has
a login", not "this site harvests credentials"; motcuadientu.backan.gov.vn, a provincial e-government
portal, trips it exactly as a bank clone does. Separating the two needs the brand-mismatch judgement
this pass deliberately refuses to make, so the tier carries no information here.

Coverage is the second failure and it bounds the first: only 24/144 listed rows (17%) resolved at
all, because 72 have no retrievable archived body and 48 are JavaScript shells. Among those 24 the
pass found 0 legitimate, which puts a 95% Wilson upper bound of 13.8% on the feed-error rate AMONG
RESOLVABLE ROWS ONLY — a statement about 17% of the arm, not about the arm.

The conclusion to carry forward is therefore about method, not composition: this corpus cannot be
adjudicated from web archives, and the human audit's `unsure` share will be large for the same
reason. That is a finding for the datasheet's limitations, and it is the only thing here worth
quoting. Do not quote the verdict percentages.

RUN:  python scripts/audit/machine_pass_composition.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs  # noqa: E402
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
import vn_filter
EVID = os.path.join(ROOT, "data", "docs", "verify", "EVIDENCE.csv")
OUT = os.path.join(ROOT, "data", "docs", "verify", "MACHINE_PASS.csv")
UA = "Mozilla/5.0 (research; contact nvthai@utc2.edu.vn)"

CRED_FIELD = re.compile(r'<input[^>]*type\s*=\s*["\']?password', re.I)
CRED_NAME = re.compile(r'(?:name|id)\s*=\s*["\'][^"\']*(?:pass|matkhau|mat_khau|otp|pin|cvv|'
                       r'sotaikhoan|so_the|cardnum)', re.I)
CRED_TEXT = re.compile(r'mật khẩu|đăng nhập|otp|mã xác (?:thực|minh)|số thẻ|internet banking|'
                       r'smart ?otp|tên đăng nhập', re.I)
GAMBLE = re.compile(r'tài xỉu|nổ hũ|cá cược|nhà cái|đặt cược|xóc đĩa|baccarat|casino|'
                    r'quay hũ|game bài|đánh bài|soi cầu|lô đề|kèo nhà cái', re.I)
ADULT = re.compile(r'phim sex|khiêu dâm|jav |xvideos|porn', re.I)
INVEST = re.compile(r'lợi nhuận cao|sàn giao dịch|đầu tư sinh lời|forex|ủy thác đầu tư|'
                    r'lãi suất cao|chốt lời', re.I)
# A REGISTRATION NUMBER, not a company name and not a copyright line. The first version of this
# rule accepted "công ty cổ phần", "bản quyền thuộc" and a bare (c) year, and it called bidv.xyz --
# a BIDV lookalike on .xyz -- legitimate, on boilerplate copied wholesale from the bank it clones.
# That is the one error this pass must not make: a false `legitimate` inflates the feed-error rate,
# which is the only number anybody would quote from it.
BIZREG = re.compile(r'mã số (?:thuế|dn|doanh nghiệp)\s*:?\s*\d{10}|'
                    r'giấy (?:phép|chứng nhận) (?:đăng ký )?(?:kinh doanh|đầu tư)\s*(?:số)?\s*:?\s*\d{5}',
                    re.I)


def lookalike(domain: str) -> bool:
    """Does the domain carry a Vietnamese brand token? Reuses the registry-generated token list the
    rest of the repo already uses, so this veto stays in step with the brand machinery rather than
    drifting against a second hand-kept copy."""
    d = vn_filter.host_of(domain)
    return bool(vn_filter.VN_TOKENS.search(d)
                or (vn_filter.BRAND_TOKENS and vn_filter.BRAND_TOKENS.search(d)))


def searchable(html: str) -> str:
    """Visible text, plus <title> and the meta/og description.

    The meta tags are included because a large share of these pages are JavaScript shells: the body
    holds a loader and 200 characters of boilerplate while the page's own description of itself sits
    in <head>. Those tags are still content the operator wrote, not an inference of mine, so reading
    them widens coverage without weakening what a match means. Script bodies stay excluded — a
    vocabulary hit inside a bundled library says nothing about the page.
    """
    parts = []
    for pat in (r"(?is)<title[^>]*>(.*?)</title>",
                r'(?is)<meta[^>]+(?:name|property)\s*=\s*["\'](?:description|og:title|'
                r'og:description|keywords)["\'][^>]*content\s*=\s*["\']([^"\']*)',
                r'(?is)<meta[^>]+content\s*=\s*["\']([^"\']*)["\'][^>]*(?:name|property)\s*=\s*'
                r'["\'](?:description|og:title|og:description|keywords)["\']'):
        parts += re.findall(pat, html)
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", " ".join(parts) + " " + body)[:200_000]


def _decode(raw: bytes, headers) -> str:
    """Decompress and decode. Both steps had to be explicit: the archive serves gzip for a large
    share of snapshots, and urllib does not decompress it, so bidv.xyz classified as `no-signal`
    on a body that was still compressed bytes. Charset comes from the page's own meta tag where
    it declares one -- Vietnamese pages of this era are not all UTF-8."""
    enc = (headers.get("Content-Encoding") or "").lower()
    if "gzip" in enc:
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
    elif "deflate" in enc:
        try:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
        except Exception:
            pass
    head = raw[:2048].decode("ascii", errors="ignore")
    m = re.search(r'charset\s*=\s*["\']?\s*([\w-]+)', head, re.I)
    for cs in ([m.group(1)] if m else []) + ["utf-8", "windows-1258", "latin-1"]:
        try:
            return raw.decode(cs)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="ignore")


class RateLimited(Exception):
    """The archive is refusing us, which is NOT the same fact as a missing snapshot.

    Conflating the two is how this pass would quietly lie: a throttled run records `no-content` for
    every remaining row, the unresolved share climbs, and the output still looks like a completed
    measurement of how auditable the corpus is. The first full run tripped exactly this -- the
    archive began refusing connections outright partway through -- so refusal now raises, backs off
    on a scale that outlasts a throttle, and aborts the run rather than filling the file.
    """


def fetch(url: str, retries: int = 4):
    """Returns page text, or None when the snapshot genuinely has no body. Raises RateLimited."""
    for a in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return _decode(r.read(), r.headers)
        except urllib.error.HTTPError as e:
            if e.code in (404, 403):        # indexed but not in storage — a real absence
                return None
            if e.code in (429, 503, 504):   # throttled — long backoff, then give up on the run
                if a == retries - 1:
                    raise RateLimited(f"HTTP {e.code}")
                time.sleep(30 * (a + 1))
                continue
            if a == retries - 1:
                return None
            time.sleep(2 ** a)
        except urllib.error.URLError as e:
            # ConnectionRefusedError arrives here: the archive drops the TCP connection when it
            # throttles, so this is a rate-limit signal, not a network fault of ours.
            if isinstance(getattr(e, "reason", None), ConnectionRefusedError):
                if a == retries - 1:
                    raise RateLimited("connection refused")
                time.sleep(30 * (a + 1))
                continue
            if a == retries - 1:
                return None
            time.sleep(2 ** a)
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(2 ** a)
    return None


def classify(html: str, domain: str = "", registry: bool = False):
    """Precedence: credential harvesting outranks other abuse; legitimacy needs positive evidence.

    The `legitimate` tier is deliberately the hardest to reach, because it is the tier that would
    be quoted. A registry entry is decisive on its own -- that is the codebook's source 4. Content
    alone qualifies only when the page prints a registration number AND the domain carries no
    Vietnamese brand token, since a clone inherits its victim's registration text verbatim.
    """
    if registry:
        return "legitimate", "registry"
    if not html:
        return "unsure", "no-content"
    txt = searchable(html)
    if CRED_FIELD.search(html) or CRED_NAME.search(html):
        return "phishing", "credential-input"
    if CRED_TEXT.search(txt):
        return "phishing", "credential-text"
    for name, pat, why in (("scam", GAMBLE, "gambling-text"), ("scam", ADULT, "adult-text"),
                           ("scam", INVEST, "investment-text")):
        if pat.search(txt):
            return name, why
    if BIZREG.search(txt) and len(txt) > 400:
        if lookalike(domain):
            return "unsure", "bizreg-lookalike-veto"
        return "legitimate", "business-registration"
    return "unsure", "no-signal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--restart", action="store_true", help="discard prior rows instead of resuming")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(EVID, newline="", encoding="utf-8")))
    rows = rows[:args.limit] if args.limit else rows

    # Resume, because a throttled run must be restartable without re-fetching what already
    # succeeded -- re-fetching is what provokes the throttle in the first place.
    done = {}
    if os.path.exists(OUT) and not args.restart:
        done = {r["vid"]: r for r in csv.DictReader(open(OUT, newline="", encoding="utf-8"))
                if r.get("signal") != "rate-limited"}
        if done:
            print(f"    resuming: {len(done)} rows already fetched", flush=True)

    out, stopped = [], None
    for i, r in enumerate(rows, 1):
        if r["vid"] in done:
            out.append(done[r["vid"]])
            continue
        snap = (r.get("wayback_nearest") or "").strip()
        html = None
        if snap:
            try:
                # the id_ modifier returns the archived bytes without the Wayback banner/rewrites
                html = fetch(re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap, count=1))
            except RateLimited as e:
                stopped = f"{e} at row {i}"
                out.append({"vid": r["vid"], "url": r["url"], "machine_verdict": "unsure",
                            "signal": "rate-limited", "snapshot": snap})
                break
            time.sleep(args.sleep)
        verdict, why = classify(html, r["url"], (r.get("registry") or "").strip() == "yes")
        out.append({"vid": r["vid"], "url": r["url"], "machine_verdict": verdict,
                    "signal": why, "snapshot": snap})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "url", "machine_verdict", "signal", "snapshot"])
        w.writeheader(); w.writerows(out)

    if stopped:
        print(f"\n[!] the archive throttled us ({stopped}); {len(out)} of {len(rows)} rows written.")
        print("    Re-run the same command later to resume — fetched rows are not re-fetched.")
        print("    The totals below are NOT a result: the unresolved share is inflated by the stop.")

    from collections import Counter
    c = Counter(r["machine_verdict"] for r in out)
    print(f"\n[+] {OUT}   (NOT the audit — see this file's script docstring)")
    for k, v in c.most_common():
        print(f"      {k:12s} {v:4d}  ({100.0 * v / len(out):5.1f}%)")
    dec = len(out) - c["unsure"]
    if dec:
        print(f"    resolved by content: {dec}/{len(out)}; among those, credential phishing "
              f"{100.0 * c['phishing'] / dec:.0f}%, other scam {100.0 * c['scam'] / dec:.0f}%, "
              f"legitimate {100.0 * c['legitimate'] / dec:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
