#!/usr/bin/env python3
"""
chongluadao_first_seen.py — Reconstruct per-domain FIRST-SEEN dates for the ChongLuaDao
community blacklist from public traces, since the project's API is offline and the main
mirror squashes its git history (only 3 commits survive).

Three independent traces, merged with "most-authoritative wins":

1. EXACT dates from Wayback-archived API JSON (2021-05 .. 2023-10): each entry's `_id` is a
   MongoDB ObjectId whose first 4 bytes encode the creation unix timestamp — so every entry
   present in ANY archived snapshot gets its exact insertion date, even entries created long
   before the snapshot was taken.
2. First-appearance dates from the git history of the AdGuard-generator mirror
   (phamleduy04/adguard-generator-chongluadao, ~105 real commits of blacklist.txt).
3. First-appearance dates from the main mirror's 3 surviving commits
   (elliotwutingfeng/ChongLuaDao-Phishing-Blocklist, urls.txt).

Domains whose earliest trace is the FIRST snapshot of a git history (and that have no
ObjectId date) are LEFT-CENSORED — they were blacklisted at some unknown earlier time — and
are emitted with an empty date plus censored=1, so downstream they stay in the undated
(random-split) pool rather than getting a fake date.

Output: data/raw/chongluadao/first_seen.csv  (domain, first_seen, basis, censored)
        merged by normalize_merge.load_chongluadao() into collected_at.

RUN:
  python scripts/chongluadao_first_seen.py                       # downloads + clones
  python scripts/chongluadao_first_seen.py --skip-wayback        # git histories only
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HEADERS = {"User-Agent": "research (contact: nvthai@utc2.edu.vn)"}
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
API_URL = "api.chongluadao.vn/v1/blacklist"
MIRROR_MAIN = "https://github.com/elliotwutingfeng/ChongLuaDao-Phishing-Blocklist.git"
MIRROR_ADG = "https://github.com/phamleduy04/adguard-generator-chongluadao.git"
HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$", re.I)
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def norm_host(u: str) -> str:
    """'http://sub.x.com/*' / '||x.com^' / 'x.com/path' -> 'sub.x.com' (empty if not host-like)."""
    h = (u or "").strip().lower()
    h = re.sub(r"^\|\|", "", h)
    h = re.sub(r"[\^|$].*$", "", h)   # AdGuard rule suffixes: ^, |, $all, $important, ...
    h = re.sub(r"^[a-z]+://", "", h)
    h = h.split("/")[0].split(":")[0].strip(".")
    h = h[4:] if h.startswith("www.") else h
    if HOST_RE.match(h) or IP_RE.match(h):
        return h
    return ""


def _get(url, params=None, tries=6):
    import requests
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=180)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(6 * (a + 1))
    return None


def objectid_date(oid: str) -> str:
    """First 8 hex chars of a Mongo ObjectId = unix seconds of creation."""
    try:
        ts = int(oid[:8], 16)
        if 1262304000 <= ts <= 1893456000:  # sanity: 2010..2030
            return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()
    except (ValueError, TypeError):
        pass
    return ""


def from_wayback() -> dict[str, str]:
    """domain -> exact ObjectId creation date, from every archived API snapshot."""
    r = _get(WAYBACK_CDX, {"url": API_URL, "output": "json",
                           "filter": "statuscode:200", "collapse": "digest", "limit": 200})
    if not r:
        print("[!] Wayback CDX unreachable; skipping ObjectId trace", file=sys.stderr)
        return {}
    dates: dict[str, str] = {}
    stamps = [row[1] for row in r.json()[1:]]
    for st in stamps:
        snap = _get(f"https://web.archive.org/web/{st}id_/https://{API_URL}")
        if not snap:
            print(f"  [!] snapshot {st}: fetch failed", file=sys.stderr)
            continue
        try:
            data = snap.json()
        except ValueError:
            continue
        items = data if isinstance(data, list) else data.get("data", [])
        n_new = 0
        for it in items:
            host = norm_host(str(it.get("url", "")))
            d = objectid_date(str(it.get("_id", "")))
            if host and d and (host not in dates or d < dates[host]):
                dates[host] = d
                n_new += 1
        print(f"  [wayback {st}] {len(items)} entries, +{n_new} dated")
    return dates


def from_git_history(repo_url, path, workdir, pattern="plain") -> tuple[dict[str, str], set[str]]:
    """domain -> first commit date it appeared in `path`. Returns (dates, first_commit_cohort):
    the cohort present in the FIRST commit is left-censored (added at some unknown earlier time)."""
    dst = os.path.join(workdir, re.sub(r"[^a-z0-9]+", "_", repo_url.lower())[-40:])
    if not os.path.isdir(dst):
        subprocess.run(["git", "clone", "--quiet", repo_url, dst], check=True)
    log = subprocess.run(["git", "-C", dst, "log", "--reverse", "--format=%H %cI", "--", path],
                         capture_output=True, text=True, check=True).stdout.split()
    commits = list(zip(log[0::2], log[1::2]))
    dates: dict[str, str] = {}
    censored: set[str] = set()
    for i, (sha, iso) in enumerate(commits):
        blob = subprocess.run(["git", "-C", dst, "show", f"{sha}:{path}"],
                              capture_output=True, text=True).stdout
        day = iso[:10]
        for ln in blob.splitlines():
            if not ln or ln.startswith(("#", "!")):
                continue
            host = norm_host(ln)
            if not host or host in dates:
                continue
            dates[host] = day
            if i == 0:
                censored.add(host)
    print(f"  [git {repo_url.rsplit('/', 1)[-1]}] {len(commits)} commits, "
          f"{len(dates)} hosts ({len(censored)} in first commit = censored)")
    return dates, censored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/chongluadao/first_seen.csv")
    ap.add_argument("--workdir", default="", help="Where to clone the mirrors (default: temp dir)")
    ap.add_argument("--skip-wayback", action="store_true")
    args = ap.parse_args()
    workdir = args.workdir or tempfile.mkdtemp(prefix="cld_")

    exact = {} if args.skip_wayback else from_wayback()
    g1, c1 = from_git_history(MIRROR_ADG, "blacklist.txt", workdir)
    g2, c2 = from_git_history(MIRROR_MAIN, "urls.txt", workdir)

    # merge: exact ObjectId date wins. A host present in ANY history's FIRST commit existed
    # before that history started, so a later "appearance" in another mirror is not its real
    # first-seen -> censored. Only hosts that genuinely appear MID-history get a git date.
    hosts = set(exact) | set(g1) | set(g2)
    rows = []
    for h in sorted(hosts):
        if h in exact:
            rows.append((h, exact[h], "objectid", 0))
        elif h in c1 or h in c2:
            rows.append((h, "", "censored-first-commit", 1))
        else:
            d, src = min((g[h], src) for g, src in
                         ((g1, "adguard-mirror"), (g2, "main-mirror")) if h in g)
            rows.append((h, d, src, 0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "first_seen", "basis", "censored"])
        w.writerows(rows)
    n_dated = sum(1 for r in rows if r[1])
    print(f"[+] {args.out}: {len(rows)} hosts, {n_dated} dated "
          f"({sum(1 for r in rows if r[2] == 'objectid')} exact ObjectId), "
          f"{len(rows) - n_dated} censored/undated")


if __name__ == "__main__":
    main()
