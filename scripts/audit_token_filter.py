#!/usr/bin/env python3
"""
audit_token_filter.py — Formal audit of the Vietnamese-targeting token filter (reviewer #1: "the
token audit is reported as 'spot-checked plausible' with no sample size").

The registry-extended filter's marginal contribution is the set of feed domains flagged ONLY by the
registry-generated brand tokens (data/processed/brand_tokens.json) — not by the .vn TLD and not by
the hand-curated VN_TOKENS core. Those are exactly the domains whose flagging the paper must
defend, so the audit samples from them.

  --sample (default): classify every domain in data/interim/vn_phishing_candidates.csv by match
      basis (vn_tld / static / registry), print per-basis counts, and write a blinded, seeded,
      alphabetically-stable sample of the registry-only domains to
      data/reports/token_audit_sample.csv for manual annotation. Verdicts:
        vn-target   — the name credibly imitates a Vietnamese brand/service/authority
        not-vn      — the match is spurious
        unsure      — cannot tell from the name and public captures
      Plausibility is not evidence: a name that merely could belong to a Vietnamese business is
      `unsure`, not `vn-target` (same rule as the label-audit codebook).
  --summarize: read the annotated sheet, compute precision on audited rows (vn-target /
      (vn-target + not-vn)) with a Wilson 95% interval, report the unsure share separately, and
      write papers/P1_dataset/sections/gen_token_audit.tex.

RUN:  python scripts/audit_token_filter.py [--n 100] [--seed 0]
      python scripts/audit_token_filter.py --summarize
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated
from vn_filter import VN_TOKENS, BRAND_TOKENS

CANDIDATES = os.path.join(ROOT, "data", "interim", "vn_phishing_candidates.csv")
SHEET = os.path.join(ROOT, "data", "reports", "token_audit_sample.csv")
OUT_TEX = os.path.join(ROOT, "papers", "P1_dataset", "sections", "gen_token_audit.tex")
VERDICTS = {"vn-target", "not-vn", "unsure"}


def basis_of(domain: str) -> str:
    d = domain.lower()
    if d.endswith(".vn"):
        return "vn_tld"
    if VN_TOKENS.search(d):
        return "static"
    if BRAND_TOKENS and BRAND_TOKENS.search(d):
        return "registry"
    return "none"


def make_sample(n: int, seed: int, candidates: str = CANDIDATES):
    df = pd.read_csv(candidates)
    df["basis"] = df.domain.astype(str).map(basis_of)
    counts = df.basis.value_counts().to_dict()
    base = len(df) - counts.get("registry", 0)
    print(f"[i] snapshot {candidates}: {len(df)} flagged domains")
    print(f"    per basis: {counts}")
    print(f"    registry extension grows the flagged set from {base} to {len(df)}")
    reg = df[df.basis == "registry"].copy()
    reg["matched_token"] = reg.domain.str.lower().map(
        lambda d: BRAND_TOKENS.search(d).group(1) if BRAND_TOKENS.search(d) else "")
    reg = reg.sort_values("domain").reset_index(drop=True)  # stable order before seeding
    samp = reg if len(reg) <= n else reg.sample(n=n, random_state=seed).sort_values("domain")
    sheet = samp[["domain", "sources", "matched_token"]].assign(verdict="", notes="")
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    sheet.to_csv(SHEET, index=False)
    print(f"[+] {SHEET}  (n={len(sheet)} of {len(reg)} registry-only domains, seed={seed})")
    print("    fill `verdict` with vn-target / not-vn / unsure, then rerun with --summarize")


def wilson(p: float, n: int, z: float = 1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize():
    df = pd.read_csv(SHEET)
    v = df.verdict.astype(str).str.strip()
    bad = sorted(set(v[~v.isin(VERDICTS | {"", "nan"})]))
    if bad:
        sys.exit(f"[!] unknown verdicts {bad}; allowed: {sorted(VERDICTS)}")
    if (v.isin({"", "nan"})).any():
        sys.exit(f"[!] {int(v.isin({'', 'nan'}).sum())} of {len(df)} rows unannotated")
    n = len(df)
    k = {t: int((v == t).sum()) for t in VERDICTS}
    resolved = k["vn-target"] + k["not-vn"]
    prec = k["vn-target"] / resolved if resolved else float("nan")
    lo, hi = wilson(prec, resolved)
    tex = (f"A seeded random sample of $n={n}$ of the domains flagged only by the "
           f"registry-generated tokens was manually audited under the same "
           f"plausibility-is-not-evidence rule as the label audit: {k['vn-target']} were "
           f"credibly Vietnamese-targeting (brand imitations or genuinely Vietnamese-language "
           f"names), {k['not-vn']} were spurious matches and "
           f"{k['unsure']} could not be resolved from the name and public captures. On the "
           f"resolved rows the extension's precision is {prec:.2f} (Wilson 95\\% CI "
           f"[{lo:.2f},\\,{hi:.2f}]).")
    write_generated(OUT_TEX, tex)
    print(f"[+] {OUT_TEX}")
    print(f"    n={n}  vn-target={k['vn-target']}  not-vn={k['not-vn']}  unsure={k['unsure']}"
          f"  precision={prec:.3f} [{lo:.3f},{hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", default=CANDIDATES,
                    help="unified-feed snapshot to audit; must have been fetched with the "
                         "registry extension active, or the registry-only stratum is empty")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        make_sample(args.n, args.seed, args.candidates)


if __name__ == "__main__":
    main()
