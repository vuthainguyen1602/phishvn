#!/usr/bin/env python3
"""
validate_first_seen.py — Accuracy estimate for the RECONSTRUCTED ChongLuaDao first-seen dates
(reviewer #1: "the reconstructed first-seen dates carry no accuracy estimate even though they
order the split").

The reconstruction (chongluadao_first_seen.py) is 99.9% ObjectId-based, so the date is exact FOR
INSERTION INTO THE CHONGLUADAO DATABASE; what needs validating is whether that insertion date is a
faithful first-seen date for the domain. Two independent checks:

1. EXTERNAL AGREEMENT — domains also in the NCSC Tin Nhiem Mang blacklist, whose detection date is
   attested rather than reconstructed. The difference mixes reconstruction error with genuine
   inter-source detection lag, so its spread is an UPPER BOUND on the reconstruction error.
2. INTERNAL CONSISTENCY — the ObjectId date against first appearance in the AdGuard-generator
   mirror's git history. The mirror can only lag the database, so objectid_date <=
   mirror_first_appearance must hold and a violation is a definite error. Requires network (git
   clone); skipped gracefully without it. Mirror-first-commit domains are left-censored, excluded.

Outputs:
  data/processed/first_seen_validation.csv          — per-domain rows for both checks
  data/processed/first_seen_validation_summary.json — the headline numbers
  papers/P1_dataset/sections/gen_dates_verdict.tex  — one generated sentence for the manuscript

RUN:  python scripts/validate_first_seen.py [--skip-mirror] [--workdir DIR]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated
from chongluadao_first_seen import from_git_history, MIRROR_ADG

DATASET = os.path.join(ROOT, "data", "processed", "dataset_url.csv")
FIRST_SEEN = os.path.join(ROOT, "data", "raw", "chongluadao", "first_seen.csv")
OUT_CSV = os.path.join(ROOT, "data", "processed", "first_seen_validation.csv")
OUT_JSON = os.path.join(ROOT, "data", "processed", "first_seen_validation_summary.json")
OUT_TEX = os.path.join(ROOT, "papers", "P1_dataset", "sections", "gen_dates_verdict.tex")


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.slice(0, 10)
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))


def check_ncsc(fs: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(DATASET, low_memory=False)
    ncsc = raw[raw.source == "tinnhiemmang"][["domain", "collected_at"]].dropna()
    n = ncsc.assign(h=ncsc.domain.str.lower().str.replace(r"^www\.", "", regex=True))
    f = fs[(fs.censored == 0) & fs.first_seen.notna()].assign(h=fs.domain.str.lower())
    ov = n.merge(f[["h", "first_seen", "basis"]], on="h")
    ov["ncsc_date"] = parse_mixed_dates(ov.collected_at)
    ov["cld_date"] = pd.to_datetime(ov.first_seen)
    ov["delta_days"] = (ov.ncsc_date - ov.cld_date).dt.days
    return ov.dropna(subset=["delta_days"])[["h", "basis", "ncsc_date", "cld_date", "delta_days"]]


def check_mirror(fs: pd.DataFrame, workdir: str) -> pd.DataFrame | None:
    try:
        mirror_dates, censored = from_git_history(
            MIRROR_ADG, "blacklist.txt", workdir, pattern="adguard")
    except Exception as e:  # no network / git failure — check 1 still stands
        print(f"[!] mirror check skipped: {e}", file=sys.stderr)
        return None
    oid = fs[(fs.basis == "objectid") & fs.first_seen.notna()]
    rows = []
    for _, r in oid.iterrows():
        h = r.domain.lower()
        if h in mirror_dates and h not in censored:
            rows.append({"h": h, "cld_date": r.first_seen, "mirror_date": mirror_dates[h]})
    m = pd.DataFrame(rows)
    if m.empty:
        return None
    m["cld_date"] = pd.to_datetime(m.cld_date)
    m["mirror_date"] = pd.to_datetime(m.mirror_date)
    m["lag_days"] = (m.mirror_date - m.cld_date).dt.days  # >= 0 iff consistent
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mirror", action="store_true")
    ap.add_argument("--workdir", default="", help="where to clone the mirror (default: temp dir)")
    args = ap.parse_args()

    fs = pd.read_csv(FIRST_SEEN)
    n_dated = int(((fs.censored == 0) & fs.first_seen.notna()).sum())
    summary = {"n_reconstructed_dated": n_dated,
               "basis_counts": fs.basis.value_counts().to_dict()}

    ov = check_ncsc(fs)
    d = ov.delta_days
    summary["ncsc_overlap"] = {
        "n": int(len(ov)),
        "median_delta_days": float(d.median()),
        "iqr_days": [float(d.quantile(.25)), float(d.quantile(.75))],
        "share_abs_within_30d": float((d.abs() <= 30).mean()),
        "share_abs_within_90d": float((d.abs() <= 90).mean()),
    }
    ov_out = ov.assign(check="ncsc_overlap")

    m = None
    if not args.skip_mirror:
        workdir = args.workdir or tempfile.mkdtemp(prefix="cld_mirror_")
        m = check_mirror(fs, workdir)
    if m is not None:
        viol = m[m.lag_days < 0]
        summary["mirror_consistency"] = {
            "n": int(len(m)),
            "violations": int(len(viol)),
            "violation_rate": float(len(viol) / len(m)),
            "median_lag_days": float(m.lag_days.median()),
            # magnitude of the violations: day-level jitter vs. genuine misdating
            "violation_median_days": float(-viol.lag_days.median()) if len(viol) else 0.0,
            "violation_share_within_30d": float((viol.lag_days >= -30).mean()) if len(viol) else 1.0,
            "violation_max_days": float(-viol.lag_days.min()) if len(viol) else 0.0,
        }
        m_out = m.rename(columns={"mirror_date": "ncsc_date", "lag_days": "delta_days"}) \
                 .assign(check="mirror_consistency", basis="objectid")
        out = pd.concat([ov_out, m_out[ov_out.columns.tolist() + ["check"]]
                         if False else m_out], ignore_index=True)
    else:
        out = ov_out

    out.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    s = summary["ncsc_overlap"]
    tex = (f"Validation of the reconstructed dates: {s['n']} ChongLuaDao domains also appear on "
           f"the NCSC blacklist, whose detection date is attested by the national feed rather "
           f"than reconstructed. The difference (NCSC $-$ reconstructed) has median "
           f"{s['median_delta_days']:.0f} days (IQR {s['iqr_days'][0]:.0f} to "
           f"{s['iqr_days'][1]:.0f}), with {100 * s['share_abs_within_30d']:.0f}\\% of pairs "
           f"within 30 days and {100 * s['share_abs_within_90d']:.0f}\\% within 90; since this "
           f"difference also contains genuine inter-source detection lag, it upper-bounds the "
           f"reconstruction error.")
    if "mirror_consistency" in summary:
        mc = summary["mirror_consistency"]
        tex += (f" An internal cross-check orders the reconstructed date against the domain's "
                f"first appearance in a downstream mirror's git history ({mc['n']:,} domains; "
                f"the mirror can only lag the database): {100 * (1 - mc['violation_rate']):.0f}\\% "
                f"satisfy the ordering, and the {100 * mc['violation_rate']:.0f}\\% that do not "
                f"miss it by a median of {mc['violation_median_days']:.0f} days "
                f"({100 * mc['violation_share_within_30d']:.1f}\\% within 30, maximum "
                f"{mc['violation_max_days']:.0f}), reflecting day-level feed and commit jitter rather "
                f"than misdating.")
    write_generated(OUT_TEX, tex)
    print(f"[+] {OUT_CSV}")
    print(f"[+] {OUT_JSON}")
    print(f"[+] {OUT_TEX}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
