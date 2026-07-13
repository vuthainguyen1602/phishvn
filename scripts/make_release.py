#!/usr/bin/env python3
"""
make_release.py — Package a citable PhishVN release for a DOI repository (Mendeley Data / Zenodo).

Produces two clearly separated bundles under release/:
  * PhishVN_v<ver>_open.zip   — OPEN tier (CC BY 4.0): URL table, features, labels, splits, docs,
                                 LICENSE, CITATION.cff, an auto-generated README + MANIFEST (SHA-256).
                                 This is what you upload for open access and cite in the paper.
  * PhishVN_v<ver>_pages_GATED.zip (only with --include-pages) — the captured phishing HTML +
                                 screenshots. GATED / on-request only; contains live malicious page
                                 captures. Do NOT upload to the open tier.

RUN:
  python scripts/make_release.py --version 1.0.0
  python scripts/make_release.py --version 1.0.0 --include-pages
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import hashlib
import os
import shutil
import zipfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OPEN_FILES = [
    ("data/processed/dataset_url.csv", "data/dataset_url.csv"),
    ("data/processed/vn_compphish.csv", "data/vn_compphish.csv"),
    ("data/processed/splits/url_train.csv", "data/splits/url_train.csv"),
    ("data/processed/splits/url_val.csv", "data/splits/url_val.csv"),
    ("data/processed/splits/url_test.csv", "data/splits/url_test.csv"),
    ("data/docs/datasheet.md", "docs/datasheet.md"),
    ("data/docs/schema.md", "docs/schema.md"),
    ("data/docs/data_sources.md", "docs/data_sources.md"),
    ("LICENSE", "LICENSE"),
    ("CITATION.cff", "CITATION.cff"),
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_stats(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lab = Counter(r.get("label", "") for r in rows)
    tier = Counter(r.get("tier", "") for r in rows)
    split = Counter(r.get("split", "") for r in rows)
    scen = Counter(r.get("scenario", "") for r in rows)
    return len(rows), lab, tier, split, scen


def _readme(version, stats):
    n, lab, tier, split, scen = stats
    top_scen = ", ".join(f"{k}={v}" for k, v in scen.most_common())
    return f"""# PhishVN v{version} — Vietnamese URL Phishing Dataset (open tier)

Time-stamped Vietnamese URL phishing dataset with a CompPhish-aligned 21-feature lexical schema,
impersonation-scenario labels, gold/silver label-confidence tiers, and a group-aware temporal split.
Released {_dt.date.today().isoformat()} under CC BY 4.0 (see LICENSE). Cite via CITATION.cff / the DOI.

## Contents
- `data/dataset_url.csv` — full record table ({n} rows).
- `data/vn_compphish.csv` — the same URLs re-featurised into the exact CompPhish schema.
- `data/splits/url_{{train,val,test}}.csv` — the official group-aware temporal split.
- `docs/` — datasheet, column schema, and data-source notes.
- `MANIFEST.txt` — SHA-256 checksums and row counts for every file.

## Composition
- Labels: {dict(lab)}
- Tiers: {dict(tier)}
- Splits: {dict(split)}
- Scenarios: {top_scen}

## Notes
- Benign = NCSC trusted-organisation registry (easy) + Tranco top-list sample (hard negatives).
- `is_https` is present for CompPhish compatibility but is a collection artefact — exclude it from
  modelling (all lexical features are computed on the scheme-stripped URL).
- The captured phishing HTML/screenshot subset is distributed separately as a GATED bundle
  (research-only; live malicious captures) — not part of this open tier.
- PII is redacted; no id↔PII mapping is included. Complies with Decree 13/2023/ND-CP and Vietnam's PDPL.
"""


def write_bundle(zip_path, files, extra_texts):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
        for arc, text in extra_texts.items():
            z.writestr(arc, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--out", default=os.path.join(ROOT, "release"))
    ap.add_argument("--include-pages", action="store_true",
                    help="Also build the GATED HTML/screenshot bundle (research-only).")
    args = ap.parse_args()

    os.chdir(ROOT)
    missing = [s for s, _ in OPEN_FILES if not os.path.exists(s)]
    if missing:
        raise SystemExit("Missing (run `make data` first?): " + ", ".join(missing))

    stats = _dataset_stats("data/processed/dataset_url.csv")
    readme = _readme(args.version, stats)

    # MANIFEST with checksums (computed on the real source files)
    manifest = [f"PhishVN v{args.version} — MANIFEST (SHA-256)", ""]
    for src, arc in OPEN_FILES:
        size = os.path.getsize(src)
        manifest.append(f"{_sha256(src)}  {arc}  ({size} bytes)")
    manifest_txt = "\n".join(manifest) + "\n"

    open_zip = os.path.join(args.out, f"PhishVN_v{args.version}_open.zip")
    write_bundle(open_zip, OPEN_FILES,
                 {"README.md": readme, "MANIFEST.txt": manifest_txt})
    print(f"[+] open tier  -> {open_zip}  ({os.path.getsize(open_zip)} bytes)")

    if args.include_pages:
        pages = []
        for d, arc_d in [("data/raw/landing", "pages/landing"),
                         ("data/raw/landing_shots", "pages/landing_shots")]:
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    pages.append((os.path.join(d, fn), f"{arc_d}/{fn}"))
        care = ("GATED / RESEARCH-ONLY.\n\nThese are captured LIVE PHISHING pages (HTML) and "
                "screenshots that clone real brands. Do NOT redistribute as attack material, do NOT "
                "open the HTML outside an isolated VM/container, and do NOT upload to an open-access "
                "tier. Share on request under a data-use agreement only. CC BY 4.0 does not waive "
                "these safety obligations.\n")
        gated_zip = os.path.join(args.out, f"PhishVN_v{args.version}_pages_GATED.zip")
        write_bundle(gated_zip, pages, {"HANDLE_WITH_CARE.txt": care})
        print(f"[+] gated tier -> {gated_zip}  ({len(pages)} artefacts, {os.path.getsize(gated_zip)} bytes)")

    print("Next: upload the OPEN zip to Mendeley Data (or Zenodo), get the DOI, then set it in "
          "CITATION.cff and papers/P1a_dataset/preamble.tex (\\datadoi).")


if __name__ == "__main__":
    main()
