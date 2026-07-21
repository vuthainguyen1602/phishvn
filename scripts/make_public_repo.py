#!/usr/bin/env python3
"""
make_public_repo.py — Assemble a CLEAN, public-safe code repo for PhishVN into ./public/.

Whitelist-only: copies exactly the files a public code release needs (scripts, tests, configs,
build files, licences, docs) and NOTHING else. It never copies papers/, proposal/, data/raw,
data/interim, data/processed, or data/private — so manuscripts, the roadmap, raw data and the
id<->PII mapping cannot leak. The dataset itself lives on Mendeley/Zenodo (DOI); the public repo
only points at it.

RUN:
  python scripts/make_public_repo.py            # build ./public/
  python scripts/make_public_repo.py --out /tmp/phishvn-public
"""
from __future__ import annotations
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COPY_DIRS = ["tests", "configs"]                      # safe: config + tests only
COPY_FILES = ["requirements.txt", "LICENSE", "LICENSE-CODE", "CITATION.cff"]
DOCS_FROM = "data/docs"                                # schema/datasheet/data_sources -> docs/

# Only the scripts that build/reproduce the RELEASED P1a URL dataset and its baselines are exported,
# so the public repo matches the published data. Scripts for unreleased channels/papers (SMS, email,
# images, PhoBERT, fusion, LLM red-team, drift, edge — P1b/P2/P6/P7) stay in the private repo and
# are added when those papers and their data are released.
INCLUDE_SCRIPTS = [
    "scrape_vn_phishing.py",       # collect phishing URLs (NCSC blacklist + feeds)
    "scrape_trusted_orgs.py",      # collect benign trusted-org URLs
    "fetch_tranco.py",             # hard benign negatives (Tranco)
    "fetch_urlscan.py",            # preliminary HTML/screenshot subset
    "whois_dns_enrich.py",         # optional URL/host enrichment
    "normalize_merge.py",          # build the unified URL dataset + group-aware temporal split
    "compphish_features.py",       # CompPhish-aligned URL feature schema
    "align_compphish.py",          # re-featurise URLs into the CompPhish schema
    "train_url_baseline.py",       # URL baselines (multi-seed + bootstrap CI)
    "make_verification_sample.py", # label-quality audit (Cohen's kappa)
    "make_p1a_assets.py",          # regenerate the paper's figure + tables from data
    "make_release.py",             # package the citable open/gated release
    "make_public_repo.py",         # this exporter
]

PUBLIC_GITIGNORE = """# never commit data or private material to the public repo
data/raw/
data/interim/
data/processed/
data/private/
!data/**/.gitkeep
# models / tracking / build artefacts
models/
mlruns/
release/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
.idea/
"""

DATA_README = """# Data

The PhishVN **dataset is not stored in this Git repository**. It is archived with a DOI:

- Open tier (URL table, features, labels, splits, docs) — CC BY 4.0 — Mendeley Data,
  DOI: [`10.17632/b97hxbxtpd.2`](https://doi.org/10.17632/b97hxbxtpd.2).
- Captured phishing HTML/screenshots — research-only **gated** tier, on request.

To reproduce the baselines, download the open tier and place `dataset_url.csv` (and the `splits/`)
under `data/processed/`, then run `make url`.
"""

README = """# PhishVN — Vietnamese URL Phishing Dataset & Baselines

Code and documentation for **PhishVN**, an open, time-stamped Vietnamese URL phishing dataset
(51,362 URLs: a 17,079-record verified core of 2,588 phishing / 14,491 legitimate, plus a
34,283-record community/feed expansion) with a CompPhish-aligned 21-feature lexical schema,
impersonation-scenario labels, gold/silver confidence tiers, and a group-aware temporal split.

> **Dataset (with DOI):** Mendeley Data [`10.17632/b97hxbxtpd.2`](https://doi.org/10.17632/b97hxbxtpd.2) — CC BY 4.0.
> This repository holds the **code** (MIT); the **data** is archived separately at the DOI above.

## What's here
- `scripts/` — URL data collection, normalisation, CompPhish features, baselines, audit, release tools.
- `docs/` — datasheet, column schema, data-source notes.
- `tests/`, `configs/`, `Makefile`, `dvc.yaml` — reproducibility.

## Quickstart
```bash
pip install -r requirements.txt
# download the dataset from the Mendeley DOI into data/processed/ , then:
make url          # train the URL baseline (LogReg/RF, multi-seed + bootstrap CI)
make test         # unit tests
```

## Key scripts
- `normalize_merge.py` — merge raw sources -> common schema, scheme-independent URL features,
  scenario inference, group-aware temporal split.
- `compphish_features.py` / `align_compphish.py` — CompPhish 21-feature schema (cross-dataset).
- `train_url_baseline.py` — URL baseline; `--seeds` (mean±std) and `--bootstrap` (95% CI).
- `fetch_tranco.py` — hard benign negatives from the Tranco top-list.
- `make_verification_sample.py` — two-annotator label audit + Cohen's kappa.
- `make_release.py` — build the citable open / gated release bundles.

## Citation
See `CITATION.cff`. Please cite the dataset DOI and credit the upstream sources
(NCSC "Tin Nhiem Mang" and the Tranco list).

## Licence
Code: MIT (`LICENSE-CODE`). Dataset: CC BY 4.0 (`LICENSE`).
"""

MAKEFILE = """.PHONY: install data url assets release verify clean
install:      ## install python deps
\tpip install -r requirements.txt
data:         ## build the URL dataset from data/raw
\tpython scripts/normalize_merge.py --raw data/raw --out data/processed
url:          ## train URL baselines (multi-seed + bootstrap CI)
\tpython scripts/train_url_baseline.py --in data/processed/dataset_url.csv --out models/url_rf.joblib
assets:       ## regenerate the paper figure + tables from data
\tpython scripts/make_p1a_assets.py
release:      ## package the citable open-tier release (PAGES=1 for the gated bundle)
\tpython scripts/make_release.py --version $(or $(VERSION),1.0.0) $(if $(PAGES),--include-pages,)
verify:       ## run unit tests
\tpytest -q
clean:
\trm -rf models data/processed/*.csv data/processed/splits/*.csv
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "public"))
    args = ap.parse_args()
    os.chdir(ROOT)

    # Clean existing contents but PRESERVE .git (so the repo/remote survives a re-export). Then
    # `git add -A` in the export picks up removals, dropping stale files from the tracked repo.
    if os.path.exists(args.out):
        for entry in os.listdir(args.out):
            if entry == ".git":
                continue
            p = os.path.join(args.out, entry)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    else:
        os.makedirs(args.out)

    def ignore(_dir, names):                       # keep dirs clean of caches
        return [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]

    for d in COPY_DIRS:
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(args.out, d), ignore=ignore)
    for f in COPY_FILES:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(args.out, f))

    # copy ONLY the whitelisted, P1a-relevant scripts (not the whole scripts/ dir)
    os.makedirs(os.path.join(args.out, "scripts"), exist_ok=True)
    for s in INCLUDE_SCRIPTS:
        src = os.path.join("scripts", s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "scripts", s))
    # a trimmed Makefile whose targets only reference the exported scripts
    with open(os.path.join(args.out, "Makefile"), "w", encoding="utf-8") as f:
        f.write(MAKEFILE)

    os.makedirs(os.path.join(args.out, "docs"), exist_ok=True)
    if os.path.isdir(DOCS_FROM):
        for fn in os.listdir(DOCS_FROM):
            if fn.endswith(".md"):
                shutil.copy2(os.path.join(DOCS_FROM, fn), os.path.join(args.out, "docs", fn))

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    with open(os.path.join(args.out, "data", "README.md"), "w", encoding="utf-8") as f:
        f.write(DATA_README)
    with open(os.path.join(args.out, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(PUBLIC_GITIGNORE)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)

    # safety assertion: nothing forbidden slipped in
    forbidden = ("private", "raw", "interim", "processed")
    leaked = []
    for dp, _, fns in os.walk(args.out):
        for fn in fns:
            p = os.path.relpath(os.path.join(dp, fn), args.out)
            if any(seg in p.split(os.sep) for seg in ("papers", "proposal")) or \
               p.startswith(os.path.join("data", "") ) and any(x in p for x in forbidden):
                leaked.append(p)
    if leaked:
        raise SystemExit("SAFETY: forbidden files present: " + ", ".join(leaked))

    n = sum(len(fs) for _, _, fs in os.walk(args.out))
    print(f"[+] public repo assembled at {args.out}  ({n} files)")
    print("    excluded: papers/, proposal/, data/raw, data/interim, data/processed, data/private")


if __name__ == "__main__":
    main()
