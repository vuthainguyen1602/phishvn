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

COPY_DIRS = ["scripts", "tests", "configs"]           # safe: code + config only
COPY_FILES = ["Makefile", "requirements.txt", "dvc.yaml",
              "LICENSE", "LICENSE-CODE", "CITATION.cff"]
DOCS_FROM = "data/docs"                                # schema/datasheet/data_sources -> docs/

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
  DOI: `10.17632/XXXXXXXX` *(replace after curation)*.
- Captured phishing HTML/screenshots — research-only **gated** tier, on request.

To reproduce the baselines, download the open tier and place `dataset_url.csv` (and the `splits/`)
under `data/processed/`, then run `make url`.
"""

README = """# PhishVN — Vietnamese URL Phishing Dataset & Baselines

Code and documentation for **PhishVN**, an open, time-stamped Vietnamese URL phishing dataset
(2,588 phishing / 5,922 legitimate) with a CompPhish-aligned 21-feature lexical schema,
impersonation-scenario labels, gold/silver confidence tiers, and a group-aware temporal split.

> **Dataset (with DOI):** Mendeley Data `10.17632/XXXXXXXX` *(replace after curation)* — CC BY 4.0.
> This repository holds the **code** (MIT); the **data** is archived separately at the DOI above.

## What's here
- `scripts/` — collection, normalisation, feature extraction, baselines, robustness, release tools.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "public"))
    args = ap.parse_args()
    os.chdir(ROOT)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    def ignore(_dir, names):                       # keep dirs clean of caches
        return [n for n in names if n in ("__pycache__", ".pytest_cache") or n.endswith(".pyc")]

    for d in COPY_DIRS:
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(args.out, d), ignore=ignore)
    for f in COPY_FILES:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(args.out, f))

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
