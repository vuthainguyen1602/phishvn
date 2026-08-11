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
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COPY_DIRS = ["configs"]                                # safe: config only
# dvc.yaml ships because the README lists it and both its stages (normalize, train_url) are exported.
# CITATION.cff is NOT here: it is version-bound and handled below.
COPY_FILES = ["requirements.txt", "LICENSE", "LICENSE-CODE", "dvc.yaml"]
DOCS_FROM = "data/docs"

# WHITELIST EVERYTHING THAT GROWS. tests/ and data/docs/ were copied wholesale until 2026-08-11,
# on the assumption that anything landing there is public-safe. Both grew: tests/ gained suites for
# the private papers (the claims verifier, P2's paired eval, P3's Jaccard band) and data/docs/
# gained scenario_grounding.md, which documents an unpublished paper's lure generator. A re-export
# would have published all four -- and the three test files import scripts that are deliberately
# NOT exported, so `pytest -q` on the mirror did not even collect. Name what ships, like scripts do.
INCLUDE_TESTS = ["test_pipeline.py"]                   # the only suite whose imports are exported
INCLUDE_DOCS = ["datasheet.md", "schema.md", "data_sources.md"]

# The mirror describes the deposit a reader can actually download, which is not necessarily the cut
# this tree builds. CITATION.cff tracks the local corpus: it had already moved to version 3.0.0,
# 53,116 records and the reserved `.3` DOI while the deposit serving readers was still v2. Exporting
# it verbatim would ship a citation resolving nowhere, describing a corpus nobody can fetch, and
# contradicting the README two paragraphs later. So the citation is version-bound: it is exported
# only once this constant names the version it describes, and the mirror keeps its published
# citation until then. Bump this when the next version actually goes live on Mendeley.
PUBLISHED_DOI = "10.17632/b97hxbxtpd.2"

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
    "watch_chongluadao.py",        # the live ChongLuaDao watcher feeding the corpus
    "vn_filter.py",                # is-this-VN-targeting test used across collection
    "build_brand_tokens.py",       # registry-derived brand tokens the filter matches on
    "make_p1a_assets.py",          # regenerate the paper's figure + tables from data
    "make_release.py",             # package the citable open/gated release
    "make_public_repo.py",         # this exporter
    "genfile.py",                  # atomic writer every asset generator goes through
    "figstyle.py",                 # house palette + rcParams (and it installs the axis guard)
    "axguard.py",                  # refuses to write a figure that clips its own data
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
  DOI: [`{doi}`](https://doi.org/{doi}).
- Captured phishing HTML/screenshots — research-only **gated** tier, on request.

To reproduce the baselines, download the open tier and place `dataset_url.csv` (and the `splits/`)
under `data/processed/`, then run `make url`.
"""

README = """# PhishVN — Vietnamese URL Phishing Dataset & Baselines

Code and documentation for **PhishVN**, an open, time-stamped Vietnamese URL phishing dataset
(51,362 URLs: a 17,079-record verified core of 2,588 phishing / 14,491 legitimate, plus a
34,283-record community/feed expansion) with a CompPhish-aligned 21-feature lexical schema,
impersonation-scenario labels, gold/silver confidence tiers, and a group-aware temporal split.

> **Dataset (with DOI):** Mendeley Data [`{doi}`](https://doi.org/{doi}) — CC BY 4.0.
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
    prev_cff = ""                                  # survives the clean; see PUBLISHED_DOI
    if os.path.exists(p := os.path.join(args.out, "CITATION.cff")):
        prev_cff = open(p, encoding="utf-8").read()
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

    # the citation ships only when it describes the published deposit (see PUBLISHED_DOI)
    local_cff = open("CITATION.cff", encoding="utf-8").read() if os.path.exists("CITATION.cff") else ""
    cited = re.search(r"^doi:\s*(\S+)\s*$", local_cff, re.M)
    if cited and cited.group(1) == PUBLISHED_DOI:
        open(os.path.join(args.out, "CITATION.cff"), "w", encoding="utf-8").write(local_cff)
    elif prev_cff:
        open(os.path.join(args.out, "CITATION.cff"), "w", encoding="utf-8").write(prev_cff)
        print(f"[!] CITATION.cff describes {cited.group(1) if cited else 'an unknown DOI'}, not the "
              f"published {PUBLISHED_DOI} — kept the mirror's published citation instead.")

    # copy ONLY the whitelisted, P1a-relevant scripts (not the whole scripts/ dir)
    os.makedirs(os.path.join(args.out, "scripts"), exist_ok=True)
    for s in INCLUDE_SCRIPTS:
        src = os.path.join("scripts", s)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "scripts", s))
    # a trimmed Makefile whose targets only reference the exported scripts
    with open(os.path.join(args.out, "Makefile"), "w", encoding="utf-8") as f:
        f.write(MAKEFILE)

    os.makedirs(os.path.join(args.out, "tests"), exist_ok=True)
    for t in INCLUDE_TESTS:
        src = os.path.join("tests", t)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "tests", t))

    os.makedirs(os.path.join(args.out, "docs"), exist_ok=True)
    for fn in INCLUDE_DOCS:
        src = os.path.join(DOCS_FROM, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "docs", fn))

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    with open(os.path.join(args.out, "data", "README.md"), "w", encoding="utf-8") as f:
        f.write(DATA_README.format(doi=PUBLISHED_DOI))
    with open(os.path.join(args.out, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(PUBLIC_GITIGNORE)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README.format(doi=PUBLISHED_DOI))

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

    # CLOSURE ASSERTION: an exported file may not import a script we deliberately kept back. This
    # is how a whitelist rots -- the export still succeeds, but the mirror cannot run, and the
    # failure only shows up for whoever clones it (which is a reviewer). Nested imports count:
    # test_pipeline.py reaches for its collection modules inside test bodies.
    private = {f[:-3] for f in os.listdir("scripts")
               if f.endswith(".py") and f not in INCLUDE_SCRIPTS}
    # Two deliberate exemptions, both imports on paths this mirror cannot reach:
    #   hpo_gwo backs `--tune --tune-method gwo`, an unreleased paper's study;
    #   p3_jaccard_check defines the guardrail make_release applies to the paraphrase attack set,
    #     which belongs to an unreleased paper and whose CSVs are absent here -- make_release
    #     aborts on the missing input before it can reach the import.
    private -= {"hpo_gwo", "p3_jaccard_check"}
    dangling = []
    for sub, names in (("scripts", INCLUDE_SCRIPTS), ("tests", INCLUDE_TESTS)):
        for fn in names:
            p = os.path.join(args.out, sub, fn)
            if not os.path.exists(p):
                continue
            src = open(p, encoding="utf-8").read()
            for m in re.findall(r"^\s*(?:from|import)\s+([a-z_][a-z0-9_]*)", src, re.M):
                if m in private:
                    dangling.append(f"{sub}/{fn} -> {m}")
    if dangling:
        raise SystemExit("SAFETY: exported file imports a non-exported script: "
                         + ", ".join(sorted(set(dangling))))

    n = sum(len(fs) for _, _, fs in os.walk(args.out))
    print(f"[+] public repo assembled at {args.out}  ({n} files)")
    print("    excluded: papers/, proposal/, data/raw, data/interim, data/processed, data/private")


if __name__ == "__main__":
    main()
