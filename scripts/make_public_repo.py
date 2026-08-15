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
import ast
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

# The label-audit instruments, which P1 states are released with the code. The blinded sheets and
# the codebook ship; key.csv NEVER does -- it maps each blinded id to the source label, so
# publishing it would hand a future annotator the answers and destroy the blinding the whole
# instrument depends on.
# MACHINE_PASS.csv ships because a negative result nobody can inspect is just an assertion: it holds
# the rule-based verdicts whose failure against the sample's benign control is the reason the
# datasheet says this corpus cannot be adjudicated from archives. It carries no source labels, so it
# un-blinds nothing; the codebook forbids annotators opening it before their own sheet is done.
INCLUDE_VERIFY = ["CODEBOOK.md", "annotator_A.csv", "annotator_B.csv", "MACHINE_PASS.csv", "adjudicated.csv"]
VERIFY_FROM = "data/docs/verify"
# Audit artefacts the revised manuscript declares released: the annotated token-filter sample
# (verdict + rationale per row) and the per-domain first-seen validation rows. Paths are
# (source, exported name); they land in docs/verify/ beside the label-audit instruments.
INCLUDE_AUDITS = [
    ("data/reports/token_audit_sample.csv", "token_audit_sample.csv"),
    ("data/processed/first_seen_validation.csv", "first_seen_validation.csv"),
    ("data/processed/first_seen_validation_summary.json", "first_seen_validation_summary.json"),
    # the unified-feed snapshot the token audit sampled from (836 -> 1,115 claim), archived so
    # the audited numbers are reproducible against a fixed input rather than a moving feed
    ("data/interim/vn_phishing_candidates_20260812.csv", "feed_snapshot_20260812.csv"),
]

# The mirror describes the deposit a reader can actually download, which is not necessarily the cut
# this tree builds. CITATION.cff tracks the local corpus: it had already moved to version 3.0.0,
# 53,116 records and the reserved `.3` DOI while the deposit serving readers was still v2. Exporting
# it verbatim would ship a citation resolving nowhere, describing a corpus nobody can fetch, and
# contradicting the README two paragraphs later. So the citation is version-bound: it is exported
# only once this constant names the version it describes, and the mirror keeps its published
# citation until then. Bump this when the next version actually goes live on Mendeley.
PUBLISHED_DOI = "10.17632/b97hxbxtpd.3"

# PROSE GATE. The whitelist and the closure assertion read code; nothing read the comments and
# docstrings, and those are what leaked. Four exported modules had been documented for a private
# audience -- they named the papers a defect was found in and quoted the numbers involved, from
# manuscripts that are not out. It surfaced only because a diff happened to be read before the
# push, which is not a control.
#
# A label is the tell, because attributing a finding is what turns an engineering note into a
# disclosure. "A floor pinned above a series minimum" says how the bug works and is safe;
# naming the paper and quoting the two numbers involved publishes its result. Tested against that
# incident, this catches all four files. It is a floor and not a ceiling: an unattributed leak
# reads as ordinary prose and passes, so this raises the cost of leaking carelessly rather than
# making it impossible.
#
# P1 is deliberately absent: it is the paper this mirror accompanies, so prose matching it is the
# code describing itself. Including it took the flag rate from 4/19 to 12/19, all noise.
PAPER_LABEL = re.compile(r"\bP[2-8][ab]?\b(?!\w)|papers/P[2-8]|P[2-8]_[a-z]+")

# file -> why its labels are legitimate. A reason is required, and waived labels are still printed
# on every export, so a NEW one appearing in a waived file cannot hide behind the old decision.
PROSE_WAIVERS = {
    "make_public_repo.py": "the export policy has to name which papers are held back",
    "normalize_merge.py": "\"P2 corpus\" is the external benchmark corpus, not the manuscript",
}

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
    "fetch_chongluadao.py",        # import the ChongLuaDao mirror snapshot (named in the paper)
    "chongluadao_first_seen.py",   # reconstruct per-domain first-seen dates (named in the paper)
    "validate_first_seen.py",      # accuracy estimate for the reconstructed dates (rev. #1)
    "audit_token_filter.py",       # sampled audit of the VN-targeting token filter (rev. #1)
    "export_p1a_results.py",       # the deposited benchmark-evidence bundle's generator
    "derive_abuse_type.py",        # types the positive class; its output ships in the open tier
    "collect_audit_evidence.py",   # gathers the lookup evidence the released audit sheets need
    "machine_pass_composition.py", # the archive-content pass, and the control showing it fails
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
(53,116 URLs: an 18,997-record verified core of 2,587 phishing / 16,410 legitimate, plus a
34,119-record community/feed bronze expansion) with a CompPhish-aligned 21-feature lexical
schema, impersonation-scenario labels, gold/silver/bronze confidence tiers, and a group-aware
temporal split.

> **Dataset (with DOI):** Mendeley Data [`{doi}`](https://doi.org/{doi}) — CC BY 4.0.
> This repository holds the **code** (MIT); the **data** is archived separately at the DOI above.

## What's here
- `scripts/` — URL data collection, normalisation, CompPhish features, baselines, audit, release tools.
- `docs/` — datasheet, column schema, data-source notes.
- `docs/verify/` — the completed human label audit: pre-registered codebook (with its amendment
  log), both annotators' independent sheets, the machine pass, and the arbitration record.
- `tests/`, `configs/`, `Makefile`, `dvc.yaml` — reproducibility.

## Label audit (completed 2026-08-15)
Two annotators independently re-checked a blinded, stratified 200-row sample against a four-way
codebook (credential *phishing* / other *scam* / *legitimate* / *unsure*) that forbids consulting
any blocklist and treats plausibility as non-evidence. Independent-round agreement: 0.710
(Cohen's κ 0.609) four-way; 0.820 (κ 0.725) collapsed to the abuse-vs-legitimate distinction the
released binary label makes. After documented arbitration of the 58 disagreements, 149/200 rows
resolve; against the source labels the positive arm shows **12.1% label noise** (95% CI
7.1–20.0%, concentrated in the explicitly-tagged bronze stratum) and the benign arm 4.0%.
Every number is recomputable from `docs/verify/` via `make_verification_sample.py score`; the
full record is in `docs/datasheet.md`.

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
- `machine_pass_composition.py` — rule-based pass over the archived page behind each sampled
  domain. It reports a negative result and `docs/verify/MACHINE_PASS.csv` is that result: the
  sample's known-legitimate stratum acts as a control, the credential rule fires on it as often as
  on listed domains (13.9% vs 12.5%, Fisher p = 1.00), and only 17% of listed rows resolve at all.
  Read it as evidence that this corpus cannot be adjudicated from web archives — not as a
  composition estimate, and never as a substitute for the human audit.
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


def _prose(path: str) -> str:
    """Comments and docstrings only. The code is not what leaks -- a function named
    make_p5_assets says nothing a reader could not infer from the repository layout."""
    src = open(path, encoding="utf-8").read()
    out = [m.group(1) for m in re.finditer(r"^\s*#\s?(.*)$", src, re.M)]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "\n".join(out)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append(doc)
    return "\n".join(out)


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

    os.makedirs(os.path.join(args.out, "docs", "verify"), exist_ok=True)
    for fn in INCLUDE_VERIFY:
        src = os.path.join(VERIFY_FROM, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "docs", "verify", fn))
    for src, name in INCLUDE_AUDITS:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "docs", "verify", name))

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

    for dp, _, fns in os.walk(args.out):
        if "key.csv" in fns:
            raise SystemExit("SAFETY: the audit key reached the export at "
                             + os.path.relpath(os.path.join(dp, "key.csv"), args.out)
                             + " — it un-blinds the annotation sheets and must never ship.")

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

    # PROSE ASSERTION: no exported comment or docstring may attribute anything to an unreleased
    # paper. See PAPER_LABEL above for what this does and does not buy.
    leaks = []
    for sub, names in (("scripts", INCLUDE_SCRIPTS), ("tests", INCLUDE_TESTS)):
        for fn in names:
            p = os.path.join(args.out, sub, fn)
            if not os.path.exists(p):
                continue
            labels = sorted({m.group(0) for m in PAPER_LABEL.finditer(_prose(p))})
            if not labels:
                continue
            why = PROSE_WAIVERS.get(fn)
            if why:
                print(f"[waived] {sub}/{fn} names {', '.join(labels)} — {why}")
            else:
                leaks.append(f"{sub}/{fn}: {', '.join(labels)}")
    if leaks:
        raise SystemExit(
            "SAFETY: exported prose attributes something to an unreleased paper:\n  "
            + "\n  ".join(leaks)
            + "\n  Describe the mechanism without the attribution, or add a PROSE_WAIVERS entry"
              " saying why the mention is legitimate.")

    # Skip .git: os.walk counted the mirror's own object database, so the "files" figure moved with
    # every commit made there (272 -> 282 across one commit) and read as if the export had grown.
    # A number printed by the safety script has to mean what it says.
    n = 0
    for dp, dns, fs in os.walk(args.out):
        dns[:] = [d for d in dns if d != ".git"]
        n += len(fs)
    print(f"[+] public repo assembled at {args.out}  ({n} files)")
    print("    excluded: papers/, proposal/, data/raw, data/interim, data/processed, data/private")


if __name__ == "__main__":
    main()
