# PhishVN — Vietnamese URL Phishing Dataset & Baselines

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
