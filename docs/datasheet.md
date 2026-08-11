# Datasheet — PhishVN URL Phishing Dataset (P1a)

> Following the *Datasheets for Datasets* framework (Gebru et al., 2021). Describes the released
> P1a URL dataset. Text-message/email channels and full HTML/screenshot coverage are planned
> extensions and are NOT part of this release.

## 1. Motivation
- **Purpose:** train and evaluate Vietnamese-targeted phishing **URL** detection, and provide a
  time-stamped, leakage-controlled benchmark aligned with the CompPhish feature schema.
- **Creators:** Thai Nguyen Vu, University of Transport and Communications — Ho Chi Minh City Campus
  (UTC2), nvthai@utc2.edu.vn. **Funding:** [if any].

## 2. Composition
- **Unit:** URLs (not people, not messages).
- **Size:** 53,116 records — 36,706 phishing, 16,410 legitimate. Verified gold+silver core:
  18,997 (2,587 NCSC phishing + 16,410 benign); bronze expansion: 34,119 (33,823 ChongLuaDao +
  296 OpenPhish), ~50% of the ChongLuaDao rows carry a reconstructed first-seen date.
- **Channels:** `url` (with a few `social`). No SMS/email/QR in this release.
- **Labels:** `phishing` / `benign`, inherited from the sources.
- **Confidence tiers:** `gold` (9,593: source-verified/handled phishing + curated benign),
  `silver` (9,404: under-processing NCSC phishing + Tranco benign), and `bronze` (34,119:
  ChongLuaDao community snapshot + OpenPhish feed — undated, excluded from the primary benchmark).
- **Scenario labels:** impersonation sector inferred from URL brand tokens (bank, government, tax,
  e-commerce, telecom, delivery, social, gaming, other).
- **Preliminary content subset:** 933 rendered HTML pages and 869 screenshots (209 phishing, 660 benign;
  screenshots include urlscan.io archive captures of live OpenPhish URLs), referenced by record
  id / scan UUID. Note: unauthenticated urlscan DOM downloads are rate-limited — set
  `URLSCAN_API_KEY` (free registration) before scaling the HTML capture.

## 3. Collection
- **Phishing URLs:** three sources — NCSC "Tin Nhiem Mang" blacklist (verified, time-stamped;
  gold/silver), the ChongLuaDao community blacklist (last public snapshot 2024-05-16, via the
  daily-scrape mirror, since the project's API went offline; bronze — per-domain first-seen dates
  reconstructed for ~50% via `chongluadao_first_seen.py` from Wayback-archived API ObjectIds and
  mirror git history), and the OpenPhish community feed (recent international URLs, bronze).
  623 registrable domains appear independently in both NCSC and ChongLuaDao (cross-source
  agreement). URLhaus is excluded on purpose: it tracks malware distribution rather than phishing.
- **Benign URLs:** three source-tagged strata of increasing difficulty — the certified
  trusted-organisation registry/whitelist (`tinnhiem_web`/`tinnhiem_org`, easy), a `.vn`-filtered
  Tranco slice (`tranco_vn`, popular sites Vietnamese users actually visit), and a global Tranco
  sample (`tranco`, hard negatives that break the "non-.vn → phishing" TLD shortcut).
- **Time span:** phishing first-seen dates span 2020-02-08 – 2025-02-18; benign registry entries
  are largely undated (a current snapshot). Note: the public NCSC feed has published no detections
  dated after 2025-02-18 (verified against the feed's date-filter endpoint on 2026-07-14), so this
  release captures the feed's full published history; monitoring continues in case it resumes.
- **Method:** polite web scraping (rate-limited, robots-respecting); content subset via urlscan.io
  snapshots.
- **Legal:** public NCSC data; complies with Decree 13/2023/ND-CP and the Personal Data Protection
  Law (effective 01/01/2026).

## 4. Preprocessing
- Deduplication by record and by **registrable domain** for splitting.
- URL features are computed on the **scheme-stripped** URL; `is_https` is retained for CompPhish
  schema compatibility but is a **collection artefact** and is excluded from modelling.
- **Group-aware temporal split** (70/15/15) grouped by registrable domain, so campaign subdomains
  never span train/test.
- **PII redaction** (phone/email/account numbers → tokens); no id↔PII mapping is released.
- Label-quality audit: two annotators independently re-check a blinded 200-row stratified sample
  (annotation in progress); Cohen's `κ` and the consensus-vs-source label-noise estimate will be
  reported here on completion (`make_verification_sample.py score`).

## 5. Distribution
- **Open tier (CC BY 4.0):** URL table, CompPhish-aligned features, labels, splits, documentation.
- **Gated tier (on request):** captured phishing HTML pages and screenshots (live malicious/
  brand-cloning captures; research-use only, not for redistribution).
- **License:** CC BY 4.0 (data); MIT (code). **Attribution:** NCSC "Tin Nhiem Mang"; the Tranco list.

## 6. Maintenance
- **Versioning:** semantic version + date (this is **v3.0.0**, DOI `10.17632/b97hxbxtpd.3`).
  **Changes since v2.0.0** (DOI `.2`, 51,362 records): three `normalize_merge` ingestion fixes
  (the `first_seen.csv` date file is no longer mis-read as a blocklist; canonical dedup strips
  scheme and trailing slash; no literal `nan` dates survive); site-level label-conflict resolution
  excludes 5 ambiguous loan/shop `.vn` sites; 21 reviewed `tinnhiem_org` misattributions removed
  via `exclude_domains.txt`; and the trusted-organisation registry was re-scraped with `--enrich`,
  taking `tinnhiem_org` benign rows from 20 to 2,026. Net effect is compositional: the benign pool
  moves from 56.9% to 61.6% `.vn` and P(benign | `.vn`) from 89.3% to 91.3% — the exact `.vn`
  shortcut the companion benchmark/XAI papers measure, so the artefact under study is ~2 pp larger
  in v3. An audit (2026-08-04) confirms the 2,026 new rows carry no label defect (0 appear on any
  phishing blocklist; 88% are `.gov.vn`/`.edu.vn`/`.org.vn`). **Updates:** planned to counter
  concept drift and to add channels/coverage. **Contact:** nvthai@utc2.edu.vn.

## 7. Limitations & bias
- **URL-only:** the released data is the URL modality; the "Vietnamese" signal is strongest in page
  content, which is only a small preliminary subset here.
- **Benign composition:** dominated by a curated trusted-organisation registry; the Tranco sample
  adds harder negatives but should be scaled up for stronger external validity.
- **Verified core is single-source** (NCSC); the bronze expansion adds ChongLuaDao + OpenPhish but is undated and community-labelled.
- **Content subset spans both classes but is a fraction of the URL table.** Phishing and benign
  pages are captured through one urlscan pipeline (209 phishing, 660 benign; 933 paired pages), so
  both classes share one capture method. As coverage is still well below the full URL table,
  content-level results remain preliminary.
