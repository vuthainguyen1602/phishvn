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
- **Size:** 8,510 records — 2,588 phishing, 5,922 legitimate.
- **Channels:** `url` (with a few `social`). No SMS/email/QR in this release.
- **Labels:** `phishing` / `benign`, inherited from the sources.
- **Confidence tiers:** `gold` (7,588: source-verified/handled + curated benign) and `silver`
  (922: under-processing phishing + Tranco benign).
- **Scenario labels:** impersonation sector inferred from URL brand tokens (bank, government, tax,
  e-commerce, telecom, delivery, social, gaming, other).
- **Preliminary content subset:** ~112 rendered HTML pages and ~104 screenshots (phishing side),
  referenced by record id.

## 3. Collection
- **Phishing URLs:** NCSC "Tin Nhiem Mang" blacklist (single-source in this release; international
  feeds such as PhishTank/OpenPhish/URLhaus are a planned addition).
- **Benign URLs:** the certified trusted-organisation registry (easy negatives) + a Tranco top-list
  sample (hard negatives) + the trusted-org directory.
- **Time span:** phishing first-seen dates span 2020–2025; benign registry entries are largely
  undated (a current snapshot).
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
- Label-quality audit: two annotators re-check a random sample; Cohen's `κ` = [to be filled].

## 5. Distribution
- **Open tier (CC BY 4.0):** URL table, CompPhish-aligned features, labels, splits, documentation.
- **Gated tier (on request):** captured phishing HTML pages and screenshots (live malicious/
  brand-cloning captures; research-use only, not for redistribution).
- **License:** CC BY 4.0 (data); MIT (code). **Attribution:** NCSC "Tin Nhiem Mang"; the Tranco list.

## 6. Maintenance
- **Versioning:** semantic version + date (this is v1.0.0). **Updates:** planned to counter concept
  drift and to add channels/coverage. **Contact:** nvthai@utc2.edu.vn.

## 7. Limitations & bias
- **URL-only:** the released data is the URL modality; the "Vietnamese" signal is strongest in page
  content, which is only a small preliminary subset here.
- **Benign composition:** dominated by a curated trusted-organisation registry; the Tranco sample
  adds harder negatives but should be scaled up for stronger external validity.
- **Single phishing source** (NCSC) may skew toward brand-impersonation sites.
- **Content subset covers the phishing side**; benign page capture is future work.
