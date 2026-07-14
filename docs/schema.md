# Schema / Data Dictionary — PhishVN URL Phishing Dataset (P1a)

The released dataset is the **URL** table `dataset_url.csv`, split into `train/val/test` (the
`splits/` folder) by a **group-aware temporal** rule (grouped by registrable domain). A companion
`vn_compphish.csv` re-featurises the same URLs into the exact CompPhish schema for cross-dataset
study. Text-message/email/QR channels are NOT part of this release.

## `dataset_url.csv` — core columns

| Column | Type | Meaning | Values/example |
|---|---|---|---|
| `id` | string | Anonymised id (uuid5 over content) | `a1b2...` |
| `channel` | enum | Channel | `url`, `social` |
| `label` | enum | Label | `phishing`, `benign` |
| `source` | string | Provenance | `tinnhiemmang`, `tinnhiem_web`, `tinnhiem_org`, `tranco` |
| `scenario` | enum | Impersonation sector (from URL brand tokens) | `bank`, `gov`, `tax`, `ecommerce`, `telecom`, `delivery`, `social`, `gaming`, `other` |
| `impersonated_org` | string | Impersonated org (if provided; often empty) | `Vietcombank` |
| `collected_at` | date | First-seen date (temporal split) | `2025-02-18` |
| `label_source` | enum | How the label was assigned | `feed` |
| `tier` | enum | Label-confidence tier | `gold`, `silver` |
| `status` | string | Original source handling status | `Đã xử lý`, `Đang xử lý` |
| `split` | enum | Partition | `train`, `val`, `test` |
| `lang` | enum | Language tag | `vi`, `mixed` |

## `dataset_url.csv` — basic URL features

| Column | Meaning |
|---|---|
| `url` | URL as collected |
| `url_norm` | Normalised URL (scheme stripped, lowercased) |
| `domain` | Host/domain |
| `tld` | Top-level domain |
| `url_len`, `num_dots`, `num_subdomains` | Length / dot count / subdomain depth |
| `has_ip`, `has_at`, `suspicious_tld` | IP-literal / `@` present / scam-prone TLD (0/1) |

## `vn_compphish.csv` — CompPhish-aligned modelling schema (21 features + label)

`url_len, dom_len, is_ip, tld_len, subdom_cnt, letter_cnt, digit_cnt, special_cnt, eq_cnt, qm_cnt,`
`amp_cnt, dot_cnt, dash_cnt, under_cnt, letter_ratio, digit_ratio, spec_ratio, slash_cnt, entropy,`
`path_len, query_len` — plus `label` (and `split`).

> All length/count/entropy features are computed on the **scheme-stripped** URL. The `is_https`
> column is kept for CompPhish compatibility but is a **collection artefact** and is **excluded from
> modelling**.

## Preliminary content subset (gated tier)
For ~112 phishing URLs a captured `landing/*.html` page and ~104 `landing_shots/*.png` screenshots
are available, referenced by record id, with `title`, `num_forms`, `html_len` where captured.

## Conventions
- **PII** (phone numbers, emails, names, account numbers): never in the released set; any `id↔PII`
  mapping stays private and encrypted, not released.
- **Temporal split:** ordered by `collected_at`, grouped by registrable domain; train = oldest,
  test = newest — measures drift and prevents campaign leakage.
- Never modify `raw/`; all transformations happen when building `processed/`.
