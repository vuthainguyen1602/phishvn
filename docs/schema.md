# Schema / Data Dictionary — Vietnamese Phishing Dataset (P1)

The dataset is split by **channel** at the `processed/` layer: `dataset_url.csv` (url/qr/social), `dataset_sms.csv`, `dataset_email.csv`. Each set is split into `train/val/test` **by time** (the `splits/` folder).

## Core columns (every record)

| Column | Type | Meaning | Values/example |
|---|---|---|---|
| `id` | string | Anonymized ID (uuid5 based on content) | `a1b2...` |
| `channel` | enum | Channel | `url`, `qr`, `social`, `sms`, `email` |
| `label` | enum | Label | `phishing`, `benign` |
| `source` | string | Source | `tinnhiemmang`, `openphish`, `urlhaus`, `author`, `report`, `llm` |
| `is_llm` | 0/1 | LLM-generated (adversarial set) | `0` / `1` |
| `lang` | enum | Language | `vi`, `en`, `mixed` |
| `scenario` | enum | Scenario/sector | `bank`, `gov`, `tax`, `ecommerce`, `telecom`, `delivery`, `social`, `other` |
| `impersonated_org` | string | Impersonated organization | `Viettel`, `Bộ Công An` |
| `collected_at` | date | Timestamp (for temporal splitting) | `2025-02-18` |
| `label_source` | enum | Label source | `feed`, `human`, `auto` |
| `split` | enum | Set | `train`, `val`, `test` |

## Channel-specific columns — URL / QR / social (`dataset_url.csv`)

| Column | Meaning |
|---|---|
| `url` | Original URL (normalized, PII stripped if any) |
| `url_norm` | Normalized URL (redundant scheme removed, lowercased) |
| `domain` | Domain name |
| `tld` | Top-level domain |
| `url_len` | URL length |
| `num_dots` | Number of dots |
| `num_subdomains` | Number of subdomains |
| `has_ip` | URL uses an IP (0/1) |
| `has_at` | Contains the `@` character (0/1) |
| `suspicious_tld` | TLD commonly used in scams: top/xyz/cc… (0/1) |
| `status` | Status (from tinnhiemmang): in progress/handled… |
| `qr_decoded_url` | (QR) URL decoded from the QR code |
| `html_captured` | Landing page crawled (0/1) — filled in a later step |
| `title` | (if crawled) page title |
| `num_forms` | (if crawled) number of forms |

## Channel-specific columns — SMS (`dataset_sms.csv`)

| Column | Meaning |
|---|---|
| `text` | SMS content **with PII removed** |
| `brandname` | Sender name/brandname |
| `has_url` | Contains a URL (0/1) |
| `msg_len` | Message length |

## Channel-specific columns — Email (`dataset_email.csv`)

| Column | Meaning |
|---|---|
| `subject` | Subject (PII removed) |
| `text` | Email content **with PII removed** |
| `sender_type` | Sender type (spoofed/internal…) |
| `has_url` | Contains a URL (0/1) |
| `has_attachment` | Has an attachment (0/1) |

## Conventions

- **PII** (phone numbers, emails, names, account numbers, national ID): **never** kept in the released set; the `id↔PII` mapping (if needed) stays in `data/private/` (encrypted, restricted access).
- **Temporal split**: ordered by `collected_at`; train = oldest, test = newest (measures concept drift).
- Never modify data in `raw/`; all transformations happen when building `processed/`.
