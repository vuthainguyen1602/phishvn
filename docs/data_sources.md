# PhishVN — Data sourcing plan (per channel)

Reference for **P1a** (dataset paper) and IRB / legal review.
Legal basis to cite: **Decree 13/2023/ND-CP** (personal data protection) and the **Personal Data Protection Law** (effective 2026-01-01). Release only after **consent + PII redaction** (phone numbers, names, account/card numbers, OTP).

Volume estimates are order-of-magnitude planning figures, not guarantees.

---

## Phishing (positive class)

| Channel | Source | How / tool | Est. volume | Legal / ethical barrier | Mitigation |
|---|---|---|---|---|---|
| **URL** | tinnhiemmang.vn (NCSC blacklist + trusted orgs) | `scrape_vn_phishing.py --endpoint filter --type fake` (full ~110k via `filterObj`; the plain `?page=` listing is capped ~1k), `scrape_trusted_orgs.py --enrich` | ~110k domains (blacklist) | Public interest data; respect robots/rate limit | Throttle (`--delay`), auto CSRF+session, no re-hosting of live payloads |
| **URL** | ChongLuaDao.vn | community API / extension export | 10k–50k | Attribution; community ToS | Credit source, store report ID only |
| **URL** | PhishTank / OpenPhish / URLhaus | feed download, filter `.vn` + VN brands | 5k–20k (VN-relevant) | Feed licenses (mostly research-OK) | Keep license note per record |
| **URL** | urlscan.io | `fetch_urlscan.py` (search VN domains) | 5k–20k scans | API quota / ToS | Use API key, store scan UUID |
| **URL enrich** | WHOIS / DNS | `whois_dns_enrich.py` | — | Rate limits | Cache, backoff |
| **Email** | Honeypot inboxes (seed addresses posted publicly) | `imap_honeypot.py` | 1k–10k / few months | No third-party PII expected; still redact | Redact headers' personal fields, keep structure |
| **Email** | User-forwarded `.eml` (students, colleagues) | `parse_eml.py` | 200–2,000 | **Consent required**; sender PII | Consent form; strip To/Cc, hash sender |
| **Email** | Intl. corpora (SpamAssassin, Nazario) — augmentation only | direct download | 5k–20k | Research licenses | English; use to enrich, label `source=intl` |
| **SMS** | User contributions via Telegram/Zalo bot | `telegram_sms_bot.py`, `sms_import.py` | 500–5,000 | **Consent**; sender numbers = PII | In-bot consent; auto-redact numbers/OTP |
| **SMS** | VN SMS-spam datasets (PTIT / Vũ Minh Tuấn; Mendeley) | direct download | 1k–5k | Dataset licenses | Cite, keep license, dedup vs ours |
| **SMS** | 156 / 5656 anti-fraud hotline reports | formal MoU with authority | large (if granted) | Government data-sharing agreement needed | Pursue only via institutional MoU + IRB |
| **QR** | Decode QR embedded in collected email/SMS/posters | `qr_decode.py` → URL back into URL pipeline | 200–2,000 | Same as underlying channel | Decode locally, never fetch payload |
| **QR** | Warning posts (news, FB groups) with QR images | manual save + `qr_decode.py` | 100–1,000 | Image copyright / platform ToS | Store decoded URL + hash, not the image |
| **QR** | Field capture (quishing stickers on posters/parking) | phone photo + decode | 50–500 | Location/context only | No personal data captured |

## Benign (negative class — needed for a usable dataset)

| Channel | Source | Tool | Notes |
|---|---|---|---|
| URL | tinnhiemmang trusted-org list; top VN sites (bank/gov/edu/e-commerce) | `scrape_trusted_orgs.py`, `safe_crawl.py` | Balance the classes; verify still-legit |
| Email | Legitimate newsletters/transactional mail (opt-in) | `parse_eml.py` | With consent |
| SMS | Legitimate brand/OTP/notification SMS (contributed) | `sms_import.py` | Redact OTP values, keep template |

---

## Practical reality

- **URL** scales automatically today → this is the backbone of P1a.
- **Email / SMS / QR** have **no large ready-to-download VN corpus** → must be self-collected via honeypots + contribution channels (Telegram/Zalo bot, `.eml` forwards). This scarcity is exactly why P1a is publishable: **no existing multi-channel Vietnamese phishing dataset.**
- **QR** is the newest, thinnest, highest-novelty slice — most QR lures ultimately resolve to a URL, so `qr_decode.py` feeds them back into the URL pipeline.

## Release checklist (before DOI)

1. Consent recorded for all user-contributed email/SMS/QR.
2. PII redaction pass (phone, name, account/card, OTP) + manual spot-check.
3. Tiered release: fully public (URLs + derived features), gated (redacted text) on request with a data-use agreement.
4. Datasheet + schema shipped (`data/docs/`).
5. Simulated/"defanged" links only for any adversarial (LLM-generated) subset.
