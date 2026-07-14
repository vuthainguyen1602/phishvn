# PhishVN — Data sources (released P1a URL dataset)

Describes the sources of the **released URL dataset**. Legal basis: **Decree 13/2023/ND-CP** and the
**Personal Data Protection Law** (effective 2026-01-01). Release only after PII redaction (phone
numbers, names, account/card numbers). Text-message/email/QR channels are planned extensions and are
not part of this release.

---

## Phishing (positive class)

| Source | How / tool | In release | Legal / ethical | Mitigation |
|---|---|---|---|---|
| tinnhiemmang.vn (NCSC "Tin Nhiem Mang" blacklist, `type=fake`) | `scrape_vn_phishing.py --endpoint filter --type fake` (full list via `filterObj`; plain `?page=` is capped ~1k) | **Yes** — 2,588 phishing URLs | Public-interest data; respect robots/rate limits | Throttle (`--delay`); auto CSRF+session; never re-host live payloads |
| urlscan.io snapshots | `fetch_urlscan.py` | Preliminary content subset (~112 HTML, ~104 screenshots) | API quota / ToS | Use existing snapshots; store scan UUID; gated tier |
| PhishTank / OpenPhish / URLhaus | feed download, filter `.vn` | **Planned** (not in v1.0) | Feed licences | Keep per-record licence note |

## Legitimate (negative class)

| Source | How / tool | In release | Notes |
|---|---|---|---|
| Certified trusted-organisation registry (Tin Nhiem Mang, `type=web`) | `scrape_trusted_orgs.py` | **Yes** — 5,879 benign (easy negatives) | Curated, verified `.vn` sites |
| Tranco top-list | `fetch_tranco.py` (stratified across rank bands) | **Yes** — hard benign negatives (23 in v1.0; scale up planned) | Avoids trusted-vs-malicious shortcut; cite the list ID |
| Trusted-org directory | `scrape_trusted_orgs.py --enrich` | **Yes** — 20 benign | Org name + website |

---

## Processing & release
- Build with `normalize_merge.py`: common schema, scheme-independent URL features, scenario inference
  from URL brand tokens, deduplication, and a **group-aware temporal split** (grouped by registrable
  domain).
- Feature parity with CompPhish via `align_compphish.py`; `is_https` excluded from modelling.
- **Open tier (CC BY 4.0):** URL features, labels, splits, docs. **Gated tier (on request):** the
  captured phishing HTML/screenshots (live malicious captures — research use only).
- Attribution: NCSC "Tin Nhiem Mang"; the Tranco list (Le Pochat et al., NDSS 2019).

## Notes
- The released benign set is dominated by the curated registry; the Tranco sample adds harder
  negatives and should be scaled up for stronger external validity.
- Phishing is currently single-source (NCSC); international feeds are a planned addition.
