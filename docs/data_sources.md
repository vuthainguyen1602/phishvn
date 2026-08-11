# PhishVN — Data sources (released P1a URL dataset)

Describes the sources of the **released URL dataset**. Legal basis: **Decree 13/2023/ND-CP** and the
**Personal Data Protection Law** (effective 2026-01-01). Release only after PII redaction (phone
numbers, names, account/card numbers). Text-message/email/QR channels are planned extensions and are
not part of this release.

---

## Phishing (positive class)

| Source | How / tool | In release | Legal / ethical | Mitigation |
|---|---|---|---|---|
| tinnhiemmang.vn (NCSC "Tin Nhiem Mang" blacklist, `type=fake`) | `scrape_vn_phishing.py --endpoint filter --type fake` (full list via `filterObj`; plain `?page=` is capped ~1k) | **Yes** — 2,588 phishing URLs, tier gold/silver. NOTE: the public feed has published no detections after 2025-02-18 (verified 2026-07-14) | Public-interest data; respect robots/rate limits | Throttle (`--delay`); auto CSRF+session; never re-host live payloads |
| ChongLuaDao community blacklist (chongluadao.vn) | `fetch_chongluadao.py` — the project's public API is offline; imports the last daily-scrape mirror snapshot (2024-05-16, github.com/elliotwutingfeng/ChongLuaDao-Phishing-Blocklist). `chongluadao_first_seen.py` reconstructs per-domain first-seen dates (~51%) from Wayback-archived API ObjectIds + mirror git history | **Yes** — 33,823 domains, tier **bronze** (community-reported; ~50% dated, rest random-split; excluded from the primary benchmark; with 296 OpenPhish URLs the bronze stratum totals 34,119) | Community project data; cite chongluadao.vn + mirror | 623 registrable domains independently confirmed by the NCSC feed (cross-source agreement) |
| OpenPhish community feed | `scrape_vn_phishing.py --feeds` | **Yes** — 296 recent URLs, tier **bronze** (international; kept out of the Vietnamese-context benchmark; role: live-URL content capture + drift) | Community feed licence (attribution) | Refresh regularly; per-record `scraped_at` |
| urlscan.io snapshots | `fetch_urlscan.py` | Preliminary content subset (HTML + screenshots) | API quota / ToS | Use existing snapshots; store scan UUID; gated tier |
| URLhaus | — | **Excluded on purpose** — tracks malware distribution, not phishing | — | — |
| PhishTank | feed download, filter `.vn` | Planned | Feed licence | Keep per-record licence note |

## Legitimate (negative class)

| Source | How / tool | In release | Notes |
|---|---|---|---|
| Certified trusted-organisation registry (Tin Nhiem Mang, `type=web`) | `scrape_trusted_orgs.py` | **Yes** — 5,879 benign (easy negatives) | Curated, verified `.vn` sites |
| Tranco top-list (global sample) | `fetch_tranco.py` (stratified across rank bands) | **Yes** — 6,000 hard benign negatives | Avoids trusted-vs-malicious and "non-.vn → phishing" shortcuts; cite the list ID |
| Tranco top-list (`.vn` slice) | `fetch_tranco.py --vn-only` | **Yes** — 2,592 popular `.vn` domains (`tranco_vn`) | Anchors FPR on sites Vietnamese users actually visit |
| Trusted-org directory | `scrape_trusted_orgs.py --enrich` | **Yes** — 2,026 benign (re-scraped with `--enrich` 2026-07-16; was 20 in v2) | Org name + website |

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
- Benign has three source-tagged strata: curated registry (easy) < `.vn` Tranco slice
  (Vietnamese-visited sites) < global Tranco sample (hard negatives).
- Phishing is multi-source: NCSC (verified, time-stamped; feed dormant since 2025-02-18),
  ChongLuaDao community snapshot (bronze; ~50% first-seen-dated via archival reconstruction),
  OpenPhish (bronze, international, live).
  The primary gold+silver benchmark uses NCSC phishing only.
