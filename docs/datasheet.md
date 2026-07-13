# Datasheet — Vietnamese Phishing Dataset (P1)

> Following the *Datasheets for Datasets* framework (Gebru et al.). Fill in/update during collection.

## 1. Motivation
- **Purpose:** train & evaluate a multi-channel Vietnamese phishing detection model, robust to LLM-generated content.
- **Creators:** Thai Nguyen Vu, University of Transport and Communications — Ho Chi Minh City Campus (UTC2), nvthai@utc2.edu.vn. **Funding:** [if any].

## 2. Composition
- **Unit:** messages (URL/QR/SMS/email) — not people.
- **Channels:** url, qr, social, sms, email. **Labels:** phishing / benign; plus `is_llm` for the adversarial set.
- **Scale (to update):** URL ~[N]; SMS ~[N]; Email ~[N]; LLM ~[N].
- **Labels:** from verified feeds (tinnhiemmang/chongluadao), the authors, or human annotators (gold set).

## 3. Collection
- **Sources:** tinnhiemmang.vn (NCSC/NCA), OpenPhish, URLhaus, author datasets (PTIT…), user-reported messages (DigiShield), LLM-generated samples.
- **Time span:** [from – to]. **Method:** crawler respecting robots.txt + rate limits; imported from reports/partners.
- **Consent/legal:** public data + a data use agreement (DUA) with partners; complies with **Decree 13/2023/ND-CP** & the **Personal Data Protection Law (effective 01/01/2026)**.

## 4. Preprocessing
- Deduplication; URL normalization; **PII anonymization** (phone/email/name/account numbers → tokens); masking of brands/logos.
- Two-class labeling (machine → human), measuring `κ` [value]; the gold set is 100% human-annotated.

## 5. Distribution
- **Open tier:** URL/QR features, LLM samples, public benign data.
- **Controlled tier:** real anonymized email/SMS — under a **data use agreement**.
- **License:** [specify]. **Source attribution:** tinnhiemmang/NCSC, OpenPhish, URLhaus, authors.

## 6. Maintenance
- **Updates:** [periodic] to counter concept drift. **Versioning:** [semver + date].
- **Contact:** [email].

## 7. Limitations & bias
- URLs/infrastructure are largely language-agnostic; the "Vietnamese" component is strongest in SMS/email content.
- Public sources skew toward brand-impersonation websites; real SMS/email are hard to collect → possible imbalance.
- LLM samples reflect the style of the LLM used → apply **leave-one-LLM-out** when evaluating generalization.
