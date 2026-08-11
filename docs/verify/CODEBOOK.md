# Label-audit codebook — PhishVN verification sample

Two annotators independently judge 200 blinded domains. The source label is withheld. Fill
`verdict` and `evidence` for every row. Do not discuss rows with the other annotator until both
sheets are complete.

## Verdicts

| verdict | assign when |
|---|---|
| `phishing` | the site impersonates a brand or service to harvest credentials, OTPs or payment details |
| `scam` | abusive but not credential-phishing: gambling, betting streams, investment or crypto fraud, counterfeit shops, adult content |
| `legitimate` | a real operator's own site — listing it would be a feed error |
| `unsure` | the evidence available does not support any of the above |

**Precedence.** A site that both impersonates a brand and runs gambling is `phishing`: credential
harvesting outranks the rest. Otherwise `scam`.

## Decision 1 — looking a domain up outside the sheet

**Allowed, and expected.** Only 12 of the 200 sampled domains have a capture inside this corpus, so
a string-only pass would return `unsure` for roughly 94% of rows and measure nothing. Consult
sources in this order and stop at the first that resolves the row:

1. **This corpus's own capture**, if the domain appears in `data/interim/content_manifest.csv` —
   contemporaneous with detection, and the strongest evidence available.
2. **A web archive** (Wayback Machine) at or near the domain's first-seen date. Judge the archived
   page, not today's parked placeholder.
3. **A public urlscan.io scan** — use its **screenshot and DOM**. Do not use urlscan's own verdict
   or engine tags.
4. **The NCSC/Tin Nhiem Mang trusted-organisation registry** (`data/raw/tinnhiem_org`) as positive
   evidence of a legitimate operator.

**Forbidden, because they re-derive the label being audited.** ChongLuaDao, NCSC blacklists,
OpenPhish, URLhaus, VirusTotal, urlvoid, any reputation or blocklist aggregator, and any search
result whose "is this a scam" content is itself sourced from a blocklist. An audit that consults
the feed it is auditing measures nothing but its own circularity.

Record which source decided the row in `evidence`: `capture`, `archive`, `urlscan`, `registry`,
`name` (the domain string alone was unambiguous — e.g. `taixiu`, `188bet`, a bank name with a
lookalike suffix), or `none`.

## Decision 2 — a name that could plausibly belong to a real business

**Plausibility is not evidence, in either direction.** `chovaytieudung.online` ("consumer lending")
could be a real lender or a loan scam; the name alone settles nothing.

- `legitimate` requires **positive evidence** of a real operator: its own branded site in a capture
  or archive, or a registry entry. Never assign it because a name sounds ordinary.
- `scam` likewise requires positive evidence: abusive content in a capture or archive, or a name
  pattern that admits no innocent reading.
- With no evidence either way, the verdict is **`unsure`** — never `legitimate`, never `scam`.

This is deliberately asymmetric against the interesting result. The audit exists to find rows where
the feed listed something legitimate, so the bar for calling a row legitimate is evidence, not
absence of suspicion.

**A large `unsure` share is a finding, not a failure.** It means this corpus cannot be audited from
URLs and archives alone and needs its capture tier extended. Report the resolvable share alongside
kappa; a noise estimate computed on 40 resolvable rows is not the same claim as one computed on 200.

## `MACHINE_PASS.csv` — do not open it before your sheet is complete

That file holds a rule-based pass over the same 200 archived pages, produced by
`scripts/machine_pass_composition.py`. It exists to estimate coverage — how much of this corpus can
be adjudicated from archives at all — and it is **not** an annotator. It casts no vote, it is not
one of the two sheets, and κ is not computed against it. Reading it before you judge would replace
your independent judgement with its rules and destroy the only quantity the audit produces. Compare
against it afterwards if you like; that comparison is a note, not a result.

## Before you start

Both annotators read this file, agree the four categories mean the same thing to each of them on
five practice rows drawn outside the sample, and only then open the sheets. Any rule added or
clarified after annotation begins is recorded here with its date, and the rows judged before the
change are re-judged under it.
