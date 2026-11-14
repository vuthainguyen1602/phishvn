<!-- Public copy: hostnames are replaced by [HOST] (hosting platforms kept). SHA-256 of the committed original: e3196ccb01857799c769723beb6c06bd6794e8fd6aaec80c1648815597a4666a -->

# pre-specification — subdomain proliferation as phishing infrastructure evasion, bound to code

**Registered:** 2026-09-06. Written after descriptive measurement of the PhishVN collector feeds
(`urlscan_brands`, `vn_phishing_live`, `chongluadao_live`) and **before any inferential analysis
is run**. The descriptive quantities known at registration are listed below. No model has been
fitted, no survival curve estimated, no blocklist simulation run.

---

## What was already known at registration

The following quantities were computed and read before this file was written, and are therefore
**descriptive for the life of this study**:

### From `data/raw/host_infra/host_infra.csv` (50,303 rows, captured 2026-09-06)

- **Phishing registered domains:** 8,009 unique eTLD+1 across all phishing-labelled rows.
- **Solo domains (1 hostname):** 4,959 (61.9%).
- **Proliferators (2+ hostnames):** 3,050 (38.1%).
- **Overall inflation ratio:** 2.11× (hostnames / registered domains, combined feeds).
- **Spike at 4 hostnames:** 2,003 domains — requires investigation (possible collector artifact
  from 4-rotation probe, or a genuine 4-subdomain kit deployment pattern).
- **Max subdomains per domain:** 93 (`weebly.com`), 77 (`[HOST]`),
  73 (`[HOST]`).

### From `data/raw/urlscan_brands/detections.csv` (971 hostname rows)

- **Unique registered domains:** 644. Inflation ratio: 1.51×.
- **Multi-subdomain domains:** 72 (11.2%). Smaller share than the general feed because branded
  phishing favours freshly registered domains over free-hosting abuse.

### From `data/raw/vn_phishing_live/detections.csv` (2,183 hostname rows)

- **Unique registered domains:** 859. Inflation ratio: 2.54×.
- **Multi-subdomain domains:** 649 (75.6%). Much higher because `vn_phishing_live` aggregates
  feeds that include free-hosting platforms (`blogspot.com` 240 subdomains, `vercel.app` 19,
  `pages.dev` 17, `weebly.com` 14, `000webhostapp.com` 13).

### Top registrars in proliferators (host_infra, phishing arm)
`MarkMonitor` (122), `NAMECHEAP` (89), `Registrar.eu` (81), `GoDaddy` (75), `NameSilo` (61),
`DYNADOT` (38), `P.A. Viet Nam` (37). Most rows have no WHOIS registrar (`unknown`: 10,221)
because `.vn` WHOIS is gated and many foreign domains do not expose it.

---

## Research Questions (confirmatory unless marked exploratory)

### RQ1 — Perishability: do proliferators survive longer than solo domains?

**Hypothesis (pre-specified direction):** Proliferators have **higher resolvability** at 32–64h
lag than solo domains, measured on the phishing arm of `host_infra.csv`.

Rationale: a domain with multiple subdomains may represent a more deliberate infrastructure
investment (kit-based deployment), and may therefore be kept alive longer by the attacker.
Alternative: proliferators are free-hosting abusers, and free-hosting platforms may take down
accounts faster than registrars take down domains.

**Primary test:** Mann-Whitney U on resolvability rate (binary) at the 32–64h bin, comparing
solo vs proliferator registered domains. Benjamini-Hochberg correction across RQ1–RQ3 (m = 3).
Significance threshold: q ≤ 0.05.

**Registered negative result:** If U-test fails to reject H0, report the confidence interval on
the difference in resolvability rates and conclude that proliferation status does not predict
survival within this dataset.

### RQ2 — Free hosting vs registered domain proliferators

**Hypothesis:** Free-hosting-platform proliferators (`blogspot`, `pages.dev`, `weebly`, `vercel`,
`000webhostapp`, `glitch.me`, `duckdns.org`, `appspot.com`) have **shorter survival** than
attacker-registered proliferators, measured by resolvability at 64–128h lag.

Rationale: platform abuse reports tend to trigger rapid account suspension; attackers who register
their own domain retain control longer.

**Primary test:** Mann-Whitney U, same correction family as RQ1.

**Operationalisation:** A registered domain is classified as "free-hosting" if its eTLD+1 appears
in the `HOSTED_SUFFIXES` list used by `scripts/audit_capture_labels.py` at analysis time.
The list is frozen at the commit hash recorded in `HOW_TO_RUN.md`.

### RQ3 — Blocklist evasion gain from subdomain proliferation

**Hypothesis:** A hostname-level blocklist blocks a smaller fraction of phishing traffic than a
registered-domain-level blocklist, and the gap is **larger for proliferators than for solo
domains**.

**Primary test:** Simulation on `host_infra.csv`. For each domain, compute:
- `hostname_coverage`: fraction of its hostnames that appear on a reference blocklist
  (OpenPhish + PhishTank snapshots, fetched at analysis time).
- `rd_coverage`: 1 if the registered domain itself appears on the blocklist, 0 otherwise.

Report the mean difference `rd_coverage − hostname_coverage` separately for solo and proliferator
groups. Test difference-in-differences with a permutation test (10,000 shuffles, q ≤ 0.05,
same BH family).

**Registered negative result:** If the simulation shows no gap, report it and discuss whether
the absence reflects blocklist design or insufficient sample size.

---

## Related Work and Differentiation

The literature on phishing infrastructure spans URL obfuscation, fast-flux DNS, bulletproof hosting, and domain reputation. However, subdomain proliferation as an active evasion and infrastructure lifecycle strategy occupies a distinct, under-studied niche. We situate this study against prior and contemporary work across five key axes:

### 1. Free Web Hosting Abuse vs. Attacker-Controlled Proliferation
- **FreePhish (Acharya et al., IMC 2023):** *FreePhish: Characterizing Phishing on Free Web Hosting Services* is the closest baseline. Acharya et al. analyzed phishing hosted on free hosting services and website builders (e.g., Weebly, Wix, Google Sites, GitHub Pages, Firebase), quantifying discovery delays, blocklist lag, and evasion techniques.
- **PhiShark (2026):** Recent active-web measurement (*PhiShark2026: A Multi-Layer Active-Web Raw-Evidence Dataset*) establishes explicit data partitions for free-hosting subdomains to prevent provider-level artifacts from poisoning machine-learning models.
- **Differentiation:** Prior work treats free hosting as a third-party abuse platform to be cleaned up or partitioned out. Our study:
  1. **Expands the taxonomy:** We define the *proliferator* abstraction across *both* free-hosting platforms and attacker-registered custom domains (e.g., `[HOST]` with 77 subdomains), comparing the empirical survival dynamics between the two (RQ2).
  2. **Shifts focus from detection lag to survival & evasion advantage:** FreePhish asks *"how long do free-hosting phishing sites take to be blocklisted?"*. We ask whether multi-subdomain proliferation provides an empirical *resolvability/survival advantage* (RQ1) over solo domains, and directly simulate the *evasion gain* (RQ3) enabled by hostname-level vs registered-domain blocklisting.
  3. **Unit of analysis:** Our primary sampling unit is the *registered domain* (eTLD+1), investigating how attackers manage subdomains under a single root, rather than treating individual URLs/hostnames as isolated incidents.

### 2. Longitudinal Phishing, ccTLDs, and Early Detection (2024–2025)
- **ccTLD Measurement (ACM CCS 2024):** Large-scale longitudinal measurement of phishing across ccTLDs observed attackers systematically burying phishing content in subdomains and subdirectories to evade automated crawlers that only inspect apex domains.
- **Subdomain Detection Patterns (NDSS 2025 / Sensors; Nosyk et al.):** Examines how cybercriminals increasingly place phishing payloads on subdomains of registered domains to evade detection, extracting lexical subdomain patterns as early classification indicators.
- **Differentiation:** CCS 2024 and NDSS 2025 treat subdomains as *features for URL classifiers* or *crawler obstacles*. In contrast, we investigate subdomains as an *operational infrastructure strategy*: we quantify the structural inflation ratio (2.11× across 8,009 domains), test whether proliferation predicts longer resolvability (RQ1), and evaluate defense efficacy when defenders block at the hostname vs domain level (RQ3). Furthermore, our dataset captures the specific socio-technical dynamics of Vietnamese-targeting campaigns (e.g., `.vn` gating, domestic registrars such as P.A. Viet Nam).

### 3. Application/DOM-Layer Evasion vs. DNS Infrastructure Agility (2024–2026 Trends)
- **Content & LLM Evasion (USENIX Security 2024, NDSS/MADWeb 2026):** Contemporary 2024–2026 literature focuses intensely on client-side evasion—e.g., logo-less phishing detection (*KnowPhish*, USENIX Security 2024), reference-less detection, and GenAI-based dynamic runtime page assembly (*Paladin*, MADWeb 2026).
- **Differentiation:** While contemporary literature focuses on evading visual and lexical content analysis inside the browser, subdomain proliferation is an orthogonal, lower-layer architectural evasion. It operates at feed ingestion and DNS resolution time, diluting URL reputation feeds before content-level or LLM-based classifiers are ever triggered.

### 4. Subdomain Hijacking vs. Active Proliferation
- **Subdomain Takeover & SubdoMailing (Guardio 2024; Schwittmann et al., USENIX Security 2021):** Subdomain takeover research focuses on *passive vulnerability exploitation*, where attackers claim abandoned DNS records (dangling CNAMEs pointing to expired cloud services) primarily for email spoofing (SPF/DKIM alignment) or reputation hijacking.
- **Differentiation:** In our corpus, subdomains are not abandoned third-party resources reclaimed through dangling pointers. They are *actively, systematically provisioned* by the attacker—either via DNS wildcard/A-record manipulation on registered domains or automated script creation on free hosting—purpose-built to distribute distinct phishing lures while amortizing root domain acquisition costs.

### 5. Application-Layer Agility vs. Fast-Flux
- **Fast-Flux DNS (Holz et al., NDSS; Perdisci et al.):** Classic fast-flux relies on rapid IP rotation behind static domain names to survive IP-based blocking.
- **Differentiation:** Subdomain proliferation is an orthogonal, application-layer agility: static or slow-moving IP infrastructure hosting dozens of ephemeral hostname entry points to fragment URL-level reputation and exploit defensive blind spots in feed ingestion.

---

## Explicitly out of scope for confirmatory analysis

- Subdomain naming patterns / NLP clustering of subdomain labels (exploratory only).
- Cross-country comparison outside the Vietnamese-targeting corpus.
- Any feature derived from DOM content (not collected at registered-domain level).
- Any analysis of the "4-hostname spike" until its cause is determined. If it is a real pattern,
  it may be added as an amendment before the analysis runs.

---

## Data and code binding

Analysis runs on `data/raw/host_infra/host_infra.csv` at the commit frozen in `HOW_TO_RUN.md`.
Scripts will live in `scripts/`. No script in that directory
exists at registration time.

The classification of "free-hosting" vs "registered" uses `audit_capture_labels.HOSTED_SUFFIXES`
as of the same frozen commit — the list must not be extended after the analysis has run.

Resolvability is defined as: `a_records` field is non-empty AND does not consist solely of
non-routable addresses (RFC 1918 / RFC 4193 ranges as filtered in `p4_live_resolve_cache.py`).

---

## Amendments (append-only, dated, must precede any analysis of the amended item)

_None yet._

---

## Suggested venue (not binding, decided after results)

| Outcome | Candidate venue |
|---|---|
| RQ1 + RQ2 both positive | **Computers & Security** (Elsevier, Q1, IF 6.8) |
| Mixed results | **JISA** (Elsevier, Q1) |
| All null | **ACM DTRAP** (informative failure track) |
| Strong RQ3 (blocklist simulation) | **IEEE S&P Workshop on Measurements** or **IMC** |

## Implementation audit amendment — 2026-09-08

The local input inspected during implementation contains 49,109 rows, ending on
2026-09-05, rather than the 50,303-row snapshot described above. The original
commit-bound input was not recorded and is not reconstructed. Descriptive schema
and source-count inspection preceded this amendment; no RQ1–RQ3 inferential test
has been run. The original registration and its numbers remain unchanged above.

Collector inspection shows retries are offered only to hostnames that have never
returned an A record. Thus later captures are selected by earlier outcomes.
RQ1 and RQ2 survival hypotheses are not identifiable from these records; they will
be reported as not evaluated, not accepted or rejected. An empty A field is not
a cause-specific takedown observation. RQ3 lacks paired contemporaneous OpenPhish
and PhishTank snapshots and traffic weights; a present-day feed fetch would not
recover historical coverage. No registered inferential p-values or BH family will
be reported. Failure to reject would not establish absence of an effect in any case.

The implementation instead produces an explicitly exploratory measurement audit:
recompute ICANN-only and private-suffix/hosted-tenant keys; retain raw hostname
counts and report a separate leading-www sensitivity; audit stored-domain changes;
report first-capture public-A availability separately from failure-conditioned
retries; report lag-bin observation support without calling it survival; and
calculate an algebraic one-seed-per-group hostname-expansion scenario without
calling it observed blocklist coverage, traffic protection, or evasion. The
four-hostname frequency is audited descriptively and its cause is left unresolved
unless independently established. Source labels remain candidate labels.

Inputs will be frozen as compressed copies with SHA-256 hashes before running
these analyses. The manifest binds the input bytes, parser suffix snapshot, hosted
suffix list, analysis script and parent commit; it does not pretend a new commit
was registered before inspection. The hosted list is read without executing the
audit module. No live DNS or HTTP requests are made by the analysis.

## Prospective registration — 2026-09-16, 23:10 Asia/Ho_Chi_Minh, before any observation

The 8 September audit concluded that the existing ledger cannot answer RQ1–RQ3 because later
observations are conditioned on earlier DNS outcomes and no time-aligned blocklist snapshot
exists. This amendment registers the prospective collection the audit specified. It is written
before the collector has admitted a single hostname: admission opens at
**2026-09-17T00:00:00 (device clock, Asia/Ho_Chi_Minh)**, and hostnames first captured before
that instant are never admitted.

### Collectors (code-bound)

- `scripts/watch_revisit.py`, hourly (cron `15 * * * *`,
  wrapper `scripts/ops/revisit_run.sh`). Admits every hostname whose **first** capture in
  `data/raw/host_infra/host_infra.csv` is at or after the admission start; records the DNS state
  seen at that capture; then resolves A, AAAA and CNAME at **fixed lags of 8, 32, 64, 128, 256
  and 512 hours after admission, regardless of any earlier answer**. Outcome states:
  `public_a` (≥1 global IPv4), `private_only`, `empty` (NOERROR, no A), `nxdomain`, `servfail`,
  `timeout`, `error`. An observation later than 1.5× its planned lag is flagged `late`, never
  dropped. No HTTP request, no WHOIS.
- `scripts/fetch_blocklists.py`, daily (cron `5 3 * * *`,
  wrapper `scripts/ops/blocklist_snapshot_run.sh`). Stores the OpenPhish community feed and
  PhishTank `online-valid.csv` as dated gzip files with SHA-256 in `data/raw/blocklists/manifest.csv`.
  A failed fetch is a manifest row, not a gap.
- Benign control: about 25 benign-labelled first captures a day, chosen by a hash of (day,
  hostname), follow the same schedule. If the control's `public_a` retention at 128 h falls below
  90 % in any calendar week, that week's phishing observations are flagged as vantage-suspect.

Both scripts are bound to the commit that carries this amendment; any later change to lags,
states, admission rule or cap is a further dated amendment, and observations already recorded
under the old rule are analysed under it.

### Unit, landmark and covariates

The unit is the admitted hostname. Group keys (ICANN and tenant, as defined in Section 3 of the
audit) are computed at analysis time from hostname strings with the PSL snapshot and hosted list
frozen in `data/processed/subdomain/input_manifest.json`; the device computes no key.
**Multiplicity is a landmark covariate:** host $h$ is *multi* if at least one other admitted
hostname shares its tenant key and was admitted no later than $\mathrm{admitted}(h) + 32$ h.
A sensitivity definition counts every hostname in `host_infra.csv` sharing the key, whatever its
capture date. *Hosted* means the tenant key ends in a suffix of the frozen hosted list.

### Confirmatory analyses (Benjamini–Hochberg, m = 3, q ≤ 0.05)

- **RQ1′ (retention by multiplicity).** Population: phishing-labelled admitted hosts with
  `public_a` at admission, tenant key not hosted. Outcome: `public_a` at the 128 h observation.
  Contrast: multi vs solo. Test: two-sided Fisher exact on the 2×2; effect reported as the
  difference in retention with a Newcombe 95 % CI. Registered direction: multi ≥ solo. Secondary,
  descriptive only: the same at 256 h and 512 h.
- **RQ2′ (hosted vs registered among multi).** Population: multi hosts with `public_a` at
  admission. Outcome: `public_a` at 256 h. Contrast: hosted vs not. Fisher exact, same family.
  Registered direction: hosted < not hosted.
- **RQ3′ (blocklist coverage and expansion).** For each phishing-labelled admitted host, take the
  latest blocklist snapshot fetched **before** its 128 h observation. Two operators, both
  reported: exact-hostname membership; and group expansion, membership of *any* admitted
  hostname sharing the tenant key. Gap = expansion − exact, mean per host. Contrast: gap for
  multi minus gap for solo, tested with a permutation test (10,000 relabellings of *multi*
  within source strata). Registered direction: gap larger for multi. Solo hosts have gap 0 by
  construction, which is stated, so the test is of whether expansion reaches any multi host the
  exact list misses.

### Handling of missing, late and stratified observations

A missing 128 h observation (collector outage) is reported as missing and never imputed; late
observations enter the primary analysis and are excluded in a sensitivity analysis. Every RQ is
also reported per source (`urlscan_brands`, `vn_phishing_live`, `chongluadao_live`), and the
pooled test is the primary one. Failure to reject is reported with its CI and is not read as
absence of an effect.

### Stopping rule and analysis code

Analysis runs when **≥ 300 phishing hosts with `public_a` at admission have completed the 512 h
schedule, or on 2026-12-01**, whichever comes first. If either arm of RQ1′ holds fewer than 100
hosts at 128 h, the contrast is reported as underpowered with its CI and no test is claimed. The
analysis script, `scripts/analyze_revisit.py`, will be written
and bound by commit hash in a further amendment **before** the stopping rule triggers; until
then the accruing observations are read only for collector health (row counts, late rate,
control retention).

### Venue

Unchanged from the table above, decided after results.

## Amendment — 2026-09-19, 00:20 Asia/Ho_Chi_Minh: the realised control rate, and a fourth source that was dry at registration

Written after 48 hours of admission and before any analysis. What had been read from the accruing
data when this was written is what the registration permits: row counts of `cohort.csv` by day,
label and source, the late count, and counts of first captures by day in `host_infra.csv`. No
retention figure, by arm, source or lag, was computed for this amendment or informed it. Neither
collector is changed by it: `watch_revisit.py` and `fetch_blocklists.py` stay at the commit the
16 September registration bound.

### 1. The benign control admits about 60 hosts a day, not about 25

The registered rule is a hash of (day, hostname); the registered *rate* was a forecast. The
collector turns its cap of 25 benign a day into a fixed sampling fraction by assuming 400 benign
first captures a day (`25 * 1000 // 400` = 62 per mille). The ledger has carried 823 to 1,071
benign first captures a day since 10 September, so the same 62 per mille admits 59 and 60 hosts
on the first two days (1,071 and 1,020 benign first captures on 17 and 18 September), and will go
on admitting about 60. The 400 was wrong when it was written: the sibling
cloaking design, drafted on 18 September from the same ledger, records 820 to 1,070 a day.

The rule is **not** corrected. The fraction stays at 62 per mille, for three reasons. Membership
is decided by the hash alone, so the larger sample is the same kind of sample, not a different
one. The control enters no confirmatory test: its only registered use is the weekly
vantage-suspect flag (128 h `public_a` retention below 90 %), and a flag estimated on about 420
hosts a week is tighter than one estimated on 175, never looser. And lowering the fraction now
would make the control two regimes split at a date chosen after admission began, which is worse
than a control that is larger than forecast. The cost is about 210 extra observations a day
on the device at full schedule (35 hosts, six lags), which the hourly tick's cap of 400
observations absorbs. The manuscript's
"about twenty-five" is replaced by the fraction and the realised number.

### 2. `ct_brands` hosts are a separately reported stratum, outside the confirmatory population

At registration three sources produced phishing-labelled first captures (`urlscan_brands`,
`vn_phishing_live`, `chongluadao_live`), and the registration names those three. A fourth,
`ct_brands` (certificate-transparency names that begin with a brand token), is mapped to the
label `phish` by `watch_host_infra.py` and would have been admitted all along; it was not,
because it produced nothing: its crt.sh query pattern could not match hostnames
(`docs/decisions/ct-brands-prefix-query.md`). The pattern was repaired on the device at
2026-09-18 23:45 and the source is expected to produce from 2026-09-19. **At the time of this
amendment the cohort holds no `ct_brands` host** (0 of 205 rows: 84 `phish`, 121 `benign`; the three `ct_brands` rows in
the ledger predate the admission start), so the rule below is set before the first host it
governs is admitted.

A certificate shows that a name was set up, not that a lure was served. The repaired query's
first live run returned, among plausible lookalikes, names that are probably the brand's own
(`[HOST]`, `[HOST]`). Pooling such hosts into "phishing-labelled admitted hosts"
would dilute the population every registered contrast is defined on, and would bring the
n = 300 stopping rule forward on the weakest labels in the ledger.

From this amendment:

- **Confirmatory population.** "Phishing-labelled admitted hosts" in RQ1′, RQ2′ and RQ3′ means
  admitted hosts labelled `phish` whose `source` in `cohort.csv` is `urlscan_brands`,
  `vn_phishing_live` or `chongluadao_live`. The Benjamini–Hochberg family stays at m = 3.
- **Multiplicity and group expansion.** The landmark definition of *multi* and the expansion
  operator of RQ3′ count hostnames of that confirmatory population only: a
  `ct_brands` host sharing a tenant key neither makes a confirmatory host *multi* nor extends
  its expansion set. The registered sensitivity definition (every hostname in `host_infra.csv`
  sharing the key, whatever its capture date) is unchanged and therefore does include them.
- **Stopping rule.** The 300 hosts are counted in the confirmatory population only. The
  2026-12-01 backstop is unchanged.
- **The fourth stratum.** `ct_brands` hosts are admitted and followed on the same schedule, and
  reported descriptively as their own stratum: counts, state at admission, and retention at each
  lag with intervals. No test is claimed on them, and they join the per-source reporting of the
  registration as a fourth line, marked as outside the family.
- **Displacement, stated.** The ledger captures a hostname once, under the source that reported
  it first, and certificate transparency tends to see a name before anyone submits it to
  urlscan. A lookalike that `urlscan_brands` would have reported a day later is therefore
  admitted under `ct_brands`, with its lags counted from an earlier instant, and is lost to the
  confirmatory population. Such hosts are not moved back: their clock started at a different
  event. Their number is reported (admitted `ct_brands` hosts that later appear in a named
  source's `detections.csv`), and one descriptive sensitivity analysis repeats each contrast with
  all four sources pooled. It is outside the family and no conclusion rests on it.

`analyze_revisit.py` is still unwritten. The amendment that binds it must implement this
population rule, and is still owed before the stopping rule triggers.

## Amendment — 2026-09-19, 23:41 Asia/Ho_Chi_Minh: hosts that exist only because this project scanned something are outside the confirmatory population; a wildcard probe is added

### What was read before this was written, and what was not

More than the registration permits, and it is stated here for that reason. A data-soundness check
on 2026-09-19 compared scan identifiers between `urlscan_brands/detections.csv` and the project's
own submission ledgers, found the loop described below, and, to size it, tabulated
`observations.csv` for the affected hosts against the rest of the `urlscan_brands` hosts, by lag
and state:

| | lag 8 h | lag 32 h |
|---|---|---|
| affected (39 hosts) | 37 `public_a`, 0 other | 27 `public_a`, 0 other |
| other `urlscan_brands` hosts (59) | 53 `public_a`, 1 `timeout` | 31 `public_a`, 2 `timeout` |

So the direction of this amendment's effect on short-lag retention was known when it was
written: removing the affected hosts cannot raise it. No comparison by multiplicity, by hosting
class or against any blocklist was computed, and no figure for any registered contrast. The
criterion below does not use an outcome: it is decided by a scan identifier recorded at
discovery. The affected hosts are nonetheless reported in full (item 4), so that a reader can
see what the amendment removed.

Two further readings belong in this account. Health checks of the collection on 18 and 19
September printed, for the whole cohort and once split into `ct_brands` against the registered
sources, the count of observations by planned lag and DNS state (on the evening of 19 September:
lag 8 h 301 `public_a`, 43 `empty`, 13 `timeout`; lag 32 h 168, 2 and 5 with one `private_only`;
lag 64 h 40 and 1 `timeout`). They were read to see whether the collector was alive, not to
compare groups, and no contrast of this registration was computed from them; but they were read.
And the text above was drafted at about 16:30 on 19 September on the morning's copy of the data;
the counts in item 1 are recomputed from `cohort.csv` as synced at 23:40 for this adoption, by
source and admission day only.

### 1. What happened

`watch_urlscan_brands.py` searches urlscan with the project's API key, and urlscan returns a
key's own unlisted scans to it. `qr_submit.py` (an unrelated probe) submits names from
`host_infra.csv`, newest first, with the same key. When the submitted name is a wildcard-parked
typo domain, the scan lands on that name with `17.` or `ww17.` prepended; the new hostname
carries a brand token, the brand feed records it as a detection, the infrastructure watcher
captures it, this study's collector admits it on that first capture, and four hours later the
probe submits it and the cycle repeats one level deeper. By scan identifier, 166 of the feed's
1,834 identities had been reached through a scan the project itself had submitted when the fix
took effect; **41 of the 112 `urlscan_brands` hosts in `cohort.csv`** at this adoption (admitted
17 Sept: 23, 18 Sept: 12, 19 Sept: 6) are among them. (The table above describes the 39 of 98 that
the morning's copy held.) They are not phishing infrastructure that
anyone deployed; they are a parking service answering this project's own requests, and being
wildcard names they resolve indefinitely. Left in, they would enter RQ1′ as long-lived hosts
under registered domains with many hostnames, which is the hypothesised direction.

### 2. The criterion

A cohort host is **self-induced** if the `scan_uuid` recorded for its hostname in
`data/raw/urlscan_brands/detections.csv` appears in the `scan_uuid` column of any ledger listed in
`scripts/own_scan_ledgers.local` at the commit this amendment is adopted in
(`qr_submit_ledger.csv`, `ct_brands/captures.csv`, and the `detections.csv` of
`chongluadao_live`, `vn_phishing_live` and `tinnhiem_benign`). All of those files are append-only
and the identifier is written at discovery, before any observation of this study. The shape of
the hostname is **not** part of the criterion.

### 3. What changes

- **Confirmatory population.** Self-induced hosts are excluded from "phishing-labelled admitted
  hosts" in RQ1′, RQ2′ and RQ3′, from the landmark definition of *multi*, from the expansion
  operator of RQ3′, and from the 300 of the stopping rule. The Benjamini–Hochberg family stays at
  m = 3.
- **The collector that feeds admission changes behaviour**, which the registration did not
  foresee: from its deploy on the device (**2026-09-19 22:10:54 Asia/Ho_Chi_Minh**, 15:10:54Z) `watch_urlscan_brands.py` no
  longer records a hit on the project's own scan as a detection (it logs it to
  `urlscan_brands/self_induced.csv`), and `qr_submit.py` no longer queues such names.
  `watch_revisit.py` and `fetch_blocklists.py` are unchanged. Hosts admitted before that time
  stay in `cohort.csv`, which is never rewritten, and keep being observed on the registered
  schedule.
- **`analyze_revisit.py`**, when it is bound, computes the self-induced set from the files named
  in item 2 and nothing else.

### 4. What is reported regardless

The number of self-induced hosts by admission day; their retention at every registered lag, as
a separate descriptive stratum beside `ct_brands`; and one sensitivity analysis of each
confirmatory contrast with them included, outside the family. If the two readings of a contrast
disagree, the manuscript says so in the abstract.

### 5. A second artefact of the same family, and a measure that is weaker on purpose

**What was read.** Beyond item 1: the number of distinct hostnames per registered domain in the
phishing arm of `host_infra.csv` (the registration's own "what was already known" section reads
the same file the same way), and the stored captures and live DNS of one such domain. No
observation of this study was read for this item.

**What it showed.** The second-largest family in the phishing arm is
`[HOST]` with 84 hostnames (`admin`, `grafana`, `kibana`, `sso`, `giris`,
`cuenta`, `area-clienti` ...). The domain resolved every label (wildcard DNS); 57 of its 82 stored
captures are one 308-byte default page of a hosting company; the names are a subdomain wordlist
that some third party ran through urlscan, each scan becoming an identity of the brand feed. One
of them is in the cohort (`account-sso.`, admitted 2026-09-17). Nobody created those 84
subdomains. Under the registered landmark definition the domain is nonetheless *multi*, with the
largest multiplicity in the ledger after a hosting platform, and a wildcard name resolves for as
long as its parent does. `[HOST]` (32 hostnames, `ww38.` parking names) is the
same kind. This is a confound in the hypothesised direction of RQ1′, and unlike item 1 it is not
of this project's making, so provenance cannot separate it.

**What is adopted, and what is deliberately not.** The registered definition of *multi* is
**not** changed: redefining the exposure after hostname counts have been read would be exactly
the flexibility a registration exists to remove. Instead:

- a one-off **wildcard probe** is taken for every registered domain in `cohort.csv` on the day
  this amendment is adopted, and for every newly admitted registered domain thereafter, by a
  separate script that does not touch `watch_revisit.py`: one lookup of a fixed unlikely label
  under the registered domain, recorded with its date, resolver and answer in
  `data/raw/revisit/wildcard_probe.csv` (the script is
  `scripts/probe_wildcard.py`; it refuses to probe the cohort
  until the registration names that file, i.e. until this amendment is appended). A domain is *wildcard-resolving* if the unlikely label
  returns an address. Domains that no longer resolve at all on the probe day are recorded as
  `undetermined`, not as negative;
- RQ1′ is reported as registered, **and** once more with wildcard-resolving and undetermined
  domains removed from both arms, outside the Benjamini–Hochberg family. If the two disagree,
  the abstract says so and the wildcard-free reading is the one the discussion is built on;
- the share of *multi* hosts that sit under a wildcard-resolving domain is reported whatever the
  result.

The probe reads DNS, which is this study's outcome variable, but it reads the domain's answer for
a name nobody uses, once, and never the cohort host's own state, so it cannot be tuned on an
outcome.

### 6. Consequence for the timetable

With the 41 removed, the confirmatory population holds 75 phishing-labelled hosts after three
days of admission rather than 116, about 25 a day. The registered stopping rule (300 hosts through
512 h, or 2026-12-01) is unchanged; at that rate the 300th admission falls in the last days of
September and the 512 h mark around 20 October, before the date limit.

### 7. What is done on the day of this adoption

`probe_wildcard.py --cohort` is run once on the collection device for every registered domain in
`cohort.csv`, and an hourly run is scheduled there so that each newly admitted registered domain
is probed once. `analyze_revisit.py` does not exist yet; it is written against this registration
and its three amendments before any contrast is computed, and it reads the self-induced set, the
`ct_brands` stratum and the wildcard probe from the files named here and nowhere else.

## Amendment — 2026-09-20, 00:35 Asia/Ho_Chi_Minh: the collection network loses negative DNS answers; the wildcard probe's transport; the analysis code is bound

### What was read before this was written, and what was not

- The wildcard probe's own output (`wildcard_probe.csv`: statuses, details, answers).
- The `rcode` column of `observations.csv`, totalled over **all** rows and not split by label,
  source, lag or exposure: 555 `NOERROR`, 19 blank (the collector's `timeout`), **0
  `NXDOMAIN`** in 574 rows. It was read for one question only: does the defect found in the probe
  (item 1) also reach the registered collector.
- `analyze_revisit.py --status`: cohort 393 (216 phishing-labelled, 177 control), 75 confirmatory
  hosts, 41 self-induced removed, 574 observations, none late. These are the counts the
  23:41 amendment already states.
- **Not read:** any retention by exposure, any 2×2, any blocklist match. `analyze_revisit.py`
  refuses to compute them (item 3) and was not run with the refusal switched off on real data;
  its tests use synthetic hosts only.

### 1. The collection device's network does not deliver negative answers over UDP

Found on 2026-09-20 between 00:00 and 00:20, because the first probe run (item 2) came back
`undetermined` for two domains in three. Measured from the device and, identically, from a
second machine on the same network:

- every nonexistent name (`<random>.google.com`, a nonexistent `.com`, the probe label under any
  domain) **times out over UDP**, through the system resolver, through the ISP's two resolvers,
  and when 8.8.8.8 or 1.1.1.1 is addressed directly; a nonexistent top-level name comes back
  `NOERROR` with an empty answer instead;
- `o-o.myaddr.l.google.com TXT @8.8.8.8` returns an address of the access provider, not of
  Google: UDP port 53 is answered by the provider's resolver whatever address it is sent to;
- the same question over **TCP** to 8.8.8.8 returns `NXDOMAIN` in 0.1 s;
- positive answers and `NOERROR`/no-data answers arrive normally over UDP.

Since when is not known. That the registered collector has recorded no `NXDOMAIN` in 574
observations since 2026-09-17 is what one would expect if it has been so for the cohort's whole
life.

**The registered collector is not changed.** `watch_revisit.py` asks the system resolver over
UDP and keeps doing so. The reasons: (i) the registered outcome is whether a host still has a
public A record; a name that has ceased to exist is recorded here as `timeout` or `empty`
instead of `nxdomain`, and all three are "not `public_a`", so **no registered contrast, interval
or sensitivity analysis changes**; (ii) moving the collector to another transport or resolver in
the middle of the cohort would put a discontinuity into the outcome measurement itself, which is
worse than a known, constant loss of detail; (iii) the control cohort shares the path.

What is lost, and is stated as a limitation wherever states are tabulated: at this vantage
`nxdomain`, `timeout` and part of `empty` are **one indistinguishable state**, "did not
resolve". The descriptive tables report them merged, with this paragraph as the reason. A
`timeout` can therefore not be read as a sign of vantage failure; the registered guard against
vantage failure remains the control cohort's weekly retention at 128 h (< 90 % flags the week),
which does not depend on negative answers. Positive answers come from the access provider's
cache; nothing here can bound its staleness beyond the records' TTLs, and the control shares
that too.

### 2. The wildcard probe: what went wrong twice, what was fixed, what is on file

The probe's negative finding *is* a negative answer, so item 1 broke it where it leaves the
registered collector intact.

- First run, 2026-09-19 23:42 → 2026-09-20 00:00 (stopped by hand): 210 domains, of which 142
  `undetermined` with detail `probe TIMEOUT`, 51 `wildcard`, 3 `no_wildcard` (no-data answers,
  which do arrive), 6 `platform`.
- Commit `713280eb` retried a UDP timeout over TCP through the system resolver. Not enough: the
  device's stub forwards over UDP whatever it is asked on (13 rows, 12 of them
  `probe TIMEOUT tcp`; stopped by hand).
- Commit `0f1b35ae` sends the TCP retry to 8.8.8.8 and 1.1.1.1 directly and writes that into the
  row's `resolver` field. Run 00:10 → 00:34 over the whole cohort.

**Rules, fixed here.** Rows are never rewritten or deleted; all 462 stay. A domain's **latest**
row is the one that counts. A domain is asked again only if its latest row is `undetermined`
with `probe TIMEOUT` in the detail, i.e. the probe's own lookup failed on the transport — a
criterion on the probe's lookup, never on any cohort host's state. A domain that is
`undetermined` because it has no name servers any more is not asked again. The 23:41 amendment
said "once"; 160 rows are the exception just described, and they are all in the file.

State after the 00:34 run, by latest row, 303 registered domains: **201 `no_wildcard`, 74
`wildcard`, 20 `undetermined`, 8 `platform`.** The 74 wildcard answers are parking and
sinkhole addresses of the usual operators, not an address of the access provider: the network
does not manufacture positive answers for the probe label (201 domains returned none).

The hourly run promised in item 7 of the 23:41 amendment is scheduled on the device from
2026-09-20 00:34, at minute 45, under a lock.

### 3. The analysis code is bound

`scripts/analyze_revisit.py` as of commit `713280eb`
(SHA-256 `2cdb7a66dbf71c27ebc73a3d409d8646f5bfd89ecf775e2f177d56954af44593`) is the confirmatory
analysis. It refuses to compute any contrast until 300 confirmatory hosts that were `public_a`
at admission have a 512 h observation, or the date is 2026-12-01 or later; until then
`--status` prints counts, the late rate and the control's weekly retention, and nothing else.
Any later change to that file is a further dated entry here, with its reason, made before the
changed code is run on real data.

The registration left six things to the code. They were fixed in that commit, before any
blocklist match or contrast had been looked at, and are repeated here so that they can be
disagreed with:

1. A blocklist URL's hostname and a cohort hostname are compared lower-cased, each with one
   leading `www.` removed.
2. A host is judged against, per feed, the latest successful snapshot fetched strictly before
   that host's 128 h observation, both converted to UTC (the device stamps observations in
   Asia/Ho_Chi_Minh, the fetcher in UTC); the two feeds' hostnames are pooled.
3. A host with no 128 h observation is counted as missing and enters no 2×2 and no gap.
4. The interval for a difference of proportions is Newcombe's hybrid-score interval (his
   method 10).
5. RQ3′'s permutation test relabels *multi* within source strata 10,000 times, seed 20260916,
   two-sided on the difference in mean gap.
6. Benjamini–Hochberg is applied to the three primary p-values, and the family stays three when
   RQ1′ falls under the fewer-than-100-per-arm rule and its test is not claimed.

The self-induced stratum is what the 23:41 criterion removed from the three registered sources
(41 hosts on 19 September). A `ct_brands` host whose own capture later echoed back through the
brand feed is already outside the confirmatory population by the 00:20 amendment and is reported
in the `ct_brands` stratum, not twice.

## Amendment — 2026-09-26, 08:00 Asia/Ho_Chi_Minh: two brand tokens added to the shared feed list, disclosed before their first tick

**What changed.** The shared brand-token list in `watch_urlscan_brands.py` (imported unchanged by
`watch_ct_brands.py`) gains two tokens, `phatnguoi` and `thongbaophatnguoi`, in the
traffic-violation group. Reason: the mini-app kit family
`[HOST]` (first seen 31 July,
latest skin 27 August) was caught only through its bank and carrier tokens; a skin of the same kit
that names no brand would have passed unseen. The official lookup domain, `[HOST]`, is already on
the official-domain whitelist, so the tokens admit lookalikes, not the real service.

**What this touches, and what it does not.** `urlscan_brands` is one of the three named
confirmatory sources, so this widens the admission stream of the running cohort from the date
above. It does not touch the revisit collector, the lag schedule, the stopping rule, the analysis
code (`analyze_revisit.py`, hash unchanged), or any already-admitted host. The change is disclosed
here before the tokens' first collection tick, and the analysis will carry one sentence noting the
admission stream widened on 2026-09-26; hosts admitted via the new tokens are identifiable by
their `brand` column (`phatnguoi`, `thongbaophatnguoi`) if a sensitivity read excluding them is
ever wanted. Nothing was looked at beyond the already-public detection ledgers quoted above.

**Timing precision, added 2026-09-26 08:35.** The exact order was: amendment written 08:00,
collector deployed 08:01, one ct_brands tick ran before the registering commit landed at 08:17
(66c567cc). That tick returned zero certificates for both new tokens — they appear only in the
"no certs and no history" line — so no observation entered any ledger through them before
registration. Nothing to exclude; recorded for the order's sake.

## Amendment — 2026-09-26, 10:15 Asia/Ho_Chi_Minh: one abused-platform token, and why the broad platforms stay out

`makefamousapp` joins the shared token list (193 tokens). It is a hosting platform, not a brand:
three phishing skins sat on `service.makefamousapp.com` between 31 July and 27 August
(`[HOST]`), each admitted only through its bank or
carrier token, so a fourth skin naming no brand would pass unseen. On the urlscan side the token
covers every scanned subdomain of the platform; on the CT side the platform's wildcard
certificate makes it inert. The same reasoning was weighed for the broad free-tier platforms
(`pages.dev`, `workers.dev`, `web.app`) and they are deliberately excluded: their tokens would
match an overwhelmingly benign population, invert the base rate the label gate assumes, and
exhaust the query budget — hosted-subdomain abuse at that scale is characterized by the P4b
hosted stratum and this study's wildcard probe, not hunted by token. Same disclosure terms as
this morning's amendment: the admission stream widens from this date, hosts admitted via the new
token are identifiable by their `brand` column, the analysis carries the same one-sentence
disclosure, and nothing else changes.

## Deviation record — 2026-10-02, 01:30 Asia/Ho_Chi_Minh: a 25-hour outage, and the collection network changed under the registered collector

Not an amendment: nothing registered is changed here. This records two events and what was read
while finding them, so that any later amendment can be judged against it.

**1. Outage.** The collection device lost power. The last observation before it is
2026-09-30 22:15:51; the first after it is 2026-10-01 23:15:12 (local time, as `observed_at`
stores it). No observation was recorded in between. As registered (§ prospective collection), a
lag that fell in the gap is not imputed: the collector took each missed lag at its first tick
after restart, so 488 observations through 2026-10-02 00:18 carry `lag_actual_h` up to about 25 h
beyond `lag_planned_h`. The collector's own `late` rule flagged 56 of them (all at 8 h and 32 h);
none at 64, 128 or 256 h is flagged although the delay there reached 23–25 h. The registered
`late` rule is applied as written; the analysis reports the realised delay of every observation
taken in the 24 h after restart.

**2. The access network changed.** On restart the device joined a different Wi-Fi network
("[network]", DHCP, a different access provider whose resolvers are 116.97.90.124/.127), and the
old network's profile was deleted on 2026-10-01 at the author's instruction. The registered
collector, its transport (system stub, UDP) and its code are unchanged. The path is not: on
this network a nonexistent name returns `NXDOMAIN` (tested directly on 2026-10-02 00:5x), which
the old network never delivered (§ amendment 2026-09-20 item 1). Observed consequence: 0
`NXDOMAIN` in every row before 2026-10-01, 15 in the 488 rows after.

Disposition, as far as it follows from what is already registered: the registered outcome is
"has a public A record", and `nxdomain`, `timeout` and `empty` all count as "not `public_a`", so
the change in how a dead name is reported does not change any registered contrast. What may
change is the vantage itself: a different provider's cache answers the positive queries, and a
different provider may block or sinkhole a different set of names (the 2026-09-22 cloaking
finding of a 127.0.0.1 sinkhole on the old path). Whether the confirmatory analysis is split or
adjusted at 2026-10-01 22:57 is **not decided here**; if it is, that is a dated amendment, with a
new hash for `analyze_revisit.py`, before the stopping rule triggers. Until then the descriptive
tables report the switch date beside the merged "did not resolve" state.

**What was read to find this.** The per-day count of `state` over all rows (not split by label,
source or exposure), the `resolver` column, and, before the per-day table, a `label × state`
cross-tabulation of the 488 post-restart rows and of the rows from 2026-09-29 to the outage
(post-restart: benign 267 `public_a` of 291, phish 161 of 197; before: benign 567 of 611, phish
364 of 436). That cross-tabulation is an outcome by arm, read during an operational check and
not for any contrast; it is disclosed here because it was read. No exposure split, 2×2,
blocklist match or retention by lag was computed.

## Amendment — 2026-10-02, Asia/Ho_Chi_Minh: one network per cohort, as a sensitivity analysis outside the family

Follows the deviation record above. Nothing in the registered analysis changes:
`analyze_revisit.py` keeps its hash (`2cdb7a66…`), its population, contrasts, family of three
and stopping rule. Written after the deviation record's reads and before any contrast; no
retention by exposure, 2×2 or blocklist match has been computed on any part of the data.

**Added.** `scripts/sensitivity_revisit_network.py` reruns
`analyze_revisit.analyse`, unchanged, on two single-network cohorts split at the device's boot
on the new network, 2026-10-01 22:57:39 device-local time:

- *old network*: hosts admitted before the switch, with only the observations taken before it;
- *new network*: hosts admitted at or after the switch.

A host admitted before the switch contributes nothing observed after it, so neither cohort mixes
two vantages. Each cohort's RQ1–RQ3 are reported with raw p-values and BH-adjusted within that
cohort only, next to the registered result and labelled as outside the registered family. They
cannot replace or rescue the registered result; agreement in direction is reported as
agreement, disagreement as a vantage dependence the conclusion must carry.

**Why this split, and not "observations before the switch".** At the switch no host had reached
512 h and the RQ2 lag of 256 h had been reached only by hosts admitted 17–20 September, so a
pre-switch-only rerun would be close to empty at every registered lag. The new-network cohort is
the one that will hold most hosts when the stopping rule triggers.

**Gate.** The script refuses until the registered stopping rule holds on the full data, as
`analyze_revisit.py --run` does. On 2026-10-02 it refuses (0 of 300 hosts at 512 h).

**Bound file.** SHA-256 of `sensitivity_revisit_network.py`: `66acc3ea03f367f3809653d72481aab105980f48d79ba62ceecb5784c0e037b7`.
`tests/test_sensitivity_network.py` checks that each host and observation falls on one network
only, that the gate refuses, and that the unchanged analysis runs once per network.

**Correction (2026-10-02, to the deviation record above).** Its heading gives 01:30; the record
was committed at 2026-10-02 00:50:27 +07, which is the time that counts. The direct NXDOMAIN test
it mentions was run between 00:42 and 00:50 that night, not at "00:5x". Nothing else in the record
changes.

## Deviation record — 2026-10-02: a brand-owned host admitted to the phishing arm from a registered source

Not an amendment: nothing registered changes. `[HOST]` was reported by `urlscan_brands`
on 2026-10-02 at 01:00 (+07) with the brand token `bidv` and admitted to this cohort with label
`phish`, inside the confirmatory population. The same day the author identified it as BIDV's news
portal, and the registrar-API record confirms the bank as registrant (the same registrant as
`[HOST]`, name servers matching live DNS). It is a label error, not a lure.

Disposition: the cohort file is append-only and the host stays in it, observed on the registered
schedule; the bound analysis script is unchanged and counts it as admitted. The name joined the
`urlscan_brands` official set on 2026-10-02 (`docs/decisions/urlscan-brands-official.md`), so it
is not reported again; that change narrows the admission stream by this one name from the deploy
onward. When results are reported, the confirmatory figures are stated once as registered and the
host is named as a known mislabel beside them. No outcome for this host was read to write this
record.

**Addendum to the record above (2026-10-02, later the same morning).** A second brand-owned host
from the same source and tick sits in the confirmatory population: `[HOST]`
(reported with the token `vnpt`; `[HOST]` is registered to VNPT Group on the registrar-API
record, and the host resolves to VNPT's own network). The same disposition applies: it stays in
the cohort, the bound analysis is unchanged, and it is named as a known mislabel beside the
confirmatory figures. A third name, `[HOST]` (an Agribank branch), was admitted
through `ct_brands`, outside the confirmatory population. No outcome for either host was read.

**Second addendum (2026-10-02, late morning).** The registrant check added the same day
(`check_brand_registrants.py`, flags only) matched eight `.vn` names to the brand's own registrant,
and the author confirmed all eight as brand-owned. One of them reaches this cohort:
`[HOST]` (registrant Tổng Công ty Cổ phần Bưu chính Viettel), whose subdomains were admitted
from `urlscan_brands` with label `phish`: ten hosts from 2026-09-19 onward, among them `[HOST]`, the company's own log-in console. Same disposition as above: the hosts stay,
the bound analysis is unchanged, and they are named as known mislabels beside the confirmatory
figures. Reading the matches required registrant strings and cohort membership only; no outcome
for any of these hosts was read. With more than ten brand-owned hosts now known in the
confirmatory arm, whether to add a sensitivity analysis that removes registrant-confirmed hosts is
an open question for a dated amendment before the stopping rule; it is not decided here.

## Amendment — 2026-10-02: a sensitivity analysis without brand-owned hosts, outside the family

**Why.** On 2026-10-02 registrar records showed that hosts admitted to the confirmatory phishing
arm belong to the brand they carry (the deviation record and its two addenda above). The bound
analysis keeps them, as admitted, and that stays the registered result. Without a second figure,
a reader cannot tell whether those hosts move it.

**Which hosts are removed, fixed now.** A phishing-labelled host is removed when it equals, or is a
subdomain of, a name in the dated table of `docs/decisions/urlscan-brands-official.md` whose
date is on or before the day the registered stopping rule is met (the day the 300th confirmatory host with a public A record at admission completes its 512 h observation, or 2026-12-01, whichever is earlier). A name added to that table later does not enter, so the removed set
cannot be chosen after the result is known; every name in it rests on a registrar record and a
dated decision by the author. Benign controls are never removed.

**Reporting.** The script reruns the bound analysis unchanged on the reduced cohort and reports
its contrasts, raw p-values and BH adjustment within that run only, beside the registered result,
outside the registered family, together with the list of removed hosts. It cannot replace or
rescue the registered result.

**What was read.** Registrant strings and cohort membership of the brand-owned hosts, as the
deviation record states. No outcome of any host, no contrast and no retention figure.

**Gate.** `scripts/sensitivity_revisit_brandowned.py` refuses until
the registered stopping rule holds on the full data. On 2026-10-02 it refuses.

**Bound file.** SHA-256 of `sensitivity_revisit_brandowned.py`: `cbc40240490b5cb79bca7c55e128e3c655995441bc1e1db278bb1350b981fe3d`.
`tests/test_sensitivity_brandowned.py` checks the date cutoff, that a lookalike name is not
removed, that benign controls stay, and the gate.

**Third addendum (2026-10-02, midday).** The registrant check's third pass matched `[HOST]`
to VNPT Group as registrant, and the author confirmed it. The name itself was admitted from
`urlscan_brands` with label `phish`: admitted on 2026-09-29. Same disposition: it stays, the bound analysis is
unchanged, it is named as a known mislabel, and, its decision row being dated 2026-10-02, it falls
within the brand-owned sensitivity analysis registered the same day. No outcome was read.

**Fourth addendum (2026-10-03).** The registrant check's fourth pass matched six more names to
the brand's own registrant, and the author confirmed all six. Three reach the confirmatory
population from `urlscan_brands` with label `phish`, seven hosts in all: `[HOST]`
(registrant Ngân hàng TMCP Quân Đội), admitted 2026-09-22; `[HOST]` and
`ns1`–`[HOST]` (registrant VNPAY), admitted 2026-09-26 and 2026-09-27; and
`[HOST]` (registrant a VNPT solutions centre), admitted 2026-09-28. Same disposition:
they stay, the bound analysis is unchanged, they are named as known mislabels, and, their decision
rows being dated 2026-10-03, before the stopping rule, they fall within the brand-owned
sensitivity analysis registered on 2026-10-02. The other three (`[HOST]`,
`[HOST]`, `[HOST]`, Viettel's own) came through `ct_brands` into that
separately reported stratum, outside the confirmatory population (see
`docs/decisions/ct-brands-cohort-consequences.md`). No outcome was read.

**Fifth addendum (2026-10-04).** The registrant check's fifth pass matched one more name to the
brand's own registrant, and the author confirmed it: `[HOST]` (registrant Công ty Công nghệ
Thông tin VNPT, a branch of VNPT Group; host on VNPT's AS135905), which reached the confirmatory
population from `urlscan_brands` with label `phish`, admitted 2026-10-04, the name itself. Same
disposition: it stays, the bound analysis is unchanged, it is named as a known mislabel, and, its
decision row being dated 2026-10-04, before the stopping rule, it falls within the brand-owned
sensitivity analysis registered on 2026-10-02. No outcome was read.

**Sixth addendum (2026-10-08).** The brand-registrant check of 2026-10-07 matched two more names to
the brand's own registrant, and the author confirmed both on 2026-10-08. `[HOST]`
(registrant Tổng Công ty Dịch vụ Viễn thông, VNPT VinaPhone; registered 2014; name servers matching)
reached the confirmatory population from `urlscan_brands` with label `phish`, admitted 2026-10-08
(local), the name itself. Same disposition: it stays, the bound analysis is unchanged, it is named
as a known mislabel, and, its decision row being dated 2026-10-08, before the stopping rule, it
falls within the brand-owned sensitivity analysis registered on 2026-10-02. `[HOST]`
(registrant MobiFone's Vinh Long branch) came through `ct_brands` into the separately reported
stratum, outside the confirmatory population, and joins `CT_BRAND_OWNED`. No outcome was read.

**Seventh addendum (2026-10-09).** The brand-registrant check of 2026-10-08 matched one more name to
the brand's own registrant, and the author confirmed it on 2026-10-09. `[HOST]` (registrant
Tổng Công ty Hạ tầng mạng, VNPT Net; registered 2015; name servers at the same provider, Mat Bao,
under a different suffix; host on VNPT's AS135905) reached the confirmatory population from
`urlscan_brands` with label `phish`, admitted 2026-10-08 (local), the name itself. Same
disposition: it stays, the bound analysis is unchanged, it is named as a known mislabel, and, its
decision row being dated 2026-10-09, before the stopping rule, it falls within the brand-owned
sensitivity analysis registered on 2026-10-02. No outcome was read.

**Eighth addendum (2026-10-11).** The brand-registrant check of 2026-10-10 matched one more name to
the brand's own registrant, and the author confirmed it on 2026-10-11. `[HOST]` (registrant
VNPT Group; registered 2022; Cloudflare name servers matching live DNS) came through `ct_brands` with
label `phish` into the revisit cohort's separately reported `ct_brands` stratum, outside the confirmatory population, admitted 2026-10-10 (local), and joins `CT_BRAND_OWNED`. The confirmatory analysis is unaffected.
No outcome was read.

**Ninth addendum (2026-10-12).** The brand-registrant check of 2026-10-11 matched one more name to
the brand's own registrant, and the author confirmed it on 2026-10-12. `[HOST]` (registrant
Viễn thông Đà Nẵng, VNPT's Da Nang unit; registered 2011; name servers matching live DNS; hosts on
VNPT's AS135905 and AS45899) reached the confirmatory population from `urlscan_brands` with label `phish`, two
hosts: the name itself, admitted 2026-10-11 (local), and `[HOST]`, admitted
2026-10-12 (local). Same disposition: they stay, the bound analysis is unchanged, they are named as
known mislabels, and, their decision row being dated 2026-10-12, before the stopping rule, they fall within the
brand-owned sensitivity analysis registered on
2026-10-02. No outcome was read.

**Tenth addendum (2026-10-17).** The brand-registrant check of 2026-10-17 (12:30 UTC) matched one
more name to the brand's own registrant, and the author confirmed it the same day. `[HOST]`
(registrant VNPT's IT company, its Digital Business Solutions Centre branch; registered 2020; name
servers matching live DNS; hosts on VNPT's AS135905; `[HOST]` serves a Zimbra webmail
login) was detected by `urlscan_brands` on 2026-10-17 and had not reached the confirmatory population
when the name was added to the official set, so no cohort row is affected and nothing needs a
disposition. No outcome was read.

**Eleventh addendum (2026-10-23).** The brand-registrant check of 2026-10-22 (05:30 UTC) matched
one more name to the brand's own registrant, and the author confirmed it on 2026-10-23.
`[HOST]` (registrant MobiFone's IT centre; registered 2020; name servers matching live
DNS; hosts on MobiFone's AS131429) came through `ct_brands` with label `phish` into the revisit
cohort's separately reported `ct_brands` stratum, outside the confirmatory population, admitted
2026-10-22 (local), and joins `CT_BRAND_OWNED`. The
confirmatory analysis is unaffected. No outcome was read.

**Twelfth addendum (2026-10-29).** The brand-registrant check of 2026-10-28 (06:30 UTC) matched
one more name to the brand's own registrant, and the author confirmed it on 2026-10-29.
`[HOST]` (registrant VNPAY; registered 2017; name servers matching live DNS and those of
`[HOST]`; hosts on VNPAY's own AS63743; the company's payment-gateway sandbox) reached the
confirmatory population from `urlscan_brands` with label `phish` as two hosts, `[HOST]` and
`[HOST]`, admitted 2026-10-28 (local), after the registered stopping rule was met
(2026-10-19). Its decision row is dated after that day, so by the brand-owned amendment of
2026-10-02 the name does not enter the removed set; wherever the two hosts enter an analysed
cohort, they are known mislabels that the sensitivity analysis does not remove. No outcome was read.

**Thirteenth addendum (2026-11-01).** The brand-registrant check of 2026-10-31 (09:30 and 11:30
UTC) matched four names to the brand's own registrant, and the author confirmed them on
2026-11-01: `[HOST]` (registrant Vietlott), `[HOST]` (Viettel's telecom
corporation; Viettel's own name servers and AS7552), `[HOST]` and
`[HOST]` (Viettel's Nghe An and Ho Chi Minh City branches). All four came from
`ct_brands` in the release of 2026-10-31 (deviation record of 2026-11-01) and were admitted with
label `phish` to the separately reported `ct_brands` stratum on 2026-10-31 (local), after the
registered stopping rule was met (2026-10-19); none is in the confirmatory population. Their
decision rows are dated after that day, so by the brand-owned amendment of 2026-10-02 they do not
enter the removed set; they are known mislabels in the `ct_brands` stratum. No outcome was read.

## Amendment — 2026-10-06, 19:05 Asia/Ho_Chi_Minh: a fifth feed behind `vn_phishing_live`, admitted on the `.vn` suffix alone, disclosed before its first tick

**What was read before this was written.** Row counts and the `sources` column of
`data/raw/vn_phishing_live/detections.csv`, and the 2026-10-05 PhishTank snapshot that
`fetch_blocklists.py` already stores daily for RQ3 of this registration. No revisit observation,
no cohort row and no outcome was read. The numbers: the thirty days to 2026-10-06 added 76 rows
to the ledger, 71 listed by `openphish`, 5 by `chongluadao`, 0 by `phishdb`. Phishing.Database
still returns the largest list but every Vietnamese-targeting name on it is already in
`seen_domains.txt`, so new intake rests on one free feed capped at 300 global entries, and 39 of
the 76 are `*.blogspot.com` hosts.

**What changed.** `fetch_phishing_feeds.py` gains a fifth source, `phishtank`, read from the
daily snapshot (or the live list when the snapshot is older than 48 hours). It is admitted on the
`.vn` suffix only. The shared token rule was tried first and rejected on the evidence: of the 176
snapshot hosts it passed, 164 were not yet seen, and 151 of those were two foreign campaigns the
rule cannot tell from Vietnamese ones (`brt.<random>.click|cc|help|buzz`, `bradesco.*`,
`*-topaz.*`). Thirteen were `.vn`, and those are compromised Vietnamese sites of the kind
Phishing.Database no longer surfaces. A name entering through the new feed carries `phishtank`
in the ledger's `sources` column; a name listed by several feeds carries all of them.

**What this touches, and what it does not.** `vn_phishing_live` is one of the three named
confirmatory sources, and `watch_host_infra.py` captures its new names, so the admission stream of
the running cohort widens from the date above by, on the first tick, at most the thirteen `.vn`
backlog names and afterwards by the feed's trickle. It does not touch the revisit collector
(`watch_revisit.py` unchanged), the lag schedule, the control's sampling fraction, the stopping
rule, the analysis code (`analyze_revisit.py`, hash unchanged), the wildcard probe, or any host
already in `cohort.csv`. The cloaking study's collector admits from `urlscan_brands` and
`ct_brands` only and is not affected. Hosts admitted via the new feed are identifiable by joining
`cohort.csv` to `vn_phishing_live/detections.csv` on hostname and testing `sources` for
`phishtank`, if a sensitivity read excluding them is ever wanted; the analysis will carry one
sentence noting that the stream widened on 2026-10-06, alongside the 2026-09-26 sentence.

**Why now and not at the next re-pin.** The feed is added to keep the live phishing arm from
drying up while collection continues, not to change any registered quantity; left alone, the
arm's new intake for the remainder of the window would be whatever one capped free list returns.

**Order of events.** Amendment written 19:05; collector deployed and one hourly tick run by hand
after the registering commit; the tick's count is appended below under *Timing precision*.

**Timing precision, added 2026-10-06 19:50.** The exact order was: amendment written 19:05,
registering commit 30b1fa61 at 19:06, collector deployed to the device at 19:06:31, hourly tick
run by hand from 19:25:11. That tick saw 1,067 Vietnamese-targeting names across the five feeds,
14 of them new; all 14 were processed and captured, 13 carrying `phishtank` in `sources`. The
hourly cron tick at 19:00 ran the previous collector and admitted nothing through the new feed.

## Amendment — 2026-10-06, 20:16 Asia/Ho_Chi_Minh: a freshness gate on the fifth feed, nine stale names withdrawn from the ledger, and a sensitivity analysis without them, outside the family

**What was read before this was written.** The thirteen rows the 19:25 tick wrote, their
PhishTank submission dates and reported targets, their capture titles, and their presence in
`host_infra.csv` and `cohort.csv`. No revisit observation of any of them and no outcome, contrast
or retention figure was read.

**What was found.** PhishTank's *online-valid* list never clears its online flag. Of the 21
`.vn` names on the 2026-10-05 snapshot, none had a submission in the last 30 days, 2 in 90, 8 in
180, and three were over three years old. Nine of the thirteen admitted names fall outside 180
days: `[HOST]` (newest submission 2023-08-14, an open redirect), `[HOST]`
(2023-01-23), `[HOST]` (2025-09-29), `[HOST]` (2025-08-16, a web-shell path under
`vendor/phpunit`), `[HOST]` (2025-04-21, an IRS lure behind a Cloudflare challenge),
`[HOST]` (2026-02-06), `[HOST]` (2026-02-16, a Plala lure), `[HOST]`
(2026-03-19) and `[HOST]` (2026-03-13, a Bank of America lure). The reported lures sit at
deep paths on compromised but legitimate Vietnamese sites and target foreign victims; the
collector captures the host root, so eight of the nine captures are the sites' home pages (one
failed). All thirteen reached `host_infra.csv` as first captures between 20:05:03 and 20:06:03,
and the 20:15 revisit tick admitted all thirteen to the cohort, label `phish`, source
`vn_phishing_live`, before any of this was acted on.

**What changed.** (1) `fetch_phishing_feeds.py`: a name from PhishTank is admitted only when its
newest submission is at most 180 days old (`PHISHTANK_MAX_SUBMISSION_AGE_D`); on today's snapshot
that admits 4 of the 13. (2) The nine rows above are withdrawn from `vn_phishing_live/detections.csv`
by `scripts/ops/quarantine_phishtank_stale.py` into
`vn_phishing_live/quarantine_phishtank_stale_20261006.csv`, with a `.bak` of the ledger and
`seen_domains.txt` untouched, under the project's quarantine protocol. The four fresher names
(`[HOST]`, `[HOST]`, `[HOST]`,
`[HOST]`) stay.

**What this does not change.** `host_infra.csv` and `cohort.csv` are append-only and are not
rewritten: the nine hosts stay in the cohort as admitted, keep being observed on the registered
schedule, and the bound analysis (`analyze_revisit.py`, hash unchanged) keeps them. That stays
the registered result. The lag schedule, the control's sampling fraction, the stopping rule, the
Benjamini–Hochberg family (m = 3) and the wildcard probe are untouched.

**Sensitivity analysis, outside the family.**
`scripts/sensitivity_revisit_stalefeed.py` reruns the bound
analysis unchanged with the phishing-labelled cohort hosts named in the quarantine file removed,
and reports its contrasts, raw p-values and BH adjustment within that run only, beside the
registered result, together with the list of removed hosts. A name enters the removed set only
if its `quarantined_at` is on or before the registered stopping day, so the set cannot be chosen
after the result is known. Benign controls are never removed. The script refuses until the
registered stopping rule holds on the full data; on 2026-10-06 it refuses.

**Bound file.** SHA-256 of `sensitivity_revisit_stalefeed.py`:
`d31e9b83a0d0230c6df9e147c0ef46fd4099f4e651a3a1b902de01b3570c0e44`.
`tests/test_sensitivity_stalefeed.py` checks the date cutoff, that a fresh name and a benign
control stay, and the gate.

**Order of events.** Gated collector and quarantine script deployed to the device at 20:15:31
and the withdrawal dry-run only (nothing moved); amendment written 20:16; registering commit
64829e26 at 20:17:23; the withdrawal run right after it, its count appended below.

**Timing precision, added 2026-10-06 20:20.** The withdrawal ran at 20:17:23 on the device and
moved exactly the nine rows named above (ledger 2,273 → 2,264 rows; backup written; a second,
dry run moved nothing). The clock on the device and the author's machine agreed to the second.
The heading first carried the time 20:35 by a clerical slip and was corrected to 20:16 in the
commit that adds this paragraph, before any revisit observation of the nine hosts was read.

## Amendment — 2026-10-07, 23:52 Asia/Ho_Chi_Minh: four more names withdrawn from the ledger after a by-hand review, and a sensitivity analysis without every withdrawn host, outside the family

**What was read before this was written.** The six rows `vn_phishing_live` admitted on 2026-10-06:
their feed, capture title, and, for the PhishTank names, the snapshot's submission date, reported
target and reported URL path; and whether each host is present in `host_infra.csv`, `cohort.csv`
and the cloaking cohort. No revisit observation of any of them and no outcome, contrast or
retention figure was read.

**What was found.** Four of the six are not a Vietnamese-targeting phishing page captured as such.
`[HOST]` (PhishTank, newest submission 2026-05-03, reported path
`/us?lt_utm_source=…`) and `[HOST]` (2026-07-31, path `/plo/`) are legitimate Vietnamese
sites with a lure at a deep path, and the captures are their home pages, titled with the
association's and the business's own names. `[HOST]` (2026-04-26)
is a Mondial Relay (French parcel carrier) lure on a `.vn` subdomain, aimed at French recipients.
`[HOST]`, listed by ChongLuaDao, carries the name of a Slovenian operator's customer
portal, and nothing in the name or the feed record targets Vietnamese users. The 20:16 amendment of
2026-10-06 kept the first three on the freshness gate alone, having read the same titles; this
amendment adds a judgement about the captured object that the freshness gate does not make.
All four reached `host_infra.csv`, `cohort.csv` (label `phish`) and the cloaking cohort's
confirmatory stratum on 2026-10-06. The other two rows (`[HOST]`,
`[HOST]`) stay.

**What changed.** The four rows are withdrawn from `vn_phishing_live/detections.csv` by
`scripts/ops/quarantine_vnphish_review.py` into `vn_phishing_live/quarantine_review_20261007.csv`,
with a `.bak` of the ledger and `seen_domains.txt` untouched, under the project's quarantine
protocol. The list is explicit and closed: the script names these four and no others. The
collector is not changed. Capturing the reported path rather than the host root, and a target rule
for `.vn`-suffix admissions, are left for a later dated amendment if they are made.

**What this does not change.** `host_infra.csv` and `cohort.csv` are append-only and are not
rewritten. The four hosts stay in the cohort as admitted and keep being observed on the registered
schedule, and the bound analysis (`analyze_revisit.py`, hash unchanged) keeps them. That stays the
registered result. The stale-feed sensitivity of 2026-10-06 is unchanged and still removes its nine
names only. The lag schedule, the control's sampling fraction, the stopping rule, the
Benjamini–Hochberg family (m = 3) and the wildcard probe are untouched.

**Sensitivity analysis, outside the family.**
`scripts/sensitivity_revisit_withdrawn.py` reruns the bound
analysis unchanged with every phishing-labelled cohort host named in either quarantine file
(`quarantine_phishtank_stale_20261006.csv`, `quarantine_review_20261007.csv`) removed, and reports
its contrasts, raw p-values and BH adjustment within that run only, beside the registered result,
together with the list of removed hosts. A name enters the removed set only if its
`quarantined_at` is on or before the registered stopping day. Benign controls are never removed.
The script refuses until the registered stopping rule holds on the full data; on 2026-10-07 it
refuses.

**Bound file.** SHA-256 of `sensitivity_revisit_withdrawn.py`:
`0a47dc0608e50a54e765303e05f23b039a18572cb777e30c5c0cb25d79dd3416`.
`tests/test_sensitivity_withdrawn.py` checks the date cutoff across both files, that a benign
control and a kept phishing host stay, the gate, and that the review list is exactly these four.

**Order of events.** Amendment written 23:52; registering commit 84c848fa at 23:53:30; quarantine
script and both sensitivity scripts deployed to the device right after; dry run 23:53:53 (four rows
named, nothing moved); withdrawal run 23:53:59.

**Timing precision, added 2026-10-07 23:56.** The withdrawal moved exactly the four rows named above
(ledger 2,265 → 2,261 rows; backup written; `seen_domains.txt` untouched; a second, dry run moved
nothing). Their `quarantined_at` is 2026-10-07T23:53:59. The heading first carried 23:55 by a
clerical slip and was corrected to 23:52 in the commit that adds this paragraph, before any revisit
or cloaking observation of the four hosts was read.

## Amendment — 2026-10-17, 14:15 Asia/Ho_Chi_Minh: four game top-up and fake-order tokens added to the shared feed list, disclosed before their first tick

**What changed.** The shared brand-token list in `watch_urlscan_brands.py` (imported unchanged by
`watch_ct_brands.py`) gains four tokens: `lienquan`, `napthe`, `napkimcuong` and `goshopee`. The
source is the public [HOST] warnings "Giả mạo Garena" (486 reported names, fake Liên Quân and
Free Fire top-up pages that take scratch-card codes and game or Facebook accounts) and "Go Shopee"
(fake-order scams). Each token passed the same 30-day urlscan volume test as the 2026-09-13
additions (41, 72, 2 and 0 scans), and `garena`, `freefire`, `giftcode`, `kimcuong`, `sukien`,
`nhanqua` and `quanhuy` failed it or were rejected as ordinary words and are recorded as rejected
in the code. Garena Vietnam's `[HOST]` and its official top-up portal `[HOST]` join the
official-domain whitelist the same commit (registrar-API record read by the author,
`docs/decisions/urlscan-brands-official.md`), so the tokens admit lookalikes and not the real
services.

**What this touches, and what it does not.** `urlscan_brands` is a named admission source here, so
this widens the admission stream from the deploy onward. Nothing else changes: not the observation
schedule, the stopping rule, the bound analysis script or any admitted host. Hosts admitted through
the new tokens carry them in their `brand` column, so a sensitivity read without them is possible
if ever wanted. Most of these pages are scams rather than credential phishing, which the analysis
will state in one sentence beside the admission count. No outcome was read to write this.

## Amendment — 2026-10-19, 19:09 Asia/Ho_Chi_Minh: the name filter reads the decoded spelling of punycode hosts with both token tiers, disclosed before its first tick

**What was read before this was written.** The Phishing.Database ACTIVE list as served on
2026-10-19 (392,178 entries, 115 punycode hosts), every punycode host present in
`data/raw`, `data/processed` and `data/interim`, and the 2026-10-05 PhishTank snapshot: 417
distinct punycode hosts in all. Hostnames only. No revisit observation, no cohort row and no
outcome was read.

**What changed.** `is_vn_target` in `scripts/vn_filter.py` matched both token tiers (the
hand-curated core and the registry-generated brand tokens) against the raw host only. A punycode
label never matches there, so a Vietnamese brand spelled with one swapped diacritic on a foreign TLD
reached only the fused rule, which by design runs the core tier alone. The filter now also matches
both tiers against the decoded spelling: each `xn--` label decoded, diacritics stripped, dots and
hyphens kept. Nothing is fused, so the reason the fused rule excludes the registry tier does not
apply. The foreign lure-word gate runs on both spellings, and the ccTLD gate still runs first. ASCII
hosts are untouched.

**Why.** One host shows the gap. `[HOST]` (decoded `pay.pubgm.zỉ[HOST]`, a fake
PUBG Mobile top-up page under VNG's Zing brand) was admitted to `vn_phishing_live` at 09:00 on
2026-08-18 by the widened filter of that morning. The 11:04 correction (9e22f80c) kept the decoded
spelling for the fused core only and did not quarantine this host. Every version since then rejects
the name the ledger already holds, so a rebuild that re-applies the filter disagrees with the
collected data on this row.

**Measured effect.** Admissions under the committed filter and the new one, compared on the 417
punycode hosts: 53 and 55. The two added hosts are `[HOST]`, already in
`seen_domains.txt` and `host_infra.csv`, and its registrable domain `[HOST]`, which no
feed lists. None was lost. Against the live feed alone the change admits nothing new: of the 115
punycode hosts on Phishing.Database, the only one it adds is already in `seen_domains.txt`. Read by
hand, none of the 110 still rejected is Vietnamese-targeting. `online.mbạ[HOST]` is Poland's mBank,
and the `cỉmbbank` names are CIMB Philippines, held out by the ccTLD gate.

**What this touches, and what it does not.** `vn_phishing_live` is a named confirmatory source, so
the admission rule of that stream changes from the deploy onward. On the measurement above it adds
no host on its first tick. It does not touch the revisit collector, the lag schedule, the stopping
rule, the bound analysis script, or any host in `cohort.csv`. The cloaking collector does not read
this feed. A host admitted through the decoded spelling alone shows up as one that `is_vn_target`
at 8d13e046 rejects, if a sensitivity read without such hosts is ever wanted. The analysis carries
one sentence noting the change beside the earlier stream-widening sentences.

**Order of events.** Amendment written 19:09; registering commit, deploy and the next hourly cron
tick follow, and their times are appended below. A tick is not run by hand: downloading
Phishing.Database alone takes about eight minutes on the device, and a hand tick would contend for
the lock with the 20:00 cron tick.

**Timing precision, added 2026-10-19 20:15.** Registering commit 9fe5ae10 at 19:11:02; filter
deployed to the device at 19:11:17 (md5 8dc69659), and on the device it accepts
`[HOST]` and rejects `[HOST]`. The 19:00 tick ran the
previous filter. The 20:00 cron tick was the first under the new one: Vietnamese-targeting names
across the feeds went from 1,058 to 1,060, one of them new, `[HOST]` from
`openphish`, an ASCII host the change does not touch. The other is consistent with
`[HOST]`, which is on that day's Phishing.Database list and already in
`seen_domains.txt`, so it was not processed again; the tick log prints counts, not names, so this
is inferred rather than read. No host entered the ledger through the decoded spelling.

## Deviation record and amendment — 2026-10-23, 08:43 Asia/Ho_Chi_Minh: URLhaus's public dump retired, the feed read through its keyed API, disclosed before its first tick

**What was read before this was written.** The collector's tick log, HTTP status codes from
abuse.ch, the ledger's `sources` column, and the `source` column of `cohort.csv` joined to it on
hostname. No revisit observation and no outcome was read.

**What happened.** abuse.ch retired every unauthenticated URLhaus download. The dump
`fetch_phishing_feeds.py` read, `downloads/csv_online/` (every URL currently online, about 16,000),
answers 410 Gone, and so do `csv_recent`, `text_recent`, `json_online` and `hostfile` (checked
2026-10-23 08:0x). The collector logs a failing source and carries on with the others, so
`vn_phishing_live` kept running on four feeds. The tick log carries no timestamps; counted back
from the 08:00 tick of 2026-10-23, 33 consecutive hourly ticks failed with `HTTPError`, the first
being the 00:00 tick of 2026-10-22. Three earlier isolated `HTTPError`/`ReadTimeout` lines are
transient and older.

**What changed.** `src_urlhaus` now reads the URLhaus API's recent list
(`/v1/urls/recent/limit/1000/`: URLs added in the last three days, at most 1,000) with a free
Auth-Key from `ABUSECH_AUTH_KEY`, and keeps those whose `url_status` is `online` (commit a8cf7a24,
deployed 08:10; key in place on the device about 08:20). Without a key it raises a named error
instead of returning an empty list. The population is the same, URLhaus URLs still online; the
window is narrower, which costs nothing here, because the collector only admits names it has not
seen, hourly. A dry run of the five feeds on the device at about 08:30, written to a temporary
file and not to the ledger, returned 652 online URLhaus entries, none Vietnamese-targeting.

**How much the feed matters.** URLhaus is a malware feed. Over the whole ledger 12 of 2,293 rows
list it, all of them listed by it alone: 9 in July, 1 in August, 2 in October (`[HOST]`
on 2026-10-13 and `[HOST]` on 2026-10-14). Its last admission was nine days before the
outage, so the 33 lost ticks most likely lost nothing; that cannot be shown, because the retired
dump cannot be replayed.

**What this touches, and what it does not.** `vn_phishing_live` is a named confirmatory source, so
for 33 hours its admission stream ran without one of its feeds, and from the 09:00 tick of
2026-10-23 that feed is read differently. Two hosts in `cohort.csv` came through URLhaus alone,
`[HOST]` (admitted 2026-10-13 14:05) and `[HOST]` (2026-10-14 02:05, local),
both before the outage and before the stopping rule was met (its 300th host reached 512 h on 2026-10-19). Neither changes. Nothing else changes: not the
revisit collector, the lag schedule, the stopping rule, the bound analysis script or any admitted
host. The analysis carries one sentence noting the gap beside the earlier stream sentences.

**Order of events.** Written 08:43, before the 09:00 tick, the first to read the API with a key;
that tick's count is appended below.

**Timing precision, added 2026-10-23 09:10.** Registering commit 66258964 at 08:42:10. The 09:00
cron tick was the first with the key in place; it finished at 09:07:45 with no URLhaus error line
and saw 1,054 Vietnamese-targeting names across the five feeds, none new. The tick log prints
counts, not per-feed lines, so the absence of the error line is what was read; the 08:30 dry run
gives the feed's size (652 online, 0 Vietnamese-targeting). No host entered the ledger through
URLhaus at this tick.

## Amendment — 2026-10-31, 11:35 Asia/Ho_Chi_Minh: the `ct_brands` collector windows crt.sh on its id, disclosed before its first tick

**What changed.** `watch_ct_brands.py` no longer windows crt.sh's answers on time. Each query
pattern stores the highest crt.sh id it has returned; a later answer reports every certificate
above that id whose `not_before` is within 90 days, and an answer whose highest id is below the
stored one is counted as a failed query. The 168-hour window applies only to a pattern's first
answer. Reason: crt.sh stopped serving `entry_timestamp` (last value in the ledger 2026-08-31), so
the window had been on `not_before` alone, and crt.sh's ingestion lag grew to weeks. A no-window
audit on the device (2026-10-30 22:30 to 10-31 04:58, every pattern asked three times, the
collector's own filters) found 873 filter-passing names certified 18 September to 30 October, of
which the collector had reported 283; 544 of the 590 it missed had reached crt.sh more than 168
hours after issue, and 121 of 510 non-empty answers (24%) came from an older copy of crt.sh's
index. Code, tests and the measurement are in `docs/design-notes/watch-ct-brands.md` and
`tests/test_ct_brands_id_window.py`.

**The missed names are not admitted.** They are written once, by `--backfill`, to
`data/raw/ct_brands/backfill.csv`, which `watch_host_infra.py` does not read, and added to the
collector's `seen` list so the live tick never reports them as new. No host reaches `cohort.csv`
through the backfill.

**What this touches, and what it does not.** `ct_brands` is the separately reported fourth
stratum (amendment of 2026-09-19), outside the confirmatory population and the stopping rule,
which was met on 2026-10-19. From the deploy onward the stratum receives names crt.sh ingested
late; it under-reported them from about early October until now, which this records as a
deviation of that stratum, not repaired. The displacement stated in that amendment can only grow
by the late-ingested names that `ct_brands` reports before a named source does; with an ingestion
lag of weeks that is rare, and the reported displacement count includes them. `ct_brands` hosts
admitted after the deploy are identifiable by `source` and admission time. Nothing else changes:
not the revisit collector, the lag schedule, the stopping rule, `analyze_revisit.py` or any
admitted host.

**Order of events.** Written before the 12:35 tick; the registering commit and the deploy time are
appended below.

**Timing precision, added 2026-10-31 11:40.** Registering commit 929e2c8d at 11:36:36; deployed to
the device at 11:36:53 (`deploy_jetson.sh`, compile check passed, health check clean). The 10:35
tick was still running on the code it had loaded, so the 12:35 tick is the first under the id
window. The backfill is set to start on the device at 2026-11-01 05:00, after three night ticks
have set the marks; its count is appended when it has run.

## Note — 2026-10-31, 14:35 Asia/Ho_Chi_Minh: a language label beside `vn_phishing_live`, recorded and not used (not an amendment)

From the deploy below, `scripts/label_feed_language.py` runs after each hourly tick of
the `vn_phishing_live` collector and writes the language of each capture (`vietnamese`,
`indonesian`, `other_text`, `no_content`, `no_capture`; rules in `scripts/page_language.py`,
the same as the cloaking post-hoc check of 2026-10-30) to a separate ledger,
`data/raw/vn_phishing_live/language.csv`. It does not filter. `detections.csv`, the seen list,
`host_infra.csv` and admission to the revisit cohort are unchanged, so the admission rule of every cohort,
the running one included, is the one registered. The label was chosen over a filter for that
reason. Any analysis that reads the label is post hoc and will be reported as such.

**Timing, added 2026-10-31 15:10.** Commit 9708d3f5; deployed 14:35:51. A first run by hand at
14:36 labelled the 1,759 (domain, scan) rows then in the ledger; the 15:00 tick was the first to
run the labeller after the collector, and found nothing new. No admission changed.

## Deviation record — 2026-11-01, 19:30 Asia/Ho_Chi_Minh: the `ct_brands` id window's first marks released already-ingested certificates as new on 2026-10-31

**What happened.** The id window needs a mark per query pattern, and a pattern's first mark is the
highest id of its first answer under the new code. crt.sh serves answers from copies of its index
that lag one another (Amendment of 2026-10-31), so some first marks came from an older copy. The
next answers from a current copy then carried every certificate between that older top and the
current one as above the mark, and the collector reported them as new. These were certificates
crt.sh had already ingested before the deploy, not certificates it ingested late afterwards. The
ticks of 2026-10-31 reported 78, 228, 113, 27, 13, 50, 9 and 6 names, against 0 to 5 per tick
from 2026-11-01 once every pattern held a current mark. Of the 524 names first detected that day,
523 carry a `not_before` in September or October 2026 and one in August.

**What it touched.** Through `host_infra.csv`, 490 of the 524 were admitted to the revisit cohort under
`source=ct_brands`, the separately reported fourth stratum, on and after 2026-10-31, after the
stopping rule was met. No confirmatory host, no admission rule, and no registered analysis
changes.

**How they are identified, and how they are read.** The 524 names are listed, with their crt.sh
ids, in `data/raw/ct_brands/released_by_first_marks_20261031.csv` on the device (SHA-256
`bfac090b6cfc385f4770aed63eeee55de0c316b9f9488e398b840d92405482e2`; hostnames, kept local). They
stay in `detections.csv`, as every collected row does. Any report of the `ct_brands` stratum is
given with and without the hosts on that list, and the list is what defines them.

**The one-off backfill.** Run on the device from 2026-11-01 05:00 to completion, it wrote 2,748
names certified since 2026-08 (1,417 in August, 1,110 in September, 221 in October) to
`backfill.csv`, none to `detections.csv`, and none was admitted. Fifty-three patterns returned a
stale answer during it and were not backfilled.

**A second backfill pass, added 2026-11-03.** The 52 brand tokens behind those 53 patterns were
queried again on the device from 2026-11-01 22:00 to 23:28 (`--backfill`, same 90-day ceiling,
holding the tick lock). It wrote 274 more names from 18 brands (130 certified in August, 110 in
September, 34 in October) to `backfill.csv`, none to `detections.csv`, and none was admitted.
Thirty-four patterns failed in this pass, 27 with a stale answer and 7 unreachable, and are not
backfilled; no further pass is scheduled. `backfill.csv` now holds 3,022 names.

## Amendment — 2026-11-01, 19:20 Asia/Ho_Chi_Minh: a 48-hour warm-up for a pattern's first id mark, disclosed before its first tick

**What changed.** In `watch_ct_brands.py`, a query pattern's id mark now records when it was first
set. For 48 hours after that, a certificate above the mark is reported as a new detection only if
its `not_before` falls inside the 168-hour window; an older one goes to `backfill.csv` and the
collector's `seen` list, not to `detections.csv`. The mark itself keeps moving with each current
answer, so after the warm-up it reflects crt.sh's current index rather than whichever copy served
the first answer. Marks set before this change carry no set time and are not in warm-up; all 180
current patterns already hold such marks, so the change acts only on patterns added later. Tests:
`tests/test_ct_brands_id_window.py`.

**Why.** The deviation record above: first marks taken from an older index copy released 524
already-ingested names as new on 2026-10-31.

**What this touches, and what it does not.** The separately reported `ct_brands` stratum only, and only for brand tokens added from now on. Nothing else changes: not the admission rule,
the lags, the stopping rule, the bound analysis script or any admitted host.

**Order of events.** Written before the deploy; the commit and deploy times are appended below.

**Timing, added 2026-11-01 19:25.** Registering commit 2d5d77ce at 19:21:33; deployed to the device
at 19:21:46 (compile check passed, health check clean). The tick that started at 18:35 was running
on the code it had loaded, so the 20:35 tick is the first under the warm-up rule.

## Run record — 2026-11-08, 11:00 Asia/Ho_Chi_Minh: the registered analysis, run once

`analyze_revisit.py` (2cdb7a66…, the bound file) ran once, 11:00–11:16, on data pulled from the
device at 10:58 (md5 of each file checked against the device). The four sensitivity scripts of
the 2026-10-02, 10-06 and 10-07 amendments ran right after, each once, each matching its bound
SHA-256. Stopping rule: 804 confirmatory `public_a` hosts past 512 h (300 needed), met; the 300th
reached 512 h on 2026-10-19 16:15 device time. No control week was vantage-suspect (W39–W45,
97.6–99.4 % at 128 h).

**Inputs**, frozen before the run as xz copies in `data/processed/subdomain/inputs/revisit_run_20261108/`
(kept local; its `SHA256SUMS` is 4c81935b…), parent commit 1da015ad: `revisit/cohort.csv`
b4d5d4df…, `revisit/observations.csv` b1e73f12…, `revisit/wildcard_probe.csv` f27e52ea…,
`blocklists/manifest.csv` 7edde882…, `host_infra/host_infra.csv` 1f26c29f…,
`urlscan_brands/detections.csv` 9318e4c1…, `ct_brands/detections.csv` a2481a27…. They were
re-hashed after the last sensitivity finished and had not changed.

**Outputs.** `revisit_analysis.json` 6d0e4d55… and `revisit_sensitivity_network.json` ee6b85a0…
(committed; no hostname). `revisit_sensitivity_brandowned.json` 8a5f90f6…,
`revisit_sensitivity_stalefeed.json` 75c00ac7…, `revisit_sensitivity_withdrawn.json` e39ecbaf…
(kept local: they list the removed hostnames).

**What the registered rules now require, recorded before any text is written.**

- RQ1′ as registered: retention at 128 h 208/226 multi vs 659/728 solo, difference +0.015
  (Newcombe −0.032 to +0.052), Fisher p = 0.60, BH-adjusted 0.90; not rejected. Wildcard-free
  (2026-09-20 amendment): 125/127 vs 504/556, +0.078 (0.032 to 0.107), Fisher p = 0.0016. The two
  readings disagree, so by that amendment the abstract says so and the discussion is built on the
  wildcard-free reading. 112 of 257 multi hosts sit under a wildcard-resolving domain.
- RQ2′: the hosted arm is empty (0 of 252 multi hosts with `public_a` at admission have a hosted
  tenant key; 196 not hosted observed at 256 h). The contrast cannot be estimated; it is reported
  as not estimable, and its p-value is absent. The BH family stays at three.
- RQ3′: exact-hostname membership 1/231 multi and 15/926 solo; expansion reaches 3/231 multi; mean
  gap difference 0.0087, permutation p = 0.0076, BH-adjusted 0.023; rejected in the registered
  direction. The effect is two hosts, and exact coverage of the whole population is 16/1,157
  (1.4 %); both are reported beside the p-value.
- Late-excluded sensitivity: identical to the primary (no confirmatory late observation changes a
  cell). The four out-of-family sensitivities leave RQ1′ unrejected and RQ3′'s two-host gap in
  place; the old-network cohort falls under the fewer-than-100-per-arm rule (46 multi) and no
  test is claimed for it.

Nothing in the registration, the collector or the analysis code was changed for this run.

## Deviation record — 2026-11-08, evening: registered descriptive items the bound script omitted, computed after the run was read

Writing the Results from the run of 11:00 showed that `analyze_revisit.py` does not compute three
things this registration asks to be reported: the displacement count and the four-source pooled
reading of RQ1′–RQ3′ (amendment of 2026-09-19 00:20), and the self-induced hosts by admission day
(amendment of 2026-09-19 23:41, item 4). They are computed by a separate script,
`scripts/describe_revisit_run.py`, which imports the bound file
and calls its functions unchanged, after the registered results had been read. The same script
adds two decompositions that were never registered: the wildcard-free reading of RQ1′ split by
probe status and arm, and the count of confirmatory hosts with a hosted tenant key (RQ2′ had an
empty hosted arm). Everything it writes is descriptive, outside the family, and labelled post hoc
in the manuscript. Inputs: the frozen run inputs plus `vn_phishing_live/detections.csv` e2a74cf9…
and `chongluadao_live/detections.csv` 4733988a… (pulled the same evening, for the displacement
count). Output: `revisit_descriptive.json` 9ee28809… (committed; no hostname). The bound file and
the registered outputs are unchanged.
