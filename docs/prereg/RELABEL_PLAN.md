# P3 revision 2 — blinded relabelling of the page subset

Plan committed 2026-10-17, before any page is rated. (The sheets were generated a few minutes
before this commit, from the sampling rule stated below, and no rater had seen them.) The scoring
script (`scripts/score_p3_relabel.py`) is committed with this plan and its
SHA-256 is recorded below before any rating exists.

## Why

The 414-page subset inherits its labels from the feeds: 121 of its 207 phishing pages carry no
evidence tier and none is gold, and the only label-noise estimate (12.1%) comes from an audit of
the corpus, not of this subset. Reviewer 1 (revision-1 round) notes that the label-error rate is
"disclosed but not resolved", and Reviewer 3 (comment 8) asks how far label quality limits the fusion
study. This audit measures the labels of the subset itself and re-asks the paper's main contrasts
on the pages whose label two independent raters confirm.

## Sample

- All 207 phishing pages of the subset, and 100 of its 207 benign pages drawn with
  `random.Random(20261017)`. The benign pages are there so that a rater cannot infer the class
  from the sheet, and to estimate noise in the benign arm.
- The 307 items are shuffled with `random.Random(20261017)` and renamed `R001`–`R307`. The key
  (item → manifest row, source label, feed, tier) is written to `data/docs/verify_p3/key.csv` and
  is not given to the raters.

## Raters and blinding

Two native speakers of Vietnamese (A and B), working independently, who do not see the source
label, feed, evidence tier, model scores or each other's sheet until both sheets are complete.
They are the two raters of the revision-1 paraphrase rating and have not seen these pages.

## Material per item

The domain, the first 1,500 characters of the page's visible text, and the screenshot of the
capture the models read (three items have no screenshot and are rated on text). An offline viewer
(`xem_trang.html`) shows them item by item, and the verdicts go into the rater's own `.xlsx`.

## Codebook

The four verdicts of the published corpus audit (`data/docs/verify/CODEBOOK.md`), unchanged in
meaning: `phishing` (impersonates a brand or service to harvest credentials, OTPs or payment data),
`scam` (abusive but not credential phishing: gambling, betting, investment or crypto fraud,
counterfeit shops, adult content, predatory lending), `legitimate` (a real operator's own page),
`unsure`. Precedence: credential harvesting outranks the rest.

Evidence, in order: (1) the capture shown on the sheet, (2) a web archive of the domain near the
capture date, (3) the NCSC trusted-organisation registry. **Forbidden**, because they re-derive the
label being audited: ChongLuaDao, NCSC blacklists, OpenPhish, URLhaus, VirusTotal, any reputation
or blocklist service, and urlscan.io's own verdicts or tags. Plausibility is not evidence:
`legitimate` and `scam` each need positive evidence, and without it the verdict is `unsure`. Each row records the
deciding source in `evidence` (`capture`, `archive`, `registry`, `name`, `none`).

Before starting, both raters judge five practice pages from outside the sample (gated captures
dropped by the off-corpus cut) and compare readings of the categories, then rate alone.

## Outcomes, fixed now

1. **Agreement.** Raw agreement, Cohen's κ and PABAK on the four verdicts and on the collapsed
   verdict (abuse = `phishing` ∪ `scam`, `legitimate`, `unsure`), over all 307 items.
2. **Consensus label.** `abuse` if both raters say abuse, `legitimate` if both say legitimate,
   and `unresolved` otherwise. There is no arbitration round.
3. **Label noise in the subset.** Phishing arm: consensus-legitimate items over consensus-resolved
   phishing items, with a Wilson 95% interval, and the share unresolved. Benign arm: consensus-
   abuse items over consensus-resolved benign items, with a Wilson interval.
4. **Sensitivity of the main contrasts.** The verified phishing arm is the phishing pages with
   consensus `abuse`. The benign arm is the 207 benign pages of the subset minus every rated
   benign page that either rater called abuse (unrated benign pages stay). The benign arm is then
   down-sampled to the size of the verified phishing arm with `random.Random(42)`. On that subset,
   the paper's protocol is re-run unchanged: 20 stratified 70/30 splits (seeds 0–19), the
   standardised linear head, five content encoders, both directions of the ablation, four
   metrics, corrected resampled t with Benjamini–Hochberg within each column, and the MDE.
   Reported in the Supplementary Material, with one paragraph in Section 6.1. The main tables stay
   on the 414-page subset.
5. **Reading.** If a contrast that is resolved on the 414 pages loses its resolution, or one that
   is unresolved gains it, on the verified subset, the article says so in Section 6.1 and the
   discussion. If the sign of a resolved contrast reverses, the abstract says so too. A smaller
   verified subset lowers power, so a lost resolution is reported together with the verified
   subset's MDE and is not read as evidence of no effect.
6. **Descriptive only.** The phishing arm's composition (credential phishing against scam) and the
   deciding-evidence shares.

The analysis runs once, after both sheets are complete. Sheets are due back on 2026-10-24.

## Script lock

`score_p3_relabel.py` SHA-256 `1e41cd94b95811e9c504efcac9c5305c6997ec86e802aefd15f2a94922c406c5`, recorded before the sheets went out. The script
was dry-run once end to end on randomly filled sheets to check that it runs. Those sheets and
their outputs were deleted and are not results.

## Amendment 1 (2026-10-17, before any rating)

Reading the subset's pages while building the sheets showed two things, recorded here before any
rater has seen a page so that the decisions they lead to are fixed in advance.

**Naming the positive class.** The phishing arm looks like a mixture of credential phishing, scam
and some apparently legitimate pages. The article will name its positive class by the raters'
consensus composition of the phishing arm: if consensus `phishing` pages are the majority of the
consensus-abuse pages, the class stays "phishing pages"; otherwise the article calls it "phishing
and scam pages reported by the feeds" in the title, abstract, highlights and text, and keeps
"phishing" only where it names the feeds' label. The rule is applied whatever the result.

**Non-content pages that passed the gate.** The parked/error gate matches English phrases and
misses Vietnamese ones. The following pattern, fixed now, flags pages whose first 600 visible
characters say the domain has expired, is unpublished, is for sale or shows a default page:

```
tên miền của website này đã hết hạn|trang chưa được xuất bản|trỏ tên miền thành công|
website hiện không hoạt động|trang web mặc định|mua bán hoặc thuê tên miền|bán tên miền|
tên miền chưa được xác nhận|bạn muốn mua tên miền
```

(case-insensitive, one alternation). On the subset it flags 16 phishing and 5 benign pages. A
robustness check, separate from the rating, drops the flagged pages from both arms, down-samples
the remaining benign pages to the size of the remaining phishing arm with `random.Random(42)`, and
re-runs the same protocol as outcome 4 (the `rerun` routine of the locked scoring script, imported,
not edited). It is reported in the Supplementary Material with one sentence in Section 6.1, and the
article states that the gate missed these pages. The main tables stay on the 414-page subset.
Script: `scripts/p3_dead_page_check.py`.

## Amendment 2 (2026-10-17, before any rating)

The `evidence` field is not collected, to keep the raters' task to one required field per page.
The column stays in the sheets, hidden, because the locked scoring script reads columns by
position, and the script is not edited. Outcome 6 therefore reports the composition of the
phishing arm only, not the deciding-evidence shares. Nothing else changes.

## Amendment 3 (2026-10-17, before any rating)

Many phishing-arm pages were captured years after their domains were listed: the ChongLuaDao
public list the watcher reads has not been updated since 2024-05-16, so its "live" captures of
July and August 2026 show what domains listed in 2021–2024 served two to five years later, and
some have changed hands (for example a bank-named domain now serving a gambling page). The guide
now states the rule the codebook already implies by putting the capture first: raters judge the
page the capture shows, not the domain name, and use the name alone only when the page shows
nothing. The consensus label therefore describes the page the models read, which is the label
the sensitivity analysis needs.

## Amendment 4 (2026-10-17, before any rating)

All 414 pages of the subset are rated: the 207 phishing pages and all 207 benign pages, instead of
a sample of 100 benign pages. The items are re-shuffled with the same seed and renamed
`R001`–`R414`; `key.csv` is regenerated. Outcome 3 then estimates benign-arm noise on every benign
page, and outcome 4 drops every benign page that either rater calls abuse (the rule already
written for rated pages, now applied to all of them). The scoring script is unchanged: it reads
whatever items the key lists. Expected rating time is about seven hours per rater. Four of the 414 pages have no screenshot and are rated on text.

## Amendment 5 (2026-10-17, before any rating)

The scoring script's generated sentence had the earlier sample sizes typed into it (207 phishing,
100 benign, 307 items). It now prints the counts it reads from the key, and a clause joined by a
semicolon is rewritten. No computation changes. The script was re-checked on randomly filled
414-item sheets with `--no-rerun`; those sheets and outputs were deleted. The locked SHA-256 is now
`4ce31b819cafbe432bfcdee6633f480097b388edddd3917b7c7db596b1c2b239` (it replaces the one recorded above).

## Amendment 6 (2026-10-17, before any rating)

The 21 non-content pages flagged by the pattern of amendment 1 (16 phishing, 5 benign) are not
rated: they show nothing to judge. The sheets hold the remaining 393 pages (191 phishing, 202
benign), re-shuffled with the same seed and renamed `R001`–`R393`. In outcome 4 these pages are
also kept out of the verified subset: phishing pages could not reach a consensus anyway, and the
benign pool now excludes them explicitly (`make_p3_relabel_sheets.dead_rows`), since the earlier
rule kept unrated benign pages. Outcome 3's noise rates are computed over the rated pages. The
scoring script was dry-run end to end on randomly filled 393-item sheets and those outputs were
deleted. Its generated sentence now names the rated counts and calls the benign arm an arm. The
locked SHA-256 is now `4622ffd2037d8f90fd33975203f95266a51730c717134ecd36bff6c615d1a295` (it replaces the ones recorded above).

## Amendment 7 (2026-10-17, before any rating)

About one screenshot in ten is blank or shows a server error, because the renderer captured the
page before it loaded or while its server failed, while the stored DOM, which the models read,
holds the page's text. The guide now tells raters to judge such pages by the text shown under the
screenshot, and to choose `unsure` when neither shows what the page is. The sample, the sheets
and the scoring script are unchanged.

## Amendment 8 (2026-10-17, before any rating)

A secondary, exploratory re-run on credential phishing alone. Positive pages are the phishing-arm
pages that **both** raters call `phishing` (not `scam`). The benign arm is built as in outcome 4
(all rated benign pages minus any that either rater calls abuse, minus the non-content pages) and
down-sampled to the size of the positive set with `random.Random(42)`. The same protocol is run
(20 stratified 70/30 splits, standardised linear head, five encoders, both directions, four
metrics, BH within each column, MDE). It runs only if at least 30 pages reach this consensus;
below that, only the count is reported, because a test fold would hold about nine phishing pages.
It is reported as exploratory, in the Supplementary Material with one sentence in Section 6.1,
beside its MDE. Script: `scripts/p3_relabel_phishing_only.py`, which imports
the locked scoring script's routines and does not edit it.

## Deviation record (2026-10-17, 15:00): the author rated the sheet, and it is not used

Not an amendment: nothing registered changes. The author filled rater A's sheet himself on
2026-10-17 (saved 14:43) and compared it with the key in one descriptive cross-tabulation. The
author is not blind (he built the corpus and has read the source labels), so that sheet is not a
rater's sheet and enters no analysis. It is kept outside the package as
`data/docs/verify_p3/hold/author_reading_20261017.xlsx`. The package already sent to the raters
(`cham_trang_P3.zip`, built 00:22) holds two blank sheets, and the copy in `package/` was restored
blank. Raters A and B are the two raters named above, and neither is shown the author's sheet or
its counts. No amendment to this plan, its threshold or its scripts follows this record.

## Note on the public copy (2026-10-24, after scoring)

Not an amendment. The sheets came back on 2026-10-23 (rater A) and 2026-10-24 (rater B), and the
scoring script ran once, unedited, at the SHA-256 locked in amendment 6. The public mirror's copy
(`scripts/score_p3_relabel.py`) differs from the locked file in one docstring line only: the
exporter rewrites the `RUN:` path for the mirror's flat layout, so its hash is not the locked
one. Restoring that line to `RUN:    python scripts/score_p3_relabel.py [--no-rerun]`
reproduces the locked hash. The key and the raters' sheets are not released.
