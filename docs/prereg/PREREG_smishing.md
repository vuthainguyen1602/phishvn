# pre-specification — the SMS text/URL fusion comparison, bound to code

**Registered:** 2026-08-30, after the corpus was ingested and characterised and **before any model
was fitted**. No encoder has been run, no baseline trained, no fusion result exists; there is no
`models/` directory for this study and the claim suite checks that there is not.

## What was already known at registration

Unlike the companion quishing prereg, this one is not written mid-run — but it is written after a
descriptive pass, and that pass changed the hypothesis. Known and published in
`sections/02_dataset_and_huggingface.tex` before this file:

- 2,991 messages, 2,193 ham / 798 phishing; 362 exact duplicates.
- Shortening is **commoner in ham** (11.7% of ham URLs) than in phishing (4.1%).
- Phishing spends 443 registrable domains on 534 URLs; ham uses 218 on 1,268.
- Suffix split: phishing on `.top`/`.cc`/`.vip`/`.online`, ham on `.vn`.
- The publisher's split shares 27 texts across train and test (47 of 597 test rows).
- The phishing arm stops at the end of July 2026; 2026-08 is 627 ham and no phishing.

Nothing about model performance has been observed, because nothing has been fitted. The
descriptive facts above are **not** confirmatory outcomes of this study and are reported as
corpus description.

## Analysis code binding

Ingestion is `scripts/sms_corpus_import.py` as of commit `c3c913b`; figures and tables are
`scripts/make_smishing_assets.py`. The split is keyed on SHA-1 of the message text with a
20% holdout, so every copy of a repeated message lands on one side. The publisher's `train.csv`
and `test.csv` are **not used** and the reason is measured, not asserted.

**Unit of analysis is the message.** n = 2,991 rows over 2,629 distinct texts. Metrics are computed
over test rows; any bootstrap resamples distinct texts, not rows, because 362 rows are duplicates
and resampling rows would count some messages twice.

**Feature set, frozen.** URL channel = the 21 CompPhish columns the web studies model (not the 25
the file carries; `is_https` and the three other collection artefacts are excluded there and are
excluded here, or the numbers are not comparable and the paper says so instead). Text channel =
PhoBERT base, mean-pooled, no fine-tuning in the primary comparison. Fusion = concatenation into
R^789 and one MLP head. No architecture search, no threshold tuning per arm.

## Confirmatory tests (Benjamini–Hochberg, m = 2)

**T1 — does text add anything to a URL model that is already well placed?**
Fusion against the URL-only baseline on the same 21 columns and the same split, paired over 10
seeds, primary metric F1 on the phishing class. Test: Nadeau–Bengio corrected paired t, two-sided.

- **Success:** fusion beats URL-only by ≥ 3 points of F1, with the interval excluding zero.
- **Negative result, and the one this design expects:** the difference is under 1 point, or its
  interval spans zero. Then the message text adds nothing to a lexical URL model on this corpus,
  and the paper reports that as the finding. It does not then go looking for a subgroup where
  fusion wins.

**T2 — is the URL channel doing the work a text channel is credited with?**
Text-only against URL-only, same split, same seeds, same metric. Registered prediction: URL-only
beats text-only by ≥ 5 points of F1, because 443 throwaway domains on abuse-friendly suffixes is a
stronger signal than 160 characters of Vietnamese.

- **Negative result:** text-only matches or beats URL-only. Then the corpus's separability is
  carried by message wording rather than by the URL, the reversal reported in §2 does not have the
  consequence claimed for it, and §2's argument is rewritten.

## Registered diagnostics (descriptive, whatever they show)

- Per-arm performance on the 22 shortened-URL phishing messages, reported with its n and never as a
  stratified test: the stratum this study was designed around has 22 rows and cannot carry one.
- A temporal-split replication (train ≤ 2026-07-15, test after), reported alongside the random
  split. Expected to be worse for both arms; the comparison of interest is whether the *gap* moves.
- Performance on messages carrying no URL at all, where the URL arm has nothing to read.
- Confusion counts at the operating point, and the duplicate-text rate within each split.

## Explicitly out of scope

Fine-tuning PhoBERT and reporting it as the same comparison; adding features to either arm after
seeing a result; relabelling the one near-duplicate pair with conflicting labels (COM_2055 /
COM_2990); dropping the 6 messages with residual e-mail from the fit to improve a number; and any
claim about deployment or about Vietnamese smishing prevalence, which a 67-day corpus with one
phishing-free month cannot support.

## Outcome record (dated, append-only; written after the fits, as a record and not a change)

> **2026-08-30, both confirmatory tests run once at 10 seeds. T2's negative result fired.**
>
> Arms, phishing-class F1 on the test split: URL-only 0.607, text-only 0.929, fusion 0.927.
>
> **T1** (fusion − URL-only) = +0.320, p < 0.0001, 10/10 seeds — above the 3-point success bar.
> Recorded as such, and read only alongside T2, because on its own it says nothing about fusion:
> the baseline it beats is the weak arm.
>
> **T2** (text-only − URL-only) = +0.322, p < 0.0001, 10/10 seeds. The registered prediction was
> URL-only ahead by ≥ 5 points. It is behind by 32. This is the negative result named above, it
> fires, and the consequence registered for it has been carried out: §2's claim that a lexically
> rich URL leaves little room for the text channel is retracted in the paper rather than defended.
>
> *Where the gap comes from, from the registered diagnostics.* 256 of 566 test messages carry no
> URL and the URL arm scores 0.000 on them, which accounts for part of it but not the rest: on the
> 310 messages that do carry a URL the URL arm still trails, 0.686 against 0.919 for text.
>
> *Post-hoc, and labelled so in the paper.* Fusion − text-only = −0.002, p = 0.85, 5/10 seeds. The
> URL channel adds nothing to the text channel. This contrast was not registered — the registered
> pair both use the URL arm as the reference — and it is reported as descriptive.
>
> *A second post-hoc probe, and a hypothesis of ours that failed.* Asked why the URL block adds
> nothing, we predicted redundancy: PhoBERT reads the whole message, so the CompPhish columns are
> derived from a substring the encoder already saw. Wrong. Masking every URL out of the message
> before embedding costs the text arm nothing (0.935 against 0.929, +0.006, p = 0.62), and adding
> the columns back on top of the masked text gains nothing either (+0.006, p = 0.60). The URL is
> close to inert on this corpus whether it arrives as characters or as features; the separating
> signal is the non-URL wording. Recorded here because the prediction was made before the run and
> it failed, and a record that only keeps the guesses that worked is not one.
>
> *Not run.* The temporal-split replication, a registered diagnostic. The shortened-URL stratum
> holds 4 test messages and is reported with that n, no test, as registered.
>
> *Environment note.* PhoBERT ships no safetensors and transformers 5.13 refuses `torch.load` on
> torch 2.5.1 (CVE-2025-32434). Rather than amend the registered encoder or upgrade a pinned
> dependency, the weights were converted locally with `torch.load(weights_only=True)`, which is the
> safe path the policy exists to enforce. The registered encoder is the one that ran.

## Amendments (dated, append-only; valid only while committed before the first fit)

*(none — the design was not changed after registration)*

## Post-registration methods audit (2026-09-02; does not amend the registration)

The registered Nadeau--Bengio test is invalid for the data actually supplied to it. The ten values
are optimization-seed fits evaluated on one fixed holdout, not repeated random train/test splits;
the correction therefore has no applicable resampling unit. The historical p-values above remain
in this append-only record as provenance but are withdrawn from the manuscript and must not be
interpreted. A post-hoc sensitivity analysis now cluster-bootstraps distinct test texts (retaining
duplicate rows within each text group) and recomputes the mean contrast across the frozen fits.
Those intervals quantify test-sample uncertainty conditional on the seed ensemble and do not
retroactively become confirmatory evidence.

The source card defines label 1 as **spam/scam**, not phishing alone. The manuscript now uses the
publisher's broader label semantics and does not claim smishing-specific sensitivity or prevalence.

## Deviation record, 2026-10-03 (post hoc)

Written after the manuscript was audited against this file. Nothing above is changed; this
section records where the paper now departs from what was registered, and why.

- **Temporal-split replication (registered diagnostic, "train ≤ 2026-07-15, test after, all
  three arms").** Registered; the outcome record above says "Not run" and gives no reason. The
  paper now states, in §2 and in its limitations, that the replication was not run and that its
  three retrospective checks (`audit_sms_temporal.py`, one linear probe, one seed) are post-hoc
  substitutes and not the registered replication. For the record, 634 of the 798 positive-labelled
  messages are dated after 2026-07-15 and the last positive-labelled message is dated 2026-07-25,
  so the registered split would have placed most of the positive class on the test side. That is
  a description of the corpus, not a reason the run was skipped: no reason was recorded at the
  time and none is supplied now.
- **Confusion counts at the operating point (registered diagnostic).** `train_sms_fusion.py`
  stores per-arm F1, precision and recall averaged over seeds and did not retain the counts. The
  paper states that they are not reported. The temporal checks store theirs
  (`temporal_audit.json`).
- **Duplicate-text rate within each split (registered diagnostic).** Now reported in §2 from
  `make_smishing_assets.py`: 308 duplicate rows of 2,425 in train (12.7%), 54 of 566 in test (9.5%).
- **Benjamini–Hochberg, m = 2.** The seed-level test it was to correct is withdrawn (audit of
  2026-09-02 above), so no p-value exists to correct. The paper now says so where T1 and T2 are
  reported, instead of leaving the family unmentioned.
- **Has-URL subgroup scores: 0.686 / 0.919 in the outcome record against 0.809 / 0.924 in the
  paper.** The outcome record was written from the 2026-08-30 run, whose subgroup diagnostics
  used the predictions of the first seed only (`preds.setdefault(name, p)`). Commit f2103bff
  (2026-09-02) changed the diagnostics to the mean over all ten seeds' predictions, and the paper
  prints that mean. The whole-test arm scores (0.607 / 0.929 / 0.927) did not move. The record
  above is left as written; the paper's numbers are the ten-seed means.
- **[TB] / [QC] tokens.** The paper's earlier text called these redaction marks. They are the
  prefixes Vietnamese regulation requires on brand-name SMS (thông báo, quảng cáo) and are
  content a deployed detector sees. The shallow-cue floor (0.621 tokens alone, 0.708 with
  diacritics and length) is unchanged in value; its reading now says that it mixes release
  artefact with a genuine sender-register signal, in shares the probe does not separate.
- **Class term.** The paper now uses "positive class" / "positive-labelled" throughout for
  label 1, following the card's "spam/scam"; the word "phishing" in this file's own text and in
  field and macro names is left as written.

### Deviation record, 2026-10-07 (appended; nothing above is edited)

- **What `date` measures.** This registration, and the paper until 2026-10-07, treated the
  publisher's `date` column as the day a message was sent or received, and named the temporal
  split on that basis (cutoff 2026-07-15). The card does not define the column. On inspection
  it takes 21 distinct values over 2,976 dated rows; its last value, 2026-08-03, holds 627 rows
  (all ham) and is the day the Hugging Face repository was created (Hub API `createdAt`); 83
  repeated texts carry more than one date. The paper now reads the field as a batch (entry or
  contribution) date and words the three post-hoc checks as checks across batches. No number
  changes. The registered temporal-split replication stays unrun; had it been run, it would have
  split collection batches, not time.
- **Source revision.** The corpus is analysed at Hugging Face revision
  `a90e2bd7e8df2376939658075ba8094dcee488ad`; on 2026-10-07 the Hub copy of
  `full_dataset.csv` matched the local file byte for byte (SHA-1 7dcbfdbd…).

### Deviation record, 2026-10-08 (appended; nothing above is edited)

Written after the manuscript was audited against the corpus a second time. The registered
contrasts, split, seeds and arms are unchanged and were refitted only to retain quantities the
first run did not store; their F1 values reproduce exactly (0.607 / 0.929 / 0.927).

- **URL extractor.** The ingestion code bound above (`c3c913b`) takes any host-like regex match as
  a URL. 109 of its 1,802 matches end in no Public Suffix List suffix (sentence fragments such as
  `ngay.tcqc` ×23 and `tp.hcm` ×9), and 56 messages count as URL-bearing only because of them. The
  registered arms keep the registered extractor. The corpus description now uses a suffix rule
  (`has_public_suffix`), and a sensitivity refit with that rule (`train_sms_fusion.py --url-rule
  psl`, `fusion_results_psl.json`) is reported beside the registered numbers: URL-only 0.604,
  fusion 0.934, fusion − text +0.005 [−0.002, +0.013]. No conclusion changes.
- **Shortener list.** The bound list counted `zalo.me`, a Zalo profile/chat link (16 positive,
  3 ham URLs), and missed app deep-link services that send a short opaque path to a redirect
  (`go.link`, `onelink.me`, `onelink.to`, `lzd.co`, `grb.to`, `lnkd.in`). The registered
  shortened-URL stratum keeps the bound list (4 test messages, as recorded above). The corpus
  description uses the corrected list: positive 7 of 510 URLs (1.4%), ham 206 of 1,183 (17.4%), of
  which 177 are Viettel's own app deep links; without them ham is 2.9%. The "shortening is three
  times commoner in ham" statement in "What was already known at registration" described the
  bound list and is not repeated in the paper. The premise this study was designed around still
  fails, for a narrower reason: positive-labelled URLs are rarely shortened.
- **Confusion counts (registered diagnostic, previously not reported).** `train_sms_fusion.py`
  now stores per-seed TP/FP/FN/TN, FPR and threshold-free scores, and the paper reports the mean
  counts and FPR for all three arms. The 2026-10-03 entry saying they were not retained no longer
  describes the paper.
- **Shortened stratum and no-URL rows (registered diagnostics).** The paper now prints every arm on
  both: on the 4 shortened positive test messages URL-only flags 1 on average, text and fusion 4;
  on the 256 no-URL test rows text-only scores 0.937 and fusion 0.940.
- **Late-fusion stack.** Its out-of-fold probabilities were fitted on folds stratified over rows,
  so repeated texts could sit in the fold that scored them. The folds are now grouped by message
  text; the stack moves from 0.9305 to 0.9308 and its interval still includes zero.
- **Post-hoc analyses added, all labelled post-hoc in the paper.** Batch × class and a
  leave-batches-out test (`sms_batch_audit.py`: two batches hold 528 of 798 positive rows; text
  models fitted without them flag 0.47 (frozen-PhoBERT head) and 0.13 (TF–IDF probe) of their
  positive texts, against 0.80–0.84 and 0.70–0.74 for a random hold-out of the same size); a
  non-linear and format-feature shallow floor (boosted trees 0.794 and 0.885); a character n-gram
  URL probe and truncation shares (`sms_extra_diagnostics.py`); the URL arm's ceiling on the
  registered split (38.5% of positive test messages carry no URL, so URL-only F1 ≤ 0.762); and an
  error analysis of the text arm's false positives.
- **Registered rule for T1.** The paper now states the registered success rule (≥ 3 points with an
  interval excluding zero) where T1 is reported, and reads it descriptively because the registered
  test is withdrawn.

### Deviation record, 2026-10-08, second entry (appended; nothing above is edited)

Written after a second audit of the revised manuscript. Registered quantities are unchanged.

- **Batch analysis redesigned.** The first entry reported recall at the 0.5 threshold for a
  hold-out of the two positive batches against a random hold-out of the same number of positive
  *texts*. That control trained on more positive rows (294–316 against 269), kept near-duplicates
  of its held-out texts in training, and recall at 0.5 mixes ranking with the training class
  share. `sms_batch_audit.py` now (i) keeps near-duplicate components together, (ii) holds a fixed
  20% of ham components out of every fit and reports ROC-AUC and recall at 2% ham FPR beside
  recall at 0.5, (iii) matches the control's positive training rows exactly (268), (iv) adds the
  reverse direction and (v) holds every batch out in turn. Results: two-batch hold-out probe
  recall@0.5 0.14, AUC 0.974, recall@2%FPR 0.44 against 0.65–0.69, 0.988–0.992 and 0.83–0.93;
  text arm 0.54 / 0.940 / 0.68 against 0.78–0.87 / 0.971–0.986 / 0.84–0.90; reverse AUC 0.971;
  leave-one-batch-out F1 0.699 and AUC 0.958 against 0.928 and 0.991 for component-grouped
  five-fold. The paper's reading changes from "text models miss most of the batches' positives"
  to "held-out batches are still ranked well, but the default threshold no longer separates them".
  The pure-to-mixed analysis is dropped from the paper.
- **Shortener list, third version.** `t.ly`, `ln.run` and `qrco.de` (public shorteners occurring
  only in positive messages) are added under one stated rule. Positive-class shortening is now
  11 of 510 URLs (2.2%); ham without Viettel's app links stays 2.9%.
- **Registered shortened stratum.** Three of its four test messages qualify only through
  `zalo.me`; under the corrected rule it holds one. The paper says so.
- **Surface-feature floor.** A strict variant without token identities (no `[TB]`/`[QC]`, no
  bracketed names) reaches 0.847; the text arm exceeds the full floor by +0.043 [+0.008, +0.082]
  and the strict one by +0.082 [+0.040, +0.128] (cluster bootstrap over distinct test texts).
- **Suffix-rule deltas** are now the seed-paired means, the same estimator as the registered ones
  (T1 +0.330, T2 +0.325).

### Deviation record, 2026-10-08, third entry (appended; nothing above is edited)

- **The second entry's reading is withdrawn.** It read leave-one-batch-out as "held-out batches
  are still ranked well, but the default threshold no longer separates them". Recall at a 2% ham
  false-positive rate on the pooled out-of-group scores falls from 0.924 (component-grouped
  five-fold) to 0.385, so the loss sits in the low false-positive region a detector uses and is a
  ranking loss there, not only a misplaced cut-off. The paper now says that ROC-AUC stays high but
  recall at a low false-positive rate falls by more than half (0.248 for the two positive batches,
  0.612 for the other positives).
- **Leave-one-batch-out grouping corrected.** The second entry said every design kept
  near-duplicates together; leave-one-batch-out grouped by batch only, and 79 near-duplicate
  components span more than one batch. Each component now goes whole to the batch most of its
  texts belong to, and the undated rows form one group (22 groups). Results move slightly against
  the batch: F1 0.699 → 0.682, ROC-AUC 0.958 → 0.952.
- **Reverse direction controlled.** A matched random hold-out of the same number of positive texts
  (five draws) gives ROC-AUC 0.988–0.993 and recall at 2% FPR 0.88–0.93, against 0.971 and 0.61
  for the reverse direction.

### Deviation record, 2026-10-08, fourth entry (appended; nothing above is edited)

- **Size of the leave-one-batch-out loss depends on calibration.** The third entry pooled the
  out-of-group scores of 22 fold models under one threshold and read the fall in recall at 2% ham
  FPR (0.924 → 0.385) as "more than half" and as a ranking loss. The fold models train on positive
  shares from about 0.20 to 0.35, so their scores are not on one scale. With balanced class
  weights recall is 0.495, and with each fold model thresholded on a fixed held-out ham set it is
  0.748; component-grouped five-fold stays at 0.924 under all three. Random pseudo-batches with
  each real batch's class counts, pooled the same way, keep 0.85–0.88, so the batch effect is not
  class composition. The paper now reports the range 0.385–0.748 and drops "more than half" and
  "ranking loss".
- **Headline floor.** The abstract and conclusion now lead with the strict surface floor (0.847,
  diacritics, length and format, no token names) instead of 0.885, since bracketed tokens include
  abbreviated words and brand names.
- **August false-positive rates.** The 2.50% and 9.12% rates come from different test rows and
  training class shares; the paper no longer reads their difference as the size of a batch effect.
- **Shortener list, fourth version.** `cps.onl`, `o2o.vn` (`l.o2o.vn`) and `id.vin`, short-link
  domains with opaque paths in ham, are added. Ham shortening without Viettel's links is 3.3%.

### Deviation record, 2026-10-08, fifth entry (appended; nothing above is edited)

- **Leave-one-batch-out with the text arm.** The frozen-PhoBERT head (registered head, seed 0) under
  the same groups gives recall at 2% ham FPR 0.461 pooled and 0.741 with per-model thresholds,
  against 0.904 and 0.906 for component-grouped five-fold, the same pattern as the probe. Within
  the 13 batches that hold both classes, the probe's ROC-AUC ranges 0.81–1.00. Both are reported.

### Deviation record, 2026-10-08, sixth entry (appended; nothing above is edited)

- **Pseudo-batch control under every calibration.** The fourth entry's pseudo-batch control was run
  with pooled scores only. Under balanced class weights it gives 0.88–0.91 and with per-model
  thresholds 0.92–0.93, against 0.495 and 0.748 for the real batches, so each leave-one-batch-out
  figure now has its own control. The explanation of the calibration dependence is reworded: a
  pooled threshold compares scores across fold models.
- **Within-batch ROC-AUC** is now reported for groups with at least five positive texts (8 groups,
  0.980–1.000). The earlier 0.81–1.00 over 13 groups counted groups with one to three positives,
  after near-duplicate components were moved to their majority batch.
