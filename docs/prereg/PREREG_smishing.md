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
