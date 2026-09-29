# P3 — successor study: paraphrase evasion with a controlled attack strength

**Written and committed 2026-08-07, before any rewrite was recalibrated and before the study was
run.** The repository history timestamps this file against the commit carrying the results.

## Why there is a successor study

The predecessor ran twice (`PARAPHRASE_PROTOCOL.md`) and resolved nothing both times. Its own
post-mortem identified the defect, and the defect is in the *design*, not the sample size: the
protocol specified what a paraphrase must **preserve** — scenario, call to action, urgency
device, register — and never what it must **change**. Attack strength was therefore free to
drift, and it did: two batches of rewrites written by the same author to the same instructions
came out at mean token Jaccard 0.28 and 0.41 with their sources. The larger corpus was also a
gentler attack, so sample size and attack strength moved together and neither run measured a
fixed quantity.

"Does paraphrasing evade the detector?" is not a well-posed question until *how far the
paraphrase travels* is part of the specification. This study makes it part of the specification.

## The controlled quantity

**Attack strength is defined as token Jaccard** between a source lure and its rewrite, computed
on URL-stripped, lowercased, whitespace-split tokens — the same function the analysis uses, so
the controlled quantity and the reported one cannot diverge.

**Target band: J ∈ [0.20, 0.30].** Chosen before recalibration, for three reasons stated now so
they cannot be rationalised later: (i) it sits at the strong end of what the predecessor
actually produced (its pilot averaged 0.276), so the band is demonstrably reachable by the same
author under the same guardrails; (ii) it is narrow enough — 0.10 wide, against the predecessor's
observed spread of roughly 0.11 standard deviation *within* a batch — that between-batch drift of
the kind that wrecked the predecessor cannot recur inside it; (iii) it is not so aggressive as to
force degenerate text, which would replace one confound with another.

**Both variants of every source must fall in the band.** If the training rewrite and the test
rewrite differ in strength, the paraphrase-adversarial arm is trained at one attack strength and
scored at another, and its apparent efficacy is confounded. Role disjunction (variant `a` trains,
variant `b` tests, splits taken over source messages) is inherited unchanged from the predecessor.

## Calibration procedure, and the firewall that makes it legitimate

Rewrites are revised until they land in the band. This is instrument calibration, and it is only
legitimate under a strict separation:

1. Revision is driven by **the Jaccard metric and the guardrails, and by nothing else**. No
   detector is trained, no miss rate is computed, and no analysis script is run at any point
   during calibration. The calibration tool (`scripts/p3_jaccard_check.py`) reports Jaccard,
   band membership and guardrail violations; it has no access to a model or an outcome.
2. The band was fixed **before** the first rewrite was revised, in this file, in a commit that
   precedes them.
3. Every source is calibrated. There is no discretion to drop a message that resists the band —
   sources are excluded only if the band proves unreachable for them after revision, and any
   such exclusion is counted and reported in the paper, not silently dropped.

Everything else about the corpus is inherited unchanged: the same 193 sources, the same guardrails
(simulated non-resolving link, generic sector references, no real brand, no targeting, no
operational instruction), the same unaccented orthography control.

## Analysis: unchanged from the predecessor

Same three detectors (naive / character-adversarial / paraphrase-adversarial), same three test
conditions (clean / character-obfuscated / paraphrased), same 20 stratified 70/30 splits over
source messages, same five contrasts, same corrected resampled *t* (Nadeau–Bengio) with
Benjamini–Hochberg. Nothing is re-tuned for the recalibrated attack.

## Pre-specified expectations, including the ways this can fail

- **Primary.** With the attack held in a narrow strong band, H1 (paraphrase vs clean, naive
  detector) is expected to be *larger* than the 3.6 pp the uncontrolled 193-lure run measured,
  because that run's attack averaged 0.356 and this one averages roughly 0.25.
- **This may not happen, and the failure is informative.** If the effect does not grow once
  strength is controlled, then the predecessor's shrinkage was not caused by attack drift, the
  post-mortem's diagnosis was wrong, and the paper says so.
- **A null remains possible.** Controlling a confound does not create an effect. If nothing
  clears the corrected test again, the reported result is that the detector's paraphrase
  robustness cannot be resolved at this corpus size *even with attack strength fixed*, which is a
  materially stronger negative than the predecessor's.

## Stopping rule

**One run. No third corpus, no fourth study, no band adjustment after seeing the outcome.** The
predecessor's failure mode was chasing a *p*-value with more data; the corresponding failure mode
here would be chasing it with a different band. If this study does not resolve the question, the
paper reports it as unresolved and the matter is closed for this manuscript.

## Secondary, descriptive, and explicitly not a test

The predecessor's uncontrolled runs and this controlled one share their sources, so the pair
invites a comparison of the same detector under a drifting versus a fixed attack. That
comparison is reported descriptively, as context for the band's effect on measurement, and no
inferential claim is attached to it: the runs are not independent, and the difference between
them was not randomised.

## Outcome record

**Completed 2026-08-07, one run as committed.** Calibration brought all 386 rewrites (both
variants of all 193 sources) into the band — achieved mean *J* = 0.247, range 0.200–0.300, no
source excluded, no guardrail relaxed.

The primary pre-specified expectation held, and by a wide margin. H1 grew from +3.6 pp
uncontrolled to **+14.7 pp** controlled (miss rate 3.2% → 17.9%, 20/20 splits, corrected
*p* < 0.001). Four of the five contrasts now survive Benjamini–Hochberg where none had before:
the paraphrase attack beats the character attack by +12.1 pp (H1b), character-level adversarial
training leaves paraphrasing *worse* rather than better (+4.8 pp, H2 — it repairs the character
attack it was built for, 5.9% → 0.7%, and does nothing for this one), and training on disjoint
paraphrases repairs it (−12.1 pp, H3, 17.9% → 5.9%). The character-attack control remains
non-significant (+2.7 pp), which is consistent with the premise that motivated the whole study.

**What this establishes about the predecessor.** The post-mortem's diagnosis was correct: attack
drift, not sample size, was the binding constraint. The corpus size is identical to the
uncontrolled second run — 193 sources, same splits, same statistics, same code path — and the
only change is that the attack became a fixed quantity. Three times the data would not have
bought this; defining the construct did.

Per the stopping rule, no further run, no band adjustment.
