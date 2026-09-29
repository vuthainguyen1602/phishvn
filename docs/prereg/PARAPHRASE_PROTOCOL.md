# P3 — paraphrase-evasion experiment: protocol

**Written and committed 2026-08-07, before the experiment was run.** The repository history
timestamps this file against the commit that adds the results. Deviations, if any prove
necessary, are recorded at the bottom rather than silently absorbed.

## Why this experiment exists

P3 currently reports one evasion study: character-level obfuscation (zero-width spaces,
homoglyphs, leetspeak) applied to held-out simulated phishing. The content detector barely
moves (miss 3.6% clean → 5.5% obfuscated). That result is real but **weak evidence for the
claim the title used to make**, because a char-`n`-gram TF-IDF model is close to
*constitutionally* robust to character noise: perturbing 10% of characters leaves most 2–5
grams intact. The experiment therefore cannot distinguish "the detector understands the lure"
from "the attack was too shallow to matter".

An LLM attacker does not need character tricks. It rewrites. The question this experiment
asks is the one a Q1 reviewer will ask:

> Does the content detector survive an **intent-preserving paraphrase** — the same scam, the
> same call to action, realised in different words?

## Population

The P2 simulated corpus already in the repo: 73 simulated phishing + 54 benign messages
(`data/processed/dataset_{sms,email}.csv`), one generator (`claude-fable-5`), 7 scenarios ×
2 channels. Guardrail links (`sim.example.vn`) are stripped before vectorising, as in the
existing study, so the detector must learn the lure and not the URL marker.

## The attack

**A2 (paraphrase).** For each of the 73 source phishing messages, two rewrites that preserve:
the scenario, the call to action, the urgency device, the channel register (SMS terse / email
formal), and the simulated link. What changes: word choice, sentence structure, ordering,
and the framing of the pretext.

**Orthography is held constant — a deliberate control.** Every source message is written in
unaccented Vietnamese (verified: 0/73 contain diacritics). The paraphrases are written the
same way. Restoring diacritics would change the character set wholesale and a char-`n`-gram
model would collapse for reasons that have nothing to do with paraphrasing; that would be a
third surface attack wearing a semantic disguise. Holding orthography fixed makes the
contrast isolate **lexical and syntactic** change.

**Role-disjoint variants, split by source message.** Variant `a` of a message may only enter
*training* augmentation; variant `b` may only enter the *test* attack. Splits are taken over
source messages, so no paraphrase of a training message is ever tested and no paraphrase of a
test message is ever trained on. Without this, adversarial training would be scored on
rewrites of sentences it had already seen — the leakage this paper's own thesis condemns.

**Guardrails, machine-checked** (same as the corpus they extend, `p2_generate_corpus.py`):
simulated non-resolving link only; generic sector references, never a real brand; no real
person targeted; no operational attack instruction. `p3_paraphrase_corpus.py` validates all
of these and raises rather than emitting a violating row.

## Design

Three detectors × three test conditions, all on the same splits:

| | A0 clean | A1 char-obfuscated | A2 paraphrased |
|---|---|---|---|
| **D0** naive (trained on clean) | baseline | published result | **the attack** |
| **D1** adversarially trained on char-obfuscation | | published result | **cross-attack transfer** |
| **D2** adversarially trained on paraphrases (variant `a`) | | | **does the right defence work?** |

The two bold-right cells are the point. D1×A2 asks whether robustness bought against surface
noise transfers to semantic rewriting — if it does not, then "adversarially trained" in the
existing table means "trained against the attack that did not matter".

## Pre-specified hypotheses

- **H1.** A2 raises the miss rate over A0 for the naive detector D0. *(The attack works where
  character noise did not.)*
- **H2.** D1 does not reduce the A2 miss rate relative to D0. *(Char-level adversarial
  training does not transfer to paraphrase.)*
- **H3.** D2 reduces the A2 miss rate relative to D0. *(Augmenting with disjoint paraphrases
  does help — the defence is attack-specific, not general.)*

H2 is a null and is reported as one either way; the repo's practice is that nulls are results.

## Statistics

20 stratified 70/30 splits over source messages, paired by seed. Differences are tested with
the **corrected resampled `t`** (Nadeau–Bengio) through `scripts/paired_eval.py` — the shared
implementation every content experiment in this repo routes through; the naive paired `t`
inflates `t` by ≈3.1× at this K and split ratio. Benjamini–Hochberg across the family of
comparisons reported in the table.

**Resolution, stated up front.** 73 phishing messages × 30% ≈ 22 per test split, so one
missed message ≈ 4.5 pp. This experiment can resolve large effects only; it is a pilot, and
the table reports it as one. Effects smaller than ~5 pp are not interpretable here.

## Known limitation, and its direction

The paraphrase attacker is the same model that authored the source corpus. Shared authorship
plausibly leaves shared style, which makes the paraphrases *closer* to the training
distribution than an independent attacker's would be. That biases the measured attack
**downward**: a real third-party attacker should do at least this well. The limitation is
therefore conservative with respect to H1, and the leave-one-LLM-out protocol already
specified in §4 remains the way to remove it entirely.

## Deviation record

### 2026-08-07 — corpus extension to the projected size, committed before generating it

**Cause.** The pilot ran as specified and resolved nothing: all five contrasts landed in the
time-stamped pre-specified direction and none survived the corrected test (adjusted *p* = 0.36), with the
projection putting H1 at ≈2.5× the corpus. The pilot's own conclusion was that the question is
answerable only with more lures.

**The honesty problem this entry exists to address.** Extending the corpus *after* seeing the
outcome is outcome-conditional data generation — the failure mode this project's
pre-specification discipline exists to prevent. Writing it down before generating does not make
it as clean as pre-specification; it makes it auditable. What follows is fixed now, and the
repository history timestamps it against the commit that adds the new lures.

**Fixed before generation:**

1. **Size.** 73 → **193** simulated phishing lures (+120). Chosen as the pilot's own H1
   projection (2.5 × 73 = 183), rounded up to fill the scenario cells evenly. Benign controls
   scale with it, 54 → **143** (+89), holding the pilot's 1.35:1 phishing-to-benign ratio so
   the decision threshold is not moved by composition instead of by the attack.
2. **Composition.** The same 14 (channel × scenario) cells as the pilot, deepened
   proportionally. No new scenario, no new channel, no change of guardrails.
3. **Paraphrases.** Every new lure gets the same two role-disjoint rewrites under the same
   rules (variant `a` train-only, variant `b` test-only, orthography held unaccented,
   guardrails machine-checked).
4. **Analysis: unchanged.** Same script, same 20 splits, same three detectors and three
   conditions, same corrected resampled *t* with Benjamini–Hochberg over the same five
   contrasts. Nothing in the analysis is re-tuned for the larger corpus.
5. **One re-run.** The extended experiment is run **once** and its result is what the paper
   reports, significant or not. The corpus is not extended again in response to the outcome.
   If the contrasts still do not resolve, that is the finding.

**What this cannot fix, and is therefore disclosed in the paper.** The author of the new lures
knew the pilot's direction. The mitigations are structural rather than blinding: the
composition rule above was fixed before writing, the lures were written to the original
generation recipe rather than selected for effect, and the pilot's numbers stay in the paper
beside the extended ones so a reader can see whether the effect moved when the corpus grew.

**What may well happen.** The projection assumed the pilot's effect sizes and per-message
variance carry over to the new lures. They need not. A smaller effect at *n* = 193 is a real
possibility and would be reported as such rather than explained away.

**Outcome, recorded 2026-08-07 after the single committed re-run.** That is what happened. The
leading contrast roughly halved (H1 +8.4 → +3.6 pp) and nothing cleared the corrected test;
H1b effectively vanished (+5.7 → +0.9 pp, 10/20 splits). Two causes, reported together because
we cannot separate them: projections extrapolate from the noisy estimate that motivated them,
*and* the extension's rewrites turned out to diverge less from their sources than the pilot's
(mean token Jaccard 0.41 vs 0.28), making the larger corpus a gentler attack. Per the rule
above the corpus was **not** extended a second time. The protocol's own gap is now the paper's
main finding here: it fixed what a paraphrase must *preserve* and never what it must *change*,
so attack strength was free to drift between batches. Any successor study must pre-specify a
target lexical-divergence band and report the achieved one.
