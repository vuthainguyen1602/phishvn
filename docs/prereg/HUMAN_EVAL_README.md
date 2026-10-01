# Human evaluation of band-controlled paraphrases (revision R2.8 / R3.6)

Sheet: `human_eval_sheet.csv` — 50 source–rewrite pairs drawn with seed 20260928 from the
band set's test-role rewrites (all inside J ∈ [0.20, 0.30]). Drawn 2026-09-28, BEFORE any
label was collected.

## Protocol (fixed before labelling)

Two native-speaker annotators (A and B) label independently, without discussing pairs, and
without seeing any detector output. For each pair, two binary judgements:

- `meaning_preserved_*`: 1 if the rewrite communicates the same situation and the same
  requested action as the source; 0 otherwise.
- `lure_intact_*`: 1 if the rewrite, sent as a real SMS/e-mail, would still function as a
  phishing lure (a plausible pretext + a call to action toward the link); 0 if it reads as
  broken, incoherent, or no longer asks the reader to do anything.

Judge the text as a message, not the grammar: unaccented telegraphic Vietnamese is the corpus
register, not a defect.

## Reporting (what goes in the paper)

Per question: proportion positive with a 95% Wilson interval, and inter-annotator agreement as
raw agreement + Cohen's kappa. The paper reports this as a post-hoc addition at revision.

Fill A's columns first, then hand the sheet (with A's columns hidden/removed) to B.

## Amendment 1 (2026-09-29, before any label was collected)

Both sheets were checked empty (0 of 100 cells in columns D–E filled in `cham_A.xlsx` and
`cham_B.xlsx`) when this amendment was written. At the author's request a third judgement is
added to each pair:

- `more_persuasive_*` (sheet column F): which message would more easily deceive a recipient,
  `A` (the source), `B` (the rewrite), or `ngang` (about equal / cannot tell).

Why: the two binary questions ask whether the rewrite is still a usable lure; they cannot say
whether rewriting weakened it. A rewrite can be a working lure and still be weaker than its
source, and the two findings answer different questions.

Reporting for the new column, fixed now: the share of pairs judged `A`, `B` and `ngang` per
annotator, with 95% Wilson intervals; the share judged "rewrite at least as persuasive" (`B` or
`ngang`); and inter-annotator agreement as raw agreement plus unweighted Cohen's kappa over the
three categories. No test is run on it. Two limits are stated with the result: the order is
fixed (A is always the source, and the sheet names B as the rewrite), so annotators are not
blind to which message is which; the judgement is therefore descriptive, not a blinded
preference test. The two binary questions and their reporting are unchanged, and they remain
the answer to the reviewers' question (R2.8, R3.6); column F is reported next to them, not in
place of them.

## Analysis script (bound 2026-10-01, before any rating was seen)

Both sheets were checked empty again (0 of 150 rating cells in each) when the scoring script was
committed. `scripts/score_p3_human_eval.py` implements the reporting above
and nothing else: it refuses to score unless every row's messages match `human_eval_sheet.csv`
and all 300 rating cells hold an allowed value. `--status` counts filled cells without scoring.

SHA-256 at binding: `e0530ffd97cf8e84ba6b432a40b8eee7a65ba402671fd9664e828cd22194a5c9`

Any later edit to the script changes this hash and must be stated as a deviation next to the result.

Correction (2026-10-01, after scoring): the heading above first said "before any label was
collected". That was too strong. When the script was committed (19:03 +07) the repository copies
were empty, but an Excel lock file showed cham_A.xlsx open, and the returned sheets were saved at
19:33 (B) and 20:48 (A), so rating may already have begun on the annotators' copies. What holds is
that the script, which implements only the reporting fixed on 2026-09-28/29, was committed before
the analyst saw any rating. The script's hash is unchanged.
