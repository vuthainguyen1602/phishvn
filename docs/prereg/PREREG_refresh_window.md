# pre-specification — refresh-window confirmation of the Stack[CB+LR] crossover

**Registered:** 2026-08-17, BEFORE the next Jetson full sync + corpus regeneration
(expected ~2026-08-29). This file is frozen at registration; any amendment must be committed
before the new data is pulled, and amendments after the pull invalidate the confirmatory
status of the affected test.

## Background (already observed, NOT part of this registration)

On the canonical phishing-temporal split (cut 0.70 of the 19,635 dated phishing rows,
registrable-domain leakage guard; `data/processed/p2/p2_temporal_strict.csv` and companions):

- Stack[CB+LR] (CatBoost + LogReg base learners, stratified 5-fold OOF probabilities,
  LogisticRegression meta-learner; `scripts/run_p2_stacking_baseline.py`) vs CatBoost,
  K=20 seeds, Nadeau–Bengio corrected paired t-test:
  temporal ΔPR-AUC +0.0032 (20/20 seeds, p=0.003); random-same-rows −0.0031 (p=0.002).
- Sign stable at rolling origins 0.60 / 0.65 (`data/processed/p2_rollorigin_*.csv`).
- Known limitation motivating this registration: the phishing test window is FIXED across
  seeds, so those p-values certify robustness to benign resampling and model seeds on ONE
  future window, not a distribution over futures.

## Confirmatory family (BH-adjusted, m = 2)

**Test 1 — crossover replication on the genuinely new window.**

- Data: after the refresh lands, define `ph_te2` = dated phishing rows with
  `collected_at` strictly later than the maximum date in the CURRENT corpus
  (`data/processed/dataset_url.csv` as of this commit), with the registrable-domain guard
  applied against ALL current dated phishing (train + current test). Benign: the refresh's
  benign pool, 70/30 per seed as in the canonical protocol.
- Train window: the full CURRENT dated phishing set (train + current test) + benign train mask.
- Models: CatBoost (canonical config: 300 iterations, depth 6, lr 0.1) vs Stack[CB+LR]
  (exact `run_p2_stacking_baseline.py` configuration, unchanged).
- K = 20 seeds; metric: PR-AUC (primary, the ONLY confirmatory metric); F1 and FPR@0.90
  reported descriptively.
- Hypothesis: paired ΔPR-AUC(Stack[CB+LR] − CatBoost) > 0 on `ph_te2`.
- Test: Nadeau–Bengio corrected paired t-test (`scripts/paired_eval.corrected_paired_t`,
  test_frac=0.30), two-sided, BH-adjusted with m=2. Success: q ≤ 0.05 with positive sign.

**Test 2 — residual-λ point estimate (CONDITIONAL; slot may be forfeited).**

- Activates ONLY if experiment E4 (residual-backbone λ sensitivity, run on the CURRENT
  corpus before the refresh) shows an interior optimum λ* ∈ (0,1) whose residual curve
  exceeds the independent λ-mix control curve at λ*. If E4 shows a monotone curve or
  no residual-vs-mix separation, this slot is DROPPED and m=1 is used for Test 1.
- If activated: λ* is frozen by an amendment to this file (committed before the pull),
  and the test mirrors Test 1 with LR⊕λ*·CB in place of Stack[CB+LR].

## Explicitly NOT confirmatory (exploratory/diagnostic, whatever their outcome)

FCTS forward-chained OOF (E5), the shift-localization discriminator matrix (E2), the
novelty↔CB-error diagnostic (E3), the internal-decay sanity design (E1), and any post-hoc
subgroup reading of `ph_te2`. Negative or null results of these appear in the paper as
diagnostics, with no significance claims.

## Analysis code

The analysis will be run by extending `run_p2_stacking_baseline.py` /
`run_p2_temporal_strict.py` with a `--test-after <date>` window definition; the paired test
exactly as in `make_p2_bench_assets.gen_stacking_verdict`. No other model, metric, seed
count, or split variant will be added to the confirmatory family after the data arrives.

## Amendment 2026-08-22 — Test 2 forfeited, analysis code bound

**Committed before any corpus regeneration.** The raw capture directories were rsynced from the
edge device on 2026-08-22 (a routine pull), but `data/processed/dataset_url.csv` — the corpus
whose maximum dated phishing date defines `ph_te2` — is unchanged (max date 2025-02-18). No
refresh-window row has been looked at.

**E4 outcome (exploratory, run on the CURRENT corpus; `scripts/run_p2_residual_lambda.py`,
`data/processed/p2/p2_residual_lambda.csv`, 5 seeds, canonical phishing-temporal split).** The
residual-backbone curve sigmoid(logit_LR + λ·f_CB|baseline=LR) has an interior optimum at
λ* = 0.7 (PR-AUC 0.9044 ± 0.0024 vs 0.9016 at λ=1 and 0.8905 at λ=0), but it does NOT exceed
the independent λ-mix control at the same λ (0.9052). By the rule registered above, **Test 2 is
dropped and the confirmatory family has m = 1.** Under random-same-rows both curves are
monotone to λ=1, so the interior optimum is a temporal-only feature, consistent with the
crossover. Either λ-blend gains ≈+0.003 PR-AUC over CatBoost alone — the same size as the
Stack[CB+LR] effect, which is the meta-learner learning this mix.

**Test 1 analysis code, bound by this amendment's commit.**
- `scripts/run_p2_temporal_strict.py --test-after 2025-02-18 --seeds 20
  --out data/processed/p2/p2_refresh_cb_k20.csv`
- `scripts/run_p2_stacking_baseline.py --test-after 2025-02-18 --seeds 20
  --bases CatBoost+LogReg --out data/processed/p2/p2_refresh_cblr_k20.csv`
- `split_phishing(ph, cut, test_after)` in `run_p2_temporal_strict.py` is the single window
  definition both runners share: train = every dated phishing row with `collected_at` ≤
  2025-02-18 (the current train + current test), test = strictly later rows, registrable-domain
  guard applied against all of train; benign 70/30 per seed as canonical.
- Verdict: `make_p2_bench_assets.gen_refresh_verdict` → `sections/gen_refresh_verdict.tex`;
  Nadeau–Bengio corrected paired t on PR-AUC (test_frac 0.30), two-sided, `REFRESH_BH_M = 1`.
  Success: q ≤ 0.05 with positive ΔPR-AUC(Stack[CB+LR] − CatBoost).
- Smoke test on the current corpus (`--test-after 2022-11-15 --seeds 2`, scratch output only):
  reproduces the canonical split to within the 51 boundary-date rows (CB 0.8984, Δ +0.0058 vs
  canonical seeds 0–1 +0.0056).

Nothing else in the family changes. E5 (FCTS) remains exploratory and has not been run.
