# The guard-matched control: why it tests rank correlation, not rank equality

**Applies to:** the `gen_guard_control` guard in `scripts/make_p2_bench_assets.py`.

The claim the paragraph rests on is that the guard contributes a near-**constant** offset, so the
family ordering of `Delta_proto` survives even though its level does not. That is what makes the
published `Delta_proto` readable as "protocol plus a constant" rather than as an artefact.

**Why a guard at all:** if a re-run broke the rank ordering, the sentence would be false while
every number in it stayed true of the run that produced it. Nothing else in the suite would catch
that.

**Why correlation rather than equality:** two families sit 0.0000 apart in the unguarded column
and swap places under the guard. That is a tie breaking one way rather than a reordering, and an
equality test would fail on it while the claim stayed true. The extremes carry the claim, so they
are pinned exactly and the middle is checked by Spearman ρ ≥ 0.9.
