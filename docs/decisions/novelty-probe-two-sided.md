# The novelty probe: why the bar is `max(AUC, 1-AUC)`

**Applies to:** the novelty-probe paragraph in `scripts/make_p2_bench_assets.py`.

**Raised in round-2 review.** The sentence used to read AUC 0.42 as "nothing to key on", while the
transfer analysis a few pages earlier reads a below-0.5 AUC as an **inverted** ranking rather than
an absent one. Both readings cannot be right in the same report.

The two-sided one is right, and it is the stronger result here: the probe is below chance on every
seed, so novelty is **anti-predictive** — the model errs on *familiar* rows — which closes the
gated-backoff route by pointing the wrong way rather than by being silent.

The bar a gate has to clear is therefore `max(AUC, 1-AUC)`, and the text states it as such.
