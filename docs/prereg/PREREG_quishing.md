# pre-specification — the quishing decoder benchmark, bound to code

**Registered:** 2026-08-30. Written with the synthetic sweep at 140,000 of 792,000 renders (17.7%)
and committed at 195,000 (24.6%), still running; and the in-the-wild arm at 616 pages examined. This file exists because
`sections/01_introduction.tex` claims the study follows the discipline of stating what would count
as a negative result before running anything, and until today no such statement existed.

## What was already seen at registration — read this before the tests

Registering after looking is not registering. The following quantities had been computed and read
before this file was written, and are therefore **descriptive for the life of this study**, whatever
the completed sweep shows:

- **Overall decode failure rate per decoder**, on the first 141,217 rows per decoder:
  OpenCV 62.8%, PyZbar 62.9%, WeChatQR 42.9%. The direction (WeChatQR ahead) was known.
- **The pilot resolution table** in `sections/03_methodology.tex`: `rotate` 25.2% at 2 px against
  0.3% at 4 px, `perspective` +15.8 pp, `saltpepper` +12.1 pp, `blur` +2.0 pp, `logo` +1.1 pp.
  Sample size not recorded at the time, which is itself the reason it is labelled provisional in
  the paper.
- **The in-the-wild result**: 1 page carrying a QR over 616 pages examined, 0 quishing.

No cut by decoder x transform, by error-correction level, or by corpus label had been read.

## Analysis code binding

The sweep runs `scripts/gen_synthetic_qr.py` as of commit `c916ac9`, invoked as
`--n 2000 --box 2,3,4 --stream`, decoders `opencv,pyzbar,wechat`, seed as coded. The grid is
frozen: 2,000 URLs stratified by URL length x 4 EC levels x 3 module sizes x (1 clean + 8
transforms x 4 strengths). A decode counts only when the SHA-1 of the returned payload matches the
SHA-1 of the URL encoded. No transform, strength, module size, EC level or decoder will be added,
dropped or re-parameterised after this file is committed; a re-run with a changed grid is a new
study and says so.

**Unit of analysis is the URL, not the render.** The 792,000 rows are 2,000 independent draws from
the corpus crossed with a fixed design. Every paired comparison below is computed per URL and
aggregated across URLs, so the effective n is 2,000 and not 792,000. Reporting a test on the render
count would be reporting the design matrix as evidence.

## Confirmatory tests (Benjamini-Hochberg, m = 2)

**T1 — render scale changes what a decoder survives.**
For each URL and each decoder, DFR at `box_size = 2` minus DFR at `box_size = 4`, computed over the
`rotate` transform pooled across its four strengths and four EC levels; paired per URL, two-sided
Wilcoxon signed-rank on the 2,000 differences.

- **Success:** mean difference >= +10 pp for at least two of the three decoders.
- **Partial:** >= +5 pp for at least two decoders.
- **Negative result, reported as such:** < +5 pp for two or more decoders, or a reversed sign for
  any decoder. This outcome retires the paper's resolution-calibration argument, and
  Section 3.3 is rewritten to say the pilot was noise at a small sample.

**T2 — the benchmark is not confounded with the corpus label.**
Overall DFR for phishing-labelled URLs minus benign-labelled URLs, per decoder, unpaired (the label
is a property of the URL), Mann-Whitney U on the per-URL DFRs.

- **Success (the intended outcome):** |difference| < 2 pp for all three decoders. The draw is
  stratified by URL length, so a null here is what the stratification was for.
- **Negative result:** |difference| >= 5 pp for any decoder. Then decodability tracks the label,
  the DFR figures cannot be read as a property of the transform, and every DFR table in the paper
  carries the confound as a stated limitation rather than being re-stratified after the fact.

## Registered diagnostics (descriptive, whatever they show)

- DFR per decoder x transform x strength, and the strength at which each transform crosses 50%.
- DFR by error-correction level, including whether H buys what its redundancy claims.
- DFR by module size across all nine transforms, not only `rotate`.
- The decoder ranking, which is already known and stays descriptive (see above).
- Renders where a decoder returned a string that was not the encoded URL, counted separately from
  renders where it returned nothing. Both are failures; only the first is a decoder that lies.

## The in-the-wild arm: what it may and may not claim

The arm reports a prevalence and nothing else. It may state: the share of Vietnamese phishing
landing pages carrying a decodable QR, and what those payloads address. It may not state campaign
lifespan, delivery tactics, or any trend, because sampling landing pages samples the destination
end of the delivery chain and not the delivery origin.

**The negative result here has already occurred and is the finding:** 1 page in 616, 0 quishing.
The paper reports it as a measurement, not as a failed collection.

**Freeze rule.** The denominator moves every hour the collectors run. The reported figure is the one
produced by `scripts/qr_prevalence.py` at the commit that finalises the paper, with the date
stated in the text; no figure is quoted without its date. Broadening the URLScan query to raise the
count is permitted only before this arm's numbers are written into a submitted draft, and any
broadening is recorded as an amendment below, because a query change moves the population and not
just the sample size.

## Explicitly out of scope

Adding decoders after seeing the ranking; tuning any decoder's parameters; re-stratifying the draw
after reading T2; reporting DFR pooled over module sizes as though scale were a nuisance parameter;
and any claim about Vietnamese quishing prevalence drawn from the synthetic arm, which contains no
Vietnamese pages and is a corpus of rendered URLs.

## Deviation record (dated, append-only)

> **2026-08-30, at 204,494 rows per decoder (25.8% of the sweep): two transforms are saturated,
> and T1's partial data was inspected.**
>
> *What was seen.* `blur` and `motion` fail at 100.0% for all three decoders at **every** strength
> and every module size, with the single exception of `blur` at strength 0.25 and box 4 (93.0%).
> The strength axis therefore carries no information for these two transforms: 8 of the 32
> non-clean strength cells are degenerate, and no graded statement about focus or camera shake can
> be made from this sweep.
>
> *Cause, and it is the paper's own argument turned on the paper.* Both magnitudes are absolute
> pixels while every other axis of the design is in modules. `t_blur` uses a Gaussian radius of
> `0.5 + 7.5s`, which is 2.4 px at the weakest setting -- more than one full module at 2 px/module
> and over half a module at 4. `t_motion` uses a directional kernel of `3 + 22s` px, 9 px at the
> weakest setting, smearing across four and a half modules at 2 px. The transforms that behave
> gradedly are the ones whose magnitude is relative: `saltpepper` is a fraction of area, `rotate`
> and `perspective` are geometric. Section 3.3 argues that rendering at a comfortable module size
> measures a decoder no user meets; calibrating a degradation in pixels rather than modules is the
> same error one level down.
>
> *Disposition.* The grid is frozen by the binding above and is **not** changed. This sweep reports
> `blur` and `motion` as saturated at all strengths, as a limitation of the degradation
> calibration and not as a property of the decoders. A recalibrated sweep with module-relative
> magnitudes would be a different grid and therefore a separate study, declared as such; it is not
> folded into this one.
>
> *Inspection of a confirmatory endpoint.* T1's quantity was read at this checkpoint: `rotate` at
> box 2 runs 12.4/10.5/36.5/39.3% across the four strengths against under 4% at box 3 and box 4.
> The design is fixed and the data accrues mechanically, so looking does not change what the test
> can conclude -- but the reading happened, it is recorded here, and the test is reported on the
> completed sweep and not on this checkpoint.

## Amendments (dated, append-only; valid only while committed before the sweep completes)

*(none)*


## Addendum, 2026-09-03: a fourth decoder, added POST-HOC and reported as such

**Registered:** nothing. This addendum registers no hypothesis and moves no bar. It records a
decision taken *after* the sweep's results existed, so that the record shows when it was taken.

**What was added.** ZXing (`zxing-cpp` 3.1.1) is measured over the same grid: same generator, same
seed (20260830), same 2,000 URLs, same four error-correction levels, three module sizes, eight
transformations and four strengths. Row for row it lines up with the registered sweep by
`sample_id`.

**Why, and why only now.** The manuscript described `OpenCV`, `PyZbar` and `WeChatQR` as widely
deployed decoders without saying which engines it left out. A reader working in this area names
ZXing immediately: it is the server-side engine most likely to sit in the pipeline the paper is
about. Omitting it does not invalidate anything measured, but leaving the omission unnamed
overstates how representative three libraries are.

**What it may not do.** T1 and T2 were registered over three decoders and are computed over those
three. The ZXing arm is **not pooled into them**, does not enter the Benjamini--Hochberg family,
and cannot change a registered verdict. It is reported beside the registered three, under its own
macros, generated by `scripts/analyze_qr_zxing.py` rather than by the registered analysis.

**What it cannot fix.** Two engines a victim's phone actually uses --- Apple's VisionKit and
Google's ML Kit --- remain unmeasured, because neither runs in this pipeline. Adding ZXing narrows
the representativeness gap; it does not close it, and the limitation says so.
