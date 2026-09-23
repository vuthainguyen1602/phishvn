# pre-specification — does a global receptive field buy what a local one cannot?

**Registered:** 2026-08-31, **before either new architecture has been built, trained, or decoded**.
No number for arm B or arm C exists anywhere at the time this file is committed — not an L1 curve,
not a DFR, not a single decoded render.

This is a separate study from `PREREG_restore.md` and does not amend it. It shares the corpus, the
generator, the split rule and the evaluation harness; it changes exactly one thing, the model
class, and it exists because the first study's own numbers named a mechanism that a different model
class would answer.

## What is already known, in full, before the tests below

Everything in `PREREG_restore.md` and everything its confirmatory run produced has been read. It is
therefore **descriptive for the life of this study** and none of it may be re-reported as a finding
here:

- The registered arm (call it **A**, four 3x3 convolutions, 9 px receptive field) reached
  T1 = **+3.5 pp** against `raw`+WeChatQR, a **PARTIAL** verdict below the +5 pp bar, and
  T2 = **+13.6 pp** against `raw`+WeChatQR read by WeChatQR, a **SUCCESS**.
- DFR raw / control / restored: OpenCV 69.6 / 53.5 / 26.5; PyZbar 66.2 / 49.7 / 25.2;
  WeChatQR 30.0 / 29.2 / 16.4. The polarity line alone is worth 16.1 pp to OpenCV and 0.8 pp to
  WeChatQR.
- Per transform, control -> restored under OpenCV: `motion` +67.1, `blur` +57.1, `saltpepper`
  +29.2, `contrast` +7.7, **`logo` +0.9**, `invert` +0.0 pp.
- Transitions: OpenCV 1,089 rescued against 9 destroyed of 4,000 renders; PyZbar 987 / 7;
  WeChatQR 535 / 22.

**The observation this study is built on, and the reason it is a study rather than a tweak.** Arm A
moves every degradation whose damage is local and does not move `logo` at all. That was predicted
before arm A ran, from its receptive field: 9 px at 2 px per module is 4.5 modules, and a central
logo consumes a region far wider than that. A model cannot reconstruct what it cannot see. The
prediction held, which makes it a mechanism worth testing rather than a post-hoc story — and the
test of a mechanism is whether removing the constraint removes the failure.

## The two new arms

Both are trained on the same pairs, with the same plumbing, the same split and the same budget as
arm A. Neither is tuned. Both are frozen at the definitions below, and a run that changes any of
them is a different study.

- **Arm B --- hybrid.** Convolutional stem, two transformer blocks at 1/4 resolution with windowless
  self-attention over the whole downsampled map, convolutional positional encoding (a depthwise
  3x3 added to the tokens, so the model stays resolution-agnostic and no learned position table
  fixes an input size), then upsampling with skip connections from the stem. Global residual.
- **Arm C --- transposed-attention (Restormer-style).** Four blocks of multi-Dconv head transposed
  attention plus a gated feed-forward network, at full resolution throughout. Attention is over
  channels rather than over pixels, so the spatial context is global while the cost stays linear in
  pixels. Global residual.

Both therefore have a receptive field covering the whole symbol, which is the single property arm A
lacks and the whole content of the hypothesis.

## Frozen with this file

The set `data/raw/qr_restore_v2` as built for the first study: 500 URLs stratified by length at seed
20260831, module size 2, all four error-correction levels, the six photometric transforms,
`--calibration modules`, 50,000 renders. The URL-hashed 25% holdout (115 URLs). Eight epochs, crop
64, batch 32, Adam at 1e-3, L1 loss, `--polarity auto`, `--binarize` off, 4,000 held-out renders
thinned deterministically. `channels = 32` for every arm.

**Equal epochs, not equal wall-clock or equal parameters.** A larger model at a fixed epoch budget
may simply be under-trained, and this design cannot separate "the model class does not help" from
"eight epochs was not enough for this model class". A negative result below is therefore a statement
about **these arms at this budget**, and it will be reported in those words. The alternative --
training each arm to its own convergence -- introduces a per-arm decision made after looking, which
is worse.

**Every arm is decoded on one machine, in one session** (`.204`, which has OpenCV, PyZbar and
WeChatQR and no torch), from PNG dumps produced on the workstation. Arm A is re-decoded there in
that same session rather than reusing the first study's CSV, so no comparison in this file crosses
two machines' decoder builds or two dates' library versions. Each dump carries its own `raw` and
`control` arms, so the plumbing check is available per architecture.

**Unit of analysis is the URL.** Every quantity is computed per URL first and aggregated across the
115 held-out URLs.

## Analysis code binding

Code as of commit `b70585a`, which is the commit that registered this file. The confirmatory run is
these commands and no others:

```
# on the workstation (torch, MPS; no PyZbar, no WeChatQR)
python3 scripts/train_qr_restore.py --dir data/raw/qr_restore_v2 --arch <arch> --epochs 8 --no-eval
python3 scripts/train_qr_restore.py --dir data/raw/qr_restore_v2 --arch <arch> \
    --eval-only --eval-limit 4000 --dump-only --dump data/raw/qr_restore_arms_<arch>
# on .204, which has all three decoders and no torch, in one session
python3 scripts/benchmark_qr.py --dir data/raw/qr_restore_arms_<arch> \
    --out data/processed/qr/qr_restore_arms_<arch>_dfr.csv
# back on the workstation
python3 scripts/analyze_qr_restore_v2.py
```

`analyze_qr_restore_v2.py` computes what this file registers and nothing else. No hyper-parameter,
arm, decoder or threshold is added, dropped or retuned after this commit; a run with any of them
changed is a different study and says so.

## Confirmatory tests

Benjamini--Hochberg over **m = 4**: two tests, each run for arm B and for arm C. Both arms are
reported whatever they show; reporting only the better one is out of scope. `D` is positive when
the new arm is ahead. Two-sided Wilcoxon signed-rank on the per-URL differences.

**T1 --- the mechanism. On `logo` renders only, OpenCV, restored arm against arm A's restored arm.**
This is the test the study exists for: if a global receptive field is what `logo` needed, the gain
appears here and it is large.

- **Success:** mean D >= **+10 pp**.
- **Partial:** 0 <= mean D < +10 pp. The mechanism is real but small, and error correction remains
  the answer to a logo overlay.
- **Negative, reported as such:** mean D < 0. Then the receptive-field explanation of arm A's
  `logo` failure is wrong, or is not the binding constraint, and the paper says so and keeps arm A.

**T2 --- the deployment question, unchanged from the first study so the two are comparable.**
Restored + OpenCV against `raw` + WeChatQR, pooled over all six transforms.

- **Success:** mean D >= **+5 pp** --- the bar arm A missed at +3.5 pp. A restorer that clears it
  changes the recommendation from "install the better library" to "install it and put this in
  front".
- **Partial:** 0 <= mean D < +5 pp.
- **Negative:** mean D < 0.

## Registered guard (a threshold, not a test)

A restorer earns nothing by rescuing renders it also breaks. For each arm and decoder, of the
held-out renders that the `control` arm decoded correctly, the share the restored arm breaks must
not exceed **2%**. Arm A's figures are known (OpenCV 9 destroyed against 1,089 rescued) and are the
comparator. An arm that fails this guard is reported as failing it, whatever T1 and T2 say.

## Registered diagnostics (descriptive, whatever they show)

- DFR per arm x decoder x transform, all three decoders.
- Rescued against destroyed, per arm and decoder.
- Parameter count and training wall-clock per arm, so that a null result carries its budget.
- Arm A re-decoded in the same invocation, as the plumbing check: its numbers must reproduce the
  first study's, and any drift means the two runs are not comparable and the comparison is void.

## Explicitly out of scope

Retraining, re-tuning, re-seeding or extending the epoch budget after reading T1 or T2; adding a
fourth architecture after seeing the ranking; choosing between B and C after the fact and reporting
only one; reporting L1 or PSNR as evidence of anything; comparing arms decoded on different
machines; and any claim that a restorer detects phishing --- it restores an image, and the phishing
signal is in the URL the code carries.

## Outcome (recorded 2026-08-31, after the confirmatory run, before anything was rewritten)

Decoded on `.204` in one session, all three arms, `benchmark_qr.py` at the md5 this file was
committed with. **The plumbing check passes**: arm A re-decodes to 69.6 / 53.5 / 26.5 (OpenCV),
66.2 / 49.7 / 25.2 (PyZbar), 30.0 / 29.2 / 16.4 (WeChatQR) — the first study's numbers to within
0.1 pp — so the comparison is not void.

| | T1 (`logo` vs arm A) | T2 (vs raw+WeChatQR) | guard: broke |
|---|---|---|---|
| B, hybrid | +8.7 pp → **PARTIAL** | +10.7 pp → **SUCCESS** | 6.4% → **FAILS** |
| C, Restormer | +8.9 pp → **PARTIAL** | −1.2 pp → **NEGATIVE** | 7.2% → **FAILS** |

- **The mechanism held and the bar did not.** Both arms move `logo` (+9.6 and +9.7 pp against arm
  A's +0.8), so a nine-pixel receptive field was the binding constraint, as predicted. Neither
  reaches the +10 pp T1 bar, and the median difference is 0.0 pp: the gain sits in a minority of
  URLs.
- **The guard is the decisive result.** Both arms break 6–8% of the renders the control arm already
  decoded, against the 2% ceiling, on all three decoders. `contrast` reverses by ~17 pp in both.
  As registered, this is reported whatever T1 and T2 say.
- **Budget, as registered:** hybrid 120,353 parameters / 12 minutes; Restormer 52,209 / 100 minutes.
  The arm that cost eight times more training returned the negative T2.
- Neither arm is retuned in response to any of the above; a fix for the destruction is a different
  study, and this file's out-of-scope list says so.

## Deviation record (dated, append-only)

**2026-08-31 — the dump batch size, not the training one.** Arm B's dump ran out of GPU memory at
the frozen batch of 32. Training is on 64 px crops, which are 16x16 tokens after the /4 stem;
inference runs at the 232 px canvas, which is 58x58 = 3,364 tokens, and a single attention matrix at
batch 32 is about 5.8 GB. The dumps for arms B and C were therefore produced at **batch 4**. This
changes no weight, no hyper-parameter and no output: there is no batch normalisation in any arm and
LayerNorm is per token, so a render's restored image is identical at either batch. Recorded here
because a number frozen in this file was changed, whatever its effect.

## Amendments (dated, append-only; valid only while committed before the confirmatory run)

*(none)*
