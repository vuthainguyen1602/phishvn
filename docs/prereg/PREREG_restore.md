# pre-specification — the restoration network, bound to code

**Registered:** 2026-08-30, after a pilot was run and read, and **before the comparison this file
exists for has been computed anywhere**. The restorer's own effect on OpenCV is already known and
is descriptive for the life of this study; whether it beats the decoder a defender could simply
install instead is not known, because no machine in this project has had PyTorch and WeChatQR at
the same time, and that is the question registered below.

This is a separate study from `PREREG_quishing.md` and does not amend it. It shares the corpus and
the generator; it does not share the grid, the calibration, or the unit under test.

## What was already seen at registration — read this before the tests

Registering after looking is not registering. A pilot ran on 2026-08-30: 200 URLs, module size 2,
four error-correction levels, the six photometric transforms under `--calibration modules`, 19,200
pairs, 157 training URLs against 43 held out, eight epochs, **OpenCV only**. Everything below was
read and is therefore **descriptive whatever the confirmatory run shows**:

- The polarity stage is worth **+16.1 pp** of DFR on its own (OpenCV 70.4% -> 54.3%), and takes
  `invert` from 100% to 3.2%.
- The network is worth **+20.7 pp** on top of that control (54.3% -> 33.6%), 312 renders rescued
  against 2 destroyed, paired per URL p = 2e-13.
- Per transform, control -> restored: `motion` +47.0, `blur` +43.0, `saltpepper` +28.7,
  `contrast` +4.8, `logo` +0.4, `invert` +0.0 pp.
- Training on a set containing `invert` without the polarity stage collapses the network to flat
  grey at L1 0.367. This was measured, not predicted.
- Two plumbing decisions were made by looking at their consequences: cropping inside the symbol
  rather than the padded canvas, and the polarity stage itself. Both are recorded as what they are
  -- choices informed by pilot output -- and neither is re-opened after this file is committed.

**Not seen, by anyone, anywhere:** any number for PyZbar or WeChatQR on any restored or control
image; any cross-decoder comparison; any run at a URL count other than 200.

## Analysis code binding

Code as of commit `44ffc0a`. The confirmatory run is these four commands and no others:

```
python3 scripts/gen_synthetic_qr.py --n 500 --box 2 --ec L,M,Q,H --seed 20260831 \
    --transforms blur,motion,saltpepper,contrast,invert,logo --calibration modules \
    --out data/raw/qr_restore_v2
python3 scripts/train_qr_restore.py --dir data/raw/qr_restore_v2 --epochs 8 \
    --dump data/raw/qr_restore_arms --eval-limit 4000
# on the second Jetson, which has OpenCV, PyZbar and WeChatQR and no torch:
python3 scripts/benchmark_qr.py --dir data/raw/qr_restore_arms \
    --out data/processed/qr/qr_restore_arms_dfr.csv
python3 scripts/train_qr_restore.py --from-dfr data/processed/qr/qr_restore_arms_dfr.csv
```

Frozen with them: 500 URLs stratified by length at seed 20260831, module size 2, all four EC
levels, the six photometric transforms, `--calibration modules`; the URL-hashed 25% holdout; eight
epochs, crop 64, batch 32, Adam at 1e-3, L1; `--polarity auto`, `--binarize` off; 4,000 held-out
renders thinned deterministically across the held-out set. No hyper-parameter, transform, arm or
decoder is added, dropped or retuned after this file is committed. A run with any of them changed
is a different study and says so.

**Both confirmatory tests are decoded on one machine, in one invocation.** The arms are produced on
the workstation and decoded on the second Jetson, so the dump carries all three arms -- the raw one
included -- and a comparison across two machines' decoder builds is never made.

**Unit of analysis is the URL.** The renders are a fixed design crossed with the held-out draw;
every quantity is computed per URL first and aggregated across URLs.

## Confirmatory tests (Benjamini-Hochberg, m = 2)

Let D be positive when the restorer is ahead, i.e. `DFR(comparator) - DFR(restorer arm)`, computed
per URL over the held-out renders pooled across transforms, strengths and EC levels; two-sided
Wilcoxon signed-rank on the per-URL differences.

**T1 — does the restorer beat the decoder a defender could simply install?**
`restored` + OpenCV against `raw` + WeChatQR.

- **Success:** mean D >= +5 pp.
- **Partial:** 0 <= mean D < +5 pp. The restorer matches a library swap and costs a GPU; the paper
  says so.
- **Negative result, reported as such:** mean D < 0. Then the honest recommendation is to install
  WeChatQR rather than train anything, the restoration section becomes a limitation of the
  approach rather than a contribution, and Section 3.4's framing is rewritten accordingly.

**T2 — is anything left to win once the strongest decoder is already in place?**
`restored` + WeChatQR against `raw` + WeChatQR.

- **Success:** mean D >= +5 pp. Restoration helps even the best decoder, which is the strongest
  form of the claim.
- **Partial:** 0 <= mean D < +5 pp.
- **Negative:** mean D < 0 — the restorer damages what WeChatQR could already read, and that is
  reported as a cost of putting it in front of a detector-based decoder.

T1 and T2 can disagree, and the pair is more informative than either: a restorer that rescues
OpenCV and damages WeChatQR is a statement about which decoder it was implicitly trained to please.

## Registered diagnostics (descriptive, whatever they show)

- DFR per arm x decoder x transform, all three decoders, `motion` reported on its own (it is the
  case the plan named as least likely to be recoverable).
- What the polarity line alone is worth **per decoder**. Known for OpenCV; unseen for the other two,
  and WeChatQR is expected to gain nothing because it already reads inverted codes at 7.8% DFR.
- Renders rescued against renders destroyed, per decoder. A net gain hides two populations.
- The plumbing check: with `--polarity auto` the raw and control arms differ by the polarity line
  alone, and any other difference is a bug rather than a finding.

## Explicitly out of scope

Retraining, re-tuning or re-seeding after reading T1 or T2; adding a decoder after seeing the
ranking; changing the polarity rule; reporting the pilot's OpenCV figures as confirmatory;
comparing arms decoded on different machines; and any claim that the network detects phishing --
it restores an image, and the phishing signal is in the URL the code carries.

## Deviation record (dated, append-only)

*(none)*

## Amendments (dated, append-only; valid only while committed before the confirmatory run)

*(none)*
