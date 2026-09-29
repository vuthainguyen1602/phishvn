# Real-message transfer check — analysis plan, fixed before any result is seen

Written and committed 2026-09-28, before the script below was run for the first time.
Status: POST HOC (added at revision in response to R3's comments on corpus realism). It does
not replace or modify any pre-specified analysis; the paraphrase band study is untouched.

## Commitment

The result is reported in the revised manuscript and the response letter **whatever it shows**,
in the form fixed here. No variant below is dropped, added or re-run with other settings after
the numbers are seen. If the code has a bug that is found later, the fix and the reason are
recorded in this file with a date before the number changes.

## Question

Do the three detectors of the band study (D0 naive, D1 character-adversarial, D2 paraphrase-
augmented), trained on the synthetic single-generator corpus exactly as in the paper, detect real
Vietnamese scam SMS and leave real legitimate SMS alone?

## Data

- External test set: the public Vietnamese SMS corpus on HuggingFace by Tran et al. (CC BY 4.0),
  `data/raw/sms_hf_full/full_dataset.csv`: 2,991 rows, label 1 = spam/scam (the source card's
  definition, broader than phishing) and 0 = ham. Real messages, collected with contributor consent.
- Deduplication: exact duplicate raw message texts are collapsed to one row (first kept) before
  anything else.
- Preprocessing, identical in spirit to the synthetic corpus: URLs removed with the band study's
  own `strip_url`; Vietnamese diacritics removed (NFD, combining marks dropped, đ→d), because the
  synthetic corpus is unaccented throughout and the real corpus carries diacritics unevenly by
  class (≈70% of scam vs ≈39% of ham messages), which would otherwise hand the detector an
  orthography shortcut. The source's redaction tokens ([TIME], [MONEY], [NUMBER], …) are left as
  released; this is a known domain difference and is reported as such. Rows empty after
  preprocessing are dropped and counted.

## Detectors and procedure

- D0, D1, D2 are built by the band study's own code (`make_p3_band_assets.load` +
  `make_p3_paraphrase_assets` vectoriser/head/perturbation), on each of the same 20 training
  splits (seed s, 70%, stratified), exactly as for the in-corpus results.
- Each of the 60 fitted detectors scores the whole preprocessed real corpus at the fixed
  threshold 0.5. No threshold, hyperparameter or preprocessing choice is tuned on real data.
- Reported per detector, mean ± sd over the 20 splits: miss rate on real scam, false-positive
  rate on real ham, macro-F1. Shown beside the same detectors' in-corpus clean figures.

## Reference anchor (what the architecture reaches with real training data)

The same char-n-gram TF-IDF + class-balanced logistic regression, preprocessed identically,
trained on the publisher's `train.csv` and tested on the publisher's `test.csv`, with test rows
whose preprocessed text also occurs in train removed first. Same three metrics, single fit.

## What is not done

No paraphrase attack on real messages, no retraining on real data for D0–D2, no combination of
real and synthetic training data, no significance test (a descriptive external check).
