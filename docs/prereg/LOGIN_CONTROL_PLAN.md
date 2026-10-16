# P3 revision 2 — page-type-matched benign control (R3.2), plan written before collection

Committed 2026-10-16 before any control page is captured or scored.

## Question

Reviewer 3 (R2 round, comment 2): the phishing pages are mostly credential, payment or
verification forms, the benign pages mostly homepages, so the content channel may separate
*page function* rather than phishing. Two measurements answer it.

1. **Page-type composition of the 414-page subset.** Per page, from the stored DOM: a
   password input (`<input type=password>`), and Vietnamese/English login or payment vocabulary
   in the visible text (đăng nhập, mật khẩu, login, sign in; số thẻ, CVV, OTP, số tài khoản,
   card number). Reported per class as counts and shares. This is a heuristic, not an
   annotation, and is reported as one.
2. **Matched control.** Genuine login/account pages of Vietnamese banks, wallets, telcos,
   utilities, government portals, logistics, e-commerce and game top-up services, i.e. the
   organisation types the phishing arm impersonates. Targets: `data/raw/login_control/targets.csv`
   (59 URLs, fixed now). Each is captured through the same urlscan.io pipeline as the benign arm
   (`urlscan_submit.py`, unlisted scans, run on the collector that holds the key), then passed
   through the *same* gates as the subset (near-empty < 40 characters, parked/error/challenge,
   no Vietnamese text). Whatever survives the gates is the control set; its size is reported
   as is, with the per-gate losses. No target is added or replaced after seeing a score.

## Scoring

For each of the 20 splits of the main study, the models trained on that split's training
portion (content only, URL only, content+URL; every content encoder; per-block standardised
linear head as in the revised main tables) score every control page. Reported:

- the false-positive rate on the control at τ = 0.5, averaged over splits, beside the
  benign-test-fold FPR of the same model on the same splits;
- the mean phishing score of the control against the benign test pages and against the
  phishing test pages.

Reading, fixed now: if the content model's FPR on genuine login pages stays near its
benign-test FPR, page function is not what it learned; if it rises towards the phishing
recall, the content signal is at least partly a page-type signal and the paper says so in
the abstract. Either way the result is reported. No statistical test is planned beyond a
Wilson interval on the pooled control FPR: the control set is small and its pages are scored by
twenty overlapping models, so a per-split test would overstate the evidence.

## Amendment 1 (2026-10-16, before any control page was scored)

The 59 fixed targets were guessed URLs: 23 returned no usable capture, and after the gates only
26 pages remain (counted with `load_controls()`; no model has scored any control page). A
second batch is added by a rule fixed here, so the added pages are chosen by the rule and not
by hand:

- **Organisations:** `data/raw/login_control/orgs_batch2.csv` (registrable domains of Vietnamese
  banks, insurers, brokers, wallets and payment gateways, e-commerce, telcos, utilities,
  government portals, logistics, airlines and game/account services), fixed now.
- **Selection:** for each domain, query urlscan.io's search for existing public scans of that
  domain or its subdomains whose URL contains `login`, `dang-nhap`, `dangnhap`, `signin`,
  `sign-in`, `ebank`, `ibank` or `account`; drop hostnames containing `dev`, `test`, `uat`,
  `staging`, `api`, `itsm`, `webmail`, `mail`, `mkt`, `crm`, `vpn` or `admin` (internal systems,
  not pages a customer meets); take the most recent remaining scan with HTTP status 200; at most
  one page per organisation domain, and none whose hostname is already in batch 1.
- **Capture:** the DOM of that existing scan (`https://urlscan.io/dom/<uuid>/`), i.e. the same
  urlscan renderer as the benign arm, without a new submission.
- Both batches then pass the same gates and are scored together, as one control set; the
  report gives the per-batch gate losses. Nothing else in the plan changes.
- Clarification (same day, still before scoring): in both batches, a control page whose hostname
  (without `www.`) is a domain of the content manifest is dropped, so no control page can share
  a host with a training page.
