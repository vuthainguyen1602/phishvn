#!/usr/bin/env python3
"""
make_verification_sample.py — Label-quality audit for PhishVN (P1a).

Two modes:
  1) DRAW a random stratified verification sample (by label x tier) for two human annotators to
     independently re-check. Writes one CSV per annotator with an empty `verdict` column plus a
     blinded id and the URL — the source label is withheld so annotation is not primed.
  2) SCORE: given the two filled files, compute observed agreement and Cohen's kappa, and (against
     the dataset's source labels) how the feed's positives break down.

WHY THE VOCABULARY IS FOUR-WAY, NOT phishing/benign/unsure. The dominant feed is ChongLuaDao, a
community ANTI-FRAUD list ("chong lua dao"), not an anti-phishing list; the "Phishing-Blocklist"
name belongs to the GitHub mirror, not to the project's scope. Sampling the positive class shows
what that implies: gambling, betting-stream, investment/crypto and adult domains sit alongside
credential-phishing, and 91% of positives carry no recognisable brand token at all. A three-way
verdict cannot express that. Facing www.taixiu66.club an annotator must call it "phishing" (it
impersonates nobody), "benign" (an anti-fraud list named it) or "unsure" (information thrown
away) — so the audit would have hidden the very composition it was drawn to measure.

Separating them answers two different questions with one annotation pass:
  * is the feed's LISTING defensible?  phishing or scam = yes; legitimate = a real mislabel
  * what is the positive class MADE OF?  the phishing-versus-scam split among defensible rows

RUN:
  # 1) draw a 200-row sample -> two blind sheets
  python make_verification_sample.py draw --in data/processed/dataset_url.csv --n 200 \
      --out data/docs/verify
  # 2) after both annotators fill verify/annotator_{A,B}.csv:
  python make_verification_sample.py score --a data/docs/verify/annotator_A.csv \
      --b data/docs/verify/annotator_B.csv --key data/docs/verify/key.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import random
from collections import Counter, defaultdict


EVIDENCE = {
    "capture": "a capture of this domain inside the corpus (content_manifest.csv)",
    "archive": "a web archive of the page at or near its first-seen date",
    "urlscan": "a public urlscan.io scan — its screenshot and DOM, never its verdict",
    "registry": "the NCSC trusted-organisation registry",
    "name": "the domain string alone admitted no other reading",
    "none": "nothing resolved the row",
}

VERDICTS = {
    "phishing": "impersonates a brand or service to harvest credentials, OTPs or payment details",
    "scam": "abusive but not credential-phishing: gambling, betting streams, investment or crypto "
            "fraud, counterfeit shops, adult content — the rest of what an anti-fraud feed lists",
    "legitimate": "a real operator's own site; listing it is a feed error",
    "unsure": "the domain string alone does not support any of the above",
}


def draw(args):
    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # stratify by (label, tier) so rare cells (e.g. silver benign) are represented
    strata = defaultdict(list)
    for r in rows:
        strata[(r.get("label", ""), r.get("tier", ""))].append(r)
    rng = random.Random(args.seed)
    picked = []
    total = sum(len(v) for v in strata.values())
    for cell, items in strata.items():
        k = max(1, round(args.n * len(items) / total))    # proportional allocation
        rng.shuffle(items)
        picked += items[:k]
    rng.shuffle(picked)
    picked = picked[:args.n]

    os.makedirs(args.out, exist_ok=True)
    # blind id so annotators can't infer order/label; key keeps id -> source label + tier
    blind = [f"V{ i:04d}" for i in range(len(picked))]
    with open(os.path.join(args.out, "key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["vid", "url", "source_label", "tier"])
        for b, r in zip(blind, picked):
            w.writerow([b, r.get("url", ""), r.get("label", ""), r.get("tier", "")])
    for who in ("A", "B"):
        with open(os.path.join(args.out, f"annotator_{who}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["vid", "url", "verdict", "evidence"])
            for b, r in zip(blind, picked):
                w.writerow([b, r.get("url", ""), ""])
    # The codebook is a document, not a generated artefact: it records decisions a human made
    # (what may be looked up, how to treat an ordinary-sounding name) and regenerating it from code
    # would quietly overwrite them. Refuse to hand out sheets that have no rules attached.
    cb = os.path.join(args.out, "CODEBOOK.md")
    if not os.path.exists(cb):
        raise SystemExit(f"No codebook at {cb}. Two annotators judging 'phishing or not' without "
                         f"written rules measure shared intuition, not a protocol — write it first.")
    print("    Read CODEBOOK.md, then fill 'verdict' (" + ", ".join(VERDICTS)
          + ") and 'evidence' (" + ", ".join(EVIDENCE) + ").")


def _cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists."""
    assert len(a) == len(b) and a, "need equal, non-empty annotation lists"
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0, po


def score(args):
    def load(p):
        with open(p, newline="", encoding="utf-8") as f:
            rr = list(csv.DictReader(f))
            return ({r["vid"]: (r.get("verdict") or "").strip().lower() for r in rr},
                    {r["vid"]: (r.get("evidence") or "").strip().lower() for r in rr})
    (A, EA), (B, EB) = load(args.a), load(args.b)
    vids = [v for v in A if v in B and A[v] and B[v]]
    a = [A[v] for v in vids]; b = [B[v] for v in vids]
    if not vids:
        raise SystemExit("No overlapping filled rows — did both annotators complete the sheets?")
    unknown = sorted({v for v in a + b if v not in VERDICTS})
    if unknown:
        raise SystemExit("Unrecognised verdict(s): " + ", ".join(unknown)
                         + "\nAllowed: " + ", ".join(VERDICTS)
                         + "\nFix the sheets rather than the scorer — a typo silently becomes its "
                           "own category and deflates kappa.")
    bad_ev = sorted({e for e in [EA[v] for v in vids] + [EB[v] for v in vids]
                     if e and e not in EVIDENCE})
    if bad_ev:
        raise SystemExit("Unrecognised evidence code(s): " + ", ".join(bad_ev)
                         + "\nAllowed: " + ", ".join(EVIDENCE))
    kappa, po = _cohen_kappa(a, b)
    print(f"Annotated pairs: {len(vids)}")
    resolvable = sum(1 for v in vids if EA[v] not in ("", "none") and EB[v] not in ("", "none"))
    print(f"Resolvable by both (evidence other than 'none'): {resolvable}/{len(vids)} "
          f"({100.0 * resolvable / len(vids):.0f}%) — every rate below is a claim about these rows, "
          f"not about the corpus")
    print(f"Observed agreement A vs B: {po:.3f}   Cohen's kappa: {kappa:.3f}   (4-way)")

    # The dataset's own label is binary, so also score the collapse it actually asserts.
    def abuse(v):
        return "abuse" if v in ("phishing", "scam") else v
    ka, poa = _cohen_kappa([abuse(x) for x in a], [abuse(x) for x in b])
    print(f"Observed agreement A vs B: {poa:.3f}   Cohen's kappa: {ka:.3f}   "
          f"(collapsed to abuse vs legitimate vs unsure — the distinction the label makes)")

    if args.key and os.path.exists(args.key):
        with open(args.key, newline="", encoding="utf-8") as f:
            key = {r["vid"]: r["source_label"].strip().lower() for r in csv.DictReader(f)}
        cons = [(v, A[v]) for v in vids if A[v] == B[v] and A[v] != "unsure"]
        pos = [(v, lab) for v, lab in cons if key.get(v) == "phishing"]
        if pos:
            # A gambling domain on an anti-fraud list is not a feed error, so it cannot count as
            # label noise; only a consensus of "legitimate" is one. Reporting the two separately is
            # the whole reason the vocabulary is four-way.
            defensible = sum(1 for _, lab in pos if lab in ("phishing", "scam"))
            phish = sum(1 for _, lab in pos if lab == "phishing")
            print(f"\nOn {len(pos)} consensus rows the feed marked positive:")
            print(f"  listing defensible (phishing or scam): {defensible / len(pos):.3f}"
                  f"   => label noise {1 - defensible / len(pos):.3f}")
            print(f"  of those, credential phishing:         {phish / defensible:.3f}"
                  if defensible else "  of those, credential phishing:         n/a")
            print("  the second line is what the positive class is made of, and is a property of "
                  "the feed's scope rather than an error rate.")
        neg = [(v, lab) for v, lab in cons if key.get(v) == "benign"]
        if neg:
            ok = sum(1 for _, lab in neg if lab == "legitimate") / len(neg)
            print(f"On {len(neg)} consensus rows the feed marked benign: {ok:.3f} confirmed "
                  f"legitimate => label noise {1 - ok:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw"); d.add_argument("--in", dest="inp", default="data/processed/dataset_url.csv")
    d.add_argument("--n", type=int, default=200); d.add_argument("--out", default="data/docs/verify")
    d.add_argument("--seed", type=int, default=42); d.set_defaults(func=draw)
    s = sub.add_parser("score"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s.add_argument("--key", default=""); s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
