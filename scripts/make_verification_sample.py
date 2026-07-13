#!/usr/bin/env python3
"""
make_verification_sample.py — Label-quality audit for PhishVN (P1a).

Two modes:
  1) DRAW a random stratified verification sample (by label x tier) for two human annotators to
     independently re-check. Writes one CSV per annotator with an empty `verdict` column
     (fill with: phishing / benign / unsure) plus a blinded id and the URL — the source label is
     withheld so annotation is not primed.
  2) SCORE: given the two filled files, compute observed agreement and Cohen's kappa, and (against
     the dataset's source labels) the annotators' agreement with the feed — an estimate of label
     noise for the datasheet.

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
            w = csv.writer(f); w.writerow(["vid", "url", "verdict"])  # verdict: phishing/benign/unsure
            for b, r in zip(blind, picked):
                w.writerow([b, r.get("url", ""), ""])
    print(f"[+] wrote {len(picked)} blinded rows -> {args.out}/annotator_A.csv, annotator_B.csv, key.csv")
    print("    Annotators fill 'verdict' (phishing/benign/unsure) independently, then run 'score'.")


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
            return {r["vid"]: (r.get("verdict") or "").strip().lower() for r in csv.DictReader(f)}
    A, B = load(args.a), load(args.b)
    vids = [v for v in A if v in B and A[v] and B[v]]
    a = [A[v] for v in vids]; b = [B[v] for v in vids]
    if not vids:
        raise SystemExit("No overlapping filled rows — did both annotators complete the sheets?")
    kappa, po = _cohen_kappa(a, b)
    print(f"Annotated pairs: {len(vids)}")
    print(f"Observed agreement A vs B: {po:.3f}")
    print(f"Cohen's kappa   A vs B: {kappa:.3f}")
    if args.key and os.path.exists(args.key):
        with open(args.key, newline="", encoding="utf-8") as f:
            key = {r["vid"]: r["source_label"].strip().lower() for r in csv.DictReader(f)}
        # consensus = annotators agree; compare consensus to the feed's source label
        cons = [(v, A[v]) for v in vids if A[v] == B[v] and A[v] in ("phishing", "benign")]
        if cons:
            match = sum(key.get(v) == lab for v, lab in cons) / len(cons)
            print(f"Consensus-vs-feed agreement: {match:.3f} on {len(cons)} consensus rows "
                  f"(estimated source-label accuracy; 1 - this = label-noise estimate)")


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
