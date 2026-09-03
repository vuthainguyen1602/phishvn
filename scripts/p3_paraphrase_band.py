#!/usr/bin/env python3
"""
p3_paraphrase_band.py — build the BAND-CONTROLLED paraphrase attack set.

The predecessor set let attack strength drift: two batches written to the same instructions came
out at mean token Jaccard 0.28 and 0.41, so sample size and attack strength moved together and
neither run measured a fixed quantity. Here strength is the controlled variable — J in [0.20, 0.30]
for BOTH variants of every source — under a protocol committed before recalibration
(papers/P3_multimodal/protocols/PARAPHRASE_BAND_PROTOCOL.md).

RUN:  python scripts/p3_paraphrase_band.py
The band, the revision loop and the guardrails: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from p3_jaccard_check import BAND, jaccard, guardrail_problems, sources

SRC_TSV = os.path.join(ROOT, "data", "raw", "author", "p3_band_rewrites.tsv")
OUT = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_band.csv")
GEN_MODEL = "claude-fable-5"


def load_tsv() -> dict[tuple[str, str], str]:
    if not os.path.exists(SRC_TSV):
        raise SystemExit(f"{SRC_TSV} missing — the recalibration output has not been assembled.")
    out: dict[tuple[str, str], str] = {}
    with open(SRC_TSV, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise SystemExit(f"malformed TSV line: {ln[:70]!r}")
            sid, variant, text = (p.strip() for p in parts)
            if variant not in ("a", "b"):
                raise SystemExit(f"bad variant {variant!r} for {sid}")
            if (sid, variant) in out:
                raise SystemExit(f"duplicate row for {sid}/{variant}")
            out[(sid, variant)] = text
    return out


def main() -> None:
    src = sources()
    rw = load_tsv()
    lo, hi = BAND

    missing = [f"{i}/{v}" for i in src for v in ("a", "b") if (i, v) not in rw]
    orphan = [f"{i}/{v}" for (i, v) in rw if i not in src]
    if missing or orphan:
        raise SystemExit(f"[!] attack set out of sync with the corpus: {len(missing)} missing "
                         f"({missing[:3]}), {len(orphan)} orphan ({orphan[:3]})")

    problems: list[str] = []
    rows = []
    import pandas as pd
    meta = pd.concat([pd.read_csv(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))
                      for t in ("sms", "email")], ignore_index=True)
    meta = {r["id"]: r for _, r in meta[meta["label"] == "phishing"].iterrows()}

    for (sid, variant), text in sorted(rw.items()):
        j = jaccard(src[sid], text)
        bad = guardrail_problems(text)
        if not (lo <= j <= hi):
            problems.append(f"{sid}/{variant}: J={j:.3f} outside [{lo}, {hi}]")
        problems += [f"{sid}/{variant}: {b}" for b in bad]
        m = meta[sid]
        rows.append({"src_id": sid, "variant": variant,
                     "role": "train" if variant == "a" else "test",
                     "channel": m["channel"], "scenario": m["scenario"], "label": "phishing",
                     "is_llm": 1, "attack": "paraphrase_band", "gen_model": GEN_MODEL,
                     "jaccard": round(j, 4), "text": text})

    # a and b must differ, or the training and test attacks are the same string
    same = [sid for sid in src if rw[(sid, "a")] == rw[(sid, "b")]]
    problems += [f"{sid}: variants a and b are identical" for sid in same]
    train_txt = {r["text"] for r in rows if r["role"] == "train"}
    test_txt = {r["text"] for r in rows if r["role"] == "test"}
    if train_txt & test_txt:
        problems.append(f"{len(train_txt & test_txt)} rewrite(s) appear in both roles")

    if problems:
        for p in problems[:20]:
            print(f"[!] {p}")
        raise SystemExit(f"[!] {len(problems)} problem(s) — refusing to write a partially "
                         "calibrated attack set; that defect is what this study removes.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    js = [r["jaccard"] for r in rows]
    print(f"[+] {OUT}: {len(rows)} rewrites over {len(src)} sources, all in band "
          f"[{lo:.2f}, {hi:.2f}] — mean J={sum(js)/len(js):.3f}, "
          f"min {min(js):.3f}, max {max(js):.3f}")


if __name__ == "__main__":
    main()
