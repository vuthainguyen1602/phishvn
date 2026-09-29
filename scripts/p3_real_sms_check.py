#!/usr/bin/env python3
"""
p3_real_sms_check.py — real-message transfer check for P3's paraphrase detectors (revision, post hoc).

Implements papers/P3_multimodal/REVISION1/REAL_SMS_CHECK_PLAN.md, committed (48c50fa5) before
this script was first run. Change nothing here without recording it in that plan, dated.

D0/D1/D2 are fitted exactly as in the band study on each of its 20 training splits, then score
the public HuggingFace Vietnamese SMS corpus (Tran et al., CC BY 4.0) at threshold 0.5.
Anchor: the same architecture trained on the publisher's train split, tested on its test split.

RUN:  python scripts/p3_real_sms_check.py
OUT:  data/processed/p3/p3_real_sms_check.csv
      papers/P3_multimodal/sections/gen_real_sms.tex
"""
import csv
import os
import random
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "core", "lib"))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

import make_p3_band_assets as band
from make_p3_paraphrase_assets import vec, strip_url, SEEDS
from train_fusion import perturb
from genfile import write_generated

REAL_DIR = os.path.join(ROOT, "data", "raw", "sms_hf_full")
OUT_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_real_sms_check.csv")
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")


def unaccent(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def prep(texts):
    return [" ".join(unaccent(strip_url(t)).split()) for t in texts]


def load_real(name):
    d = pd.read_csv(os.path.join(REAL_DIR, name), encoding="utf-8-sig")
    tcol = "message"
    d = d.dropna(subset=[tcol])
    return d[tcol].astype(str).tolist(), d["label"].astype(int).to_numpy()


def head():
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def metrics(y, pred):
    pos, neg = y == 1, y == 0
    return {"miss": float(np.mean(pred[pos] == 0)), "fpr": float(np.mean(pred[neg] == 1)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def main():
    # ---- external set: full corpus, exact-duplicate texts collapsed, then preprocessed ----
    raw_t, raw_y = load_real("full_dataset.csv")
    n_raw = len(raw_t)
    seen, t_d, y_d = set(), [], []
    for t, y in zip(raw_t, raw_y):
        if t in seen:
            continue
        seen.add(t); t_d.append(t); y_d.append(y)
    n_dedup = len(t_d)
    t_p = prep(t_d)
    keep = [i for i, t in enumerate(t_p) if t]
    real_x = [t_p[i] for i in keep]
    real_y = np.array([y_d[i] for i in keep])
    n_empty = n_dedup - len(keep)
    print(f"[i] real corpus: {n_raw} rows -> {n_dedup} after dedup -> {len(real_x)} non-empty "
          f"({int(real_y.sum())} scam / {int((real_y == 0).sum())} ham)")

    # ---- D0/D1/D2 exactly as in the band study, one fit per training split ----
    df, _ = band.load()
    txt, y = df["text"].to_numpy(), df["y"].to_numpy()
    pa = df["para_a"].to_numpy()
    res = {d: [] for d in ("D0", "D1", "D2")}
    for s in range(SEEDS):
        rng = random.Random(s)
        tr, _te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=s)
        ytr = y[tr]
        v0 = vec(); c0 = head().fit(v0.fit_transform(txt[tr]), ytr)
        tr_ph = txt[tr][ytr == 1]
        aug1 = list(txt[tr]) + [perturb(t, rng) for t in tr_ph for _ in range(2)]
        y1 = list(ytr) + [1] * (2 * len(tr_ph))
        v1 = vec(); c1 = head().fit(v1.fit_transform(aug1), y1)
        aug2 = list(txt[tr]) + [t for t in pa[tr][ytr == 1] if t]
        y2 = list(ytr) + [1] * int(sum(1 for t in pa[tr][ytr == 1] if t))
        v2 = vec(); c2 = head().fit(v2.fit_transform(aug2), y2)
        for name, (c, v) in (("D0", (c0, v0)), ("D1", (c1, v1)), ("D2", (c2, v2))):
            res[name].append(metrics(real_y, c.predict(v.transform(real_x))))

    # ---- anchor: same architecture, trained and tested on the real corpus's own split ----
    tr_t, tr_y = load_real("train.csv")
    te_t, te_y = load_real("test.csv")
    tr_p, te_p = prep(tr_t), prep(te_t)
    tr_set = set(tr_p)
    te_keep = [i for i, t in enumerate(te_p) if t and t not in tr_set]
    va = vec()
    ca = head().fit(va.fit_transform([t for t in tr_p if t]), tr_y[[i for i, t in enumerate(tr_p) if t]])
    anchor = metrics(te_y[te_keep], ca.predict(va.transform([te_p[i] for i in te_keep])))
    n_anchor_test = len(te_keep)

    rows = []
    for d, ms in res.items():
        for k in ("miss", "fpr", "macro_f1"):
            vals = np.array([m[k] for m in ms])
            rows.append({"detector": d, "metric": k, "mean": vals.mean(), "sd": vals.std()})
    for k, v in anchor.items():
        rows.append({"detector": "anchor_real_trained", "metric": k, "mean": v, "sd": 0.0})
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[+] {OUT_CSV}")

    def mm(d, k):
        vals = np.array([m[k] for m in res[d]])
        return vals.mean(), vals.std()

    cells = []
    for d, lab in (("D0", "D0"), ("D1", "D1"), ("D2", "D2")):
        mi, fp, f1 = mm(d, "miss"), mm(d, "fpr"), mm(d, "macro_f1")
        cells.append(f"{lab} misses ${100*mi[0]:.1f}\\%\\pm{100*mi[1]:.1f}$ of real scam and "
                     f"flags ${100*fp[0]:.1f}\\%\\pm{100*fp[1]:.1f}$ of real ham "
                     f"(Macro-F1 ${f1[0]:.3f}$)")
    frag = (
        "\\textbf{Do these detectors transfer to real messages?} A post-hoc check, with its "
        "analysis plan fixed before any result was seen, scores the band study's own detectors "
        "(fitted exactly as above on each of the 20 training splits) on a public corpus of real "
        "Vietnamese SMS~\\cite{vietsmsdata} (CC BY 4.0). "
        f"After collapsing exact duplicates it holds {len(real_x):,} distinct messages "
        f"({int(real_y.sum()):,} labelled spam/scam, {int((real_y == 0).sum()):,} ham). Links are "
        "removed and diacritics stripped so that orthography cannot separate the classes; the "
        "source's redaction tokens are left as released. At the fixed threshold, "
        + ". ".join(cells) +
        ". For reference, the same architecture trained on the real corpus's own training split "
        f"and tested on its {n_anchor_test:,} de-duplicated test messages misses "
        f"${100*anchor['miss']:.1f}\\%$ and flags ${100*anchor['fpr']:.1f}\\%$ "
        f"(Macro-F1 ${anchor['macro_f1']:.3f}$). These numbers bound what the synthetic corpus "
        "licenses: the in-corpus miss and false-positive rates reported above are properties of "
        "one generator's writing, and the figures here show how far they are from real "
        "traffic"
    )
    write_generated(os.path.join(SEC, "gen_real_sms.tex"), frag + ".\n")
    for d in res:
        print(f"[i] {d}: miss={mm(d,'miss')[0]:.3f} fpr={mm(d,'fpr')[0]:.3f} f1={mm(d,'macro_f1')[0]:.3f}")
    print(f"[i] anchor: {anchor} (test n={n_anchor_test})")


if __name__ == "__main__":
    main()
