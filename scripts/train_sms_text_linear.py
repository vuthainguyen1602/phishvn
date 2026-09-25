#!/usr/bin/env python3
"""
train_sms_text_linear.py — the cheapest text model on the frontier, fitted and persisted.

WHY IT EXISTS. The edge study lists three text-channel candidates spanning two orders of magnitude
of model size: a linear n-gram classifier, a character-CNN, and a distilled PhoBERT. Only the
expensive two were ever going to be built, and a frontier that contains no cheap candidate cannot
show an expensive one being dominated. This fits the cheap one and writes it to `models/` so the
edge assets can score it and report its footprint from the file rather than from an estimate.

WHAT IT IS. Word 1--2-gram TF-IDF over the raw message, logistic head. No Vietnamese tokeniser, no
embedding table, no accelerator: on this corpus the separating signal is largely surface form --
diacritics present or absent, carrier boilerplate, the publisher's redaction tokens -- and n-grams
read exactly that.

WHICH SPLIT. The smishing study's registered split (hashed on the message text, so exact repeats
cannot cross), read from the `split` column of sms_messages.csv. It is NOT the URL temporal split
the edge paper's URL rows use; the two are different corpora and different tasks, and the table
that carries both says so in a heading rather than letting a reader average them.

RUN
    python3 scripts/train_sms_text_linear.py
    python3 scripts/train_sms_text_linear.py --out models/sms_text_linear.joblib
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline  # noqa: E402

SRC = os.path.join(ROOT, "data", "raw", "sms_hf_full", "full_dataset.csv")
MSG = os.path.join(ROOT, "data", "processed", "sms", "sms_messages.csv")
OUT = os.path.join(ROOT, "models", "sms_text_linear.joblib")
SNAP = os.path.join(ROOT, "data", "processed", "sms", "text_linear.json")
SEED = 0


def load() -> tuple:
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        raw = {r["message_id"]: r["message"] for r in csv.DictReader(f)}
    with open(MSG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["message_id"] in raw]
    texts = [raw[r["message_id"]] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    return texts, y, split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json", default=SNAP)
    a = ap.parse_args()
    if not os.path.isfile(SRC) or not os.path.isfile(MSG):
        print(f"[!] need {SRC} and {MSG} — run scripts/sms_corpus_import.py",
              file=sys.stderr)
        return 1

    texts, y, split = load()
    tr, te = split == "train", split == "test"
    Xtr = [t for t, m in zip(texts, tr) if m]
    Xte = [t for t, m in zip(texts, te) if m]
    print(f"[*] {len(texts):,} messages; train {int(tr.sum()):,} / test {int(te.sum()):,}")

    pipe = make_pipeline(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=3000, C=4, random_state=SEED))
    pipe.fit(Xtr, y[tr])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # No compression: the edge table reports a FOOTPRINT, and a compressed file is not the thing
    # that has to be resident on the board.
    joblib.dump(pipe, a.out, compress=0)

    sc = pipe.predict_proba(Xte)[:, 1]
    pred = (sc > 0.5).astype(int)
    # Batched, on whatever CPU this runs on. It is NOT a Jetson number and the snapshot says so;
    # the edge paper's latency rows come from the hardware campaign and from nowhere else.
    t0 = time.perf_counter()
    for _ in range(20):
        pipe.predict(Xte)
    per_msg_us = (time.perf_counter() - t0) / (20 * len(Xte)) * 1e6

    res = {"accuracy": round(float((pred == y[te]).mean()), 4),
           "precision": round(float(precision_score(y[te], pred, zero_division=0)), 4),
           "recall": round(float(recall_score(y[te], pred)), 4),
           "f1": round(float(f1_score(y[te], pred)), 4),
           "roc_auc": round(float(roc_auc_score(y[te], sc)), 4),
           "pr_auc": round(float(average_precision_score(y[te], sc)), 4),
           "size_mb": round(os.path.getsize(a.out) / 1e6, 2),
           "features": int(len(pipe.steps[0][1].vocabulary_)),
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "split": "smishing registered (text-hash)",
           "host_us_per_message": round(per_msg_us, 1),
           "host_note": "batched on the workstation CPU; not the appliance, not the protocol",
           "model": a.out.replace(ROOT + os.sep, "")}
    json.dump(res, open(a.json, "w", encoding="utf-8"), indent=2, sort_keys=True)

    print(f"  F1(phishing) {res['f1']:.3f}   ROC-AUC {res['roc_auc']:.3f}   "
          f"PR-AUC {res['pr_auc']:.3f}")
    print(f"  {res['features']:,} features, {res['size_mb']:.2f} MB on disk, "
          f"{res['host_us_per_message']:.1f} us/message batched on this host (not the board)")
    print(f"  [+] {a.out}\n  [+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
