#!/usr/bin/env python3
"""Post-hoc, NOT registered: three checks a referee asked for on 2026-10-08.

  url_char     Is "the URL channel is weak" a statement about URLs or about the 21 CompPhish
               columns? A character 2-5-gram TF-IDF over the first URL (public-suffix rule),
               logistic regression, scored on the test rows that carry a URL and on all test rows
               (a message with no URL is predicted negative, as the URL arm effectively does).
  truncation   The fine-tuned arm truncates segmented text at 128 PhoBERT tokens and the frozen
               arm at 256. What share of each class each limit cuts.
  fp_examples  The ham test rows the registered text arm flags in most seeds, for the error
               analysis (ids only; the paper quotes them shortened).

Reads   data/processed/sms/sms_messages.csv, data/raw/sms_hf_full/full_dataset.csv,
        data/processed/sms/fusion_test_predictions.csv (train_sms_fusion.py)
Writes  data/processed/sms/extra_diagnostics.json

RUN:  OPENBLAS_NUM_THREADS=1 python3 scripts/sms_extra_diagnostics.py
"""
from __future__ import annotations
import csv, json, os, sys, warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)

from train_sms_fusion import MSG, PHOBERT, PRED, SRC, first_url  # noqa: E402

OUT = os.path.join(ROOT, "data", "processed", "sms", "extra_diagnostics.json")


def main() -> int:
    raw = {r["message_id"]: r["message"] for r in
           csv.DictReader(open(SRC, newline="", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(MSG, newline="", encoding="utf-8")))
    texts = [raw[r["message_id"]] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    out: dict = {"registered": False, "status": "post-hoc, descriptive"}

    # --- URL as characters rather than 21 columns
    urls = np.array([first_url(t, "psl") for t in texts], dtype=object)
    has = np.array([bool(u) for u in urls])
    v = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    A = v.fit_transform(urls[tr & has])
    m = LogisticRegression(max_iter=3000, C=4, random_state=0).fit(A, y[tr & has])
    sel = te & has
    prob = m.predict_proba(v.transform(urls[sel]))[:, 1]
    p = (prob >= 0.5).astype(int)
    full = np.zeros(int(te.sum()), dtype=int)
    full[has[te]] = p
    out["url_char"] = {
        "train_rows_with_url": int((tr & has).sum()), "test_rows_with_url": int(sel.sum()),
        "has_url_f1": round(float(f1_score(y[sel], p)), 4),
        "has_url_roc_auc": round(float(roc_auc_score(y[sel], prob)), 4),
        "has_url_pr_auc": round(float(average_precision_score(y[sel], prob)), 4),
        "all_test_f1": round(float(f1_score(y[te], full)), 4),
        "all_test_precision": round(float(precision_score(y[te], full)), 4),
        "all_test_recall": round(float(recall_score(y[te], full)), 4),
        "model": "char 2-5-gram TF-IDF on the first public-suffix URL, logistic regression C=4"}

    # --- truncation
    from transformers import AutoTokenizer
    from sms_predecision_common import segment
    tok = AutoTokenizer.from_pretrained(PHOBERT)
    raw_len = np.array([len(tok(t)["input_ids"]) for t in texts])
    seg_len = np.array([len(tok(segment(t))["input_ids"]) for t in texts])
    out["truncation"] = {
        lab: {"raw_over_256_pct": round(100 * float((raw_len[y == k] > 256).mean()), 1),
              "seg_over_128_pct": round(100 * float((seg_len[y == k] > 128).mean()), 1),
              "seg_over_256_pct": round(100 * float((seg_len[y == k] > 256).mean()), 1),
              "seg_median_tokens": int(np.median(seg_len[y == k]))}
        for k, lab in ((0, "ham"), (1, "positive"))}

    # --- the text arm's most persistent false positives
    pr = list(csv.DictReader(open(PRED, encoding="utf-8")))
    date_of = {r["message_id"]: r["date"] for r in rows}
    fps = sorted((r for r in pr if r["label"] == "0" and int(r["text_votes"]) > 0),
                 key=lambda r: -int(r["text_votes"]))
    seen, ex = set(), []
    for r in fps:
        if r["text_sha1"] in seen:
            continue
        seen.add(r["text_sha1"])
        ex.append({"message_id": r["message_id"], "text_votes": int(r["text_votes"]),
                   "has_url": int(r["has_url"])})
    out["fp_examples"] = {"ham_rows_flagged_by_any_seed": len(fps),
                          "ham_rows_flagged_by_majority": sum(1 for r in fps
                                                              if int(r["text_votes"]) >= 6),
                          "distinct_texts_flagged_by_majority": sum(1 for e in ex
                                                                    if e["text_votes"] >= 6),
                          "majority_in_august_batch": sum(
                              1 for r in fps if int(r["text_votes"]) >= 6
                              and date_of[r["message_id"]].endswith(("/8/2026", "/08/2026"))),
                          "top": ex[:8]}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:2500])
    print(f"  [+] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
