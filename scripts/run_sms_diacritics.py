#!/usr/bin/env python3
"""
run_sms_diacritics.py — does the text separation survive with the diacritics removed?

WHY. The paper's vocabulary ranking puts accented Vietnamese function words on the positive side
and the same words unaccented on the ham side, and says a diacritic-normalised ablation "would
be needed" to separate orthography from content. This is that ablation. Every text is normalised
to NFD, its combining marks dropped and đ mapped to d, so that "bạn" and "ban" become one term,
and the text models are refitted on the registered split:

  tfidf_word / tfidf_char / tfidf_word_no_tokens   the linear probes of sms_shallow_cues.py
  tfidf_word_no_tokens_nodia                        diacritics gone AND redaction tokens gone,
                                                    the "combined normalisation" the paper names
  text_nodia                                        the frozen PhoBERT arm on normalised text,
                                                    MLP head, the registered seeds

A score that stays near the accented one says the separation is carried by which words are used;
a score that falls says orthography was carrying it. POST-HOC and UNREGISTERED.

RUN:
  python3 scripts/run_sms_diacritics.py
Writes data/processed/sms/diacritics_ablation.json.
"""
from __future__ import annotations
import argparse, json, os, re, sys

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
from sklearn.metrics import f1_score  # noqa: E402

from sms_predecision_common import (SEEDS, SMS_DIR, load, strip_diacritics,  # noqa: E402
                                    summarise, test_groups)
from train_sms_fusion import EMB, cluster_bootstrap_f1, embed, fit_score  # noqa: E402

OUT = os.path.join(SMS_DIR, "diacritics_ablation.json")
EMB_NODIA = os.path.join(SMS_DIR, "phobert_emb_nodia.npy")
TOKRX = re.compile(r"\[[A-Z_]+\]")
GRIDS = {"word": dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
         "char": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)}


def linear(train_texts, ytr, test_texts, yte, grid, strip_tokens=False):
    prep = (lambda t: TOKRX.sub(" ", t)) if strip_tokens else (lambda t: t)
    v = TfidfVectorizer(**GRIDS[grid])
    A = v.fit_transform([prep(t) for t in train_texts])
    B = v.transform([prep(t) for t in test_texts])
    p = LogisticRegression(max_iter=3000, C=4, random_state=0).fit(A, ytr).predict(B)
    return round(float(f1_score(yte, p)), 4), p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows, texts, y, tr, te, Xurl, has_url = load()
    nodia = [strip_diacritics(t) for t in texts]
    changed = sum(1 for t, n in zip(texts, nodia) if t != n)
    print(f"  {len(rows)} messages, {changed} changed by normalisation; train {int(tr.sum())} / "
          f"test {int(te.sum())}", flush=True)

    def sel(seq, m):
        return [x for x, k in zip(seq, m) if k]

    res = {"n_changed_by_normalisation": changed, "linear": {}}
    lin_preds = {}
    for label, corpus in (("accented", texts), ("nodia", nodia)):
        for grid in ("word", "char"):
            f, p = linear(sel(corpus, tr), y[tr], sel(corpus, te), y[te], grid)
            res["linear"][f"tfidf_{grid}_{label}"] = f
            lin_preds[f"tfidf_{grid}_{label}"] = p
        f, p = linear(sel(corpus, tr), y[tr], sel(corpus, te), y[te], "word", strip_tokens=True)
        res["linear"][f"tfidf_word_no_tokens_{label}"] = f
        lin_preds[f"tfidf_word_no_tokens_{label}"] = p
        print("  " + "  ".join(f"{k.replace('tfidf_', '')}={v:.3f}"
                               for k, v in res["linear"].items() if k.endswith(label)), flush=True)

    Xtxt = embed(texts, cache=EMB)
    Xnd = embed(nodia, cache=EMB_NODIA)
    per = {"text": [], "text_nodia": []}
    preds = {"text": [], "text_nodia": []}
    for s in SEEDS[:a.seeds]:
        for name, X in (("text", Xtxt), ("text_nodia", Xnd)):
            m, p = fit_score(X[tr], y[tr], X[te], y[te], s)
            per[name].append(m); preds[name].append(p)
        print(f"    seed {s}: text {per['text'][-1]['f1']:.3f}  nodia {per['text_nodia'][-1]['f1']:.3f}",
              flush=True)
    groups = test_groups(rows, te)
    res["arms"] = {k: summarise(v) for k, v in per.items()}
    res["cluster_bootstrap"] = {
        "text_nodia_minus_text": cluster_bootstrap_f1(y[te], preds["text_nodia"], preds["text"], groups),
        "tfidf_word_nodia_minus_accented": cluster_bootstrap_f1(
            y[te], [lin_preds["tfidf_word_nodia"]], [lin_preds["tfidf_word_accented"]], groups),
        "tfidf_word_no_tokens_nodia_minus_accented": cluster_bootstrap_f1(
            y[te], [lin_preds["tfidf_word_no_tokens_nodia"]], [lin_preds["tfidf_word_no_tokens_accented"]], groups),
    }
    res["seeds"] = a.seeds
    res["normalisation"] = "NFD, combining marks removed, đ/Đ -> d/D"
    res["status"] = "post-hoc, unregistered"
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2, sort_keys=True)
    for k, z in res["cluster_bootstrap"].items():
        print(f"  {k:<42} {z['mean']:+.4f} [{z['ci95'][0]:+.4f}, {z['ci95'][1]:+.4f}]")
    print(f"  [+] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
