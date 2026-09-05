#!/usr/bin/env python3
"""
train_html_baseline.py — HTML/DOM modality baseline for PhishVN (P1a).

Reads a CSV whose rows point to saved DOM files (the `dom_file` column produced by
fetch_urlscan.py) plus a `label`, extracts the page's VISIBLE TEXT, and trains a
TF-IDF + Logistic Regression / Random Forest phishing classifier.

Input CSV needs: a label column and a column with the path to each HTML file.
  Typical: join data/processed/dataset_url.csv (label, split) with data/interim/urlscan.csv
  (dom_file) on `domain`, or pass any CSV that already has both.
Rows with an empty/missing HTML file are skipped (and counted).

INSTALL: pip install pandas scikit-learn   (beautifulsoup4 optional; regex fallback otherwise)
RUN:
  python train_html_baseline.py --in data/interim/urlscan_labeled.csv \
         --dom-col dom_file --label-col label --model both
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score, roc_curve)


def fpr_at_recall(y, score, target_recall=0.90):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")

TAG_RE = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
STRIP_RE = re.compile(r"(?s)<[^>]+>")
WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Visible text from HTML. Uses BeautifulSoup if available, else a regex fallback."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.extract()
        return WS_RE.sub(" ", soup.get_text(" ")).strip()
    except Exception:
        h = TAG_RE.sub(" ", html)
        return WS_RE.sub(" ", STRIP_RE.sub(" ", h)).strip()


def to_y(v) -> int:
    s = str(v).strip().lower()
    return 1 if s in ("1", "phishing", "phish", "spam", "smishing", "malicious") else 0


def report(name, y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    print(f"  [{name}] F1={f1_score(y,pred):.3f} P={precision_score(y,pred,zero_division=0):.3f} "
          f"R={recall_score(y,pred,zero_division=0):.3f} ROC-AUC={auc:.3f} "
          f"PR-AUC={average_precision_score(y,score):.3f} FPR@R0.90={fpr_at_recall(y,score):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dom-col", default="dom_file", help="Column holding the path to each HTML file")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--model", default="both", choices=["logreg", "rf", "both"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.dom_col not in df or args.label_col not in df:
        raise SystemExit(f"Need columns '{args.dom_col}' and '{args.label_col}'. Have: {list(df.columns)}")

    texts, ys, splits, skipped = [], [], [], 0
    for _, r in df.iterrows():
        path = str(r.get(args.dom_col, "") or "").strip()
        if not path or not os.path.exists(path):
            skipped += 1
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                txt = html_to_text(f.read())
        except Exception:
            skipped += 1
            continue
        if len(txt) < 5:
            skipped += 1
            continue
        texts.append(txt)
        ys.append(to_y(r.get(args.label_col)))
        splits.append(str(r.get("split", "")))
    if len(texts) < 4 or len(set(ys)) < 2:
        raise SystemExit(f"Not enough HTML text with both classes (usable={len(texts)}, skipped={skipped}).")
    print(f"usable HTML docs={len(texts)} (skipped={skipped}) | phishing={sum(ys)} benign={len(ys)-sum(ys)}")

    ys = np.array(ys)
    # temporal split if present, else stratified random
    idx = np.arange(len(texts))
    have_split = any(s in ("train", "val", "test") for s in splits)
    if have_split:
        tr = [i for i in idx if splits[i] == "train"]
        te = [i for i in idx if splits[i] in ("val", "test")]
        if not tr or not te or len(set(ys[tr])) < 2:
            have_split = False
    if not have_split:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(idx, test_size=0.3, stratify=ys, random_state=args.seed)
    Xtxt = np.array(texts, dtype=object)

    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=50000)
    Xtr = vec.fit_transform(Xtxt[tr]); Xte = vec.transform(Xtxt[te])
    ytr, yte = ys[tr], ys[te]
    print(f"train={len(tr)} test={len(te)} | tfidf-dim={Xtr.shape[1]}")

    models = {}
    if args.model in ("logreg", "both"):
        models["LogReg"] = LogisticRegression(max_iter=3000, class_weight="balanced")
    if args.model in ("rf", "both"):
        models["RandomForest"] = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                        n_jobs=-1, random_state=args.seed)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        report(name, yte, score)


if __name__ == "__main__":
    main()
