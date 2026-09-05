#!/usr/bin/env python3
"""
train_content_fusion.py — P1b content/JS fusion detector on crawled landing pages.

Fuses up to four modalities of a captured page and ablates each: CONTENT (char/word TF-IDF or
frozen PhoBERT), JS (obfuscation features or CodeBERT), URL (CompPhish lexical) and optionally
IMAGE (frozen ResNet18 of the screenshot). Input is a manifest CSV from build_content_manifest.py.
The fusion head is Logistic Regression by default or XGBoost (--head), i.e. a hybrid
transformer->boosted-trees stack. Stratified splits, because the crawled subset is small and has no
reliable temporal order — and the paper says so.

RUN:
  python train_content_fusion.py --in data/interim/content_manifest.csv --seeds 5
  python train_content_fusion.py --in data/interim/content_manifest.csv --text-encoder phobert --js-encoder codebert --img
  python train_content_fusion.py --in data/interim/content_manifest_vi.csv --text-encoder xlm-r --head xgboost --balance
Modality details and the reported metrics: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score,
                             average_precision_score, roc_auc_score, roc_curve)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from compphish_features import extract
from train_html_baseline import html_to_text
from extract_js_features import js_vector, scripts_of, all_js, codebert_embed

URL_NUM = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
           "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
           "letter_ratio", "digit_ratio", "spec_ratio", "slash_cnt", "entropy",
           "path_len", "query_len"]  # is_https dropped (collection artifact, per P1a)


def fpr_at_recall(y, score, target=0.90):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.where(tpr >= target)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def load_pages(path, dom_col="dom_file", label_col="label"):
    """Read the manifest; for each existing HTML file extract (text, js_vector, url_features, y)."""
    df = pd.read_csv(path)
    rows = []
    skipped = 0
    for _, r in df.iterrows():
        p = str(r.get(dom_col, "") or "").strip()
        if not p or not os.path.exists(p):
            skipped += 1
            continue
        try:
            html = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            skipped += 1
            continue
        text = html_to_text(html)
        if len(text) < 5 and not scripts_of(html)[0]:
            skipped += 1
            continue
        dom = str(r.get("domain", "") or "").strip().lower()
        uf = extract(dom) if dom else {k: 0 for k in URL_NUM}
        jsd = r.get("js_dir", "")
        jsd = "" if pd.isna(jsd) else str(jsd).strip()  # external JS captured at crawl time
        rows.append({
            "text": text,
            "js": js_vector(html, jsd),
            "html": html,
            "js_dir": jsd,
            "url": [float(uf.get(k, 0)) for k in URL_NUM],
            "shot": str(r.get("shot_file", "") or "").strip(),
            "y": 1 if str(r.get(label_col, "")).lower() in ("1", "phishing", "phish", "smishing", "spam", "malicious") else 0,
            # Preserve provenance for tier-stratified evaluation.  These fields are metadata,
            # never model inputs; keeping them on the row prevents a later audit from having to
            # rejoin by domain (which is not guaranteed unique after recrawls).
            "tier": ("unassigned" if pd.isna(r.get("tier")) else
                     str(r.get("tier")).strip().lower()),
            "provenance": ("unknown" if pd.isna(r.get("provenance")) else
                           str(r.get("provenance")).strip().lower()),
        })
    print(f"usable pages={len(rows)} (skipped={skipped}) | "
          f"phishing={sum(x['y'] for x in rows)} benign={sum(1 - x['y'] for x in rows)}")
    return rows


# The standard Vietnamese encoders plus a multilingual ablation baseline. transformers 5.x
# refuses hub checkpoints that ship only pytorch_model.bin when torch < 2.6 (CVE-2025-32434);
# scripts/convert_phobert_safetensors.py run once per bin-only repo produces the local copy
# that _encoder_src() prefers. xlm-roberta ships safetensors and loads straight from the Hub.
HF_ENCODERS = {
    "phobert": "vinai/phobert-base",
    "phobert-v2": "vinai/phobert-base-v2",
    "visobert": "uitnlp/visobert",
    "xlm-r": "FacebookAI/xlm-roberta-base",
}
SEGMENTED = {"phobert", "phobert-v2"}  # PhoBERT expects word-segmented input (pyvi)


def _encoder_src(repo):
    local = os.path.join(ROOT,
                         "models", repo.split("/")[-1] + "-safetensors")
    return local if os.path.isdir(local) else repo


def _segment(texts):
    """Word-segment Vietnamese text (ngân hàng -> ngân_hàng) as PhoBERT's tokeniser expects."""
    try:
        from pyvi import ViTokenizer
        return [ViTokenizer.tokenize(t) for t in texts]
    except ImportError:
        print("[i] pyvi not installed — feeding PhoBERT unsegmented text (slightly weaker).")
        return texts


# A frozen encoder's embedding of a page does not depend on the train/test split, so cache it:
# evaluate() calls text_matrix once per split, and without the cache a 20-split run re-embeds
# every page 20 times. TF-IDF is NOT cached — it is fitted on each split's training text.
_MODEL_CACHE = {}
_EMB_CACHE = {}


def _load_encoder(encoder):
    if encoder not in _MODEL_CACHE:
        from transformers import AutoTokenizer, AutoModel
        src = _encoder_src(HF_ENCODERS[encoder])
        tok = AutoTokenizer.from_pretrained(src, use_fast=encoder not in SEGMENTED)
        mdl = AutoModel.from_pretrained(src).eval()
        _MODEL_CACHE[encoder] = (tok, mdl)
    return _MODEL_CACHE[encoder]


def text_matrix(train_txt, test_txt, encoder):
    if encoder in HF_ENCODERS:
        try:
            import torch
            tok, mdl = _load_encoder(encoder)

            def emb(texts):
                miss = list(dict.fromkeys(t for t in texts if (encoder, t) not in _EMB_CACHE))
                if miss:
                    seg = _segment(miss) if encoder in SEGMENTED else miss
                    with torch.no_grad():
                        for i in range(0, len(miss), 16):
                            enc = tok([t[:1200] for t in seg[i:i + 16]], truncation=True,
                                      padding=True, max_length=256, return_tensors="pt")
                            vecs = mdl(**enc).last_hidden_state.mean(1).numpy()
                            for t, v in zip(miss[i:i + 16], vecs):
                                _EMB_CACHE[(encoder, t)] = v
                return np.vstack([_EMB_CACHE[(encoder, t)] for t in texts])
            return emb(train_txt), emb(test_txt), "dense"
        except Exception as e:
            print(f"[i] {encoder} unavailable ({type(e).__name__}: {e}); TF-IDF content features.")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30000)
    return vec.fit_transform(train_txt), vec.transform(test_txt), "sparse"


def precompute_js(rows, encoder):
    """Attach a fixed JS embedding to every row ONCE (CodeBERT is too costly to recompute per
    seed). Falls back to the lightweight vector if CodeBERT is unavailable."""
    if encoder == "codebert":
        emb = codebert_embed([all_js(r["html"], r.get("js_dir", "")) for r in rows])
        if emb is not None:
            for r, v in zip(rows, emb):
                r["js_emb"] = v
            print(f"[i] CodeBERT JS embeddings: {emb.shape}")
            return
    for r in rows:
        r["js_emb"] = np.asarray(r["js"], dtype=float)


def js_matrix(train_rows, test_rows):
    return (np.array([r["js_emb"] for r in train_rows]),
            np.array([r["js_emb"] for r in test_rows]))


def image_block(train_rows, test_rows):
    try:
        from train_image_baseline import load_backbone
        from PIL import Image
        embed, _ = load_backbone()
    except Exception as e:
        print(f"[i] image branch skipped ({type(e).__name__}).")
        return None, None

    def emb(rows):
        paths = [r["shot"] if r["shot"] and os.path.exists(r["shot"]) else None for r in rows]
        imgs = [Image.open(p).convert("RGB") if p else Image.new("RGB", (224, 224)) for p in paths]
        return np.array(embed(imgs))
    return emb(train_rows), emb(test_rows)


def stack(text_mat, dense_blocks, kind):
    dense = np.hstack(dense_blocks) if dense_blocks else None
    if text_mat is None:
        return dense
    if kind == "sparse":
        return sparse.hstack([text_mat] + ([sparse.csr_matrix(dense)] if dense is not None else [])).tocsr()
    return np.hstack([text_mat] + ([dense] if dense is not None else []))


def make_head(head, seed, ytr):
    """The fusion head. 'logreg' is the paper's default linear head; 'xgboost' swaps in
    gradient-boosted trees over the SAME concatenated blocks (default hyperparameters, per the
    equal-budget HPO null — tuning either head would reopen that comparison). Class imbalance is
    handled symmetrically: class_weight='balanced' for the linear head, scale_pos_weight for the
    trees (=1 on the paper's 1:1 subset)."""
    if head == "xgboost":
        from xgboost import XGBClassifier
        pos = int(np.sum(ytr == 1))
        return XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1,
                             eval_metric="logloss",
                             scale_pos_weight=(len(ytr) - pos) / max(pos, 1))
    return LogisticRegression(max_iter=4000, class_weight="balanced")


def evaluate(rows, configs, text_enc, js_enc, use_img, seeds, head="logreg",
             return_scores=False, return_test_metadata=False):
    """With return_scores, also hands back the per-split (y_true, score) pairs behind the
    aggregates. The PR-curve figure needs the curves the TABLE was computed from: re-running the
    splits to draw them would be a second experiment, and a figure that disagrees with the table
    beside it by a resampling accident is worse than no figure."""
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))
    precompute_js(rows, js_enc)  # attach r["js_emb"] once (esp. for CodeBERT)
    agg = {c: {m: [] for m in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90")} for c in configs}
    curves: dict[str, list] = {c: [] for c in configs}
    for s in range(seeds):
        tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=s)
        trr = [rows[i] for i in tr]
        ter = [rows[i] for i in te]
        ytr, yte = y[tr], y[te]
        # precompute each modality once per split
        tmat_tr, tmat_te, kind = text_matrix([r["text"] for r in trr], [r["text"] for r in ter], text_enc)
        js_tr, js_te = js_matrix(trr, ter)
        url_tr = np.array([r["url"] for r in trr]); url_te = np.array([r["url"] for r in ter])
        img_tr = img_te = None
        if use_img:
            img_tr, img_te = image_block(trr, ter)

        for cfg in configs:
            txt = (tmat_tr, tmat_te, kind) if "content" in cfg else (None, None, "dense")
            blocks_tr, blocks_te = [], []
            if "url" in cfg:
                blocks_tr.append(url_tr); blocks_te.append(url_te)
            if "js" in cfg:
                blocks_tr.append(js_tr); blocks_te.append(js_te)
            if "image" in cfg and img_tr is not None:
                blocks_tr.append(img_tr); blocks_te.append(img_te)
            Xtr = stack(txt[0], blocks_tr, txt[2])
            Xte = stack(txt[1], blocks_te, txt[2])
            if Xtr is None:
                continue
            clf = make_head(head, s, ytr)
            clf.fit(Xtr, ytr)
            sc = clf.predict_proba(Xte)[:, 1]
            pred = (sc >= 0.5).astype(int)
            agg[cfg]["F1"].append(f1_score(yte, pred, zero_division=0))
            agg[cfg]["PR-AUC"].append(average_precision_score(yte, sc))
            agg[cfg]["ROC-AUC"].append(roc_auc_score(yte, sc) if len(set(yte)) > 1 else float("nan"))
            agg[cfg]["FPR@R0.90"].append(fpr_at_recall(yte, sc))
            if return_scores:
                if return_test_metadata:
                    curves[cfg].append((yte.copy(), sc.copy(), {
                        "tier": np.asarray([rows[i].get("tier", "unassigned") for i in te]),
                        "provenance": np.asarray([rows[i].get("provenance", "unknown") for i in te]),
                    }))
                else:
                    curves[cfg].append((yte.copy(), sc.copy()))
    return (agg, curves) if return_scores else agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/interim/content_manifest.csv")
    ap.add_argument("--text-encoder", default="tfidf",
                    choices=["tfidf"] + sorted(HF_ENCODERS))
    ap.add_argument("--js-encoder", default="lightweight", choices=["lightweight", "codebert"])
    ap.add_argument("--head", default="logreg", choices=["logreg", "xgboost"],
                    help="Fusion head: linear (default) or gradient-boosted trees")
    ap.add_argument("--img", action="store_true", help="Add the screenshot (ResNet18) branch")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--balance", action="store_true",
                    help="Down-sample the majority class to a 1:1 subset before splitting")
    args = ap.parse_args()

    rows = load_pages(args.inp)
    if len({r["y"] for r in rows}) < 2:
        raise SystemExit("Need both classes with usable HTML.")
    if args.balance:
        import random
        rng = random.Random(42)
        ph = [r for r in rows if r["y"] == 1]; be = [r for r in rows if r["y"] == 0]
        k = min(len(ph), len(be))
        rows = rng.sample(ph, k) + rng.sample(be, k)
        print(f"balanced subset: {k} phishing + {k} benign")

    configs = ["url", "content", "js", "content+url", "content+url+js"]
    if args.img:
        configs.append("content+url+js+image")
    agg = evaluate(rows, configs, args.text_encoder, args.js_encoder, args.img, args.seeds,
                   head=args.head)

    print(f"\ncontent={args.text_encoder} js={args.js_encoder} head={args.head} seeds={args.seeds} "
          f"(stratified splits; small crawled content subset)")
    print(f"{'configuration':<24}{'F1':>16}{'PR-AUC':>16}{'ROC-AUC':>16}{'FPR@R0.90':>16}")
    for cfg in configs:
        m = agg[cfg]
        cells = "".join(f"{np.mean(m[k]):>10.3f}±{np.std(m[k]):<5.3f}"
                        for k in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"))
        print(f"{cfg:<24}{cells}")


if __name__ == "__main__":
    main()
