#!/usr/bin/env python3
"""
train_fusion.py — Multi-channel FUSION detector (P1b core): fuse Vietnamese CONTENT features
with INFRASTRUCTURE (URL) features, with optional ADVERSARIAL TRAINING for robustness.

Content encoder:
  * --text-encoder phobert : frozen PhoBERT mean-pooled embeddings (needs transformers/torch)
  * --text-encoder tfidf   : char n-gram TF-IDF (default; runs anywhere, no GPU)  [auto-fallback]
Infrastructure: CompPhish lexical features of the URL found inside the message (zeros if none).
Visual (optional, --img-col): a frozen ResNet18 embedding of the page screenshot (zeros if the
image is missing); needs torch+torchvision+pillow, otherwise the image branch is skipped and the
model falls back to 2-modality fusion.
Fusion: concat(content, url[, image]) -> Logistic Regression.

Adversarial training (--adv-train): augment training phishing texts with light perturbations
(zero-width spaces, homoglyphs, leetspeak, URL spacing) so the detector does not over-rely on
exact tokens -> more robust to LLM-reworded / obfuscated phishing.

INSTALL: pip install pandas scikit-learn scipy   (+ transformers torch for phobert;
         + torch torchvision pillow for the --img-col visual branch)
RUN:
  python train_fusion.py --in data/processed/dataset_sms.csv --adv-train
  # 3-modality (content + url + screenshot):
  python train_fusion.py --in data/interim/urlscan_labeled.csv --img-col shot_file
"""
from __future__ import annotations
import argparse, os, random, re, sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score, roc_curve)
sys.path.insert(0, os.path.dirname(__file__))
from compphish_features import extract  # noqa: E402

URL_NUM = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
           "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
           "letter_ratio", "digit_ratio", "spec_ratio", "is_https", "slash_cnt", "entropy",
           "path_len", "query_len"]
URL_RE = re.compile(r"(https?://\S+|\b[a-z0-9-]+\.[a-z]{2,}\S*)", re.I)
HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і"}  # latin->cyrillic look-alikes


def url_feats(text: str) -> list[float]:
    m = URL_RE.search(text or "")
    if not m:
        return [0.0] * len(URL_NUM)
    f = extract(m.group(0))
    return [float(f[k]) for k in URL_NUM]


def perturb(text: str, rng: random.Random) -> str:
    """Light, meaning-preserving obfuscation to harden the model."""
    s = list(text)
    for i in range(len(s)):
        r = rng.random()
        if r < 0.05:                                  # zero-width space
            s[i] = s[i] + "​"
        elif r < 0.08 and s[i].lower() in HOMO:       # homoglyph
            s[i] = HOMO[s[i].lower()]
        elif r < 0.10 and s[i].isalpha():             # leetspeak
            s[i] = {"a": "4", "e": "3", "o": "0", "i": "1"}.get(s[i].lower(), s[i])
    return "".join(s)


def build_text_matrix(train_txt, test_txt, encoder):
    if encoder == "phobert":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
            mdl = AutoModel.from_pretrained("vinai/phobert-base").eval()

            def embed(texts):
                out = []
                with torch.no_grad():
                    for i in range(0, len(texts), 32):
                        enc = tok(list(texts[i:i + 32]), truncation=True, padding=True,
                                  max_length=128, return_tensors="pt")
                        h = mdl(**enc).last_hidden_state.mean(1)
                        out.append(h.numpy())
                return np.vstack(out)
            return embed(train_txt), embed(test_txt), "dense"
        except Exception as e:
            print(f"[!] PhoBERT unavailable ({type(e).__name__}); falling back to TF-IDF.")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    return vec.fit_transform(train_txt), vec.transform(test_txt), "sparse"


def combine(text_mat, dense_blocks, kind):
    """Concatenate the text matrix with one or more dense feature blocks."""
    dense = np.hstack(dense_blocks) if len(dense_blocks) > 1 else dense_blocks[0]
    if kind == "sparse":
        return sparse.hstack([text_mat, sparse.csr_matrix(dense)]).tocsr()
    return np.hstack([text_mat, dense])


def build_image_embeddings(df, img_col):
    """path -> ResNet18 embedding dict (frozen backbone). Returns (dict, dim) or (None, 0) if
    torch/torchvision/pillow unavailable."""
    paths = sorted({str(p).strip() for p in df.get(img_col, []) if str(p).strip()})
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"[i] no usable images in '{img_col}' -> image branch skipped (2-modality fusion).")
        return None, 0
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from train_image_baseline import load_backbone
        from PIL import Image
        embed, _ = load_backbone()
    except Exception as e:
        print(f"[i] image branch skipped ({type(e).__name__}); install torch torchvision pillow.")
        return None, 0
    emb = {}
    for i in range(0, len(paths), 32):
        chunk = paths[i:i + 32]
        imgs = []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        for p, v in zip(chunk, embed(imgs)):
            emb[p] = v
    dim = len(next(iter(emb.values())))
    print(f"[i] image branch: embedded {len(emb)} screenshots (dim={dim}).")
    return emb, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--text-encoder", default="tfidf", choices=["tfidf", "phobert"])
    ap.add_argument("--img-col", default="", help="Column with screenshot paths (adds a visual branch)")
    ap.add_argument("--adv-train", action="store_true", help="Augment phishing train with perturbations")
    ap.add_argument("--adv-copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    df = pd.read_csv(args.inp)
    df["text"] = df["text"].fillna("").astype(str)
    df["y"] = (df["label"].astype(str).str.lower().isin(["phishing", "1", "spam", "smishing"])).astype(int)
    df = df[df["text"].str.len() > 0]
    if df["y"].nunique() < 2:
        raise SystemExit("Need both classes.")

    if "split" in df and set(df.get("split", [])) & {"train", "test"}:
        tr = df[df.split == "train"].copy(); te = df[df.split.isin(["val", "test"])].copy()
    else:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=args.seed)

    keep = ["text", "y"] + ([args.img_col] if args.img_col and args.img_col in df.columns else [])
    if args.adv_train:
        aug = []
        for _, r in tr[tr.y == 1].iterrows():
            for _ in range(args.adv_copies):
                row = {"text": perturb(r["text"], rng), "y": 1}
                if args.img_col in keep:
                    row[args.img_col] = ""      # augmented rows have no screenshot -> zero embedding
                aug.append(row)
        if aug:
            tr = pd.concat([tr[keep], pd.DataFrame(aug)], ignore_index=True)
        print(f"adv-train: +{len(aug)} perturbed phishing samples")

    # optional visual branch
    img_emb, img_dim = (build_image_embeddings(df, args.img_col) if args.img_col else (None, 0))

    def img_block(frame):
        if not img_emb:
            return None
        zero = np.zeros(img_dim, dtype=np.float32)
        return np.array([img_emb.get(str(p).strip(), zero) for p in frame.get(args.img_col, [""] * len(frame))])

    tmat_tr, tmat_te, kind = build_text_matrix(list(tr.text), list(te.text), args.text_encoder)
    Utr = np.array([url_feats(t) for t in tr.text])
    Ute = np.array([url_feats(t) for t in te.text])
    blocks_tr, blocks_te = [Utr], [Ute]
    Itr, Ite = img_block(tr), img_block(te)
    if Itr is not None:
        blocks_tr.append(Itr); blocks_te.append(Ite)
    Xtr = combine(tmat_tr, blocks_tr, kind)
    Xte = combine(tmat_te, blocks_te, kind)
    modal = "content+url" + ("+image" if Itr is not None else "")
    print(f"encoder={args.text_encoder if kind=='dense' else 'tfidf'} | train={len(tr)} test={len(te)} "
          f"| modalities={modal} (url={len(URL_NUM)}{f', img={img_dim}' if Itr is not None else ''})")

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xtr, tr.y)
    score = clf.predict_proba(Xte)[:, 1]
    pred = (score >= 0.5).astype(int)
    yte = te.y.to_numpy()
    auc = roc_auc_score(yte, score) if len(set(yte)) > 1 else float("nan")
    fpr90 = float("nan")
    if len(set(yte)) > 1:
        fpr, tpr, _ = roc_curve(yte, score)
        hit = np.where(tpr >= 0.90)[0]
        fpr90 = float(fpr[hit[0]]) if len(hit) else float("nan")
    print(f"  [fusion{'+adv' if args.adv_train else ''}] "
          f"F1={f1_score(yte,pred):.3f} P={precision_score(yte,pred,zero_division=0):.3f} "
          f"R={recall_score(yte,pred,zero_division=0):.3f} ROC-AUC={auc:.3f} "
          f"PR-AUC={average_precision_score(yte,score):.3f} FPR@R0.90={fpr90:.3f}")
    if "is_llm" in te:
        for k, g in te.assign(_s=score).groupby(te["is_llm"].fillna(0).astype(int)):
            gp = (g._s >= 0.5).astype(int)
            tag = "LLM" if k == 1 else "clean"
            print(f"    [{tag}] n={len(g)} R(catch)={recall_score(g.y, gp, zero_division=0):.3f}")


if __name__ == "__main__":
    main()
