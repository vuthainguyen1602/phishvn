#!/usr/bin/env python3
"""
train_image_baseline.py — Visual (screenshot) modality baseline for PhishVN (P1a).

Transfer learning: a frozen ImageNet-pretrained ResNet18 backbone is used as a fixed feature
extractor over each page screenshot; a Logistic Regression head is trained on the embeddings
(fast, works on CPU, robust with a few hundred images/class). Optionally exports the per-image
embeddings so the fusion model can reuse them without recomputing.

Input CSV needs: a label column and a column with the path to each screenshot PNG
  (the `shot_file` column produced by `fetch_urlscan.py --save-shots`).
Rows with a missing image are skipped (and counted).

INSTALL: pip install pandas scikit-learn torch torchvision pillow
  (if torch/torchvision are unavailable the script prints how to install and exits cleanly.)
RUN:
  python train_image_baseline.py --in data/interim/urlscan_labeled.csv \
         --img-col shot_file --label-col label --emb-out data/interim/img_emb.npz
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, roc_auc_score, roc_curve)


def fpr_at_recall(y, score, target_recall=0.90):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def to_y(v) -> int:
    s = str(v).strip().lower()
    return 1 if s in ("1", "phishing", "phish", "spam", "smishing", "malicious") else 0


def load_backbone():
    """Return (embed_fn, preprocess) using a frozen ResNet18, or raise with a clear message."""
    import torch
    import torchvision as tv
    from torchvision import transforms
    weights = tv.models.ResNet18_Weights.IMAGENET1K_V1
    net = tv.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()          # 512-d embedding
    net.eval()
    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def embed(pil_images):
        import torch as _t
        xs = _t.stack([prep(im) for im in pil_images])
        with _t.no_grad():
            return net(xs).cpu().numpy()
    return embed, prep


def report(name, y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    print(f"  [{name}] F1={f1_score(y,pred):.3f} P={precision_score(y,pred,zero_division=0):.3f} "
          f"R={recall_score(y,pred,zero_division=0):.3f} ROC-AUC={auc:.3f} "
          f"PR-AUC={average_precision_score(y,score):.3f} FPR@R0.90={fpr_at_recall(y,score):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--img-col", default="shot_file", help="Column with the path to each screenshot")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--emb-out", default="", help="Optional .npz to cache embeddings (id-less, row-aligned)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from PIL import Image
        embed, _ = load_backbone()
    except Exception as e:
        raise SystemExit(f"[skip] visual baseline needs torch+torchvision+pillow ({type(e).__name__}: {e}).\n"
                         f"       Install: pip install torch torchvision pillow")

    df = pd.read_csv(args.inp)
    if args.img_col not in df or args.label_col not in df:
        raise SystemExit(f"Need columns '{args.img_col}' and '{args.label_col}'. Have: {list(df.columns)}")

    paths, ys, splits, skipped = [], [], [], 0
    for _, r in df.iterrows():
        p = str(r.get(args.img_col, "") or "").strip()
        if not p or not os.path.exists(p):
            skipped += 1
            continue
        paths.append(p); ys.append(to_y(r.get(args.label_col))); splits.append(str(r.get("split", "")))
    if len(paths) < 4 or len(set(ys)) < 2:
        raise SystemExit(f"Not enough screenshots with both classes (usable={len(paths)}, skipped={skipped}).")
    print(f"usable screenshots={len(paths)} (skipped={skipped}) | phishing={sum(ys)} benign={len(ys)-sum(ys)}")

    # embed in batches
    embs = []
    for i in range(0, len(paths), args.batch):
        batch_imgs = []
        for p in paths[i:i + args.batch]:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                batch_imgs.append(Image.new("RGB", (224, 224)))
        embs.append(embed(batch_imgs))
    X = np.vstack(embs); ys = np.array(ys)
    if args.emb_out:
        os.makedirs(os.path.dirname(args.emb_out) or ".", exist_ok=True)
        np.savez_compressed(args.emb_out, emb=X, y=ys, paths=np.array(paths))
        print(f"cached embeddings -> {args.emb_out} ({X.shape})")

    idx = np.arange(len(paths))
    have_split = any(s in ("train", "val", "test") for s in splits)
    if have_split:
        tr = [i for i in idx if splits[i] == "train"]
        te = [i for i in idx if splits[i] in ("val", "test")]
        if not tr or not te or len(set(ys[tr])) < 2:
            have_split = False
    if not have_split:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(idx, test_size=0.3, stratify=ys, random_state=args.seed)
    print(f"train={len(tr)} test={len(te)} | emb-dim={X.shape[1]}")

    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X[tr], ys[tr])
    score = clf.predict_proba(X[te])[:, 1]
    report("ResNet18+LogReg", ys[te], score)


if __name__ == "__main__":
    main()
