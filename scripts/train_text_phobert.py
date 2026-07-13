#!/usr/bin/env python3
"""
train_text_phobert.py — Fine-tune PhoBERT to classify phishing by CONTENT (Vietnamese SMS/email).

Input: data/processed/dataset_sms.csv or dataset_email.csv (columns 'text','label','split').
Model: vinai/phobert-base (default). TIME-BASED split via the 'split' column. Reports F1/PR-AUC.

INSTALL:  pip install "transformers>=4.40" "torch>=2.1" scikit-learn pandas
  (GPU recommended; CPU works but is slow. PhoBERT ideally uses VnCoreNLP for word segmentation,
   here we use the tokenizer directly for simplicity — can be upgraded later.)
RUN:
  python train_text_phobert.py --in data/processed/dataset_sms.csv --out models/phobert_sms --epochs 4
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd


def build_splits(df):
    if "split" in df and df["split"].notna().any() and set(df["split"]) & {"train", "test"}:
        tr = df[df.split == "train"]; va = df[df.split.isin(["val", "test"])]
        te = df[df.split == "test"] if (df.split == "test").any() else va
    else:
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=0); va = te
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="models/phobert_text")
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=128)
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

    df = pd.read_csv(args.inp)
    df = df[df["label"].isin(["phishing", "benign"])].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[df["text"].str.len() > 0]
    if df["y"].nunique() < 2:
        raise SystemExit("Need BOTH classes (phishing + benign) in the text data.")

    tr, va, te = build_splits(df)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    class DS(Dataset):
        def __init__(self, frame):
            self.enc = tok(list(frame.text), truncation=True, padding="max_length",
                           max_length=args.maxlen)
            self.y = list(frame.y)
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def metrics(p):
        logits, y = p
        prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        pred = (prob >= 0.5).astype(int)
        return {"f1": f1_score(y, pred), "pr_auc": average_precision_score(y, prob),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0)}

    import importlib.util
    _rep = ["mlflow"] if importlib.util.find_spec("mlflow") else []   # optional experiment tracking
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", logging_steps=20, report_to=_rep)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(tr),
                      eval_dataset=DS(va), compute_metrics=metrics)
    trainer.train()
    print("== Results on TEST (time-based split) ==")
    print(trainer.evaluate(DS(te)))
    trainer.save_model(args.out); tok.save_pretrained(args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
