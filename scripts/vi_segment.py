#!/usr/bin/env python3
"""
vi_segment.py — Vietnamese word segmentation for PhoBERT (PhoBERT expects word-segmented input).

Backend (auto): py_vncorenlp -> underthesea -> identity (no-op) if neither installed.
Use as a CLI to add a segmented column, then train on that column.

INSTALL (best): pip install py_vncorenlp   (needs Java)  OR  pip install underthesea
RUN:
  python vi_segment.py --in data/processed/dataset_sms.csv --col text --out-col text_seg \
         --out data/processed/dataset_sms_seg.csv
  # then: python train_text_phobert.py --in data/processed/dataset_sms_seg.csv  (point text col to text_seg)
"""
from __future__ import annotations
import argparse, os


def get_segmenter():
    try:
        import py_vncorenlp
        os.makedirs("vncorenlp", exist_ok=True)
        try:
            py_vncorenlp.download_model(save_dir="vncorenlp")
        except Exception:
            pass
        model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="vncorenlp")
        return (lambda t: " ".join(model.word_segment(t)) if t else ""), "vncorenlp"
    except Exception:
        pass
    try:
        from underthesea import word_tokenize
        return (lambda t: word_tokenize(t, format="text") if t else ""), "underthesea"
    except Exception:
        pass
    return (lambda t: t), "identity(no-op)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--col", default="text")
    ap.add_argument("--out-col", default="text_seg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pandas as pd
    seg, backend = get_segmenter()
    print(f"segmenter backend = {backend}")
    df = pd.read_csv(args.inp)
    df[args.out_col] = df[args.col].fillna("").astype(str).map(seg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows with '{args.out_col}' -> {args.out}")


if __name__ == "__main__":
    main()
