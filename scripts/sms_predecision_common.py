#!/usr/bin/env python3
"""Shared loading for the three 2026-09-16 pre-decision arms of the smishing study.

Everything here is read from the registered artefacts (the importer's message table with its
`split` column, the raw corpus for the text) and from train_sms_fusion.py's own helpers, so the
three arms train on exactly the rows and seeds the registered comparison used. Nothing re-splits.
"""
from __future__ import annotations
import csv, os, sys, unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from compphish_features import extract  # noqa: E402
from train_sms_fusion import COMPPHISH, MSG, SEEDS, SRC, first_url  # noqa: E402

SMS_DIR = os.path.join(ROOT, "data", "processed", "sms")


def load():
    """rows, texts, y, split mask pair, URL feature matrix, has_url — as train_sms_fusion builds them."""
    raw = {r["message_id"]: r["message"] for r in
           csv.DictReader(open(SRC, newline="", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(MSG, newline="", encoding="utf-8")))
    texts = [raw.get(r["message_id"], "") for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    urls = [first_url(t) for t in texts]
    has_url = np.array([bool(u) for u in urls])
    Xurl = np.zeros((len(rows), len(COMPPHISH)), dtype=np.float32)
    for i, u in enumerate(urls):
        if u:
            f = extract(u)
            Xurl[i] = [float(f.get(c, 0) or 0) for c in COMPPHISH]
    return rows, texts, y, split == "train", split == "test", Xurl, has_url


def test_groups(rows, te):
    return [rows[i]["text_sha1"] for i in np.where(te)[0]]


def strip_diacritics(text: str) -> str:
    """Vietnamese without its diacritics: NFD, drop combining marks, map đ/Đ, which are base
    letters in Unicode and survive NFD, to d/D."""
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D")


def segment(text: str) -> str:
    """PhoBERT's expected input: word-segmented Vietnamese (pyvi), syllables of a word joined by
    underscores. The registered text arm skipped this step."""
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(text or "")


def summarise(per_seed_metrics: list) -> dict:
    keys = ("f1", "precision", "recall")
    return {k: round(float(np.mean([m[k] for m in per_seed_metrics])), 4) for k in keys} | {
        "f1_sd": round(float(np.std([m["f1"] for m in per_seed_metrics], ddof=1)), 4)
        if len(per_seed_metrics) > 1 else 0.0}


__all__ = ["ROOT", "SMS_DIR", "SEEDS", "COMPPHISH", "load", "test_groups", "strip_diacritics",
           "segment", "summarise"]
