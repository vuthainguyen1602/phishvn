#!/usr/bin/env python3
"""
run_p2_charcnn.py — a character-CNN over the raw URL string, under the same protocols as the
seven tabular families.

WHY. §II cites a CNN-based malicious-URL detector on a Vietnamese venue and answers it by SCOPE
("URL-only, English-centric corpora") rather than by measurement, which leaves the obvious
referee question unasked: the benchmark evaluates classification from a URL, so how does a model
that reads the string itself compare with gradient boosting over 21 hand-built features? This
runs it. The benchmark's own decomposition predicts a null -- between-family spread is 0.014 F1
on the dated rows against 0.076 for the design steps -- and a measured null closes the question
where a scope argument does not.

IT SEES THE SAME ROWS, NOT A SIMILAR SPLIT. `load`, `split_phishing` and the per-seed benign mask
are imported from run_p2_temporal_strict and consumed in the identical order, so for a given
(protocol, seed) this arm trains and tests on exactly the rows every other family did. That is
what makes the comparison paired: no re-draw, no separate harness, and the output carries the
canonical schema so the rows sit in the existing tables.

THE MODEL IS DELIBERATELY ORDINARY: character embedding, parallel convolutions of width 3-6,
global max pooling, dropout, one linear layer. The claim under test is whether reading the string
beats reading the features, not whether a tuned architecture can win; a bespoke model would make
a loss uninterpretable ("you tuned it badly") and a win unattributable.

TWO HONEST LIMITS, both worth reading before the number is quoted. The corpus is host-level for
most rows (median URL length 22 characters, p99 43), so the convolutions see far less string than
the URLNet-style literature assumes. And the vocabulary is built on TRAIN ONLY, per seed, with
unseen characters mapped to UNK -- fitting it on the whole corpus would leak the test window's
alphabet, which on a temporal split is exactly the kind of leak this paper exists to measure.

RUN:
  python scripts/run_p2_charcnn.py --seeds 5
  python scripts/run_p2_charcnn.py --seeds 1 --epochs 3      # wiring check
Design note: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from train_url_baseline import _metrics
from run_p2_benchmark import pr_curve_row, write_curves
from run_p2_temporal_strict import load, split_phishing

MAX_LEN = 128          # p99 is 43; 20 rows of 53,116 are longer than this and are truncated
EMB = 32
WIDTHS = (3, 4, 5, 6)
FILTERS = 128
DROPOUT = 0.3
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.10        # carved from TRAIN, never from the test window


def encode(urls, vocab):
    """URLs -> (n, MAX_LEN) int matrix. 0 is PAD, 1 is UNK, so a character the training window
    never showed cannot silently become a different known character."""
    out = np.zeros((len(urls), MAX_LEN), dtype=np.int64)
    for i, u in enumerate(urls):
        for j, ch in enumerate(str(u)[:MAX_LEN]):
            out[i, j] = vocab.get(ch, 1)
    return out


def build_vocab(urls):
    chars = sorted({ch for u in urls for ch in str(u)[:MAX_LEN]})
    return {ch: i + 2 for i, ch in enumerate(chars)}


def make_net(vocab_size, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size + 2, EMB, padding_idx=0)
            self.convs = nn.ModuleList(
                [nn.Conv1d(EMB, FILTERS, w, padding=w // 2) for w in WIDTHS])
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(FILTERS * len(WIDTHS), 1)

        def forward(self, x):
            h = self.emb(x).transpose(1, 2)
            h = torch.cat([c(h).relu().amax(dim=2) for c in self.convs], dim=1)
            return self.fc(self.drop(h)).squeeze(1)

    return Net()


def fit_predict(tr_urls, ytr, te_urls, seed, epochs, patience=2):
    """Train on TRAIN (minus a seeded validation slice) and score the test window.

    Early stopping reads the validation slice only. The best state is restored before scoring, so
    a run that overfits late scores as its best epoch rather than its last -- otherwise `epochs`
    becomes a hyperparameter nobody chose and the seeds disagree for the wrong reason."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    rng = np.random.RandomState(seed)
    vmask = rng.rand(len(tr_urls)) < VAL_FRAC
    vocab = build_vocab(np.asarray(tr_urls)[~vmask])          # TRAIN-only alphabet

    Xtr = torch.from_numpy(encode(np.asarray(tr_urls)[~vmask], vocab))
    Xva = torch.from_numpy(encode(np.asarray(tr_urls)[vmask], vocab))
    Xte = torch.from_numpy(encode(np.asarray(te_urls), vocab))
    ytr_t = torch.from_numpy(np.asarray(ytr)[~vmask].astype(np.float32))
    yva_t = torch.from_numpy(np.asarray(ytr)[vmask].astype(np.float32))

    net = make_net(len(vocab), seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr_t), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = lossf(net(Xva), yva_t).item()
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xte)).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--cut", type=float, default=0.70)
    ap.add_argument("--test-after", default=None)
    ap.add_argument("--out", default="data/processed/p2/p2_charcnn.csv")
    ap.add_argument("--curves", default="data/processed/p2/p2_pr_curves_charcnn.csv")
    a = ap.parse_args()
    os.chdir(ROOT)

    df = load()
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, a.cut, a.test_after)
    print(f"phishing dated: {len(ph)}  train ({len(ph_tr)})  test ({len(ph_te)}; "
          f"{n_leaked} guard-dropped)  benign pool: {len(be)}")

    rows, curves = [], []
    for proto in ("temporal_strict", "random_same_rows"):
        for s in range(a.seeds):
            rng = np.random.RandomState(s)                 # the canonical per-seed benign mask
            bmask = rng.rand(len(be)) < a.cut
            if proto == "temporal_strict":
                tr = pd.concat([ph_tr, be[bmask]])
                te = pd.concat([ph_te, be[~bmask]])
            else:
                pool = pd.concat([ph_tr, ph_te])
                pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                tr = pd.concat([pool[pmask], be[bmask]])
                te = pd.concat([pool[~pmask], be[~bmask]])

            t0 = time.time()
            score = fit_predict(tr["url"].tolist(), tr["y"].to_numpy(int),
                                te["url"].tolist(), s, a.epochs)
            yte = te["y"].to_numpy(int)
            met = _metrics(yte, score)
            met.update({"family": "CharCNN", "seed": s, "protocol": proto,
                        "fit_seconds": round(time.time() - t0, 2)})
            rows.append(met)
            curves.append({"family": "CharCNN", "protocol": proto, "seed": s,
                           "precision": pr_curve_row(yte, score)})
            print(f"  CharCNN  {proto:<17} seed={s} F1={met['F1']:.3f} "
                  f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f} "
                  f"({met['fit_seconds']}s)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    if a.curves:
        write_curves(curves, a.curves)
    print(f"[+] {a.out}")


if __name__ == "__main__":
    main()
