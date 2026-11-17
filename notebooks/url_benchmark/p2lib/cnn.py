"""Notebook 04, first half: the character CNN, its benign-source probe and its transfer matrix."""
import os, re, time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .transfer import load_corpus, in_dataset_split
from .common import (COMPPHISH, URL_CSV, _metrics, benign_mask, load, make_any_model, pr_curve_row,
                     split_phishing, weighting_params, write_curves)


MAX_LEN = 128          # p99 is 43; 20 rows of 53,116 are longer than this and are truncated


EMB = 32


WIDTHS = (3, 4, 5, 6)


FILTERS = 128


DROPOUT = 0.3


BATCH = 256


LR = 1e-3


VAL_FRAC = 0.10        # carved from TRAIN, never from the test window


def encode(urls, vocab):
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


def fit_predict(tr_urls, ytr, te_urls, seed, epochs, patience=2, device="cpu"):
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

    net = make_net(len(vocab), seed).to(device)
    Xva, yva_t = Xva.to(device), yva_t.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr_t), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
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
        # score in slices: the test corpus can be 53k rows and the whole tensor need not sit on
        # the accelerator at once
        out = [torch.sigmoid(net(Xte[i:i + 4096].to(device))).cpu()
               for i in range(0, len(Xte), 4096)]
        return torch.cat(out).numpy()


def run_charcnn(**overrides):
    a = SimpleNamespace(**{**{'seeds': 5, 'epochs': 12, 'cut': 0.7, 'test_after': None, 'out': 'data/processed/p2/p2_charcnn.csv', 'curves': 'data/processed/p2/p2_pr_curves_charcnn.csv', 'device': 'cpu', 'benign_grouped': False, 'strip_scheme': False}, **overrides})
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
            bmask = benign_mask(be, rng, a.cut, a.benign_grouped)
            if proto == "temporal_strict":
                tr = pd.concat([ph_tr, be[bmask]])
                te = pd.concat([ph_te, be[~bmask]])
            else:
                pool = pd.concat([ph_tr, ph_te])
                pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                tr = pd.concat([pool[pmask], be[bmask]])
                te = pd.concat([pool[~pmask], be[~bmask]])

            t0 = time.time()
            enc = (lambda xs: [re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", str(u)) for u in xs]) \
                if a.strip_scheme else (lambda xs: xs)
            score = fit_predict(enc(tr["url"].tolist()), tr["y"].to_numpy(int),
                                enc(te["url"].tolist()), s, a.epochs, device=a.device)
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


TRANCO = {"tranco", "tranco_vn"}


TINNHIEM = {"tinnhiem_org", "tinnhiem_web"}


def with_source(df):
    src = pd.read_csv(URL_CSV, usecols=["url_norm", "source"], dtype=str)
    return df.merge(src.rename(columns={"url_norm": "url"}), on="url", how="left")


def run_source_probe(**overrides):
    a = SimpleNamespace(**{**{'seeds': 3, 'epochs': 8, 'cut': 0.7, 'families': ['CatBoost', 'LogReg'], 'out': 'data/processed/p2/p2_source_probe.csv'}, **overrides})
    df = with_source(load())
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, a.cut)
    print(f"benign by family: tranco {be.source.isin(TRANCO).sum()}  "
          f"tinnhiem {be.source.isin(TINNHIEM).sum()}  (unmapped {be.source.isna().sum()})")
    rows = []
    for train_fam, name in ((TRANCO, "tranco"), (TINNHIEM, "tinnhiem")):
        be_tr = be[be.source.isin(train_fam)]
        be_te = be[~be.source.isin(train_fam) & be.source.notna()]
        for s in range(a.seeds):
            tr = pd.concat([ph_tr, be_tr])
            te = pd.concat([ph_te, be_te])
            yte = te["y"].to_numpy(int)

            t0 = time.time()
            score = fit_predict(tr["url"].tolist(), tr["y"].to_numpy(int),
                                te["url"].tolist(), s, a.epochs)
            met = _metrics(yte, score)
            met.update({"family": "CharCNN", "seed": s, "train_benign": name,
                        "fit_seconds": round(time.time() - t0, 1)})
            rows.append(met)
            print(f"  CharCNN   train_benign={name:<9} seed={s} F1={met['F1']:.3f} "
                  f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")

            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte = te[feats].to_numpy(float)
            for fam in a.families:
                t0 = time.time()
                m = make_any_model(fam, s, weighting_params(fam))
                m.fit(Xtr, ytr)
                sc = m.predict_proba(Xte)[:, 1]
                met = _metrics(yte, sc)
                met.update({"family": fam, "seed": s, "train_benign": name,
                            "fit_seconds": round(time.time() - t0, 1)})
                rows.append(met)
                print(f"  {fam:<9} train_benign={name:<9} seed={s} F1={met['F1']:.3f} "
                      f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[+] {a.out}")


XD_CNN_CORPORA = [("PhishVN", "data/processed/vn_compphish.csv"),
                  ("PhiUSIIL", "data/processed/external/phiusiil_compphish.csv"),
                  ("ISCXURL2016", "data/processed/external/iscx_compphish.csv"),
                  ("PhishStorm", "data/processed/external/phishstorm_compphish.csv")]


_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


def strip_scheme(u) -> str:
    return _SCHEME.sub("", str(u))


def run_charcnn_xdata(**overrides):
    a = SimpleNamespace(**{**{'seeds': 5, 'epochs': 12, 'device': 'cpu', 'out': 'data/processed/p2/p2_charcnn_xdata.csv', 'matrix_prefix': 'data/processed/p2/cross_dataset_'}, **overrides})
    corpora = {}
    for name, path in XD_CNN_CORPORA:
        df = load_corpus(path)
        df["url_s"] = df["url"].map(strip_scheme)
        corpora[name] = df
        print(f"[i] {name}: {len(df)} rows (phishing={int(df.y.sum())}), "
              f"{df.url_s.str.len().median():.0f} chars median")
    names = [n for n, _ in XD_CNN_CORPORA]
    rows = []
    for src in names:
        S = corpora[src]
        for s in range(a.seeds):
            # diagonal: the grouped in-corpus split, same draw as the tabular matrix
            tr, te = in_dataset_split(S, s)
            t0 = time.time()
            sc = fit_predict(tr.url_s.tolist(), tr.y.to_numpy(int), te.url_s.tolist(),
                             s, a.epochs, device=a.device)
            m = _metrics(te.y.to_numpy(int), sc)
            m.update(train=src, test=src, seed=s, n_train=len(tr), n_test=len(te),
                     fit_seconds=round(time.time() - t0, 1), device=a.device)
            rows.append(m)
            print(f"  {src:<12}-> {src:<12} seed={s} F1={m['F1']:.3f} ROC={m['ROC-AUC']:.3f} "
                  f"({m['fit_seconds']}s, in-dist)", flush=True)
            # off-diagonal: one fit on the whole source, scored on each whole target. The
            # network is trained once and only scored three times, so the fit time is charged
            # to the first target and zero to the others; the CSV says so.
            targets = [t for t in names if t != src]
            t0 = time.time()
            net_scores = fit_predict(S.url_s.tolist(), S.y.to_numpy(int),
                                     pd.concat([corpora[t].url_s for t in targets]).tolist(),
                                     s, a.epochs, device=a.device)
            fit_s = round(time.time() - t0, 1)
            off = 0
            for t in targets:
                T = corpora[t]
                sc = net_scores[off:off + len(T)]
                off += len(T)
                m = _metrics(T.y.to_numpy(int), sc)
                m.update(train=src, test=t, seed=s, n_train=len(S), n_test=len(T),
                         fit_seconds=fit_s if t == targets[0] else 0.0, device=a.device)
                rows.append(m)
                print(f"  {src:<12}-> {t:<12} seed={s} F1={m['F1']:.3f} "
                      f"ROC={m['ROC-AUC']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)  # checkpoint after every seed
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"[+] {len(out)} rows -> {a.out}")
    if a.matrix_prefix:
        for metric in ("F1", "ROC-AUC"):
            mat = (out.pivot_table(index="train", columns="test", values=metric, aggfunc="mean")
                      .reindex(index=names, columns=names).round(3))
            mat.index.name = None
            path = f"{a.matrix_prefix}{metric}_CharCNN.csv"
            mat.to_csv(path)
            v = mat.to_numpy(float)
            d, o = np.diag(v).mean(), v[~np.eye(len(v), dtype=bool)].mean()
            print(f"[+] {path}  diag={d:.3f} off={o:.3f} gap={d - o:.3f}")
            print(mat.to_string())
