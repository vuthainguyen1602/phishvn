"""Notebook 02: transfer across the four corpora, pooled training, bootstrap intervals and SHAP."""
import os, time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import common
from .common import (COMPPHISH, DETERMINISTIC, GroupShuffleSplit, MODEL_NAMES, _metrics, add_label,
                     f1_score, make_any_model, make_model, roc_auc_score, train_test_split)


LEGACY_SPLIT = False


def load_corpus(path):
    df = pd.read_csv(path)
    df = add_label(df)
    for c in COMPPHISH:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if df["y"].nunique() < 2:
        raise SystemExit(f"{path}: needs both classes (phishing + benign).")
    return df


def in_dataset_split(df, seed):
    if "split" in df and df["split"].astype(str).isin(["train", "test"]).any():
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        if len(te) and te["y"].nunique() > 1:
            return tr, te
    if LEGACY_SPLIT or "dom" not in df:
        return train_test_split(df, test_size=0.3, stratify=df["y"], random_state=seed)
    gi, gj = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(df, groups=df["dom"]))
    tr, te = df.iloc[gi], df.iloc[gj]
    if te["y"].nunique() < 2:  # degenerate draw: fall back rather than crash the cell
        return train_test_split(df, test_size=0.3, stratify=df["y"], random_state=seed)
    return tr, te


def _psd_power(C, p):
    w, V = np.linalg.eigh(C)
    return (V * np.clip(w, 0, None) ** p) @ V.T


class CoralAligner:

    def __init__(self, Xs, Xt, lam=1.0):
        Xs = np.asarray(Xs, float)
        Xt = np.asarray(Xt, float)
        self.mu_s, self.sd_s = Xs.mean(0), Xs.std(0) + 1e-9
        self.mu_t, self.sd_t = Xt.mean(0), Xt.std(0) + 1e-9
        Zs = (Xs - self.mu_s) / self.sd_s
        Zt = (Xt - self.mu_t) / self.sd_t
        d = Zs.shape[1]
        Cs = np.cov(Zs, rowvar=False) + lam * np.eye(d)
        Ct = np.cov(Zt, rowvar=False) + lam * np.eye(d)
        self.A = _psd_power(Cs, -0.5) @ _psd_power(Ct, 0.5)

    def source(self, X):
        return ((np.asarray(X, float) - self.mu_s) / self.sd_s) @ self.A

    def target(self, X):
        return (np.asarray(X, float) - self.mu_t) / self.sd_t


def _cell_metric(y, score, metric):
    if metric in ("MCC", "BalAcc"):
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
        pred = (np.asarray(score) >= 0.5).astype(int)
        return (matthews_corrcoef(y, pred) if metric == "MCC"
                else balanced_accuracy_score(y, pred))
    return _metrics(y, score)[metric]


def cell_score(train_df, test_df, same, metric, seeds, model="RandomForest",
               feats=COMPPHISH, adapt="none"):
    n_seed = 1 if model in DETERMINISTIC else seeds
    params = {}
    vals = []
    for s in range(n_seed):
        if same:
            tr, te = in_dataset_split(train_df, s)
        else:
            tr, te = train_df, test_df
        if model in MODEL_NAMES:
            # the baseline module's class_weight=balanced is dropped for the families that carry it.
            if common.UNWEIGHTED and model in {"LogReg", "RandomForest", "HistGB"}:
                params = {**params, "class_weight": None}
            m = make_model(model, s, params)
        else:
            m = make_any_model(model, s)
        if adapt == "coral" and not same:  # diagonal cells stay unadapted by construction
            al = CoralAligner(tr[feats].to_numpy(), te[feats].to_numpy())
            m = m.fit(al.source(tr[feats]), tr["y"])
            score = m.predict_proba(al.target(te[feats]))[:, 1]
        else:
            m = m.fit(tr[feats], tr["y"])
            score = m.predict_proba(te[feats])[:, 1]
        vals.append(_cell_metric(te["y"].to_numpy(), score, metric))
    return float(np.mean(vals)), float(np.std(vals))


def run_cross_dataset(**overrides):
    args = SimpleNamespace(**{**{'corpora': None, 'metric': 'F1', 'seeds': 5, 'model': 'RandomForest', 'drop': '', 'adapt': 'none', 'legacy_split': False, 'out': ''}, **overrides})
    if args.corpora is None:
        args.corpora = CORPUS_ARGS
    drop = {f.strip() for f in args.drop.split(",") if f.strip()}
    unknown = drop - set(COMPPHISH)
    if unknown:
        raise SystemExit(f"--drop names not in the lexical URL schema: {sorted(unknown)}")
    feats = [c for c in COMPPHISH if c not in drop]
    if drop:
        print(f"[i] artefact-pruning: dropping {sorted(drop)} -> {len(feats)} features")
    corpora = {}
    for entry in args.corpora:
        if "=" not in entry:
            raise SystemExit(f"Bad --corpora entry (need name=path): {entry}")
        name, path = entry.split("=", 1)
        corpora[name] = load_corpus(path)
        print(f"[i] {name}: {len(corpora[name])} rows "
              f"(phishing={int(corpora[name]['y'].sum())})")
    names = list(corpora)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    print(f"\nComputing {len(names)}x{len(names)} matrix "
          f"(metric={args.metric}, seeds={args.seeds}) ...")
    for i in names:
        for j in names:
            mean, std = cell_score(corpora[i], corpora[j], i == j, args.metric, args.seeds,
                                   model=args.model, feats=feats,
                                   adapt=args.adapt)
            mat.loc[i, j] = round(mean, 3)
            print(f"  train={i:14s} test={j:14s} {args.metric}={mean:.3f}±{std:.3f}"
                  f"{'  (in-dist)' if i == j else ''}")
    diag = np.array([mat.loc[n, n] for n in names], dtype=float)
    off = mat.values[~np.eye(len(names), dtype=bool)].astype(float)
    print("\n=== " + args.metric + " matrix (rows=train, cols=test) ===")
    print(mat.to_string())
    print(f"\nmean diagonal (in-distribution) = {diag.mean():.3f}")
    print(f"mean off-diagonal (transfer)    = {np.nanmean(off):.3f}")
    print(f"generalisation gap              = {diag.mean() - np.nanmean(off):.3f}  "
          f"(smaller = features transfer better)")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        mat.to_csv(args.out)
        print(f"[+] saved matrix -> {args.out}")


CORPUS_ARGS = ["PhishVN=data/processed/vn_compphish.csv", "PhiUSIIL=data/processed/external/phiusiil_compphish.csv",
               "ISCXURL2016=data/processed/external/iscx_compphish.csv", "PhishStorm=data/processed/external/phishstorm_compphish.csv"]
PRUNE = "tld_len,subdom_cnt,dot_cnt"
MATRICES = []
for metric in ("F1", "ROC-AUC"):
    MATRICES += [(f"cross_dataset_{metric}.csv", dict(metric=metric)),
                 (f"cross_dataset_{metric}_LogReg.csv", dict(metric=metric, model="LogReg")),
                 (f"cross_dataset_{metric}_HistGB.csv", dict(metric=metric, model="HistGB")),
                 (f"cross_dataset_{metric}_pruned.csv", dict(metric=metric, drop=PRUNE)),
                 (f"cross_dataset_{metric}_LogReg_pruned.csv", dict(metric=metric, model="LogReg", drop=PRUNE)),
                 (f"cross_dataset_{metric}_HistGB_pruned.csv", dict(metric=metric, model="HistGB", drop=PRUNE)),
                 (f"cross_dataset_{metric}_coral.csv", dict(metric=metric, adapt="coral")),
                 (f"cross_dataset_{metric}_LogReg_coral.csv", dict(metric=metric, model="LogReg", adapt="coral"))]
MATRICES.append(("cross_dataset_F1_CatBoost.csv", dict(metric="F1", model="CatBoost")))


PROC = "data/processed"


XD_CORPORA = {
    "PhishVN": "vn_compphish.csv",
    "PhiUSIIL": "external/phiusiil_compphish.csv",
    "ISCXURL2016": "external/iscx_compphish.csv",
    "PhishStorm": "external/phishstorm_compphish.csv",
}


XD_DROP = {"tld_len", "subdom_cnt", "dot_cnt"}


PRUNED = [c for c in COMPPHISH if c not in XD_DROP]


XD_SEEDS = 5


def combined_csv() -> str:
    return os.path.join(PROC, "p2", "combined_training.csv")


def run_combined_training(**overrides):
    args = SimpleNamespace(**{**{}, **overrides})
    data = {n: load_corpus(os.path.join(PROC, f)) for n, f in XD_CORPORA.items()}
    names = list(data)
    f1_mat = pd.read_csv(os.path.join(PROC, "p2", "cross_dataset_F1.csv"), index_col=0)
    roc_mat = pd.read_csv(os.path.join(PROC, "p2", "cross_dataset_ROC-AUC.csv"), index_col=0)
    rows = []
    for held in names:
        union = pd.concat([data[n] for n in names if n != held], ignore_index=True)
        te = data[held]
        print(f"held-out={held}: union train={len(union)} rows, test={len(te)}")
        for tag, feats in (("full", COMPPHISH), ("pruned", PRUNED)):
            t0 = time.time()
            f1s, rocs = [], []
            for s in range(XD_SEEDS):
                m = make_model("RandomForest", s, {"class_weight": None} if common.UNWEIGHTED else {}).fit(
                    union[feats], union["y"])
                score = m.predict_proba(te[feats])[:, 1]
                met = _metrics(te["y"].to_numpy(), score)
                f1s.append(met["F1"])
                rocs.append(met["ROC-AUC"])
            singles_f1 = [f1_mat.loc[i, held] for i in names if i != held]
            rows.append({
                "held_out": held, "config": tag,
                "F1": float(np.mean(f1s)), "F1_std": float(np.std(f1s)),
                "ROC-AUC": float(np.mean(rocs)), "ROC-AUC_std": float(np.std(rocs)),
                "best_single_F1": float(max(singles_f1)),
                "best_single_ROC": float(max(roc_mat.loc[i, held] for i in names if i != held)),
                "indist_F1": float(f1_mat.loc[held, held]),
            })
            print(f"  [{tag:6s}] F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} "
                  f"ROC-AUC={np.mean(rocs):.3f}±{np.std(rocs):.3f} "
                  f"(best single {max(singles_f1):.3f}, in-dist {f1_mat.loc[held, held]:.3f}) "
                  f"({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(combined_csv(), index=False)
    print(f"[+] {combined_csv()}")


def _fit(model: str, seed: int):
    if model == "RandomForest":
        unweighted = os.environ.get("P2_UNWEIGHTED") == "1"
        return make_model(model, seed, {"class_weight": None} if unweighted else {})
    return make_any_model(model, seed)


def collect_scores(seeds: int = XD_SEEDS, model: str = "RandomForest"):
    data = {n: load_corpus(os.path.join(PROC, f)) for n, f in XD_CORPORA.items()}
    names = list(data)
    held = {}
    for i in names:
        for j in names:
            same = i == j
            per_seed = []
            t0 = time.time()
            for s in range(seeds):
                if same:
                    tr, te = in_dataset_split(data[i], s)
                else:
                    tr, te = data[i], data[j]
                m = _fit(model, s).fit(tr[COMPPHISH], tr["y"])
                sc = m.predict_proba(te[COMPPHISH])[:, 1]
                per_seed.append((te["y"].to_numpy().astype(int), sc))
            held[(i, j)] = per_seed
            f1 = np.mean([f1_score(y, sc >= 0.5, zero_division=0) for y, sc in per_seed])
            roc = np.mean([roc_auc_score(y, sc) for y, sc in per_seed])
            print(f"[i] {i:12s} -> {j:12s} F1={f1:.3f} ROC-AUC={roc:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return names, held


def _cell_metrics(y, sc, idx):
    yb, sb = y[idx], sc[idx]
    if len(np.unique(yb)) < 2:
        return np.nan, np.nan
    return (f1_score(yb, sb >= 0.5, zero_division=0), roc_auc_score(yb, sb))


def bootstrap(names, held, boot: int, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    keys = list(held)
    out = {k: {"F1": [], "ROC-AUC": []} for k in keys}
    for b in range(boot):
        for k in keys:
            f1s, rocs = [], []
            for y, sc in held[k]:
                idx = rng.integers(0, len(y), len(y))
                f1, roc = _cell_metrics(y, sc, idx)
                f1s.append(f1); rocs.append(roc)
            out[k]["F1"].append(np.nanmean(f1s))
            out[k]["ROC-AUC"].append(np.nanmean(rocs))
        if (b + 1) % 50 == 0:
            print(f"    ... {b+1}/{boot} replicates", flush=True)
    return out


def summarise(names, held, reps, boot):
    diag = [(n, n) for n in names]
    off = [k for k in reps if k[0] != k[1]]
    rows = []
    point = {}
    for metric in ("F1", "ROC-AUC"):
        obs = {}
        for k, per_seed in held.items():
            fn = (lambda y, sc: f1_score(y, sc >= 0.5, zero_division=0)) if metric == "F1" \
                else (lambda y, sc: roc_auc_score(y, sc))
            obs[k] = float(np.mean([fn(y, sc) for y, sc in per_seed]))
        d = float(np.mean([obs[k] for k in diag]))
        o = float(np.mean([obs[k] for k in off]))
        point[metric] = {"diag": d, "off": o, "gap": d - o, "cells": obs}
        dv = np.array([np.mean([reps[k][metric][b] for k in diag]) for b in range(boot)])
        ov = np.array([np.mean([reps[k][metric][b] for k in off]) for b in range(boot)])
        for label, series, pt in (("diagonal", dv, d), ("off-diagonal", ov, o),
                                  ("gap", dv - ov, d - o)):
            lo, hi = np.percentile(series, [2.5, 97.5])
            rows.append({"metric": metric, "quantity": label, "point": round(pt, 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "boot": boot})
        # Per-cell interval for every off-diagonal cell, so "N cells are below chance" can be
        # stated as "N cells whose whole interval is below chance" rather than as a bare count.
        for k in off:
            s = np.array(reps[k][metric])
            lo, hi = np.percentile(s, [2.5, 97.5])
            rows.append({"metric": metric, "quantity": f"cell {k[0]}->{k[1]}",
                         "point": round(obs[k], 4), "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4), "boot": boot})
    return pd.DataFrame(rows), point


BOOT_ARMS = [("CatBoost", "F1"), ("RandomForest", "ROC-AUC")]


def run_xdata_bootstrap(**overrides):
    args = SimpleNamespace(**{**{'boot': 500, 'seeds': 5, 'out': 'data/processed/p2/p2_xdata_bootstrap.csv'}, **overrides})
    # the forest is fitted unweighted; the shared bootstrap code reads the switch from the environment
    os.environ.setdefault("P2_UNWEIGHTED", "1")
    unweighted = os.environ["P2_UNWEIGHTED"] == "1"
    arms = BOOT_ARMS if unweighted else [a for a in BOOT_ARMS if a[0] == "RandomForest"]
    frames = []
    for model, metric in arms:
        print(f"\n=== {model} ({metric} matrix) ===", flush=True)
        names, held = collect_scores(args.seeds, model=model)
        reps = bootstrap(names, held, args.boot)
        df, point = summarise(names, held, reps, args.boot)
        df.insert(0, "model", model)
        frames.append(df[df.metric == metric])
        p = point[metric]
        print(f"    {model}/{metric}: diagonal {p['diag']:.3f}, off-diagonal {p['off']:.3f}, "
              f"gap {p['gap']:.3f}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[+] {args.out}")


def run_xdata_bootstrap_rf(**overrides):
    args = SimpleNamespace(**{**{'boot': 500, 'seeds': 5, 'out': 'data/processed/p2/p3_xdata_bootstrap.csv'}, **overrides})
    names, held = collect_scores(args.seeds)
    reps = bootstrap(names, held, args.boot)
    df, point = summarise(names, held, reps, args.boot)
    df.to_csv(args.out, index=False)
    print(f"[+] {args.out}")
    for metric in ("F1", "ROC-AUC"):
        p = point[metric]
        print(f"    {metric}: diagonal {p['diag']:.3f}, off-diagonal {p['off']:.3f}, "
              f"gap {p['gap']:.3f}")


def attribution(df, seeds, n_sample, weighting):
    import shap
    params = None if weighting == "balanced" else {"class_weight": None}
    imps, signs = [], []
    for s in range(seeds):
        tr, te = in_dataset_split(df, s)
        m = make_model("RandomForest", s, params).fit(tr[COMPPHISH], tr["y"])
        # the same stratified rows tab_shap explains: the draw depends on the seed only
        sub = te.groupby("y", group_keys=False).apply(
            lambda g: g.sample(min(len(g), n_sample // 2), random_state=s))
        sv = shap.TreeExplainer(m).shap_values(sub[COMPPHISH])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imps.append(np.abs(sv).mean(axis=0))
        signs.append([np.corrcoef(sub[c], sv[:, i])[0, 1] if sub[c].nunique() > 1 else np.nan
                      for i, c in enumerate(COMPPHISH)])
    imp = np.mean(imps, axis=0)
    return pd.DataFrame({"feature": COMPPHISH, "weighting": weighting,
                         "mean_abs_shap": imp, "sd_abs_shap": np.std(imps, axis=0),
                         "share": imp / imp.sum(),
                         "direction": np.nanmean(np.array(signs, dtype=float), axis=0)})


SHAP_CORPUS = "data/processed/vn_compphish.csv"


def run_shap_weighting(**overrides):
    a = SimpleNamespace(**{**{'seeds': 5, 'n_sample': 2000, 'out': 'data/processed/p2/p2_shap_weighting.csv'}, **overrides})
    df = load_corpus(SHAP_CORPUS)
    out = pd.concat([attribution(df, a.seeds, a.n_sample, w) for w in ("balanced", "none")])
    out["rank"] = out.groupby("weighting")["share"].rank(ascending=False).astype(int)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    for w, g in out.groupby("weighting"):
        g = g.sort_values("share", ascending=False)
        top = ", ".join(f"{r.feature} {100 * r.share:.1f}%" for r in g.head(5).itertuples())
        print(f"[{w}] top-3 share {100 * g.share.head(3).sum():.1f}%  |  {top}")
    print(f"[+] {a.out}")
