"""Shared code: features, model families, metrics, the temporal split and the paired test.

Every step of the walkthrough imports from here. `UNWEIGHTED` is the one switch the steps read at
run time: every family is fitted without class weights, and `legacy_weighting()` turns the
paper's earlier mixed configuration back on for one step."""
import os, re, time, math, collections, warnings
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


_TWO_LEVEL = {"com.vn", "net.vn", "org.vn", "edu.vn", "gov.vn", "ac.vn", "info.vn", "pro.vn",
              "health.vn", "int.vn", "name.vn", "biz.vn", "co.uk", "com.br", "com.au", "co.jp",
              "pages.dev", "workers.dev", "r2.dev", "netlify.app", "vercel.app", "web.app",
              "firebaseapp.com", "amplifyapp.com", "webflow.io", "weebly.com", "wixsite.com",
              "blogspot.com", "github.io", "duckdns.org", "ddns.net", "glitch.me", "repl.co"}


def apex(result) -> str:
    new = getattr(result, "top_domain_under_public_suffix", None)
    return result.registered_domain if new is None else new


def registered_domain(host: str) -> str:
    if _EXTRACT is not None:
        try:
            return apex(_EXTRACT(host)).lower() or host
        except Exception:
            pass
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


# every library the steps below use
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_predict, cross_val_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


MODEL_NAMES = ["LogReg", "RandomForest", "HistGB", "MLP"]


DETERMINISTIC = {"LogReg"}  # no seed effect; reported once


COMPPHISH = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
             "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
             "letter_ratio", "digit_ratio", "spec_ratio", "slash_cnt", "entropy",
             "path_len", "query_len"]


def add_label(df):
    # Unknown is not a negative. Respect explicit training eligibility and reject
    # unresolved labels instead of silently converting them to benign.
    if "training_eligible" in df:
        eligible = df["training_eligible"].astype(str).str.lower().isin({"1", "true"})
        if not eligible.all():
            raise ValueError("observations excluded by the selected label policy cannot be used for training/evaluation")
    lab = df["label"]
    if lab.dtype != object:
        numeric = pd.to_numeric(lab, errors="coerce")
        if not numeric.isin([0, 1]).all():
            raise ValueError("labels must be known binary outcomes, not missing or unknown")
        df["y"] = numeric.astype(int)
    else:
        pos = {"phishing", "phish", "spam", "smishing", "malicious", "1"}
        negative = {"benign", "legitimate", "ham", "0"}
        normalized = lab.astype(str).str.strip().str.lower()
        if not normalized.isin(pos | negative).all():
            raise ValueError("unknown label cannot be silently converted to benign")
        df["y"] = normalized.isin(pos).astype(int)
    return df


def fpr_at_recall(y, score, target_recall=0.90):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.where(tpr >= target_recall)[0]
    return float(fpr[idx[0]]) if len(idx) else float("nan")


def make_model(name, seed, params=None):
    # tuned params must OVERRIDE the defaults, not collide with them: passing e.g. a tuned
    # n_estimators alongside a hard-coded one raises TypeError, which the HPO fitness then
    # silently swallows as "invalid config" — so merge dicts instead of stacking kwargs.
    params = params or {}
    if name == "LogReg":
        kw = {"max_iter": 1000, "class_weight": "balanced"} | params
        return LogisticRegression(**kw)
    if name == "RandomForest":
        kw = {"n_estimators": 300, "class_weight": "balanced",
              "n_jobs": -1, "random_state": seed} | params
        return RandomForestClassifier(**kw)
    if name == "HistGB":
        kw = {"class_weight": "balanced", "random_state": seed} | params
        return HistGradientBoostingClassifier(**kw)
    if name == "MLP":  # scaled inputs matter for a neural net on tabular features
        pipe = make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                           random_state=seed))
        return pipe.set_params(**params) if params else pipe  # keys use mlpclassifier__ prefix
    raise ValueError(f"unknown model: {name}")


def _metrics(y, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    y = np.asarray(y)
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    return {"F1": f1_score(y, pred, zero_division=0),
            "PR-AUC": average_precision_score(y, score),
            "ROC-AUC": auc,
            "FPR@R0.90": fpr_at_recall(y, score)}


# every family is fitted unweighted; step 6 switches this off for the legacy arm
UNWEIGHTED = True


FAMILIES = ["LogReg", "RandomForest", "HistGB", "MLP", "XGBoost", "LightGBM", "CatBoost"]


WEIGHTED = {"LogReg", "RandomForest", "HistGB"}


def weighting_params(name, unweighted=None):
    u = UNWEIGHTED if unweighted is None else unweighted
    return {"class_weight": None} if (u and name in WEIGHTED) else None


def make_any_model(name: str, seed: int, params: dict | None = None):
    p = params or {}
    if name == "Stack[CB+LR]":
        from catboost import CatBoostClassifier
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        return StackingClassifier(
            estimators=[("CB", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                                  random_seed=seed, verbose=False)),
                        ("LR", LogisticRegression(max_iter=1000, class_weight="balanced"))],
            final_estimator=LogisticRegression(max_iter=1000),
            stack_method="predict_proba", cv=5, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                                "eval_metric": "logloss", "random_state": seed, "n_jobs": -1,
                                **p})
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.1,
                                 "random_state": seed, "n_jobs": -1, "verbosity": -1, **p})
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{"iterations": 300, "depth": 6, "learning_rate": 0.1,
                                     "random_seed": seed, "verbose": False, **p})
    return make_model(name, seed, params)


PR_GRID = np.linspace(0.02, 1.0, 200)


def pr_curve_row(y, score):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, score)
    return np.interp(PR_GRID, rec[::-1], prec[::-1])


def write_curves(curves, path):
    if not curves or not path:
        return
    import collections
    acc = collections.defaultdict(list)
    for c in curves:
        acc[(c["family"], c["protocol"])].append(c["precision"])
    rows = []
    for (fam, proto), mats in acc.items():
        mean = np.mean(mats, axis=0)
        rows += [{"family": fam, "protocol": proto, "n_seeds": len(mats),
                  "recall": round(float(r), 4), "precision": round(float(pv), 6)}
                 for r, pv in zip(PR_GRID, mean)]
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        # a partial run (--families) refreshes only its own curves and keeps everyone else's
        old = pd.read_csv(path)
        keep = ~old.set_index(["family", "protocol"]).index.isin(list(acc))
        new = pd.concat([old[keep], new], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)
    print(f"[+] {len(acc)} curve(s) -> {path}")


def run_one(name, Xtr, ytr, Xte, yte, seed, return_scores=False, unweighted=None):
    t0 = time.time()
    params = weighting_params(name, unweighted)
    m = make_any_model(name, seed, params)
    m.fit(Xtr, ytr)
    fit_s = time.time() - t0
    score = m.predict_proba(Xte)[:, 1]
    met = _metrics(yte, score)
    met.update({"family": name, "seed": seed, "fit_seconds": round(fit_s, 2)})
    return (met, np.asarray(yte), score) if return_scores else met


URL_CSV = "data/processed/dataset_url.csv"
ALIGNED = "data/processed/vn_compphish.csv"


def parse_date(s):
    s = (s if isinstance(s, str) else "").strip()[:10]
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def load():
    meta = pd.read_csv(URL_CSV, usecols=["url_norm", "label", "collected_at"], dtype=str)
    meta["date"] = meta["collected_at"].map(parse_date)
    df = add_label(pd.read_csv(ALIGNED))
    df = df.merge(meta.rename(columns={"url_norm": "url"})[["url", "date"]],
                  on="url", how="left")
    df = df.dropna(subset=[c for c in COMPPHISH if c in df.columns])
    df["rdom"] = df["dom"].astype(str).str.lower().map(registered_domain)
    return df


def split_phishing(ph: pd.DataFrame, cut: float, test_after: str | None = None):
    if test_after:
        d = pd.Timestamp(test_after)
        ph_tr, ph_te = ph[ph.date <= d], ph[ph.date > d]
    else:
        k = int(len(ph) * cut)
        ph_tr, ph_te = ph.iloc[:k], ph.iloc[k:]
    leaked = ph_te.rdom.isin(set(ph_tr.rdom))
    return ph_tr, ph_te[~leaked], int(leaked.sum())


def benign_mask(be: pd.DataFrame, rng, cut: float, grouped: bool = False) -> np.ndarray:
    if not grouped:
        return rng.rand(len(be)) < cut
    doms = be["rdom"].astype(str).to_numpy()
    uniq = np.unique(doms)
    return np.isin(doms, uniq[rng.rand(len(uniq)) < cut])


def match_prior(te: pd.DataFrame, prior: float, rng) -> pd.DataFrame:
    ph, be = te[te.y == 1], te[te.y == 0]
    k = int(round(len(ph) * (1 - prior) / prior))
    if k >= len(be):
        return te
    return pd.concat([ph, be.iloc[np.sort(rng.choice(len(be), k, replace=False))]])


DEFAULT_TEST_FRAC = 0.30


def corrected_paired_t(d, test_frac: float = DEFAULT_TEST_FRAC) -> dict:
    d = np.asarray(d, dtype=float)
    k = len(d)
    mean = float(np.mean(d))
    var = float(np.var(d, ddof=1)) if k > 1 else 0.0
    out = {"k": k, "mean": mean, "wins": int(np.sum(d > 0)),
           "p": 1.0, "p_naive": 1.0, "se": 0.0, "inflation": float("nan")}
    if k < 2 or var <= 0:
        return out
    try:
        from scipy import stats
    except ImportError:  # keep the pipeline runnable without scipy; verdicts fall back to "no evidence"
        return out
    factor = 1.0 / k + test_frac / (1.0 - test_frac)
    se = math.sqrt(factor * var)
    se_naive = math.sqrt(var / k)
    t = mean / se
    out.update(p=float(2 * stats.t.sf(abs(t), k - 1)),
               p_naive=float(2 * stats.t.sf(abs(mean / se_naive), k - 1)),
               se=se, t=float(t), inflation=se / se_naive)
    return out


def bh_adjust(pvals, alpha=0.05):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):          # walk up from the largest p-value
        i = idx[rank - 1]
        running = min(running, pvals[i] * m / rank)
        adj[i] = min(1.0, running)
    return adj, [a <= alpha for a in adj]


@contextmanager
def legacy_weighting():
    # step 6: logistic regression, random forest and HistGB with class_weight="balanced" again
    global UNWEIGHTED
    UNWEIGHTED = False
    try:
        yield
    finally:
        UNWEIGHTED = True
