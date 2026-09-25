#!/usr/bin/env python3
"""
run_sms_late_fusion.py — is "fusion adds nothing" a finding or an artefact of one fusion head?

WHY. The registered fusion concatenates 21 URL columns onto a 768-dimensional embedding and fits
one MLP, and 45% of the test rows carry no URL and so present a zero URL vector. That the fusion
minus text contrast is -0.002 could mean the URL channel is redundant, or that this head cannot
use it. Four post-hoc combinations test the second reading, on the registered split and seeds:

  early_flag   the registered concatenation plus one indicator column, has_url
  late_mean    mean of the URL arm's and the text arm's probabilities
  late_gated   the text arm's probability where the message has no URL, the mean where it has
  stack        logistic regression over (p_url, p_text, has_url), fitted on 5-fold out-of-fold
               probabilities of the training window

Every arm is compared with the registered text-only arm by the distinct-text cluster bootstrap on
identical rows. POST-HOC and UNREGISTERED; the registered T1/T2 stand as printed.

RUN:
  OPENBLAS_NUM_THREADS=1 python3 scripts/run_sms_late_fusion.py
Writes data/processed/sms/late_fusion.json.
"""
from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sms_predecision_common import SEEDS, SMS_DIR, load, summarise, test_groups  # noqa: E402
from train_sms_fusion import EMB, cluster_bootstrap_f1, embed, fit_score  # noqa: E402

OUT = os.path.join(SMS_DIR, "late_fusion.json")


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=seed,
                         early_stopping=True, n_iter_no_change=15)


def fit_proba(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    m = mlp(seed).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def oof_proba(X, y, seed):
    out = np.zeros(len(y))
    for f, (i, j) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        out[j] = fit_proba(X[i], y[i], X[j], seed * 100 + f)
    return out


def metrics(yte, p):
    return {"f1": f1_score(yte, p, pos_label=1, zero_division=0),
            "precision": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall": recall_score(yte, p, pos_label=1, zero_division=0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows, texts, y, tr, te, Xurl, has_url = load()
    Xtxt = embed(texts, cache=EMB)
    flag = has_url.astype(np.float32)[:, None]
    Xfus = np.hstack([Xtxt, Xurl])
    Xflag = np.hstack([Xtxt, Xurl, flag])
    print(f"  {len(rows)} messages; train {int(tr.sum())} / test {int(te.sum())}; "
          f"{int((~has_url[te]).sum())} test rows without a URL", flush=True)

    names = ("text", "fusion", "early_flag", "late_mean", "late_gated", "stack")
    per = {k: [] for k in names}
    preds = {k: [] for k in names}
    for s in SEEDS[:a.seeds]:
        p_url = fit_proba(Xurl[tr], y[tr], Xurl[te], s)
        p_txt = fit_proba(Xtxt[tr], y[tr], Xtxt[te], s)
        _, p_fus = fit_score(Xfus[tr], y[tr], Xfus[te], y[te], s)
        _, p_flag = fit_score(Xflag[tr], y[tr], Xflag[te], y[te], s)
        mean = (p_url + p_txt) / 2
        gated = np.where(has_url[te], mean, p_txt)
        o_url, o_txt = oof_proba(Xurl[tr], y[tr], s), oof_proba(Xtxt[tr], y[tr], s)
        Ztr = np.column_stack([o_url, o_txt, has_url[tr]])
        Zte = np.column_stack([p_url, p_txt, has_url[te]])
        stack = LogisticRegression(max_iter=2000).fit(Ztr, y[tr]).predict(Zte)
        out = {"text": (p_txt >= 0.5).astype(int), "fusion": p_fus, "early_flag": p_flag,
               "late_mean": (mean >= 0.5).astype(int), "late_gated": (gated >= 0.5).astype(int),
               "stack": stack}
        for k, p in out.items():
            per[k].append(metrics(y[te], p)); preds[k].append(p)
        print(f"    seed {s}: " + "  ".join(f"{k}={per[k][-1]['f1']:.3f}" for k in names), flush=True)

    groups = test_groups(rows, te)
    res = {"seeds": a.seeds, "n_test": int(te.sum()), "n_test_no_url": int((~has_url[te]).sum()),
           "arms": {k: summarise(v) for k, v in per.items()},
           "cluster_bootstrap": {f"{k}_minus_text": cluster_bootstrap_f1(y[te], preds[k], preds["text"], groups)
                                 for k in names if k != "text"},
           "has_url_only": {k: round(float(np.mean([f1_score(y[te][has_url[te]], p[has_url[te]], zero_division=0)
                                                    for p in preds[k]])), 4) for k in names},
           "status": "post-hoc, unregistered; the registered fusion is the concatenation MLP"}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2, sort_keys=True)
    for k in names:
        print(f"  {k:<11} F1 {res['arms'][k]['f1']:.3f}  has-URL rows {res['has_url_only'][k]:.3f}")
    for k, z in res["cluster_bootstrap"].items():
        print(f"  {k:<24} {z['mean']:+.4f} [{z['ci95'][0]:+.4f}, {z['ci95'][1]:+.4f}]")
    print(f"  [+] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
