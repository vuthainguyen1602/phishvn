#!/usr/bin/env python3
"""Post-hoc diagnostic, NOT registered: why does the URL channel add nothing?

The fusion arm came out 0.002 below text-only descriptively, and the question that
leaves is why 21 URL features add nothing at all. The hypothesis tested here was that they are
redundant because PhoBERT reads the whole message including the URL, so the CompPhish columns are
derived from a substring the encoder already saw.

That hypothesis is WRONG, and the result is stronger than it. Masking the URL out of the message
before embedding changes the point estimate by only +0.006. Adding the URL features back on top
of the masked text changes it by another +0.006. Sampling intervals are clustered by distinct
text; optimizer seeds on one fixed holdout are not treated as independent data replicates.
The separating signal on this corpus is in the non-URL wording, and the URL is close to inert
whether it arrives as characters or as engineered features.

Reported in the paper as post-hoc and labelled so. It is not one of the two registered contrasts
and must never be presented as a third.

RUN:  python3 scripts/diagnose_sms_channels.py
"""
import csv, os, sys, warnings, json
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np
from compphish_features import extract
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from train_sms_fusion import (COMPPHISH, URL_RE, first_url, SRC, MSG, embed,
                              cluster_bootstrap_f1)

raw = {r["message_id"]: r["message"] for r in csv.DictReader(open(SRC, newline="", encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(MSG, newline="", encoding="utf-8")))
texts = [raw.get(r["message_id"], "") for r in rows]
y = np.array([int(r["label"]) for r in rows]); split = np.array([r["split"] for r in rows])
masked = [URL_RE.sub(" <URL> ", t) for t in texts]
print(f"  che URL: {sum(1 for a,b in zip(texts,masked) if a!=b)} tin đổi")

Xfull = embed(texts, os.path.join(ROOT, "data/processed/sms/phobert_emb.npy"))
Xmask = embed(masked, os.path.join(ROOT, "data", "processed", "sms", "phobert_emb_masked.npy"))
Xurl = np.zeros((len(rows), len(COMPPHISH)), dtype=np.float32)
for i, t in enumerate(texts):
    u = first_url(t)
    if u:
        f = extract(u); Xurl[i] = [float(f.get(c,0) or 0) for c in COMPPHISH]

tr, te = split=="train", split=="test"
arms = {"text_full": Xfull, "text_masked": Xmask,
        "masked+url": np.hstack([Xmask, Xurl])}
res = {k: [] for k in arms}
preds = {k: [] for k in arms}
for s in range(10):
    for k, X in arms.items():
        sc = StandardScaler().fit(X[tr])
        c = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=s,
                          early_stopping=True, n_iter_no_change=15).fit(sc.transform(X[tr]), y[tr])
        p = c.predict(sc.transform(X[te])); preds[k].append(p)
        res[k].append(f1_score(y[te], p, pos_label=1, zero_division=0))
f = {k: np.array(v) for k, v in res.items()}
for k in arms: print(f"  {k:<13} F1 {f[k].mean():.3f}  (sd {f[k].std(ddof=1):.3f})")
groups = [rows[i]["text_sha1"] for i in np.where(te)[0]]
d1 = cluster_bootstrap_f1(y[te], preds["text_masked"], preds["text_full"], groups)
d2 = cluster_bootstrap_f1(y[te], preds["masked+url"], preds["text_masked"], groups)
print(f"  che URL khỏi text : {d1['mean']:+.4f}, CI [{d1['ci95'][0]:+.4f}, {d1['ci95'][1]:+.4f}]")
print(f"  thêm URL vào text đã che: {d2['mean']:+.4f}, CI [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}]")
json.dump({"text_full": float(f["text_full"].mean()), "text_masked": float(f["text_masked"].mean()),
           "masked_plus_url": float(f["masked+url"].mean()),
           "mask_effect": d1, "url_adds_to_masked": d2},
          open(os.path.join(ROOT, "data/processed/sms/why_fusion.json"), "w"), indent=2, default=float)
