#!/usr/bin/env python3
"""Post-hoc, NOT registered: how far the label follows the collection batch.

WHY THIS EXISTS. The publisher's `date` column behaves as a batch (entry) date, and the class mix
is not spread across batches: two batches, 2026-07-22 and 2026-07-24, hold 528 of the 798
positive-labelled rows and one ham row between them. A random split, even one keyed on the text,
puts rows of those batches on both sides, so a model can score well by recognising what is
specific to them. This script measures that and the related confounds a referee raised on
2026-10-08:

  * batch x class, and how many rows sit in batches that are >= 95% one class
  * the diacritic rate inside and outside the two positive-only batches
  * a leave-batches-out test: fit without the two batches, score on their positive rows, against a
    control that holds out the same number of positive texts at random
  * pure batches -> mixed batches
  * the ceiling the URL arm had on the registered split (positives with no URL score zero)
  * test rows whose text differs from a training text only in whitespace, case or punctuation
  * message_id prefix x class

Every fit here is a linear probe (word 1-2-gram TF-IDF, logistic regression, C = 4, as in
sms_shallow_cues.py) or the registered text arm's MLP on the cached frozen PhoBERT embeddings.
Nothing is tuned. Descriptive; reported as such.

Reads   data/processed/sms/sms_messages.csv, data/raw/sms_hf_full/full_dataset.csv,
        data/processed/sms/phobert_emb.npy (+ .meta.json), data/processed/sms/fusion_results.json
Writes  data/processed/sms/batch_audit.json

RUN:  OPENBLAS_NUM_THREADS=1 python3 scripts/sms_batch_audit.py
"""
from __future__ import annotations
import collections, csv, json, os, re, sys, unicodedata, warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402

MSG = os.path.join(ROOT, "data", "processed", "sms", "sms_messages.csv")
RAW = os.path.join(ROOT, "data", "raw", "sms_hf_full", "full_dataset.csv")
EMB = os.path.join(ROOT, "data", "processed", "sms", "phobert_emb.npy")
FUS = os.path.join(ROOT, "data", "processed", "sms", "fusion_results.json")
OUT = os.path.join(ROOT, "data", "processed", "sms", "batch_audit.json")
POS_BATCHES = ("22/07/2026", "24/07/2026")
PURE = 0.95
CONTROL_SEEDS = range(5)


def probe():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                         sublinear_tf=True),
                         LogisticRegression(max_iter=3000, C=4, random_state=0))


def text_arm(Xtr, ytr, seed=0):
    """The registered text arm's head, on the cached frozen embeddings."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=seed,
                                       early_stopping=True, n_iter_no_change=15)).fit(Xtr, ytr)


def norm_ws(t):
    return " ".join(t.split())


def norm_loose(t):
    t = unicodedata.normalize("NFC", t).lower()
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def main() -> int:
    from sklearn.metrics import f1_score, precision_score, recall_score
    raw = {r["message_id"]: r["message"] for r in
           csv.DictReader(open(RAW, newline="", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(MSG, newline="", encoding="utf-8")))
    texts = np.array([raw[r["message_id"]] for r in rows], dtype=object)
    y = np.array([int(r["label"]) for r in rows])
    date = np.array([r["date"] for r in rows])
    sha = np.array([r["text_sha1"] for r in rows])
    split = np.array([r["split"] for r in rows])
    acc = np.array([int(r["accented"]) for r in rows])
    out: dict = {"registered": False, "status": "post-hoc, descriptive"}

    # --- batch x class
    tab = collections.defaultdict(lambda: [0, 0])
    for d, l in zip(date, y):
        tab[d or "(undated)"][l] += 1
    batches = {d: {"ham": v[0], "positive": v[1], "pos_share": round(v[1] / sum(v), 3)}
               for d, v in tab.items()}
    pure = {d for d, v in batches.items() if max(v["pos_share"], 1 - v["pos_share"]) >= PURE}
    out["batches"] = batches
    out["n_batches"] = len(batches)
    out["rows_in_pure_batches"] = int(sum(batches[d]["ham"] + batches[d]["positive"] for d in pure))
    out["pure_threshold"] = PURE
    inb = np.isin(date, POS_BATCHES)
    out["positive_only_batches"] = {
        "dates": list(POS_BATCHES), "positive": int((inb & (y == 1)).sum()),
        "ham": int((inb & (y == 0)).sum()), "positive_total": int((y == 1).sum())}

    # --- diacritics inside / outside the two batches
    out["accented_pct"] = {
        "positive_in_batches": round(100 * acc[inb & (y == 1)].mean(), 1),
        "positive_outside": round(100 * acc[~inb & (y == 1)].mean(), 1),
        "positive_outside_n": int((~inb & (y == 1)).sum()),
        "ham": round(100 * acc[y == 0].mean(), 1),
        "positive_all": round(100 * acc[y == 1].mean(), 1)}

    # --- leave-batches-out, unit = distinct text so no copy of a held-out text is trained on
    held_txt = set(sha[inb & (y == 1)])
    held = np.isin(sha, list(held_txt)) & (y == 1)
    train = ~np.isin(sha, list(held_txt)) & ~inb
    first = {}
    for i, h in enumerate(sha):
        first.setdefault(h, i)
    held_rows = [first[h] for h in held_txt]      # one row per distinct held-out text
    m = probe().fit(texts[train], y[train])
    lbo_probe = float(m.predict(texts[held_rows]).mean())
    emb = np.load(EMB)
    lbo_arm = float(np.mean([text_arm(emb[train], y[train], s).predict(emb[held_rows]).mean()
                             for s in range(3)]))
    pos_txt = sorted(set(sha[y == 1]))
    ctrl_probe, ctrl_arm = [], []
    for s in CONTROL_SEEDS:
        pick = set(np.random.default_rng(s).choice(pos_txt, len(held_txt), replace=False))
        tr_c = ~np.isin(sha, list(pick))
        rows_c = [first[h] for h in pick]
        ctrl_probe.append(float(probe().fit(texts[tr_c], y[tr_c]).predict(texts[rows_c]).mean()))
        ctrl_arm.append(float(text_arm(emb[tr_c], y[tr_c], 0).predict(emb[rows_c]).mean()))
    out["leave_batches_out"] = {
        "held_out_texts": len(held_txt), "train_rows": int(train.sum()),
        "train_positive_rows": int((train & (y == 1)).sum()),
        "probe_recall": round(lbo_probe, 3), "text_arm_recall": round(lbo_arm, 3),
        "control_probe_recall": [round(v, 3) for v in ctrl_probe],
        "control_text_arm_recall": [round(v, 3) for v in ctrl_arm],
        "control": "same number of positive texts held out at random, 5 seeds",
        "unit": "distinct text"}

    # --- pure batches -> mixed batches
    is_pure = np.isin(date, list(pure))
    mixed_txt = set(sha[~is_pure])
    tr_p = is_pure & ~np.isin(sha, list(mixed_txt))
    te_m = ~is_pure
    p = probe().fit(texts[tr_p], y[tr_p]).predict(texts[te_m])
    out["pure_to_mixed"] = {
        "train_rows": int(tr_p.sum()), "test_rows": int(te_m.sum()),
        "test_positive": int(y[te_m].sum()),
        "f1": round(float(f1_score(y[te_m], p)), 3),
        "precision": round(float(precision_score(y[te_m], p)), 3),
        "recall": round(float(recall_score(y[te_m], p)), 3)}

    # --- the URL arm's ceiling on the registered split
    fr = json.load(open(FUS, encoding="utf-8"))
    d = fr["diagnostics"]
    pos_url, pos_all = d["url/has_url"]["n_pos"], d["url/has_url"]["n_pos"] + d["url/no_url"]["n_pos"]
    r_max = pos_url / pos_all
    out["url_ceiling"] = {"test_positive": pos_all, "test_positive_with_url": pos_url,
                          "test_positive_no_url_pct": round(100 * (1 - r_max), 1),
                          "max_recall": round(r_max, 3),
                          "max_f1": round(2 * r_max / (1 + r_max), 3),
                          "note": "perfect precision, every URL-bearing positive caught"}

    # --- twins the text hash cannot see
    tr_m, te_m2 = split == "train", split == "test"
    twins = {}
    for name, f in (("whitespace", norm_ws), ("case_punct", norm_loose)):
        trn = {f(t) for t in texts[tr_m]}
        hit = [i for i in np.where(te_m2)[0] if f(texts[i]) in trn]
        twins[name] = {"test_rows": len(hit), "ham": int((y[hit] == 0).sum()),
                       "positive": int((y[hit] == 1).sum())}
    groups = collections.defaultdict(set)
    for t, l in zip(texts, y):
        groups[norm_ws(t)].add(int(l))
    twins["cross_label_whitespace_groups"] = sum(1 for v in groups.values() if len(v) == 2)
    out["twins"] = twins

    # --- August-dated rows on each side of the registered split (batch check 1 trains on some)
    aug = np.array([d.endswith("/8/2026") or d.endswith("/08/2026") for d in date])
    out["august_rows"] = {"train": int((aug & tr_m).sum()), "test": int((aug & te_m2).sum())}

    # --- message_id prefix
    pre = collections.defaultdict(lambda: [0, 0])
    for r, l in zip(rows, y):
        pre[r["message_id"].split("_")[0]][l] += 1
    out["id_prefix"] = {k: {"ham": v[0], "positive": v[1]} for k, v in sorted(pre.items())}

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"  {out['rows_in_pure_batches']} of {len(rows)} rows in batches >= {PURE:.0%} one class")
    pb = out["positive_only_batches"]
    print(f"  {', '.join(POS_BATCHES)}: {pb['positive']} of {pb['positive_total']} positive rows, "
          f"{pb['ham']} ham")
    a = out["accented_pct"]
    print(f"  accented: positive in batches {a['positive_in_batches']}%, outside "
          f"{a['positive_outside']}% (n={a['positive_outside_n']}), ham {a['ham']}%")
    lb = out["leave_batches_out"]
    print(f"  leave-batches-out recall: probe {lb['probe_recall']}, text arm {lb['text_arm_recall']} "
          f"| random control probe {lb['control_probe_recall']}, arm {lb['control_text_arm_recall']}")
    print(f"  pure -> mixed: {out['pure_to_mixed']}")
    print(f"  URL ceiling: {out['url_ceiling']}")
    print(f"  twins: {twins}")
    print(f"  id prefix: {out['id_prefix']}")
    print(f"  [+] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
