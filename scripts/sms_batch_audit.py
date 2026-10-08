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


SIM = 0.90


def components(txts):
    """Near-duplicate components over distinct texts, as in the similarity-grouped split."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                        sublinear_tf=True).fit_transform(txts)
    return connected_components(csr_matrix(cosine_similarity(X) >= SIM), directed=False)[1]


def scores(yev, sc):
    """Recall at 0.5, ROC-AUC, and recall at the threshold that flags 2% of the evaluation ham."""
    from sklearn.metrics import roc_auc_score
    thr = np.quantile(sc[yev == 0], 0.98)
    return {"recall_at_0.5": round(float((sc[yev == 1] >= 0.5).mean()), 3),
            "roc_auc": round(float(roc_auc_score(yev, sc)), 3),
            "recall_at_2pct_fpr": round(float((sc[yev == 1] > thr).mean()), 3)}


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

    # --- leave-batches-out, matched and threshold-free (2026-10-08, second revision)
    # Unit: near-duplicate component (char 3-5-gram TF-IDF, cosine >= SIM), so no held-out text
    # has a near-twin in training. A fixed evaluation set of ham components (20%) is held out of
    # every fit, so each condition is scored by ROC-AUC and by recall at a matched 2% ham
    # false-positive rate as well as at the default 0.5 threshold, which mixes ranking with the
    # training class share. The control holds out random positive components until it trains on
    # the same number of positive rows as the batch hold-out.
    first = {}
    for i, h in enumerate(sha):
        first.setdefault(h, i)
    utxt = sorted(first)
    comp = components([texts[first[h]] for h in utxt])
    comp_of = dict(zip(utxt, comp))
    lab_of = {h: int(y[first[h]]) for h in utxt}
    rng0 = np.random.default_rng(0)
    ham_comps = sorted({comp_of[h] for h in utxt if lab_of[h] == 0})
    eval_ham_comps = set(rng0.choice(ham_comps, int(0.2 * len(ham_comps)), replace=False).tolist())
    eval_ham = [h for h in utxt if lab_of[h] == 0 and comp_of[h] in eval_ham_comps]
    emb = np.load(EMB)
    row_comp = np.array([comp_of[h] for h in sha])

    def run(held_pos, extra_excl_rows=None, arm_seeds=(0, 1, 2)):
        """Fit without the components of held_pos and of the ham evaluation set; score them."""
        held_c = {comp_of[h] for h in held_pos} | eval_ham_comps
        tr = ~np.isin(row_comp, list(held_c))
        if extra_excl_rows is not None:
            tr &= ~extra_excl_rows
        pos_eval = [h for h in held_pos if lab_of[h] == 1]
        ev = [first[h] for h in pos_eval] + [first[h] for h in eval_ham]
        yev = np.array([1] * len(pos_eval) + [0] * len(eval_ham))
        res = {"train_positive_rows": int((tr & (y == 1)).sum()), "train_rows": int(tr.sum()),
               "eval_positive_texts": len(pos_eval), "eval_ham_texts": len(eval_ham)}
        sc_probe = probe().fit(texts[tr], y[tr]).predict_proba(texts[ev])[:, 1]
        res["probe"] = scores(yev, sc_probe)
        arm = [scores(yev, text_arm(emb[tr], y[tr], s_).predict_proba(emb[ev])[:, 1])
               for s_ in arm_seeds]
        res["text_arm"] = {k: round(float(np.mean([a[k] for a in arm])), 3) for k in arm[0]}
        return res

    held_txt = sorted({h for h in sha[inb & (y == 1)]})
    batch = run(held_txt, extra_excl_rows=inb)
    target = batch["train_positive_rows"]
    pos_comps = sorted({comp_of[h] for h in utxt if lab_of[h] == 1} - eval_ham_comps)
    controls = []
    for s_ in CONTROL_SEEDS:
        order = np.random.default_rng(100 + s_).permutation(pos_comps)
        picked, held_c = [], set()
        for c in order:
            held_c.add(c)
            ntr = int((~np.isin(row_comp, list(held_c | eval_ham_comps)) & (y == 1)).sum())
            if ntr <= target:
                break
        hp = [h for h in utxt if comp_of[h] in held_c and lab_of[h] == 1]
        controls.append(run(hp))
    def span(key, metric):
        v = [c[key][metric] for c in controls]
        return [round(min(v), 3), round(max(v), 3)]
    out["leave_batches_out"] = {
        "unit": f"near-duplicate component (char 3-5-gram TF-IDF, cosine >= {SIM})",
        "eval_ham_texts": len(eval_ham), "batch": batch, "controls": controls,
        "control_span": {k: {m_: span(k, m_) for m_ in ("recall_at_0.5", "roc_auc", "recall_at_2pct_fpr")}
                         for k in ("probe", "text_arm")},
        "control_train_positive_rows": [c["train_positive_rows"] for c in controls]}

    # --- the reverse direction: trained on the two batches' positives only (plus ham)
    other_pos = sorted({h for h in sha[~inb & (y == 1)]} - set(held_txt))
    tr_rev = ~np.isin(row_comp, list({comp_of[h] for h in other_pos} | eval_ham_comps))
    ev = [first[h] for h in other_pos] + [first[h] for h in eval_ham]
    yev = np.array([1] * len(other_pos) + [0] * len(eval_ham))
    out["reverse"] = {"train_positive_rows": int((tr_rev & (y == 1)).sum()),
                      "eval_positive_texts": len(other_pos),
                      "probe": scores(yev, probe().fit(texts[tr_rev], y[tr_rev])
                                      .predict_proba(texts[ev])[:, 1])}

    # --- every batch held out in turn (texts assigned to the batch of their first row), against
    # a five-fold split grouped by near-duplicate component; pooled out-of-fold scores
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    T = np.array([texts[first[h]] for h in utxt], dtype=object)
    Y = np.array([lab_of[h] for h in utxt])
    B = np.array([date[first[h]] or "(undated)" for h in utxt])
    C = np.array([comp_of[h] for h in utxt])
    def pooled(groups, k):
        oof = np.zeros(len(Y))
        for a_, b_ in GroupKFold(k).split(T, Y, groups):
            oof[b_] = probe().fit(T[a_], Y[a_]).predict_proba(T[b_])[:, 1]
        return {"roc_auc": round(float(roc_auc_score(Y, oof)), 3),
                "f1_at_0.5": round(float(f1_score(Y, (oof >= 0.5).astype(int))), 3)}
    out["leave_one_batch_out"] = {"by_batch": pooled(B, len(set(B))),
                                  "by_component_5fold": pooled(C, 5),
                                  "batches": len(set(B)), "unit": "distinct text"}

    # --- positives without a URL: how many sit in the two batches (registered extractor)
    nurl = np.array([int(r["n_urls"]) for r in rows])
    out["no_url_positive"] = {"total": int(((y == 1) & (nurl == 0)).sum()),
                              "in_two_batches": int(((y == 1) & (nurl == 0) & inb).sum())}

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
    print(f"  leave-batches-out: batch {lb['batch']}")
    print(f"    control span {lb['control_span']}, train pos rows {lb['control_train_positive_rows']}")
    print(f"  reverse: {out['reverse']}")
    print(f"  leave-one-batch-out: {out['leave_one_batch_out']}")
    print(f"  no-URL positives: {out['no_url_positive']}")
    print(f"  pure -> mixed: {out['pure_to_mixed']}")
    print(f"  URL ceiling: {out['url_ceiling']}")
    print(f"  twins: {twins}")
    print(f"  id prefix: {out['id_prefix']}")
    print(f"  [+] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
