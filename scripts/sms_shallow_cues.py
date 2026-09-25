#!/usr/bin/env python3
"""
sms_shallow_cues.py — how much of this corpus separates without reading any Vietnamese.

WHY THIS EXISTS. The smishing study's headline is that the separating signal is the message
wording rather than the URL: text-only reaches 0.929 phishing-class F1 against 0.607 for a URL
model on the same split. That reading is only safe if the "wording" is the sender's. Two things in
this corpus are not:

  * THE REDACTOR'S TOKENS. The publisher shipped the corpus pre-redacted, with bracketed tokens
    ([MONEY], [DATE], [QC], ...). If those tokens are distributed unevenly across the classes, a
    text model can separate the classes by counting the publisher's preprocessing and never read a
    word. This script fits a logistic regression on TOKEN PRESENCE ALONE -- no characters, no
    embeddings, no URL -- on the study's own split, and reports the phishing-class F1 it reaches.
    That number is a FLOOR under every text arm in the paper.

  * DIACRITICS. The paper's qualitative section claimed attackers overwhelmingly write unaccented
    Vietnamese to save GSM-7 budget and evade diacritic-sensitive filters. Whether that is true of
    THIS corpus is a two-line count, and it was never run before 2026-08-31. It is run here.

Both are POST-HOC and UNREGISTERED. `PREREG_smishing.md` registered the fusion contrasts and its
diagnostics before anything was fitted; neither of these is among them, and the paper says so where
it reports them. They are descriptive, and they qualify a result rather than establishing one.

The split is the study's own, read from the `split` column sms_corpus_import.py wrote (keyed on a
hash of the message text, so every copy of a repeated message lands on the same side). Nothing here
re-splits, re-labels or re-reads the raw corpus: it uses the derived columns, which is why it costs
a second and can run in `make claims`.

RUN
    python3 scripts/sms_shallow_cues.py
    python3 scripts/sms_shallow_cues.py --json data/processed/sms/shallow_cues.json
"""
from __future__ import annotations
import argparse, collections, csv, json, os, re, statistics, sys

MSG = os.path.join("data", "processed", "sms", "sms_messages.csv")
OUT = os.path.join("data", "processed", "sms", "shallow_cues.json")
SEED = 0


RAW = os.path.join("data", "raw", "sms_hf_full", "full_dataset.csv")
TOKRX = re.compile(r"\[[A-Z_]+\]")
DUP_THRESHOLDS = (0.95, 0.90, 0.80)
FAMILY_SIM = 0.80


def join_raw(proc: list, raw_path: str) -> list:
    """Attach the message text to the derived rows, keyed on message_id.

    The processed CSV deliberately carries a hash and not the text (a corpus under CC-BY that has
    been redacted once should not be re-published row by row through this repository), so anything
    about the LANGUAGE has to come back to the raw file, and this returns [] when it is absent
    rather than silently analysing nothing."""
    if not os.path.isfile(raw_path):
        return []
    with open(raw_path, newline="", encoding="utf-8-sig") as f:
        raw = {r["message_id"]: r["message"] for r in csv.DictReader(f)}
    out = [dict(r, text=raw[r["message_id"]]) for r in proc if r["message_id"] in raw]
    return out if len(out) == len(proc) else out


def tfidf_baselines(train: list, test: list) -> dict:
    """What a linear model on n-grams reaches, with and without the redactor's tokens.

    This is the baseline the paper's text arm has to beat to justify a transformer, and the
    tokens-stripped variant is what says whether the redaction confound explains the text result or
    merely sits underneath it."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    ytr = [int(r["label"]) for r in train]
    yte = [int(r["label"]) for r in test]
    out = {}
    grids = (("word", dict(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True), False),
             ("char", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), False),
             ("word_no_tokens", dict(analyzer="word", ngram_range=(1, 2), min_df=2,
                                     sublinear_tf=True), True))
    for name, kw, strip in grids:
        prep = (lambda t: TOKRX.sub(" ", t)) if strip else (lambda t: t)
        v = TfidfVectorizer(**kw)
        A = v.fit_transform([prep(r["text"]) for r in train])
        B = v.transform([prep(r["text"]) for r in test])
        m = LogisticRegression(max_iter=3000, C=4, random_state=SEED).fit(A, ytr)
        out[name] = round(float(f1_score(yte, m.predict(B))), 4)
    return out


def duplicate_leak(train: list, test: list) -> dict:
    """Near-duplicate leakage across the study's own split.

    The paper rebuilt the publisher's split because 47 test rows repeated a training message
    EXACTLY. A hash of the text cannot see a message that repeats one with a different amount or a
    different date, and those are the same object for a model. Cosine over character n-grams can,
    so it is measured here rather than assumed away."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    A = v.fit_transform([r["text"] for r in train])
    B = v.transform([r["text"] for r in test])
    mx = cosine_similarity(B, A).max(axis=1)
    out = {}
    for thr in DUP_THRESHOLDS:
        out[f"{thr:.2f}"] = {}
        for lab in ("0", "1"):
            idx = [i for i, r in enumerate(test) if r["label"] == lab]
            n = int(sum(1 for i in idx if mx[i] >= thr))
            out[f"{thr:.2f}"][lab] = {"n": n, "of": len(idx),
                                      "pct": round(100 * n / max(len(idx), 1), 1)}
    return out


def families(msgs: list) -> dict:
    """Template families: messages joined when they are 80% similar, counted as components.

    A corpus of 800 phishing messages that is really 80 templates has an effective n of 80, and
    every interval in the paper would be wrong by the ratio."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(msgs)
    n, lab = connected_components(csr_matrix(cosine_similarity(X) >= FAMILY_SIM), directed=False)
    sizes = collections.Counter(lab.tolist())
    return {"messages": len(msgs), "families": int(n),
            "pct": round(100 * n / max(len(msgs), 1), 1),
            "singletons": int(sum(1 for v_ in sizes.values() if v_ == 1)),
            "largest": sorted(sizes.values(), reverse=True)[:5]}


def log_odds(rows: list, top: int = 12) -> dict:
    """Which words mark which class: log-odds ratio with an informative Dirichlet prior.

    Raw frequency ranks stopwords; a plain ratio ranks hapaxes. The informative prior (Monroe et
    al.) is what makes a 3,000-message corpus rankable at all, and the z-score it carries is what
    stops a word appearing three times from leading the table."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(analyzer="word", ngram_range=(1, 1), min_df=3,
                         token_pattern=r"(?u)\b\w[\w\-\.]+\b")
    X = cv.fit_transform([r["text"].lower() for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    vocab = np.array(cv.get_feature_names_out())
    c1 = np.asarray(X[y == 1].sum(0)).ravel().astype(float)
    c0 = np.asarray(X[y == 0].sum(0)).ravel().astype(float)
    a = (c1 + c0) * 0.01                      # prior at 1% of the observed counts
    n1, n0, A = c1.sum(), c0.sum(), a.sum()
    d = (np.log((c1 + a) / (n1 + A - c1 - a))
         - np.log((c0 + a) / (n0 + A - c0 - a)))
    z = d / np.sqrt(1.0 / (c1 + a) + 1.0 / (c0 + a))
    order = np.argsort(z)
    pick = lambda idx: [{"term": str(vocab[i]), "z": round(float(z[i]), 1),
                         "phish": int(c1[i]), "ham": int(c0[i])} for i in idx]
    return {"phishing": pick(order[::-1][:top]), "ham": pick(order[:top])}


def rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_f1(train, test, featf) -> dict:
    """Logistic regression on whatever `featf` returns, scored on the phishing class.

    Deterministic: lbfgs on a fixed design over a fixed split. A shallow-cue baseline that moved
    between runs could not be quoted in a paper as a floor."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    Xtr = np.array([featf(r) for r in train], dtype=float)
    ytr = np.array([int(r["label"]) for r in train])
    Xte = np.array([featf(r) for r in test], dtype=float)
    yte = np.array([int(r["label"]) for r in test])
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"f1": round(float(f1_score(yte, p)), 4),
            "precision": round(float(precision_score(yte, p, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, p)), 4),
            "n_train": len(ytr), "n_test": len(yte)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", default=MSG)
    ap.add_argument("--json", default=OUT)
    a = ap.parse_args()
    if not os.path.isfile(a.msgs):
        print(f"[!] {a.msgs} missing — run scripts/sms_corpus_import.py", file=sys.stderr)
        return 1
    rs = rows(a.msgs)
    tot = collections.Counter(r["label"] for r in rs)
    labs = sorted(tot)

    # --- diacritics ---------------------------------------------------------------------------
    acc = collections.Counter()
    for r in rs:
        acc[r["label"]] += int(r["accented"] or 0)
    accented = {l: round(100 * acc[l] / tot[l], 1) for l in labs}

    # --- the redactor's tokens ----------------------------------------------------------------
    tok = {l: collections.Counter() for l in labs}
    any_tok = collections.Counter()
    for r in rs:
        ts = set((r["pii_tokens"] or "").split())
        any_tok[r["label"]] += bool(ts)
        for t in ts:
            tok[r["label"]][t] += 1
    names = sorted({t for l in labs for t in tok[l]},
                   key=lambda t: -sum(tok[l][t] for l in labs))
    per_token = {t: {l: round(100 * tok[l][t] / tot[l], 1) for l in labs} for t in names}

    # --- length -------------------------------------------------------------------------------
    lens = collections.defaultdict(list)
    for r in rs:
        lens[r["label"]].append(int(r["chars"] or 0))

    # --- the baselines that read no Vietnamese -------------------------------------------------
    train = [r for r in rs if r["split"] == "train"]
    test = [r for r in rs if r["split"] == "test"]
    f_tok = lambda r: [1.0 if t in set((r["pii_tokens"] or "").split()) else 0.0 for t in names]
    f_len = lambda r: [int(r["chars"] or 0) / 100.0]
    f_acc = lambda r: [float(int(r["accented"] or 0))]
    base = {"tokens": fit_f1(train, test, f_tok),
            "tokens_and_length": fit_f1(train, test, lambda r: f_tok(r) + f_len(r)),
            "length": fit_f1(train, test, f_len),
            "accent_and_length": fit_f1(train, test, lambda r: f_acc(r) + f_len(r)),
            # the floor that matters: everything about a message that is not its language
            "all_shallow": fit_f1(train, test, lambda r: f_acc(r) + f_len(r) + f_tok(r))}

    # --- the language layer: needs the raw text, and says so when it is not there ------------
    lang = {}
    with_text = join_raw(rs, RAW)
    if with_text:
        tr_t = [r for r in with_text if r["split"] == "train"]
        te_t = [r for r in with_text if r["split"] == "test"]
        lang = {"tfidf": tfidf_baselines(tr_t, te_t),
                "near_duplicate_leak": duplicate_leak(tr_t, te_t),
                "families": {l: families([r["text"] for r in with_text if r["label"] == l])
                             for l in labs},
                "log_odds": log_odds(with_text),
                "n_joined": len(with_text)}
    else:
        print(f"[!] {RAW} missing — the language layer is skipped, not silently empty",
              file=sys.stderr)

    out = {"n": dict(tot), "accented_pct": accented, "language": lang,
           "any_token_pct": {l: round(100 * any_tok[l] / tot[l], 1) for l in labs},
           "token_pct": per_token,
           "median_chars": {l: int(statistics.median(lens[l])) for l in labs},
           "baselines": base, "split": {"train": len(train), "test": len(test)},
           "registered": False, "seed": SEED}
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=2, sort_keys=True)

    print("=== diacritics: messages carrying at least one accented character ===")
    for l in labs:
        print(f"  {'ham' if l == '0' else 'phishing':9} {accented[l]:5.1f}%   ({acc[l]:,}/{tot[l]:,})")
    print("  The unaccented half of this corpus is the LEGITIMATE half; bulk marketing pays per\n"
          "  GSM-7 segment, and an attacker sending hundreds of messages does not.")
    print("\n=== the publisher's redaction tokens, share of each class ===")
    print(f"  {'token':12} {'ham':>7} {'phish':>7}   gap")
    print(f"  {'ANY':12} {out['any_token_pct']['0']:6.1f}% {out['any_token_pct']['1']:6.1f}%  "
          f"{out['any_token_pct']['1'] - out['any_token_pct']['0']:+6.1f}pp")
    for t in names[:8]:
        h, p = per_token[t]["0"], per_token[t]["1"]
        print(f"  {t:12} {h:6.1f}% {p:6.1f}%  {p - h:+6.1f}pp")
    print("\n=== what separates WITHOUT reading a single Vietnamese word ===")
    for k, v in base.items():
        print(f"  {k:20} F1(phishing) {v['f1']:.3f}   P {v['precision']:.3f}  R {v['recall']:.3f}")
    print("  POST-HOC and UNREGISTERED. The `all_shallow` row is the floor under every text arm in\n"
          "  the paper, and the paper reports it as one.")
    if lang:
        t = lang["tfidf"]
        print("\n=== what a LINEAR model on n-grams reaches (does this corpus need a transformer?) ===")
        print(f"  word 1-2gram TF-IDF        {t['word']:.3f}")
        print(f"  char 3-5gram TF-IDF        {t['char']:.3f}")
        print(f"  word TF-IDF, tokens gone   {t['word_no_tokens']:.3f}   <- language only")
        print("  Compare the paper's PhoBERT arm. A transformer that does not beat this bought "
              "nothing here.")
        print("\n=== near-duplicate leakage in OUR OWN split (the publisher's was exact-text) ===")
        for thr, v in lang["near_duplicate_leak"].items():
            print(f"  sim>={thr}  ham {v['0']['pct']:5.1f}% ({v['0']['n']}/{v['0']['of']})   "
                  f"phishing {v['1']['pct']:5.1f}% ({v['1']['n']}/{v['1']['of']})")
        print("\n=== template families (cosine >= %.2f) ===" % FAMILY_SIM)
        for l in labs:
            f = lang["families"][l]
            print(f"  {'ham' if l == '0' else 'phishing':9} {f['messages']:4} messages -> "
                  f"{f['families']:4} families ({f['pct']:.1f}%), largest {f['largest']}")
        print("\n=== what the wording actually is (log-odds, informative Dirichlet prior) ===")
        for side in ("phishing", "ham"):
            print(f"  {side}:")
            for w in lang["log_odds"][side][:8]:
                print(f"    {w['term']:18} z={w['z']:6.1f}  (phish {w['phish']:4}, ham {w['ham']:4})")
    print(f"\n[+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
