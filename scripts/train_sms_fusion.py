#!/usr/bin/env python3
"""The registered comparison: does SMS text add anything to a URL-only model?

Runs the two confirmatory contrasts in papers/future_smishing/PREREG_smishing.md and nothing else.

  T1  fusion vs URL-only   — success needs >= 3 F1 points; under 1 point is the registered negative
  T2  text-only vs URL-only — the prediction is URL-only ahead by >= 5 points

DESIGN, as registered and not re-decided here. Split by SHA-1 of the message text, so every copy of
a repeated message lands on one side; 10 seeds; positive-class F1 on the test rows. The registered
Nadeau-Bengio test over optimization seeds is retained as an audit record but is NOT valid
inference: the split is fixed, so these are not repeated train/test samples. A post-hoc
distinct-text cluster bootstrap supplies sampling intervals instead. URL channel is the 21 CompPhish columns the web studies model,
is_https excluded there and excluded here. Text channel is PhoBERT base, mean-pooled over the last
hidden state, NOT fine-tuned. Fusion concatenates the two into R^789 and fits one MLP head.

Messages with no URL get a zero URL vector; they are a registered diagnostic, reported
separately, because the URL arm has nothing to read for them and pooling them hides that.

RUN:  python3 scripts/train_sms_fusion.py
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys, warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from compphish_features import extract  # noqa: E402
from paired_eval import corrected_paired_t  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SRC = os.path.join(ROOT, "data", "raw", "sms_hf_full", "full_dataset.csv")
MSG = os.path.join(ROOT, "data", "processed", "sms", "sms_messages.csv")
OUT = os.path.join(ROOT, "data", "processed", "sms", "fusion_results.json")
EMB = os.path.join(ROOT, "data", "processed", "sms", "phobert_emb.npy")
EMB_META_SUFFIX = ".meta.json"
PHOBERT = os.path.expanduser("~/.cache/phishvn/phobert-base-st")

# registered: the 21 the web studies fit, is_https excluded as a collection artefact
COMPPHISH = ["url_len", "dom_len", "is_ip", "tld_len", "subdom_cnt", "letter_cnt", "digit_cnt",
             "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt", "dot_cnt", "dash_cnt", "under_cnt",
             "letter_ratio", "digit_ratio", "spec_ratio", "slash_cnt", "entropy",
             "path_len", "query_len"]
URL_RE = re.compile(r"(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})(?:/[^\s,;)\]]*)?", re.I)
SEEDS = list(range(10))


def first_url(text: str) -> str:
    m = URL_RE.search(text or "")
    if not m:
        return ""
    u = m.group(0)
    return u if u.lower().startswith("http") else "http://" + u


def _cache_identity(texts) -> dict:
    h = hashlib.sha256()
    for t in texts:
        b = (t or "").encode("utf-8", "surrogatepass")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    cfg = os.path.join(PHOBERT, "config.json")
    cfg_sha = hashlib.sha256(open(cfg, "rb").read()).hexdigest() if os.path.isfile(cfg) else ""
    return {"n": len(texts), "text_sha256": h.hexdigest(), "model": os.path.realpath(PHOBERT),
            "config_sha256": cfg_sha, "pooling": "attention-mask mean", "max_length": 256}


def embed(texts, cache=EMB):
    ident = _cache_identity(texts)
    meta_path = cache + EMB_META_SUFFIX
    if os.path.exists(cache):
        e = np.load(cache)
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) else None
        if meta == ident and len(e) == len(texts):
            return e
        print(f"[!] refusing stale/unidentified embedding cache {cache}; recomputing", file=sys.stderr)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PHOBERT)
    mdl = AutoModel.from_pretrained(PHOBERT).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256,
                    return_tensors="pt")
            h = mdl(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
            if i % 640 == 0:
                print(f"    embedding {i}/{len(texts)}", flush=True)
    e = np.vstack(out).astype(np.float32)
    np.save(cache, e)
    json.dump(ident, open(meta_path, "w", encoding="utf-8"), indent=2, sort_keys=True)
    return e


def cluster_bootstrap_f1(y, pred_a, pred_b, groups, B=5000, seed=20260902) -> dict:
    """Paired bootstrap over distinct test texts, retaining duplicate rows within a cluster."""
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    group_index = {g: i for i, g in enumerate(uniq)}
    gid = np.array([group_index[g] for g in groups])
    rng = np.random.default_rng(seed)
    # Each row gives how often a text cluster was drawn. Grouped TP/FP/FN counts then make the
    # full bootstrap matrix multiplication rather than 150,000 sklearn metric calls.
    draws = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)

    def boot_f1(preds):
        scores = []
        for p in preds:
            p = np.asarray(p, dtype=np.int8)
            tp = np.bincount(gid, weights=((y == 1) & (p == 1)), minlength=len(uniq))
            fp = np.bincount(gid, weights=((y == 0) & (p == 1)), minlength=len(uniq))
            fn = np.bincount(gid, weights=((y == 1) & (p == 0)), minlength=len(uniq))
            T, Fp, Fn = draws @ tp, draws @ fp, draws @ fn
            scores.append(np.divide(2 * T, 2 * T + Fp + Fn,
                                    out=np.zeros(B), where=(2 * T + Fp + Fn) != 0))
        return np.mean(scores, axis=0)

    diffs = boot_f1(pred_a) - boot_f1(pred_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)], "B": B,
            "clusters": len(uniq), "seed": seed,
            "unit": "distinct test text; duplicate rows retained within cluster",
            "status": "post-hoc sensitivity; conditional on the fitted seed ensemble"}


def fit_score(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400, random_state=seed,
                        early_stopping=True, n_iter_no_change=15)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict(sc.transform(Xte))
    return {"f1": f1_score(yte, p, pos_label=1, zero_division=0),
            "precision": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall": recall_score(yte, p, pos_label=1, zero_division=0)}, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()

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

    print(f"  {len(rows)} messages, {int(has_url.sum())} with a URL; "
          f"train {int((split=='train').sum())} / test {int((split=='test').sum())}")
    Xtxt = embed(texts)
    Xfus = np.hstack([Xtxt, Xurl])

    tr, te = split == "train", split == "test"
    arms = {"url": Xurl, "text": Xtxt, "fusion": Xfus}
    per_seed: dict = {k: [] for k in arms}
    preds: dict = {k: [] for k in arms}
    for s in SEEDS[:a.seeds]:
        for name, X in arms.items():
            m, p = fit_score(X[tr], y[tr], X[te], y[te], s)
            per_seed[name].append(m)
            preds[name].append(p)
        print(f"    seed {s}: " + "  ".join(f"{k} F1={per_seed[k][-1]['f1']:.3f}" for k in arms),
              flush=True)

    f1 = {k: np.array([m["f1"] for m in v]) for k, v in per_seed.items()}
    t1 = corrected_paired_t(f1["fusion"] - f1["url"])
    t2 = corrected_paired_t(f1["text"] - f1["url"])
    # NOT registered. The prereg's T1 contrasts fusion against the URL arm, which turned out to be
    # the weak one; the quantity a reader will want is whether the URL channel adds anything to the
    # text channel. Reported as post-hoc and labelled so, never as a third confirmatory test.
    post = corrected_paired_t(f1["fusion"] - f1["text"])
    test_groups = [rows[i]["text_sha1"] for i in np.where(te)[0]]
    boot = {
        "T1_fusion_minus_url": cluster_bootstrap_f1(
            y[te], preds["fusion"], preds["url"], test_groups),
        "T2_text_minus_url": cluster_bootstrap_f1(
            y[te], preds["text"], preds["url"], test_groups),
        "posthoc_fusion_minus_text": cluster_bootstrap_f1(
            y[te], preds["fusion"], preds["text"], test_groups),
    }

    # registered diagnostics
    short_ids = {r["message_id"] for r in rows if int(r["n_shortened"]) > 0 and r["label"] == "1"}
    te_idx = np.where(te)[0]
    diag = {}
    for name in arms:
        pp = preds[name]
        for tag, mask in (("no_url", ~has_url[te]), ("has_url", has_url[te]),
                          ("shortened_phish",
                           np.array([rows[i]["message_id"] in short_ids for i in te_idx]))):
            if mask.sum():
                diag[f"{name}/{tag}"] = {
                    "n": int(mask.sum()),
                    "f1": round(float(np.mean([
                        f1_score(y[te][mask], p[mask], pos_label=1, zero_division=0)
                        for p in pp])), 4)}

    res = {
        "n_messages": len(rows), "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_with_url": int(has_url.sum()), "seeds": a.seeds,
        # `p` is exempt from the 4-decimal rounding every other field gets: 7.4e-06 rounds to
        # 0.0, and "p = 0.000" is both wrong and the kind of wrong a referee stops at.
        "arms": {k: {m: round(float(np.mean([x[m] for x in v])), 4)
                     for m in ("f1", "precision", "recall")} for k, v in per_seed.items()},
        "arms_sd": {k: round(float(np.std(f1[k], ddof=1)), 4) for k in f1},
        "T1_fusion_minus_url": {kk: (vv if kk in ("p", "p_naive") else round(vv, 4) if isinstance(vv, float) else vv)
                                for kk, vv in t1.items()},
        "T2_text_minus_url": {kk: (vv if kk in ("p", "p_naive") else round(vv, 4) if isinstance(vv, float) else vv)
                              for kk, vv in t2.items()},
        "posthoc_fusion_minus_text": {kk: (vv if kk in ("p", "p_naive") else round(vv, 4) if isinstance(vv, float) else vv)
                                      for kk, vv in post.items()},
        "registered_inference_audit": {
            "valid": False,
            "reason": "Nadeau-Bengio requires repeated data splits; these ten values vary only the optimizer seed on one fixed holdout.",
            "legacy_seed_tests_retained_for_provenance": True,
        },
        "cluster_bootstrap": boot,
        "diagnostics": diag,
    }
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, sort_keys=True)

    print("\n  arm        F1     P      R")
    for k in ("url", "text", "fusion"):
        m = res["arms"][k]
        print(f"  {k:<9} {m['f1']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}  "
              f"(sd {res['arms_sd'][k]:.3f})")
    print("\n  [audit] registered seed-level p-values are invalid on a fixed split; not interpreted")
    for k, label in (("T1_fusion_minus_url", "T1 fusion - url"),
                     ("T2_text_minus_url", "T2 text - url"),
                     ("posthoc_fusion_minus_text", "post-hoc fusion - text")):
        z = boot[k]
        print(f"  {label:<25}: bootstrap mean {z['mean']:+.4f}, 95% CI "
              f"[{z['ci95'][0]:+.4f}, {z['ci95'][1]:+.4f}], {z['clusters']} text clusters")
    print(f"  [+] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
