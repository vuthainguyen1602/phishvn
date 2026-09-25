#!/usr/bin/env python3
"""
sms_split_robustness.py — the registered contrasts again, on a split that near-duplicates cannot
cross. POST-HOC ROBUSTNESS, NOT A RE-RUN OF THE REGISTRATION.

WHY. Section 2 of the smishing paper rejects the publisher's split because 47 of its 597 test rows
repeat a training message exactly, and keys ours on a hash of the message text so that exact
repeats land on one side. `sms_shallow_cues.py` then measured what that hash cannot see: at a
character n-gram cosine of 0.90, 34.3% of ham and 18.9% of phishing test messages still have a
near-twin in training. A message that differs from a training one only in an amount or a date is
the same object to a model, and every score in the paper is optimistic by whatever that overlap is
worth.

This script rebuilds the split so it is not, and re-computes the same three arms and the same
contrasts on it. What it produces is a SENSITIVITY ANALYSIS. It does not replace anything:

  * `PREREG_smishing.md` registered the text-hash split. Those numbers stay the study's numbers.
  * These numbers say how much of the registered result was the split, and they are labelled
    post-hoc wherever the paper prints them.
  * `fusion_results.json` is never written here. This writes its own file.

HOW THE SPLIT IS BUILT. Messages are joined into components when their char 3--5-gram cosine
reaches 0.90, and a whole component goes to one side, chosen by hashing the component's smallest
message_id. That is the same rule the corpus importer applies to identical texts, one similarity
level up. The holdout fraction is the study's 0.25, applied to components rather than to texts, so
the test set is a similar size and a different shape.

WHAT WOULD COUNT AS EACH OUTCOME, stated before the numbers rather than after them:

  * The arms move by less than a point of F1 -> the leak was worth little, the limitation stays in
    the paper as a measured bound rather than a worry, and the registered numbers stand as they are.
  * The arms fall materially and the ORDER holds (text > url, fusion ~ text) -> the study's
    conclusions stand and its effect sizes were inflated; the paper says both.
  * The order changes -> the finding was the split. That would be the most important sentence in
    the paper, and it would have to be written.

RUN
    python3 scripts/sms_split_robustness.py
    python3 scripts/sms_split_robustness.py --sim 0.95 --seeds 10
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

# the fitting code, the feature list and the seeds are the study's own, imported rather than copied
from train_sms_fusion import (COMPPHISH, MSG, SEEDS, SRC, embed, extract, first_url,
                              fit_score, cluster_bootstrap_f1)  # noqa: E402

OUT = os.path.join(ROOT, "data", "processed", "sms", "split_robustness.json")
REGISTERED = os.path.join(ROOT, "data", "processed", "sms", "fusion_results.json")


def components(texts: list, sim: float) -> np.ndarray:
    """Component id per message: near-duplicates join, and a component never splits."""
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform(texts)
    n, lab = connected_components(csr_matrix(cosine_similarity(X) >= sim), directed=False)
    print(f"  {len(texts):,} messages -> {n:,} components at cosine {sim}")
    return lab


def split_by_component(rows: list, comp: np.ndarray, holdout: float) -> np.ndarray:
    """Whole components to one side, keyed on the component's smallest message_id.

    Hashing rather than shuffling, for the same reason the importer hashes: the assignment has to
    be reproducible on another machine without carrying a split file around."""
    rep: dict = {}
    for r, c in zip(rows, comp):
        c = int(c)
        rep[c] = min(rep.get(c, r["message_id"]), r["message_id"])
    side = {}
    for c, key in rep.items():
        h = int(hashlib.sha1(("simsplit:" + key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        side[c] = "test" if h < holdout else "train"
    return np.array([side[int(c)] for c in comp])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=0.90)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--json", default=OUT)
    a = ap.parse_args()

    raw = {r["message_id"]: r["message"] for r in
           csv.DictReader(open(SRC, newline="", encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(MSG, newline="", encoding="utf-8")))
    texts = [raw.get(r["message_id"], "") for r in rows]
    y = np.array([int(r["label"]) for r in rows])

    print("[*] rebuilding the split so that near-duplicates cannot cross it")
    comp = components(texts, a.sim)
    split = split_by_component(rows, comp, a.holdout)
    tr, te = split == "train", split == "test"
    old = np.array([r["split"] for r in rows])
    moved = int((split != old).sum())
    print(f"  train {int(tr.sum()):,} / test {int(te.sum()):,}  "
          f"({moved:,} messages change side against the study's split)")
    print(f"  test class mix: {int(y[te].sum())} phishing / {int((1 - y[te]).sum())} ham")

    urls = [first_url(t) for t in texts]
    Xurl = np.zeros((len(rows), len(COMPPHISH)), dtype=np.float32)
    for i, u in enumerate(urls):
        if u:
            f = extract(u)
            Xurl[i] = [float(f.get(c, 0) or 0) for c in COMPPHISH]
    Xtxt = embed(texts)                      # cached; the encoder is not re-run
    arms = {"url": Xurl, "text": Xtxt, "fusion": np.hstack([Xtxt, Xurl])}

    per_seed: dict = {k: [] for k in arms}
    preds: dict = {k: [] for k in arms}
    for s in SEEDS[:a.seeds]:
        for name, X in arms.items():
            m, p = fit_score(X[tr], y[tr], X[te], y[te], s)
            per_seed[name].append(m)
            preds[name].append(p)
        print("    seed %d: " % s
              + "  ".join(f"{k} F1={per_seed[k][-1]['f1']:.3f}" for k in arms), flush=True)

    # The shallow-cue floor belongs to a split, so it is recomputed here rather than quoted from
    # the other one. Comparing an arm on this split against a floor from that one would be the
    # mistake this whole check exists to avoid.
    toks = sorted({t for r in rows for t in (r["pii_tokens"] or "").split()})
    def shallow(r, with_all):
        v = [1.0 if t in set((r["pii_tokens"] or "").split()) else 0.0 for t in toks]
        return v + ([float(int(r["accented"] or 0)), int(r["chars"] or 0) / 100.0]
                    if with_all else [])
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score as _f1
    floors = {}
    for tag, with_all in (("tokens", False), ("all_shallow", True)):
        A = np.array([shallow(r, with_all) for r in rows], dtype=float)
        m = LogisticRegression(max_iter=2000, random_state=0).fit(A[tr], y[tr])
        floors[tag] = round(float(_f1(y[te], m.predict(A[te]))), 4)
    print(f"\n  shallow-cue floor on THIS split: tokens {floors['tokens']:.3f}, "
          f"all three cues {floors['all_shallow']:.3f}")

    groups = comp[te]
    res = {"arms": {k: {m: round(float(np.mean([d[m] for d in per_seed[k]])), 4)
                        for m in ("f1", "precision", "recall")} for k in arms},
           "T1_fusion_minus_url": cluster_bootstrap_f1(y[te], preds["fusion"], preds["url"], groups),
           "T2_text_minus_url": cluster_bootstrap_f1(y[te], preds["text"], preds["url"], groups),
           "posthoc_fusion_minus_text": cluster_bootstrap_f1(y[te], preds["fusion"], preds["text"], groups),
           "split": {"sim": a.sim, "holdout": a.holdout, "components": int(comp.max()) + 1,
                     "n_train": int(tr.sum()), "n_test": int(te.sum()), "moved": moved,
                     "test_phishing": int(y[te].sum()), "test_ham": int((1 - y[te]).sum())},
           "floors": floors, "seeds": a.seeds, "registered": False,
           "note": "post-hoc sensitivity analysis; the registered split is the text-hash one"}

    print("\n=== arms, similarity-aware split (post-hoc) against the registered split ===")
    reg = json.load(open(REGISTERED, encoding="utf-8")) if os.path.exists(REGISTERED) else None
    print(f"  {'arm':8} {'registered':>11} {'sim-aware':>11} {'delta':>8}")
    for k in ("url", "text", "fusion"):
        new = res["arms"][k]["f1"]
        if reg:
            r0 = reg["arms"][k]["f1"]
            res["arms"][k]["delta_vs_registered"] = round(new - r0, 4)
            print(f"  {k:8} {r0:11.3f} {new:11.3f} {new - r0:+8.3f}")
        else:
            print(f"  {k:8} {'--':>11} {new:11.3f}")
    order_holds = res["arms"]["text"]["f1"] > res["arms"]["url"]["f1"]
    res["order_holds"] = bool(order_holds)
    print(f"\n  text > url still holds: {order_holds}")
    print(f"  T1 fusion-url {res['T1_fusion_minus_url']['mean']:+.4f};  "
          f"T2 text-url {res['T2_text_minus_url']['mean']:+.4f};  "
          f"post-hoc fusion-text {res['posthoc_fusion_minus_text']['mean']:+.4f} "
          "(distinct-component cluster bootstrap)")
    print("  POST-HOC. The registered numbers are the text-hash split's; these say how much of "
          "them\n  was the split.")

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(res, open(a.json, "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(f"\n[+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
