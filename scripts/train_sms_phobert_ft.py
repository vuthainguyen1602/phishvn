#!/usr/bin/env python3
"""
train_sms_phobert_ft.py — the text arm done the way PhoBERT expects, and fine-tuned.

WHY. The registered text arm is PhoBERT-base frozen, mean-pooled, fed UNSEGMENTED text although
the model was pretrained on word-segmented Vietnamese. The paper says so and calls it "a frozen
feature baseline, not a best-practice PhoBERT benchmark". A referee will still read the sentence
"word TF-IDF at 0.943 beats PhoBERT at 0.929" as a claim about transformers unless the arm is run
properly. Two arms do that here, on the registered split and seeds, so the contrasts are paired:

  text_seg  frozen PhoBERT, mean-pooled, on pyvi word-segmented text (the registered arm's one
            omission repaired, nothing else changed)
  text_ft   PhoBERT-base fine-tuned end to end on segmented text: 3 epochs, lr 2e-5, batch 8,
            max 128 tokens (the longest message is well inside it), linear decay, no early
            stopping and no validation slice, so the test window is touched once per seed at
            the end. Batch 8 rather than 16 because the first two runs were killed for memory on
            a shared workstation; the checkpoint below resumes a killed run seed by seed.

Both are POST-HOC and UNREGISTERED; the registered comparison stands as printed. The registered
frozen arm is refitted here from its cached embeddings (deterministic per seed) only so that every
contrast is a paired difference on identical test rows with the study's distinct-text cluster
bootstrap, and the linear word TF-IDF probe is refitted (seed 0, deterministic) for the same
reason.

RUN:
  python3 scripts/train_sms_phobert_ft.py --device mps
  python3 scripts/train_sms_phobert_ft.py --seeds 1 --epochs 1 --out /tmp/x.json
Writes data/processed/sms/phobert_ft.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from sms_predecision_common import (SEEDS, SMS_DIR, load, segment, summarise,  # noqa: E402
                                    test_groups)
from train_sms_fusion import EMB, PHOBERT, cluster_bootstrap_f1, embed, fit_score  # noqa: E402

OUT = os.path.join(SMS_DIR, "phobert_ft.json")
EMB_SEG = os.path.join(SMS_DIR, "phobert_emb_seg.npy")
HF_MODEL = "vinai/phobert-base"          # what the weights are
# Loaded from the same local safetensors export the frozen arm reads: transformers 5 refuses
# the hub snapshot's pytorch_model.bin under torch < 2.6, and the two exports are one model.
MODEL_PATH = PHOBERT


def metrics(yte, p):
    return {"f1": f1_score(yte, p, pos_label=1, zero_division=0),
            "precision": precision_score(yte, p, pos_label=1, zero_division=0),
            "recall": recall_score(yte, p, pos_label=1, zero_division=0)}


def finetune(tr_texts, ytr, te_texts, seed, epochs, lr, batch, max_len, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=2, use_safetensors=True).to(device)
    enc = tok(tr_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(ytr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(loader)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    mdl.train()
    for _ in range(epochs):
        for ids, am, yb in loader:
            ids, am, yb = ids.to(device), am.to(device), yb.to(device)
            opt.zero_grad()
            mdl(input_ids=ids, attention_mask=am, labels=yb).loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            sch.step()
    mdl.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_texts), 64):
            e = tok(te_texts[i:i + 64], padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(device)
            preds.append(mdl(**e).logits.argmax(-1).cpu().numpy())
    del mdl
    if device == "mps":
        torch.mps.empty_cache()
    return np.concatenate(preds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows, texts, y, tr, te, Xurl, has_url = load()
    seg = [segment(t) for t in texts]
    print(f"  {len(rows)} messages; train {int(tr.sum())} / test {int(te.sum())}", flush=True)

    Xtxt = embed(texts, cache=EMB)                     # the registered arm's embeddings
    Xseg = embed(seg, cache=EMB_SEG)                   # segmented, otherwise identical
    tr_seg = [s for s, m in zip(seg, tr) if m]
    te_seg = [s for s, m in zip(seg, te) if m]

    per = {"text": [], "text_seg": [], "text_ft": []}
    preds = {"text": [], "text_seg": [], "text_ft": []}
    # A fine-tune is twenty minutes a seed on a busy machine, and the first run was killed for
    # memory after three of them with nothing written. Every seed's predictions are checkpointed
    # and a rerun resumes from the checkpoint rather than refitting.
    ckpt = a.out + ".ckpt.json"
    done = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    for s in SEEDS[:a.seeds]:
        key = str(s)
        if key in done and done[key]["epochs"] == a.epochs and done[key]["max_len"] == a.max_len:
            for name in per:
                preds[name].append(np.array(done[key]["preds"][name], dtype=int))
                per[name].append(metrics(y[te], preds[name][-1]))
            print(f"    seed {s}: resumed from checkpoint", flush=True)
            continue
        for name, X in (("text", Xtxt), ("text_seg", Xseg)):
            m, p = fit_score(X[tr], y[tr], X[te], y[te], s)
            per[name].append(m); preds[name].append(p)
        t0 = time.time()
        p = finetune(tr_seg, y[tr], te_seg, s, a.epochs, a.lr, a.batch, a.max_len, a.device)
        per["text_ft"].append(metrics(y[te], p)); preds["text_ft"].append(p)
        print(f"    seed {s}: text {per['text'][-1]['f1']:.3f}  seg {per['text_seg'][-1]['f1']:.3f}"
              f"  ft {per['text_ft'][-1]['f1']:.3f}  ({time.time() - t0:.0f}s fine-tune)", flush=True)
        done[key] = {"epochs": a.epochs, "max_len": a.max_len,
                     "preds": {n: [int(v) for v in preds[n][-1]] for n in per}}
        json.dump(done, open(ckpt, "w"))

    # the linear probe, refitted so the contrast is paired on the same rows
    tfidf = make_pipeline(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                          sublinear_tf=True),
                          LogisticRegression(max_iter=3000, C=4, random_state=0))
    tfidf.fit([t for t, m in zip(texts, tr) if m], y[tr])
    p_tfidf = tfidf.predict([t for t, m in zip(texts, te) if m])
    groups = test_groups(rows, te)
    boot = {
        "seg_minus_text": cluster_bootstrap_f1(y[te], preds["text_seg"], preds["text"], groups),
        "ft_minus_text": cluster_bootstrap_f1(y[te], preds["text_ft"], preds["text"], groups),
        "ft_minus_tfidf": cluster_bootstrap_f1(y[te], preds["text_ft"], [p_tfidf], groups),
        "tfidf_minus_text": cluster_bootstrap_f1(y[te], [p_tfidf], preds["text"], groups),
    }
    res = {"seeds": a.seeds, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "arms": {k: summarise(v) for k, v in per.items()},
           "tfidf_word_f1": round(float(f1_score(y[te], p_tfidf)), 4),
           "finetune": {"model": HF_MODEL, "epochs": a.epochs, "lr": a.lr, "batch": a.batch,
                        "max_len": a.max_len, "device": a.device, "segmentation": "pyvi",
                        "validation_slice": None, "early_stopping": False},
           "cluster_bootstrap": boot,
           "status": "post-hoc, unregistered; the registered text arm is frozen and unsegmented"}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=2, sort_keys=True)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    for k, v in res["arms"].items():
        print(f"  {k:<9} F1 {v['f1']:.3f} (sd {v['f1_sd']:.3f})")
    print(f"  tfidf     F1 {res['tfidf_word_f1']:.3f}")
    for k, z in boot.items():
        print(f"  {k:<18} {z['mean']:+.4f} [{z['ci95'][0]:+.4f}, {z['ci95'][1]:+.4f}]")
    print(f"  [+] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
