#!/usr/bin/env python3
r"""make_p3_llm_assets.py — Evaluate LLM adversarial text robustness for P3.

Tests content detector survival under character-perturbed lures and whether
adversarial data augmentation restores detection recall.
"""

from __future__ import annotations
import os
import random
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
from genfile import write_generated
from paired_eval import bh_adjust, corrected_paired_t, fmt_p
from train_fusion import perturb

URL_RE = re.compile(r"https?://\S+|\bsim\.example\.vn\S*", re.I)
# 20, matching the paraphrase study in the same section rather than the 5 this experiment
# started with: the two are read side by side, and a reader comparing them should not have to
# discover that one is estimated from four times fewer splits than the other.
SEEDS = 20


def load():
    import pandas as pd
    frames = []
    for t in ("sms", "email"):
        p = os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No dataset_sms/email.csv — run p2_generate_corpus.py + normalize_merge.")
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str).map(lambda t: URL_RE.sub(" ", t))
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=20000)


def run():
    df = load()
    texts = df["text"].to_numpy(); y = df["y"].to_numpy()
    res = {k: [] for k in ("f1_clean", "miss_clean", "miss_obf", "miss_obf_adv")}
    for s in range(SEEDS):
        rng = random.Random(s)
        tr, te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=y, random_state=s)
        Xtr_txt, ytr = texts[tr], y[tr]
        Xte_txt, yte = texts[te], y[te]
        ph = yte == 1
        # perturbed (obfuscated) copies of the held-out phishing
        obf_txt = np.array([perturb(t, rng) if ph[i] else t for i, t in enumerate(Xte_txt)], dtype=object)

        # baseline detector (no adversarial training)
        v = vec(); Xtr = v.fit_transform(Xtr_txt)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        pc = clf.predict(v.transform(Xte_txt))
        po = clf.predict(v.transform(obf_txt))
        res["f1_clean"].append(f1_score(yte, pc, zero_division=0))
        res["miss_clean"].append(1 - recall_score(yte, pc, pos_label=1, zero_division=0))
        res["miss_obf"].append(1 - recall_score(yte, po, pos_label=1, zero_division=0))

        # adversarially-trained detector: augment training phishing with perturbed copies
        aug_txt = list(Xtr_txt) + [perturb(t, rng) for t, yy in zip(Xtr_txt, ytr) if yy == 1 for _ in range(2)]
        aug_y = list(ytr) + [1] * (2 * int(ytr.sum()))
        va = vec(); Xtra = va.fit_transform(aug_txt)
        clfa = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtra, aug_y)
        poa = clfa.predict(va.transform(obf_txt))
        res["miss_obf_adv"].append(1 - recall_score(yte, poa, pos_label=1, zero_division=0))
    return df, res


# The two contrasts the experiment exists to make, each stated with the direction predicted
# BEFORE the numbers: obfuscation should raise the miss rate, adversarial training should lower
# it again. Naming them here keeps the family fixed at two, which is what BH is applied across.
CONTRASTS = (
    ("evasion", "miss_obf", "miss_clean",
     "obfuscation raises the naive detector's miss rate"),
    ("repair", "miss_obf", "miss_obf_adv",
     "adversarial training lowers the miss rate on the same obfuscated messages"),
)


def tests(res: dict) -> dict:
    """Paired, corrected, BH-adjusted. Every condition is scored on the SAME splits, so the
    per-seed differences are paired by construction; the corrected resampled t is the same test
    the paraphrase study in this section uses, and it is applied here because reading an ordering
    off three mean +/- sd rows is exactly the error this paper spends a paragraph warning about."""
    out = {}
    for key, hi, lo, label in CONTRASTS:
        d = np.asarray(res[hi], float) - np.asarray(res[lo], float)
        out[key] = {**corrected_paired_t(d), "label": label}
    adj, rej = bh_adjust([out[k]["p"] for k, *_ in CONTRASTS])
    for (key, *_), a, rj in zip(CONTRASTS, adj, rej):
        out[key]["p_adj"], out[key]["reject"] = a, bool(rj)
    return out


def main():
    df, res = run()
    T = tests(res)
    n_ph = int(df["y"].sum())
    n_be = int((df["y"] == 0).sum())
    gens = (sorted(df.loc[df["y"] == 1, "gen_model"].dropna().astype(str).unique())
            if "gen_model" in df else ["?"])

    def pct(k):
        v = np.asarray(res[k], float)
        return f"{v.mean() * 100:.1f}\\,$\\pm$\\,{v.std() * 100:.1f}"

    n_test_ph = n_ph * 0.3
    pp_per_msg = 100.0 / n_test_ph
    tex = f"""\\begin{{table*}}[t]
\\caption{{\\textbf{{Preliminary}} obfuscation-robustness of the content detector on the
LLM-generated adversarial corpus ({n_ph} simulated phishing $+$ {n_be} benign; generator
{', '.join(gens)}); mean\\,$\\pm$\\,std over {SEEDS} stratified splits. Both contrasts are
paired on the splits and corrected; $q$ is Benjamini--Hochberg across the two.}}
\\label{{tab:llmrobust}}
\\small
\\begin{{tabular}}{{l c c}}
\\toprule
Test condition (content detector) & Miss rate (\\%) & vs.\\ reference \\\\
\\midrule
Clean phishing (held-out) & {pct('miss_clean')} & --- \\\\
\\midrule
Obfuscated phishing, naive detector & {pct('miss_obf')} & {verdict_cell(T['evasion'])} \\\\
Obfuscated phishing, adversarially trained & {pct('miss_obf_adv')} & \
{verdict_cell(T['repair'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    os.makedirs(SEC, exist_ok=True)
    write_generated(os.path.join(SEC, "tab_llm_robust.tex"), tex)

    write_generated(os.path.join(SEC, "gen_llm_robust_verdict.tex"), verdict_sentence(res, T, n_ph))

    # The corpus size the SECTION opens with, generated rather than typed. It was typed, and it
    # went stale: the prose introduced the study as "73 ... 54" (the 2026-07 pilot) while the
    # table beside it printed 193/143, so the paper contradicted itself on the size of its own
    # corpus in two adjacent floats.
    per_channel = {}
    if "channel" in df:
        for ch in ("sms", "email"):
            sub = df[df["channel"].astype(str).str.lower() == ch]
            per_channel[ch] = (int(sub["y"].sum()), int((sub["y"] == 0).sum()))
    detail = ""
    if per_channel.get("sms") and per_channel.get("email"):
        detail = (f" (${per_channel['sms'][0]}$ SMS and ${per_channel['email'][0]}$ e-mail lures, "
                  f"against ${per_channel['sms'][1]}$ and ${per_channel['email'][1]}$ controls)")
    write_generated(
        os.path.join(SEC, "gen_llm_corpus.tex"),
        f"${n_ph}$ simulated Vietnamese phishing messages against ${n_be}$ benign "
        f"controls{detail}\n",
        f"({n_ph} phishing / {n_be} benign)")

    for key, hi, lo, label in CONTRASTS:
        t = T[key]
        print(f"    {key:8s} {label}\n"
              f"             delta {t['mean'] * 100:+.2f} pp, {t['wins']}/{t['k']} splits in "
              f"direction, corrected p={t['p']:.4f}, BH q={t['p_adj']:.4f}, "
              f"reject={t['reject']}")
    print(f"    {n_test_ph:.0f} phishing messages per test split "
          f"= {pp_per_msg:.1f} pp per message")


def verdict_cell(t: dict) -> str:
    """One table cell: the paired difference and its adjusted verdict, or an explicit null."""
    mark = "" if t["reject"] else r"\,\textsuperscript{ns}"
    return (rf"${t['mean'] * 100:+.1f}$\,pp, {fmt_p(t['p_adj'], 'q')}{mark}")


def verdict_sentence(res: dict, T: dict, n_ph: int) -> str:
    """The claim the tests support, written so it cannot outrun them.

    Whichever way the numbers fall, the sentence has to be generated: the previous version of
    this section argued a null from mean +/- sd alone, which is the error the rest of the paper
    exists to avoid, and it would have been just as wrong to argue a positive that way."""
    ev, rp = T["evasion"], T["repair"]
    mc = np.mean(res["miss_clean"]) * 100
    mo = np.mean(res["miss_obf"]) * 100
    ma = np.mean(res["miss_obf_adv"]) * 100
    per_msg = 100.0 / (n_ph * 0.3)

    def one(t, name, direction):
        if t["reject"]:
            return (f"{name} is supported: ${t['mean'] * 100:+.1f}$\\,pp paired, "
                    f"{t['wins']}/{t['k']} splits in the predicted direction, corrected "
                    f"{fmt_p(t['p'])}, BH-adjusted {fmt_p(t['p_adj'], 'q')}")
        return (f"{name} is \\emph{{not}} supported at this sample size: "
                f"${t['mean'] * 100:+.1f}$\\,pp paired, {t['wins']}/{t['k']} splits in the "
                f"predicted direction, corrected {fmt_p(t['p'])}, BH-adjusted "
                f"{fmt_p(t['p_adj'], 'q')}")

    return (
        f"On the {n_ph}-source adversarial corpus the miss rate runs ${mc:.1f}\\%$ clean, "
        f"${mo:.1f}\\%$ obfuscated and ${ma:.1f}\\%$ obfuscated against the adversarially "
        f"trained detector, and the two time-stamped pre-specified contrasts are tested rather than read off "
        f"those means. Evasion: {one(ev, 'obfuscation raising the miss rate', 1)}. Repair: "
        f"{one(rp, 'adversarial training lowering it again', -1)}. Both are paired on the "
        f"{ev['k']} splits and corrected for the resampling overlap; one misclassified message "
        f"still moves a split's miss rate by ${per_msg:.1f}$\\,pp, so the corpus bounds what "
        f"either verdict can be"
    )


if __name__ == "__main__":
    main()
