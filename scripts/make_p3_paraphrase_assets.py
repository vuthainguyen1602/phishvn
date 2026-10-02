#!/usr/bin/env python3
r"""make_p3_paraphrase_assets.py — Paraphrase-evasion experiment evaluation for P3.

Evaluates 3 detectors (naive, char-obfuscation-trained, paraphrase-trained) across
clean, obfuscated, and paraphrased test splits using corrected paired t-tests.
Emits tab_paraphrase.tex and gen_paraphrase_verdict.tex.
"""

from __future__ import annotations

import os
import random
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
PARA_CSV = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase.csv")
from genfile import write_generated
from train_fusion import perturb
from paired_eval import corrected_paired_t, bh_adjust, fmt_p

URL_RE = re.compile(r"https?://\S+|\bsim\.example\.vn\S*", re.I)
SEEDS = 20
# D1_A0/D2_A0 (clean-text misses of the trained defences) are descriptive extras added at the
# reviewers' request; no pre-specified contrast reads them and COMPARISONS is unchanged.
CELLS = ("D0_A0", "D0_A1", "D0_A2", "D1_A0", "D1_A1", "D1_A2", "D2_A0", "D2_A2")


def strip_url(s: str) -> str:
    return URL_RE.sub(" ", str(s))


def load():
    frames = [pd.read_csv(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))
              for t in ("sms", "email")
              if os.path.exists(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))]
    if not frames:
        raise SystemExit("No dataset_sms/email.csv — run p2_generate_corpus.py + normalize_merge.")
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].fillna("").astype(str).map(strip_url)
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    if not os.path.exists(PARA_CSV):
        raise SystemExit(f"{PARA_CSV} missing — run scripts/p3_paraphrase_corpus.py first.")
    para = pd.read_csv(PARA_CSV)
    para["text"] = para["text"].map(strip_url)
    wide = para.pivot(index="src_id", columns="variant", values="text")
    missing = set(df.loc[df.y == 1, "id"]) - set(wide.index)
    if missing:
        raise SystemExit(f"[!] {len(missing)} phishing messages have no paraphrase — re-run "
                         "scripts/p3_paraphrase_corpus.py (it checks corpus/paraphrase sync).")
    df["para_a"] = df["id"].map(wide["a"]).fillna("")
    df["para_b"] = df["id"].map(wide["b"]).fillna("")
    return df


def vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, max_features=20000)


def miss(clf, v, texts, y_true) -> float:
    """Miss rate = 1 - recall on the phishing rows of this test condition."""
    pred = clf.predict(v.transform(texts))
    return 1.0 - recall_score(y_true, pred, pos_label=1, zero_division=0)


def run(df: pd.DataFrame):
    txt, y = df["text"].to_numpy(), df["y"].to_numpy()
    pa, pb = df["para_a"].to_numpy(), df["para_b"].to_numpy()
    per_seed = {c: [] for c in CELLS}
    fpr = {"D0": [], "D1": [], "D2": []}

    for s in range(SEEDS):
        rng = random.Random(s)
        tr, te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=s)
        ytr, yte = y[tr], y[te]
        ph_te = te[yte == 1]                       # held-out phishing, attacked in A1/A2
        be_te = te[yte == 0]                       # benign, never attacked
        y_ph = np.ones(len(ph_te), dtype=int)

        clean_ph = txt[ph_te]
        obf_ph = np.array([perturb(t, rng) for t in clean_ph], dtype=object)
        para_ph = pb[ph_te]                        # variant 'b' — test role only

        # D0: naive
        v0 = vec(); clf0 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v0.fit_transform(txt[tr]), ytr)
        # D1: adversarially trained on character obfuscation (the published defence)
        tr_ph_txt = txt[tr][ytr == 1]
        aug1 = list(txt[tr]) + [perturb(t, rng) for t in tr_ph_txt for _ in range(2)]
        y1 = list(ytr) + [1] * (2 * len(tr_ph_txt))
        v1 = vec(); clf1 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v1.fit_transform(aug1), y1)
        # D2: adversarially trained on paraphrases, variant 'a' only (train role)
        aug2 = list(txt[tr]) + [t for t in pa[tr][ytr == 1] if t]
        y2 = list(ytr) + [1] * int(sum(1 for t in pa[tr][ytr == 1] if t))
        v2 = vec(); clf2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
            v2.fit_transform(aug2), y2)

        per_seed["D0_A0"].append(miss(clf0, v0, clean_ph, y_ph))
        per_seed["D0_A1"].append(miss(clf0, v0, obf_ph, y_ph))
        per_seed["D0_A2"].append(miss(clf0, v0, para_ph, y_ph))
        per_seed["D1_A0"].append(miss(clf1, v1, clean_ph, y_ph))
        per_seed["D1_A1"].append(miss(clf1, v1, obf_ph, y_ph))
        per_seed["D1_A2"].append(miss(clf1, v1, para_ph, y_ph))
        per_seed["D2_A0"].append(miss(clf2, v2, clean_ph, y_ph))
        per_seed["D2_A2"].append(miss(clf2, v2, para_ph, y_ph))
        for name, (c, v) in (("D0", (clf0, v0)), ("D1", (clf1, v1)), ("D2", (clf2, v2))):
            fpr[name].append(float(np.mean(c.predict(v.transform(txt[be_te])) == 1)))

    return per_seed, fpr


COMPARISONS = (
    ("H1",   "D0_A2", "D0_A0", "paraphrase vs clean, naive detector", +1),
    ("H1b",  "D0_A2", "D0_A1", "paraphrase vs character obfuscation, naive detector", +1),
    ("H2",   "D1_A2", "D0_A2", "char-adversarial training against the paraphrase attack", +1),
    ("H3",   "D2_A2", "D0_A2", "paraphrase-adversarial training against the paraphrase attack", -1),
    ("ctrl", "D0_A1", "D0_A0", "character obfuscation vs clean (replicates the published cell)", +1),
)


def corpus_multiple_for_significance(t_now: float, k: int = SEEDS, alpha: float = 0.05):
    """Estimate corpus size multiplier needed to achieve significance under corrected paired test."""
    try:
        from scipy import stats
    except ImportError:
        return float("nan")
    if not t_now or t_now != t_now:
        return float("nan")
    t_req = float(stats.t.ppf(1 - alpha / 2, k - 1))
    return float((t_req / abs(t_now)) ** 2)


def lexical_shift(df: pd.DataFrame) -> dict:
    """Mean token Jaccard similarity between source lures and test-role rewrites by stratum."""

    from p3_paraphrase_corpus import PARA as PILOT_IDS
    ph = df[df.y == 1]
    out: dict = {}
    for name, mask in (("all", ph["id"].notna()),
                       ("pilot", ph["id"].isin(PILOT_IDS)),
                       ("extension", ~ph["id"].isin(PILOT_IDS))):
        sub = ph[mask]
        js = []
        for src, para in zip(sub["text"], sub["para_b"]):
            a, b = set(str(src).lower().split()), set(str(para).lower().split())
            if a or b:
                js.append(len(a & b) / len(a | b))
        out[name] = (float(np.mean(js)) if js else float("nan"), len(js))
    return out


def main() -> None:
    df = load()
    per_seed, fpr = run(df)
    mean = {c: float(np.mean(per_seed[c])) for c in CELLS}
    sd = {c: float(np.std(per_seed[c])) for c in CELLS}

    tests = []
    for key, hi, lo, label, direction in COMPARISONS:
        d = np.array(per_seed[hi]) - np.array(per_seed[lo])
        r = corrected_paired_t(d)
        # Consistency reported in the PREDICTED direction, so a helpful defence (negative delta)
        # does not read as "0/20".
        consistent = int(np.sum(d > 0)) if direction > 0 else int(np.sum(d < 0))
        tests.append({"key": key, "hi": hi, "lo": lo, "label": label, "direction": direction,
                      "consistent": consistent,
                      "need": corpus_multiple_for_significance(r.get("t", float("nan"))), **r})
    adj, rej = bh_adjust([t["p"] for t in tests])
    for t, a, rj in zip(tests, adj, rej):
        t["p_adj"], t["reject"] = a, rj
    T = {t["key"]: t for t in tests}

    n_ph = int(df.y.sum())
    n_test_ph = n_ph * 0.30
    pp = 100.0 / n_test_ph
    jac = lexical_shift(df)

    def cell(c):
        return f"{mean[c]*100:.1f}\\,$\\pm$\\,{sd[c]*100:.1f}"

    fpr_cell = {d: f"{np.mean(fpr[d])*100:.1f}\\,$\\pm$\\,{np.std(fpr[d])*100:.1f}"
                for d in ("D0", "D1", "D2")}
    tex = rf"""\begin{{table*}}[t]
\caption{{Paraphrase evasion, attack strength uncontrolled: {n_ph} lures against
{int((df.y == 0).sum())} benign controls, mean $\pm$ std over {SEEDS} splits. \emph{{n/a}}: D2 under character obfuscation is outside the pre-specified design (no hypothesis concerns it) and was not computed.}}
\label{{tab:paraphrase}}
\small
\begin{{tabular}}{{l c c c c}}
\toprule
& \multicolumn{{3}}{{c}}{{Miss rate (\%) on held-out phishing}} & \\
\cmidrule(lr){{2-4}}
Detector & Clean & Char-obfuscated & Paraphrased & Benign FPR (\%) \\
\midrule
D0 naive & {cell('D0_A0')} & {cell('D0_A1')} & {cell('D0_A2')} & {fpr_cell['D0']} \\
D1 adv.\ trained on char.\ obfuscation & {cell('D1_A0')} & {cell('D1_A1')} & {cell('D1_A2')} & {fpr_cell['D1']} \\
D2 adv.\ trained on paraphrases & {cell('D2_A0')} & n/a & {cell('D2_A2')} & {fpr_cell['D2']} \\
\bottomrule
\end{{tabular}}\end{{table*}}"""
    os.makedirs(SEC, exist_ok=True)
    write_generated(os.path.join(SEC, "tab_paraphrase.tex"), tex)

    def verdict(key):
        t = T[key]
        # Corrected 95% CI (mean ± t_{.975,k-1} · corrected SE): the reviewers asked for the
        # interval, not only the verdict, on every main paired difference.
        ci = ""
        if t.get("se"):
            from scipy import stats as _st
            half = _st.t.ppf(0.975, t["k"] - 1) * t["se"] * 100
            ci = (f", corrected $95\\%$ CI $[{t['mean']*100 - half:+.1f}, "
                  f"{t['mean']*100 + half:+.1f}]$\\,pp")
        return (f"$\\Delta = {t['mean']*100:+.1f}$\\,pp{ci}, {t['consistent']}/{SEEDS} splits in the "
                # Both symbols printed in full: stripping the label off the adjusted value
                # produced "BH-adjusted 0.11" beside "BH-adjusted q<0.001", i.e. the same
                # quantity rendered two ways within one sentence.
                f"predicted direction, corrected {fmt_p(t['p'])}, "
                f"BH-adjusted {fmt_p(t['p_adj'], 'q')}")

    need_h1 = T["H1"]["need"]
    _, j_pilot, j_ext = jac["all"][0], jac["pilot"][0], jac["extension"][0]
    # How many contrasts the uncorrected test would have handed us, and the largest p among
    # THOSE — quoting max() over all five would misstate the threshold once one stops clearing it.
    naive_hits = [t for t in tests if t["p_naive"] < 0.05]
    max_naive_hit = max((t["p_naive"] for t in naive_hits), default=float("nan"))
    prose = (
        f"The time-stamped pre-specified directions all appear, and none of them survives the corrected test. "
        f"Against the naive detector the miss rate rises from ${mean['D0_A0']*100:.1f}\\%$ on clean "
        f"held-out phishing to ${mean['D0_A2']*100:.1f}\\%$ when the same messages are replaced by "
        f"their rewrites ({verdict('H1')}). The character attack that trains D1 gives "
        f"${mean['D0_A1']*100:.1f}\\%$ (against clean: {verdict('ctrl')}). Adversarial training "
        f"on character obfuscation repairs the character attack (${mean['D0_A1']*100:.1f}\\% "
        f"\\rightarrow {mean['D1_A1']*100:.1f}\\%$) and its effect on the "
        f"paraphrase attack is not resolved (${mean['D1_A2']*100:.1f}\\%$ against D0's "
        f"${mean['D0_A2']*100:.1f}\\%$: {verdict('H2')}). Training on \\emph{{disjoint}} "
        f"rewrites moves it to ${mean['D2_A2']*100:.1f}\\%$ (against D0: {verdict('H3')}), at a benign "
        f"false-positive cost of ${(np.mean(fpr['D2']) - np.mean(fpr['D0']))*100:+.1f}$\\,pp. "
        f"The ordinary paired $t$-test would have licensed {len(naive_hits)} of the "
        f"{len(tests)} contrasts at $p \\leq {max_naive_hit:.3f}$. The correction for overlapping "
        f"resamples, the same one applied to the fusion results, widens every interval past "
        f"significance.\n\n"
        # Trimmed again 2026-09-29: the run history (pilot, one pre-committed extension, gentler
        # second batch) is told in full once, here; Section 6 and the conclusion only point to it.
        f"\\textbf{{This is the second and final measurement of the uncontrolled design, and it "
        f"is underpowered.}} The study first ran on {jac['pilot'][1]} lures. Under a rule fixed "
        f"before any new lure was written, the corpus was extended once, to {n_ph}, the size "
        f"that run's power projection called for (\\texttt{{PARAPHRASE\\_PROTOCOL.md}}). The "
        f"leading effect roughly halved, and the data show why: the extension's rewrites share a "
        f"mean token Jaccard of ${j_ext:.2f}$ with their sources against ${j_pilot:.2f}$ for the "
        f"pilot's (a lower $J$ means stronger rewriting), so the larger corpus is also a "
        f"\\emph{{gentler attack}}. On the present estimates the leading contrast would need "
        f"about ${need_h1:.1f}\\times$ these {n_ph} lures. Per the rule the corpus is not "
        f"extended again, and the next subsection fixes attack strength instead. With "
        f"{n_test_ph:.0f} held-out phishing messages per split a single miss is worth "
        f"${pp:.1f}$\\,pp, and the paraphraser is the model that wrote the corpus, which biases "
        f"the measured attack downward.\n")
    write_generated(os.path.join(SEC, "gen_paraphrase_verdict.tex"), prose)

    # Machine-readable record so check_paper_claims.py can enforce the invariant that matters
    # here: the paper's wording must track what the correction actually licensed. Without this
    # the honest framing is one careless edit away from becoming three findings.
    stats_csv = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_stats.csv")
    pd.DataFrame([{"key": t["key"], "hi": t["hi"], "lo": t["lo"], "label": t["label"],
                   "mean_pp": t["mean"] * 100, "p": t["p"], "p_naive": t["p_naive"],
                   "p_adj": t["p_adj"], "reject": int(t["reject"]),
                   "consistent": t["consistent"], "k": SEEDS,
                   "corpus_multiple_needed": t["need"]} for t in tests]).to_csv(
        stats_csv, index=False)
    pd.DataFrame([{"cell": c, "miss_mean_pp": mean[c] * 100, "miss_sd_pp": sd[c] * 100}
                  for c in CELLS]).to_csv(
        os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase_cells.csv"), index=False)

    print(f"[+] {stats_csv}")
    for c in CELLS:
        print(f"    {c}: {mean[c]*100:5.1f}% ± {sd[c]*100:.1f}")
    for t in tests:
        print(f"    {t['key']:5s} {t['label']:<58s} Δ={t['mean']*100:+6.1f}pp  "
              f"p={t['p']:.4f} (naive {t['p_naive']:.4f}, ×{t['inflation']:.1f})  "
              f"BH={t['p_adj']:.4f} {'REJECT-NULL' if t['reject'] else 'no evidence'}  "
              f"{t['consistent']}/{SEEDS} in predicted direction  "
              f"needs {t['need']:.1f}x corpus")
    for k, (v, n) in jac.items():
        print(f"    lexical shift [{k:9s} n={n:3d}]: mean token Jaccard(source, rewrite) = {v:.3f}")
    print(f"    benign FPR: D0 {np.mean(fpr['D0'])*100:.1f}%  D1 {np.mean(fpr['D1'])*100:.1f}%  "
          f"D2 {np.mean(fpr['D2'])*100:.1f}%")


if __name__ == "__main__":
    main()
