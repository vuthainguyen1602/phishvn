#!/usr/bin/env python3
"""
p3_relabel_phishing_only.py — exploratory re-run on pages both raters call credential phishing.

Amendment 8 of papers/P3_multimodal/REVISION2/RELABEL_PLAN.md, committed before any rating.
Positives: phishing-arm pages with verdict `phishing` from both raters. Benign: rated benign pages
that no rater calls abuse, minus non-content pages, down-sampled with random.Random(42). Runs only
if at least MIN_N positives exist; otherwise reports the count.

RUN:  python scripts/p3_relabel_phishing_only.py
OUT:  data/processed/p3/results/relabel_phishing_only.json
      papers/P3_multimodal/sections/gen_phishonly.tex (+ tab_phishonly.tex when it runs)
"""
import csv
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
import pandas as pd
from genfile import write_generated, write_results
from paired_eval import fmt_p as _p

MIN_N = 30
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")


def main():
    from score_p3_relabel import read_sheet, rerun, ENCODERS, VDIR, ABUSE
    from make_p3_relabel_sheets import subset_indices, dead_rows
    from make_p3_assets import ENC_LABEL, num_word
    from train_content_fusion import load_pages
    key = {r["item"]: r for r in csv.DictReader(open(os.path.join(VDIR, "key.csv"), encoding="utf-8"))}
    A = read_sheet(os.path.join(VDIR, "returned", "cham_trang_A.xlsx"))
    B = read_sheet(os.path.join(VDIR, "returned", "cham_trang_B.xlsx"))
    pos = [i for i in key if key[i]["in_subset_arm"] == "phishing"
           and A[i][0] == "phishing" and B[i][0] == "phishing"]
    n = len(pos)
    out = {"n_consensus_phishing": n, "min_n": MIN_N, "ran": n >= MIN_N}
    if n < MIN_N:
        write_results(os.path.join(ROOT, "data", "processed", "p3", "results",
                                   "relabel_phishing_only.json"), out)
        write_generated(os.path.join(SEC, "gen_phishonly.tex"),
                        f"Only ${n}$ phishing-arm pages were called credential phishing by both raters, "
                        f"below the $30$ fixed in advance for a re-run on credential phishing alone, so "
                        f"that re-run is not reported\n")
        print(out)
        return
    path = os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv")
    man = pd.read_csv(path)
    rows = load_pages(path)
    _ph, be_sub = subset_indices(man)
    drop = {int(key[i]["manifest_row"]) for i in key if key[i]["in_subset_arm"] == "benign"
            and (A[i][0] in ABUSE or B[i][0] in ABUSE)}
    dead = dead_rows(man, be_sub)
    pool = [i for i in be_sub if i not in drop and i not in dead]
    ben = random.Random(42).sample(pool, n)
    res, mde, means = rerun([rows[int(key[i]["manifest_row"])] for i in pos] + [rows[i] for i in ben])
    out.update({"results": {"|".join(k): v for k, v in res.items()},
                "mde": {"|".join(k): v for k, v in mde.items()}, "f1_means": means})
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results",
                               "relabel_phishing_only.json"), out)

    def cell(e, d):
        t = res[(e, d, "F1")]
        body = f"{t['mean']:+.3f}"
        return (("\\textbf{" + body + "}") if t["mean"] > 0 and t["q"] < 0.05 else body) + \
            f" & {{\\scriptsize {_p(t['q'], 'q')}}}"
    body = "\n".join(
        f"{ENC_LABEL.get(e, e)} & {means[e]['url']:.3f} & {means[e]['content']:.3f} & "
        f"{means[e]['content+url']:.3f} & {cell(e, 'cu')} & {cell(e, 'uc')} & "
        f"{mde[(e, 'cu')]:.3f} \\\\" for e in ENCODERS)
    write_generated(os.path.join(SEC, "tab_phishonly.tex"), f"""\\begin{{table}}[t]
\\caption{{Exploratory re-run on credential phishing alone: ${n}$ pages both raters call phishing and
${n}$ benign pages, same protocol as the article. F1 means, paired F1 differences with BH $q$ per
column, and the MDE of the content-to-URL contrast.}}
\\label{{tab:phishonly}}
\\begin{{tabular}}{{l ccc rl rl c}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{F1}} & \\multicolumn{{2}}{{c}}{{$+$content to URL}} & \\multicolumn{{2}}{{c}}{{$+$URL to content}} & MDE \\\\
Encoder & URL & content & fusion & \\multicolumn{{2}}{{c}}{{$\\Delta$F1}} & \\multicolumn{{2}}{{c}}{{$\\Delta$F1}} & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""", f"(n={n})")
    resolved = [e for e in ENCODERS if res[(e, "cu", "F1")]["mean"] > 0 and res[(e, "cu", "F1")]["q"] < 0.05]
    lo = min(mde[(e, "cu")] for e in ENCODERS)
    hi = max(mde[(e, "cu")] for e in ENCODERS)
    write_generated(
        os.path.join(SEC, "gen_phishonly.tex"),
        f"On credential phishing alone (${n}$ pages both raters call phishing, and as many benign "
        f"pages), an exploratory re-run resolves the content-to-URL F1 gain for "
        f"{num_word(len(resolved))} of the five encoders, with an MDE of ${lo:.3f}$--${hi:.3f}$ "
        f"(Supplementary Table~\\SuppRef{{tab:phishonly}})\n")
    print(out["n_consensus_phishing"], resolved)


if __name__ == "__main__":
    main()
