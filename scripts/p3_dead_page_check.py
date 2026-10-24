#!/usr/bin/env python3
"""
p3_dead_page_check.py — the main contrasts without the non-content pages the gate missed.

Amendment 1 of papers/P3_multimodal/REVISION2/RELABEL_PLAN.md, committed before this ran. The
parked/error gate of build_content_manifest.py matches English phrases only; the Vietnamese
pattern below, fixed in the plan, flags expired, unpublished, for-sale and default pages. This
drops the flagged pages from both arms of the 207+207 subset, down-samples the remaining benign
pages to the remaining phishing count with random.Random(42), and re-runs the article's protocol
through score_p3_relabel.rerun (imported unchanged).

RUN:  python scripts/p3_dead_page_check.py
OUT:  data/processed/p3/results/dead_page_check.json
      papers/P3_multimodal/sections/gen_deadpage.tex, tab_deadpage.tex
"""
import json
import os
import random
import re
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

VI_DEAD = re.compile(
    r"tên miền của website này đã hết hạn|trang chưa được xuất bản|trỏ tên miền thành công|"
    r"website hiện không hoạt động|trang web mặc định|mua bán hoặc thuê tên miền|bán tên miền|"
    r"tên miền chưa được xác nhận|bạn muốn mua tên miền", re.I)
SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
MANIFEST = os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv")
MLAB = {"F1": "F1", "PR-AUC": "PR-AUC", "ROC-AUC": "ROC-AUC", "FPR@R0.90": "FPR@90\\%rec."}


def main():
    from vn_filter import visible_text
    from train_content_fusion import load_pages
    from make_p3_relabel_sheets import subset_indices
    from make_p3_assets import ENC_LABEL, SWEEP_METRICS, SWEEP_DIRECTIONS
    from score_p3_relabel import rerun, ENCODERS

    man = pd.read_csv(MANIFEST)
    ph, be = subset_indices(man)

    def dead(i):
        t = visible_text(open(os.path.join(ROOT, man.iloc[i]["dom_file"]), encoding="utf-8",
                              errors="ignore").read())
        return bool(VI_DEAD.search(t[:600]))
    dead_ph = [i for i in ph if dead(i)]
    dead_be = [i for i in be if dead(i)]
    keep_ph = [i for i in ph if i not in set(dead_ph)]
    pool = [i for i in be if i not in set(dead_be)]
    keep_be = random.Random(42).sample(pool, len(keep_ph))
    rows = load_pages(MANIFEST)
    if len(rows) != len(man):
        raise SystemExit("load_pages skipped rows: manifest positions no longer index rows")
    res, mde, means = rerun([rows[i] for i in keep_ph] + [rows[i] for i in keep_be])

    main_res = json.load(open(os.path.join(ROOT, "data", "processed", "p3", "results",
                                           "encoder_sweep.json"), encoding="utf-8"))
    changed = []
    for e in ENCODERS:
        for d, _ in SWEEP_DIRECTIONS:
            for m, _l, sign in SWEEP_METRICS:
                old, new = main_res[e][f"{d}|{m}"], res[(e, d, m)]
                was = sign * old["mean"] > 0 and old["q"] < 0.05
                now = sign * new["mean"] > 0 and new["q"] < 0.05
                flip = was and (old["mean"] > 0) != (new["mean"] > 0)
                if was != now or flip:
                    changed.append({"encoder": e, "direction": d, "metric": m, "was": was,
                                    "now": now, "flip": flip, "mean": new["mean"], "q": new["q"]})
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results", "dead_page_check.json"),
                  {"dead_phishing": [man.iloc[i]["domain"] for i in dead_ph],
                   "dead_benign": [man.iloc[i]["domain"] for i in dead_be],
                   "n_per_class": len(keep_ph),
                   "results": {"|".join(k): v for k, v in res.items()},
                   "mde": {"|".join(k): v for k, v in mde.items()},
                   "f1_means": means, "changed_verdicts": changed})

    def cell(e, d):
        t = res[(e, d, "F1")]
        body = f"{t['mean']:+.3f}"
        return (("\\textbf{" + body + "}") if t["mean"] > 0 and t["q"] < 0.05 else body) + \
            f" & {{\\scriptsize {_p(t['q'], 'q')}}}"
    body = "\n".join(
        f"{ENC_LABEL.get(e, e)} & {means[e]['url']:.3f} & {means[e]['content']:.3f} & "
        f"{means[e]['content+url']:.3f} & {cell(e, 'cu')} & {cell(e, 'uc')} \\\\" for e in ENCODERS)
    n = len(keep_ph)
    tex = f"""\\begin{{table}}[t]
\\caption{{The main contrasts without the {len(dead_ph) + len(dead_be)} non-content pages the gate
missed: ${n}$ phishing and ${n}$ benign pages, same protocol as the article (20 splits,
standardised linear head). F1 means and paired F1 differences with BH $q$ per column.}}
\\label{{tab:deadpage}}
\\begin{{tabular}}{{l ccc rl rl}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{F1}} & \\multicolumn{{2}}{{c}}{{$+$content to URL}} & \\multicolumn{{2}}{{c}}{{$+$URL to content}} \\\\
Encoder & URL & content & fusion & \\multicolumn{{2}}{{c}}{{$\\Delta$F1}} & \\multicolumn{{2}}{{c}}{{$\\Delta$F1}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_deadpage.tex"), tex, f"(n={n} per class)")
    verdict = ("returns the same resolved and unresolved verdicts on all $40$ comparisons"
               if not changed else
               f"changes ${len(changed)}$ of the $40$ verdicts ("
               + ", ".join(f"{ENC_LABEL.get(c['encoder'], c['encoder'])} "
                           f"{'content to URL' if c['direction'] == 'cu' else 'URL to content'} "
                           f"{MLAB[c['metric']]} {'now resolved' if c['now'] else 'no longer resolved'}"
                           + (", sign reversed" if c["flip"] else "") for c in changed) + ")")
    write_generated(
        os.path.join(SEC, "gen_deadpage.tex"),
        f"The parked and error gate matches English phrases only, and it missed "
        f"${len(dead_ph)}$ phishing and ${len(dead_be)}$ benign pages of the subset whose "
        f"Vietnamese text says the domain has expired, is unpublished, is for sale or shows a "
        f"default page. Dropping them and re-balancing to ${n}{{+}}{n}$ pages {verdict} "
        f"(Supplementary Table~\\SuppRef{{tab:deadpage}})\n")
    print(len(dead_ph), len(dead_be), n, len(changed), changed)


if __name__ == "__main__":
    main()
