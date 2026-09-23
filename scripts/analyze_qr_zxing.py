#!/usr/bin/env python3
"""analyze_qr_zxing.py — the post-hoc ZXing arm, reported beside the registered three.

WHY THIS IS SEPARATE FROM analyze_qr_dfr.py, AND STAYS SEPARATE. The sweep's grid, its two
hypotheses and their bars were registered over THREE decoders. ZXing was added on 2026-09-03,
after those results existed, because the paper called its three "widely deployed" while omitting
the server-side engine a reader is most likely to name. Folding it into the registered analysis
would change pooled numbers fixed in advance, so it does not go there: this script writes its own
macros, the paper reports the arm as post-hoc, and T1 and T2 are untouched.

The grid is reproduced rather than re-derived: same generator, same seed, same 2,000 URLs, so a
row here lines up with a row there by sample_id.

RUN  python3 scripts/gen_synthetic_qr.py --n 2000 --stream --decoders zxing \\
         --dfr-out data/processed/qr/qr_dfr_zxing.csv
     python3 scripts/analyze_qr_zxing.py
"""
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
ZX = os.path.join(ROOT, "data", "processed", "qr", "qr_dfr_zxing.csv")
SNAP = os.path.join(ROOT, "data", "processed", "qr", "dfr_snapshot.json")
OUT_SNAP = os.path.join(ROOT, "data", "processed", "qr", "zxing_snapshot.json")
OUT_TEX = os.path.join(ROOT, "papers", "future_quishing", "sections", "gen_qr_zxing.tex")


def main() -> int:
    if not os.path.exists(ZX):
        print(f"[!] {ZX} missing — run the streaming sweep with --decoders zxing first",
              file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(ZX, newline="", encoding="utf-8")))
    if not rows:
        print("[!] no rows", file=sys.stderr)
        return 1

    by_tr, tot, bad = {}, 0, 0
    for r in rows:
        ok = r["correct"] == "1"
        n, f = by_tr.get(r["transform"], (0, 0))
        by_tr[r["transform"]] = (n + 1, f + (0 if ok else 1))
        tot += 1
        bad += 0 if ok else 1
    dfr = {t: 100.0 * f / n for t, (n, f) in by_tr.items()}
    overall = 100.0 * bad / tot

    reg = json.load(open(SNAP, encoding="utf-8"))
    three = reg["by_transform"]
    better, worse = [], []
    for t, v in sorted(dfr.items()):
        others = [three[t][d] for d in reg["decoders"] if t in three and d in three[t]]
        if not others:
            continue
        if v < min(others):
            better.append(t)
        elif v > max(others):
            worse.append(t)
    reg_overall = {d: sum(three[t][d] for t in three) / len(three) for d in reg["decoders"]}

    snap = {"rows": tot, "overall_dfr": round(overall, 2),
            "by_transform": {t: round(v, 2) for t, v in sorted(dfr.items())},
            "beats_all_three": sorted(better), "worse_than_all_three": sorted(worse),
            "registered_mean_by_decoder": {d: round(v, 2) for d, v in reg_overall.items()},
            "note": "post-hoc; the registered tests were computed over three decoders"}
    json.dump(snap, open(OUT_SNAP, "w", encoding="utf-8"), indent=2, sort_keys=True)

    m = lambda k, v: "\\newcommand{\\QrZx%s}{%s}\n" % (k, v)
    tex = (m("Rows", f"{tot:,}") + m("Dfr", f"{overall:.1f}")
           + m("Best", str(len(better))) + m("Worst", str(len(worse))))
    for t, v in sorted(dfr.items()):
        tex += m(t.capitalize(), f"{v:.1f}")
    for d, v in sorted(reg_overall.items()):
        tex += m("Reg" + d.capitalize(), f"{v:.1f}")
    from genfile import write_generated
    write_generated(OUT_TEX, tex)
    print(f"  zxing: {tot:,} rows, overall DFR {overall:.1f}%; best of four on {len(better)}, "
          f"worst on {len(worse)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
