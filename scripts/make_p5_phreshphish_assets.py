#!/usr/bin/env python3
"""
make_p5_phreshphish_assets.py — the sentence on PhreshPhish's date order, read off the data.

The pre-specified test in run_p5_phreshphish.py could not run its true-order stream: in date order
the corpus's windows are sometimes all phishing, so its date order records when each class was
collected rather than when pages arrived. This writes what was measured, on the same sample and
the same windows: the per-window phishing share and the longest run of months with no benign row.

Reads data/external/phreshphish/phreshphish_dated.csv (fetch_phreshphish.py). Writes
papers/P5_temporal_drift/sections/gen_phreshphish.tex and
data/processed/p5/results/p5_phreshphish.json.

RUN:  python scripts/make_p5_phreshphish_assets.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from genfile import write_generated, write_results
from make_p5_assets import WINDOWS
from retrain_drift import load
from run_p5_phreshphish import STREAM, build_stream

SEC = os.path.join(ROOT, "papers", "P5_temporal_drift", "sections")
RESULTS = os.path.join(ROOT, "data", "processed", "p5", "results", "p5_phreshphish.json")


def benign_gap(df: pd.DataFrame) -> tuple[pd.Period, pd.Period, int]:
    """The longest run of calendar months inside the span with no benign row (empty months count)."""
    months = pd.period_range(df["_t"].min().to_period("M"), df["_t"].max().to_period("M"), freq="M")
    has = set(df.loc[df.y == 0, "_t"].dt.to_period("M"))
    best, cur = (None, None, 0), []
    for m in months:
        cur = cur + [m] if m not in has else []
        if len(cur) > best[2]:
            best = (cur[0], cur[-1], len(cur))
    return best


def main():
    corpus = build_stream()
    df, _ = load(STREAM, "collected_at", spread_undated=False)
    idx = [i for i in np.array_split(np.arange(len(df)), WINDOWS) if len(i)]
    share = [float(df.y.iloc[i].mean()) for i in idx]
    g0, g1, glen = benign_gap(df)
    write_results(RESULTS, {"corpus": corpus, "window_phishing_share": share,
                            "all_phishing_windows": sum(s == 1.0 for s in share),
                            "benign_gap": {"first": str(g0), "last": str(g1), "months": glen}})
    num = lambda v: f"{int(v):,}".replace(",", "{,}")
    body = (
        f"A public corpus that dates every row of both classes does not supply one either. "
        f"PhreshPhish~\\cite{{phreshphish2025}}, sampled by hash to this corpus's ${num(corpus['rows'])}$ "
        f"rows and cut into the same {len(idx)} date-ordered windows, runs from "
        f"{100 * min(share):.0f}\\% to {100 * max(share):.0f}\\% phishing per window, "
        f"{sum(s == 1.0 for s in share)} of them all phishing, because its benign rows were collected "
        f"in campaigns: none is dated between {g0.strftime('%B %Y')} and {g1.strftime('%B %Y')}. "
        f"Its date order records the collection schedule"
        + (", and a model fitted on its first window never sees a benign page." if share[0] == 1.0 else "."))
    write_generated(os.path.join(SEC, "gen_phreshphish.tex"), body)


if __name__ == "__main__":
    main()
