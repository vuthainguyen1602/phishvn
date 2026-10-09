#!/usr/bin/env python3
"""
drift_detectors.py — P5 drift-detection accuracy/lag + labelling-budget frontier.

Extends retrain_drift.py with the two analyses its results section needs: (A) five monitors
(ADWIN over the error stream, PSI, SHAP-shift, ADWIN+PSI, fixed schedule) scored against a
MODEL-INDEPENDENT drift ground truth built from a Bonferroni-corrected two-sample KS test, giving
detection rate, false-alarm rate, mean lag and precision; and (B) sustained accuracy (AUTC) against
the per-period labelling budget for static / periodic / drift-triggered / active-learning policies.

RUN:  python scripts/drift_detectors.py --in data/interim/drift_compphish.csv
      (build the input with scripts/make_p5_assets.py or the snippet in retrain_drift docs)
Why KS is the ground truth, and how independent each monitor is of it:
kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from retrain_drift import load, psi  # reuse loader (spread-undated, date parsing) + PSI


# ----------------------------- ADWIN (Bifet & Gavaldà, 2007) -----------------------------
class ADWIN:
    """Adaptive windowing drift detector over a stream of real values (here: per-sample error,
    0/1). Maintains an exponential histogram of buckets; when the means of two adjacent
    sub-windows differ by more than the Hoeffding-style bound at confidence delta, it drops the
    older sub-window and flags a change. Compact, dependency-free implementation."""

    def __init__(self, delta=0.002, max_buckets=5):
        self.delta = delta
        self.max_buckets = max_buckets
        self.buckets = []  # each: [total_sum, count] ; counts are powers of two per "row"
        self.total = 0.0
        self.width = 0
        self.variance = 0.0

    def _insert_bucket(self, value):
        self.buckets.insert(0, [value, 1])
        self.width += 1
        # compress: merge when more than max_buckets of the same size accumulate
        i = 0
        size = 1
        while i < len(self.buckets):
            # count buckets of current 'size' from position i
            same = [b for b in self.buckets if b[1] == size]
            if len(same) > self.max_buckets:
                # merge the two oldest of this size
                idxs = [j for j, b in enumerate(self.buckets) if b[1] == size]
                a, b = idxs[-2], idxs[-1]
                self.buckets[a][0] += self.buckets[b][0]
                self.buckets[a][1] += self.buckets[b][1]
                self.buckets.pop(b)
            size *= 2
            i += 1

    def update(self, value):
        """Feed one value; return True if a change was detected (older window dropped)."""
        self.total += value
        self._insert_bucket(value)
        return self._detect()

    def _detect(self):
        changed = False
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            changed = True
        return changed

    def _find_cut(self):
        n = sum(b[1] for b in self.buckets)
        if n < 2:
            return None
        total = sum(b[0] for b in self.buckets)
        n0 = 0.0
        s0 = 0.0
        # try each split point between buckets (older side = suffix)
        for i in range(len(self.buckets) - 1):
            n0 += self.buckets[i][1]
            s0 += self.buckets[i][0]
            n1 = n - n0
            if n1 < 1:
                break
            m0 = s0 / n0
            m1 = (total - s0) / n1
            # Hoeffding bound with Bonferroni over window
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            dd = math.log(2.0 * math.log(max(n, 2)) / self.delta)
            eps = math.sqrt(2.0 / m * 0.25 * dd) + 2.0 / 3.0 * dd / m
            if abs(m0 - m1) > eps:
                # drop the OLDER sub-window (the suffix buckets)
                self.buckets = self.buckets[: i + 1]
                self.width = int(sum(b[1] for b in self.buckets))
                return True
        return None


# ----------------------------- stream + detectors -----------------------------
def fit(df, feats, seed=0):
    m = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed)
    m.fit(df[feats], df.y)
    return m


def shap_share(model, X, n=400, seed=0):
    """Normalised mean-|SHAP| attribution share of the deployed model on a window subsample —
    'what the model currently leans on'. Returns None when shap is not installed (the SHAP-shift
    detector is then skipped)."""
    try:
        import shap
    except ImportError:
        return None
    sub = X.sample(min(n, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(sub)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
    imp = np.abs(sv).mean(axis=0)
    s = imp.sum()
    return imp / s if s else imp


def ks_ground_truth(ref, cur, feats, alpha=0.01, d_min=0.15):
    """Model-independent drift flag: a window is TRUE drift when some feature's KS two-sample
    statistic against the previous window is both statistically significant (Bonferroni-corrected
    asymptotic critical value, no SciPy) AND of practical size (max KS D > d_min). The effect-size
    floor matters: at ~2500 samples/window plain significance flags trivially small shifts, so
    without d_min almost every window looks like drift and there are no negatives to score against."""
    n1, n2 = len(ref), len(cur)
    if n1 < 20 or n2 < 20:
        return False, 0.0
    c_alpha = math.sqrt(-0.5 * math.log((alpha / len(feats)) / 2.0))
    crit = c_alpha * math.sqrt((n1 + n2) / (n1 * n2))
    worst = 0.0
    for c in feats:
        a = np.sort(ref[c].values)
        b = np.sort(cur[c].values)
        grid = np.union1d(a, b)
        ca = np.searchsorted(a, grid, side="right") / n1
        cb = np.searchsorted(b, grid, side="right") / n2
        worst = max(worst, float(np.max(np.abs(ca - cb))))
    return (worst > max(crit, d_min)), worst


def run_detection(df, feats, windows, period, adwin_delta, psi_det=0.10, shap_tau=0.05):
    # NOTE 2026-08-17: this signature previously also took a `psi_tau` that nothing used —
    # the PSI *detector* fires at psi_det=0.10 (the conventional "moderate shift" cut), which
    # is deliberately distinct from the PSI *retraining trigger* (0.2) in retrain_drift. The
    # dead parameter made the two look coupled when they are not; it is gone, and the paper
    # quotes psi_det explicitly.
    """Ground truth = a step change vs the PREVIOUS window (consecutive KS with an effect-size
    floor). Detectors watch the same stream: ADWIN over a FIXED deployed model's per-sample error
    (continuous, never reset — the textbook setup: alarm when deployed error grows), PSI over
    consecutive feature distributions, SHAP-shift over the deployed model's attribution shares
    (model-aware yet label-free: fires when HOW the model decides changes, measured as the total-
    variation distance between consecutive windows' normalised mean-|SHAP| vectors), their OR
    combos, and a naive every-k schedule."""
    idx = np.array_split(np.arange(len(df)), windows)
    chunks = [df.iloc[i] for i in idx if len(i)]
    base = chunks[0]
    model = fit(base, feats)          # fixed deployed model (never retrained here)
    adwin = ADWIN(delta=adwin_delta)
    for e in (model.predict(base[feats]) != base.y).astype(float):
        adwin.update(e)
    s_prev = shap_share(model, base[feats])   # None when shap is unavailable

    truth, psi_fire, adw_fire, shp_fire, fix_fire = [], [], [], [], []
    prev = base
    for w in range(1, len(chunks)):
        cur = chunks[w]
        is_drift, _ = ks_ground_truth(prev, cur, feats)
        truth.append(is_drift)
        mean_psi = float(np.mean([psi(prev[c].values, cur[c].values) for c in feats]))
        psi_fire.append(mean_psi > psi_det)
        err = (model.predict(cur[feats]) != cur.y).astype(float)
        adw_fire.append(bool(any(adwin.update(e) for e in err)))
        if s_prev is not None:
            s_cur = shap_share(model, cur[feats])
            shp_fire.append(0.5 * float(np.abs(s_cur - s_prev).sum()) > shap_tau)
            s_prev = s_cur
        fix_fire.append(w % period == 0)
        prev = cur
    fires = {"ADWIN (error stream)": adw_fire,
             "PSI (feature distribution)": psi_fire}
    if shp_fire:
        fires["SHAP-shift (attribution)"] = shp_fire
    fires["ADWIN + PSI (combined)"] = [a or p for a, p in zip(adw_fire, psi_fire)]
    fires["Fixed-schedule (naive)"] = fix_fire
    return truth, fires


def score_detector(truth, fire, horizon=2):
    """Event-based scoring. Each true-drift window is an event; it is DETECTED if the monitor
    alarms within [onset, onset+horizon]. detection rate = detected events / events; mean lag =
    windows from onset to the crediting alarm; false-alarm rate = alarms that credit no event,
    over non-drift windows; precision = crediting alarms / all alarms."""
    onsets = [i for i, t in enumerate(truth) if t]
    detected, lags, credited = 0, [], set()
    for o in onsets:
        for j in range(o, min(len(fire), o + horizon + 1)):
            if fire[j]:
                detected += 1
                lags.append(j - o)
                credited.add(j)
                break
    alarms = [j for j, f in enumerate(fire) if f]
    fp = [j for j in alarms if j not in credited]
    neg = len(truth) - len(onsets)
    det_rate = detected / len(onsets) if onsets else float("nan")
    fa_rate = len(fp) / neg if neg else float("nan")
    prec = len(credited) / len(alarms) if alarms else float("nan")
    mean_lag = float(np.mean(lags)) if lags else float("nan")
    return det_rate, fa_rate, mean_lag, prec


# ----------------------------- budget frontier -----------------------------
def run_budget(df, feats, windows, budgets, period=3, psi_tau=0.10, f1_drop=0.1, seed=0):
    """AUTC (mean per-window F1) and total labels spent per strategy at each per-retrain budget B.
    The four columns isolate two design axes against a common backbone:
      static   : never retrain (labels = 0 after init).
      periodic : retrain every `period` windows, labelling B RANDOM samples per retrain.
      drift    : retrain when the window's mean PSI against the last refit window exceeds psi_tau,
                 labelling B RANDOM samples per retrain (same selection as periodic, different
                 TRIGGER). The trigger is label-free here: until 2026-10-09 it also fired on an F1
                 drop computed from the window's FULL labels, which the budget never counted, so the
                 trigger spent labels the frontier did not charge. Every label it uses is now one it
                 pays for: B per retrain, exactly as periodic and active.
      active   : retrain every `period` windows, labelling the B most-UNCERTAIN samples per retrain
                 (same trigger as periodic, different SELECTION -> label efficiency).
    So periodic-vs-active isolates sample selection; periodic-vs-drift isolates the trigger.
    `seed` sets the forest and the random draws together; seed 0 is the run the paper's grid prints."""
    idx = np.array_split(np.arange(len(df)), windows)
    chunks = [df.iloc[i] for i in idx if len(i)]
    base = chunks[0]
    out = {}
    for B in budgets:
        f1s = {k: [] for k in ("static", "periodic", "drift", "active")}
        labels = {k: 0 for k in f1s}
        models = {k: fit(base, feats, seed) for k in f1s}
        pools = {k: base.copy() for k in f1s}
        ref = {k: base for k in f1s}
        last_f1 = {k: 1.0 for k in f1s}
        # per-strategy RNG so a column's random draws never depend on how often another retrains
        rngs = {k: np.random.RandomState(seed) for k in f1s}
        for w in range(1, len(chunks)):
            cur = chunks[w]
            for k in f1s:
                p = models[k].predict_proba(cur[feats])[:, 1]
                f1 = f1_score(cur.y, (p >= 0.5).astype(int), zero_division=0)
                f1s[k].append(f1)
                if k == "static":
                    last_f1[k] = f1
                    continue
                if k in ("periodic", "active"):
                    fire = (w % period == 0)
                else:  # drift: the label-free PSI term of the trigger (see the docstring)
                    psi_mean = float(np.mean([psi(ref[k][c].values, cur[c].values) for c in feats]))
                    fire = psi_mean > psi_tau
                last_f1[k] = f1
                if not fire:
                    continue
                if k == "active":
                    # kind="stable" is load-bearing, not tidiness. A 200-tree forest emits very
                    # few distinct probabilities -- one window here has 2,656 rows across 154
                    # distinct uncertainty values, the largest tie group holding 995 -- so "the B
                    # most uncertain rows" is undefined for most B, and the default quicksort
                    # resolved it differently between runs. That made this column irreproducible
                    # (the B=1600 cell moved in the third decimal), which for a number a paper
                    # prints is a defect regardless of its size. Ties now break by row order.
                    take = cur.iloc[np.argsort(np.abs(p - 0.5), kind="stable")[:B]]
                else:
                    take = cur.sample(min(B, len(cur)), random_state=rngs[k])
                pools[k] = pd.concat([pools[k], take])
                labels[k] += len(take)
                models[k] = fit(pools[k], feats, seed)
                ref[k] = cur
        out[B] = {k: (float(np.mean(f1s[k])), labels[k]) for k in f1s}
    return out


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/interim/drift_compphish.csv")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--period", type=int, default=3)
    ap.add_argument("--psi", type=float, default=0.2)
    ap.add_argument("--adwin-delta", type=float, default=0.002)
    ap.add_argument("--shap-tau", type=float, default=0.05,
                    help="TV-distance threshold for the SHAP-shift attribution monitor")
    ap.add_argument("--budgets", default="50,200,800")
    args = ap.parse_args()

    df, feats = load(args.inp, "collected_at", spread_undated=True)
    print(f"n={len(df)} features={len(feats)} windows={args.windows}")

    truth, fires = run_detection(df, feats, args.windows, args.period, args.adwin_delta,
                                 shap_tau=args.shap_tau)
    print(f"\n== Drift detection ({sum(truth)}/{len(truth)} windows are KS-true drift) ==")
    print(f"{'detector':<30}{'det.rate':>9}{'FA.rate':>9}{'mean.lag':>9}{'prec':>7}")
    for name, fire in fires.items():
        dr, fa, lag, pr = score_detector(truth, fire)
        print(f"{name:<30}{dr:>9.2f}{fa:>9.2f}{lag:>9.2f}{pr:>7.2f}")

    budgets = [int(b) for b in args.budgets.split(",")]
    bud = run_budget(df, feats, args.windows, budgets, period=args.period, psi_tau=args.psi)
    print("\n== Labelling-budget frontier (AUTC = mean F1; labels = total spent) ==")
    print(f"{'B/retrain':>9}  " + "".join(f"{k:>18}" for k in ("static", "periodic", "drift", "active")))
    for B in budgets:
        r = bud[B]
        print(f"{B:>9}  " + "".join(f"{r[k][0]:>10.3f}/{r[k][1]:<7}" for k in ("static", "periodic", "drift", "active")))


if __name__ == "__main__":
    main()
