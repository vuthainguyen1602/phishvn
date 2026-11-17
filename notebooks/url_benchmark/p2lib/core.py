"""Notebook 01: the seven-family benchmark and the temporal protocol with its controls."""
import os, collections
from types import SimpleNamespace

import numpy as np
import pandas as pd
from .common import (COMPPHISH, DETERMINISTIC, FAMILIES, _metrics, add_label, benign_mask,
                     f1_score, load, make_any_model, match_prior, pr_curve_row,
                     precision_recall_curve, registered_domain, run_one, split_phishing,
                     train_test_split, weighting_params, write_curves)



def run_benchmark(**overrides):
    args = SimpleNamespace(**{**{'inp': 'data/processed/vn_compphish.csv', 'families': None, 'seeds': 5, 'out': 'data/processed/p2/p2_benchmark.csv', 'curves': 'data/processed/p2/p2_pr_curves.csv', 'unweighted': True}, **overrides})
    if args.families is None:
        args.families = FAMILIES
    df = add_label(pd.read_csv(args.inp))
    feats = [c for c in COMPPHISH if c in df.columns]
    if len(feats) != len(COMPPHISH):
        raise SystemExit(f"corpus lacks the lexical URL schema ({len(feats)}/{len(COMPPHISH)} features)")
    df = df.dropna(subset=feats)
    protocols = {}
    X, y = df[feats].to_numpy(float), df["y"].to_numpy(int)
    protocols["random"] = lambda s: train_test_split(X, y, test_size=0.30, stratify=y,
                                                     random_state=s)
    if "split" in df.columns:
        tr, te = df[df.split == "train"], df[df.split == "test"]
        Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
        # bind the bundled split now: the loop below reassigns Xtr/Xte/ytr/yte, and a closure over
        # the names would hand the temporal protocol the last random split instead (the bug in the
        # runs before 2026-11-17, whose "temporal" rows were the random split of seed 4)
        split = (Xtr, Xte, ytr, yte)
        protocols["temporal"] = lambda s, split=split: split
        print(f"temporal split: train={len(tr)} test={len(te)}")
    else:
        print("[!] no `split` column — temporal protocol skipped")
    print(f"corpus: {len(df)} rows, {int(y.sum())} phishing / {len(y) - int(y.sum())} benign")
    rows, curves = [], []
    for name in args.families:
        for proto, splitter in protocols.items():
            seeds = [0] if (name in DETERMINISTIC and proto == "temporal") else range(args.seeds)
            for s in seeds:
                Xtr, Xte, ytr, yte = splitter(s)
                met, y_t, sc = run_one(name, Xtr, ytr, Xte, yte, s, return_scores=True,
                                       unweighted=args.unweighted)
                met["protocol"] = proto
                rows.append(met)
                curves.append({"family": name, "protocol": proto, "seed": s,
                               "precision": pr_curve_row(y_t, sc)})
                print(f"  {name:<13} {proto:<9} seed={s} "
                      f"F1={met['F1']:.3f} PR-AUC={met['PR-AUC']:.3f} "
                      f"FPR@R0.90={met['FPR@R0.90']:.3f} ({met['fit_seconds']}s)")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    agg = (out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
              .agg(["mean", "std"]).round(4))
    print(agg.to_string())


def run_strict(**overrides):
    args = SimpleNamespace(**{**{'families': None, 'seeds': 5, 'cut': 0.7, 'test_after': None, 'out': 'data/processed/p2/p2_temporal_strict.csv', 'curves': 'data/processed/p2/p2_pr_curves_strict.csv', 'guard_control': False, 'benign_grouped': False, 'match_prior': False}, **overrides})
    if args.families is None:
        args.families = FAMILIES
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, args.test_after)
    if ph_te.empty:
        raise SystemExit(f"no dated phishing after {args.test_after}: the refresh window has "
                         f"not arrived (corpus max date {ph.date.max().date()})")
    print(f"phishing dated: {len(ph)}  train<= {ph_tr.date.max().date()} ({len(ph_tr)})  "
          f"test>= {ph_te.date.min().date()} ({len(ph_te)}; {n_leaked} test rows dropped as "
          f"same-registrable-domain re-detections)")
    print(f"benign pool: {len(be)} ({'grouped by registrable domain' if args.benign_grouped else 'random'}"
          f" {args.cut:.2f} per seed)")
    protos = ("temporal_strict", "random_same_rows_guarded") if args.guard_control \
        else ("temporal_strict", "random_same_rows")
    rows, curves = [], []
    for name in args.families:
        for proto in protos:
            # even "deterministic" families need all seeds here: the benign 70/30 mask is drawn
            # per seed, so no family is deterministic under either protocol
            for s in range(args.seeds):
                rng = np.random.RandomState(s)
                bmask = benign_mask(be, rng, args.cut, args.benign_grouped)
                if proto == "temporal_strict":
                    tr = pd.concat([ph_tr, be[bmask]])
                    te = pd.concat([ph_te, be[~bmask]])
                else:
                    pool = pd.concat([ph_tr, ph_te])
                    pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                    p_tr, p_te = pool[pmask], pool[~pmask]
                    if proto == "random_same_rows_guarded":
                        # The temporal arm's guard, applied to the random control: drop from TEST
                        # any row whose registrable domain occurs in TRAIN. Same direction, same
                        # granularity, same PSL logic -- so the only remaining difference between
                        # the two arms is how rows were assigned.
                        p_te = p_te[~p_te.rdom.isin(set(p_tr.rdom))]
                    tr = pd.concat([p_tr, be[bmask]])
                    te = pd.concat([p_te, be[~bmask]])
                    if args.match_prior:
                        te = match_prior(te, len(ph_te) / (len(ph_te) + int((~bmask).sum())), rng)
                met, y_t, sc = run_one(name, tr[feats].to_numpy(float),
                                       tr["y"].to_numpy(int), te[feats].to_numpy(float),
                                       te["y"].to_numpy(int), s, return_scores=True)
                met["protocol"] = proto
                rows.append(met)
                curves.append({"family": name, "protocol": proto, "seed": s,
                               "precision": pr_curve_row(y_t, sc)})
                print(f"  {name:<13} {proto:<17} seed={s} F1={met['F1']:.3f} "
                      f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


PRIOR_ARMS = ("full_natural", "full_matched", "dated_natural", "dated_matched")


def prior_max_f1(y, score) -> float:
    p, r, _ = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.nanmax(f))


def draw(arm: str, ph_all: pd.DataFrame, ph_dated: pd.DataFrame, be: pd.DataFrame,
         rng: np.random.RandomState) -> pd.DataFrame:
    n_dated, n_be = len(ph_dated), len(be)
    prior_full = len(ph_all) / (len(ph_all) + n_be)
    if arm == "full_natural":
        return pd.concat([ph_all, be])
    if arm == "full_matched":
        idx = rng.choice(len(ph_all), n_dated, replace=False)
        return pd.concat([ph_all.iloc[np.sort(idx)], be])
    if arm == "dated_natural":
        return pd.concat([ph_dated, be])
    if arm == "dated_matched":
        n_keep = int(round(n_dated * (1 - prior_full) / prior_full))
        idx = rng.choice(n_be, n_keep, replace=False)
        return pd.concat([ph_dated, be.iloc[np.sort(idx)]])
    raise ValueError(arm)


def run_prior_control(**overrides):
    a = SimpleNamespace(**{**{'families': None, 'seeds': 5, 'out': 'data/processed/p2/p2_prior_control.csv'}, **overrides})
    if a.families is None:
        a.families = FAMILIES
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph_all = df[df.y == 1].reset_index(drop=True)
    ph_dated = ph_all[ph_all.date.notna()].reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    print(f"phishing {len(ph_all)} (dated {len(ph_dated)}), benign {len(be)}; "
          f"prior full {len(ph_all)/(len(ph_all)+len(be)):.3f}, "
          f"dated {len(ph_dated)/(len(ph_dated)+len(be)):.3f}")
    rows = []
    for arm in PRIOR_ARMS:
        for s in range(a.seeds):
            pool = draw(arm, ph_all, ph_dated, be, np.random.RandomState(1000 + s))
            X, y = pool[feats].to_numpy(float), pool["y"].to_numpy(int)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                                  random_state=s)
            for name in a.families:
                met, y_t, sc = run_one(name, Xtr, ytr, Xte, yte, s, return_scores=True)
                met.update(arm=arm, prior=round(float(y.mean()), 4), n_rows=len(pool),
                           n_phish=int(y.sum()), n_benign=int(len(y) - y.sum()),
                           n_test=len(yte), maxF1=prior_max_f1(y_t, sc))
                rows.append(met)
                print(f"  {arm:<14} seed={s} {name:<13} F1={met['F1']:.3f} "
                      f"maxF1={met['maxF1']:.3f} PR-AUC={met['PR-AUC']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["arm", "family"])[["F1", "maxF1", "PR-AUC"]].mean().round(4)
             .unstack("arm").to_string())


DESIGNS = ("random_full", "random_same_rows", "temporal_strict")


VAL_FRAC = 0.10


def best_threshold(y, score) -> float:
    p, r, thr = precision_recall_curve(y, score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)[:-1]
    return float(thr[int(np.nanargmax(f))])


def carve(tr: pd.DataFrame, design: str, seed: int):
    rng = np.random.RandomState(10_000 + seed)
    if design == "temporal_strict":
        ph = tr[tr.y == 1].sort_values("date")
        be = tr[tr.y == 0]
        k = int(len(ph) * (1 - VAL_FRAC))
        bm = rng.rand(len(be)) < VAL_FRAC
        fit = pd.concat([ph.iloc[:k], be[~bm]])
        val = pd.concat([ph.iloc[k:], be[bm]])
        return fit, val
    fit, val = train_test_split(tr, test_size=VAL_FRAC, stratify=tr.y, random_state=seed)
    return fit, val


def run_val_threshold(**overrides):
    a = SimpleNamespace(**{**{'families': None, 'seeds': 5, 'cut': 0.7, 'out': 'data/processed/p2/p2_val_threshold.csv'}, **overrides})
    if a.families is None:
        a.families = FAMILIES
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, a.cut)
    rows = []
    for design in DESIGNS:
        for s in range(a.seeds):
            if design == "random_full":
                tr, te = train_test_split(df, test_size=0.30, stratify=df.y, random_state=s)
            else:
                rng = np.random.RandomState(s)          # the canonical per-seed benign mask
                bmask = rng.rand(len(be)) < a.cut
                if design == "temporal_strict":
                    tr = pd.concat([ph_tr, be[bmask]])
                    te = pd.concat([ph_te, be[~bmask]])
                else:
                    pool = pd.concat([ph_tr, ph_te])
                    pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                    tr = pd.concat([pool[pmask], be[bmask]])
                    te = pd.concat([pool[~pmask], be[~bmask]])
            fit, val = carve(tr, design, s)
            for name in a.families:
                m = make_any_model(name, s, weighting_params(name))
                m.fit(fit[feats].to_numpy(float), fit.y.to_numpy(int))
                sv = m.predict_proba(val[feats].to_numpy(float))[:, 1]
                tau = best_threshold(val.y.to_numpy(int), sv)
                st = m.predict_proba(te[feats].to_numpy(float))[:, 1]
                yte = te.y.to_numpy(int)
                met = _metrics(yte, st)                      # F1 here is at tau = 0.5
                met.update(family=name, seed=s, protocol=design, tau=round(tau, 4),
                           F1_val=f1_score(yte, (st >= tau).astype(int)),
                           F1_oracle=prior_max_f1(yte, st),
                           n_fit=len(fit), n_val=len(val), n_test=len(te),
                           val_last_date=str(val[val.y == 1].date.max().date())
                           if design == "temporal_strict" else "")
                rows.append(met)
                print(f"  {design:<17} seed={s} {name:<13} tau*={tau:.3f} "
                      f"F1@0.5={met['F1']:.3f} F1@tau*={met['F1_val']:.3f} "
                      f"oracle={met['F1_oracle']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["protocol", "family"])[["F1", "F1_val", "F1_oracle"]].mean().round(4)
             .to_string())


def guard(te: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    return te[~te.rdom.isin(set(tr.rdom))]


def run_twosided(**overrides):
    a = SimpleNamespace(**{**{'families': None, 'seeds': 5, 'cut': 0.7, 'out': 'data/processed/p2/p2_temporal_twosided.csv'}, **overrides})
    if a.families is None:
        a.families = FAMILIES
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    dated = df[df.date.notna()].sort_values("date").reset_index(drop=True)
    ph = dated[dated.y == 1]
    be = dated[dated.y == 0]
    cut_date = ph.date.iloc[int(len(ph) * a.cut) - 1]      # the canonical phishing cut
    tr_all = dated[dated.date <= cut_date]
    te_all = guard(dated[dated.date > cut_date], tr_all)
    n_dropped = int((dated.date > cut_date).sum()) - len(te_all)
    print(f"cut {cut_date.date()}: train {len(tr_all)} (phish {int(tr_all.y.sum())}, benign "
          f"{int((tr_all.y == 0).sum())}), test {len(te_all)} (phish {int(te_all.y.sum())}, benign "
          f"{int((te_all.y == 0).sum())}); {n_dropped} test rows dropped by the domain guard")
    rate = {1: (tr_all.y == 1).sum() / len(ph), 0: (tr_all.y == 0).sum() / len(be)}
    rows = []
    for name in a.families:
        for proto in ("temporal_both", "random_both_guarded"):
            for s in range(a.seeds):
                if proto == "temporal_both":
                    tr, te = tr_all, te_all
                else:
                    rng = np.random.RandomState(s)
                    mask = np.array([rng.rand() < rate[int(y)] for y in dated.y])
                    tr, te = dated[mask], guard(dated[~mask], dated[mask])
                met, _, _ = run_one(name, tr[feats].to_numpy(float), tr.y.to_numpy(int),
                                    te[feats].to_numpy(float), te.y.to_numpy(int), s,
                                    return_scores=True)
                met.update(protocol=proto, n_train=len(tr), n_test=len(te),
                           n_test_benign=int((te.y == 0).sum()), cut_date=str(cut_date.date()))
                rows.append(met)
                print(f"  {name:<13} {proto:<20} seed={s} F1={met['F1']:.3f} "
                      f"ROC={met['ROC-AUC']:.3f} PR={met['PR-AUC']:.3f} "
                      f"FPR@0.90={met['FPR@R0.90']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"\n[+] {len(out)} runs -> {a.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


OUT_PROTO = "data/processed/p2/p2_dup_leakage_protocols.csv"


def _group_labels(keys, labels):
    acc = collections.defaultdict(list)
    for k, lab in zip(keys, labels):
        acc[k].append(int(lab))
    return acc


def _rates(fv, rdom, itr, ite):
    tr_fv, tr_rd = set(fv[itr]), set(rdom[itr])
    return (float(np.mean([k in tr_fv for k in fv[ite]])),
            float(np.mean([r in tr_rd for r in rdom[ite]])))


def protocol_rates(seeds: int, out_path: str):
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]

    def keys(frame):
        return (np.array([hash(t) for t in map(tuple, frame[feats].to_numpy(float))]),
                frame["rdom"].to_numpy())

    rows = []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        bmask = rng.rand(len(be)) < 0.70
        pool = pd.concat([ph_tr, ph_te])
        pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
        for proto, tr, te in (
                ("temporal_strict", pd.concat([ph_tr, be[bmask]]),
                 pd.concat([ph_te, be[~bmask]])),
                ("random_same_rows", pd.concat([pool[pmask], be[bmask]]),
                 pd.concat([pool[~pmask], be[~bmask]]))):
            fv = np.concatenate([keys(tr)[0], keys(te)[0]])
            rd = np.concatenate([keys(tr)[1], keys(te)[1]])
            itr = np.arange(len(tr))
            ite = np.arange(len(tr), len(tr) + len(te))
            twin, dom = _rates(fv, rd, itr, ite)
            # the phishing side alone is where the guard operates and where drift lives
            pv_tr = keys(tr[tr.y == 1])
            pv_te = keys(te[te.y == 1])
            fvp = np.concatenate([pv_tr[0], pv_te[0]])
            rdp = np.concatenate([pv_tr[1], pv_te[1]])
            twin_p, dom_p = _rates(fvp, rdp, np.arange(len(pv_tr[0])),
                                   np.arange(len(pv_tr[0]), len(fvp)))
            rows.append({"protocol": proto, "seed": s,
                         "test_twin_rate": round(twin, 4),
                         "test_regdom_shared_rate": round(dom, 4),
                         "phish_test_twin_rate": round(twin_p, 4),
                         "phish_test_regdom_shared_rate": round(dom_p, 4)})
            print(f"  {proto:<17} seed={s} twin={twin:.4f} regdom={dom:.4f} "
                  f"(phishing side {twin_p:.4f}/{dom_p:.4f})")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"[+] {len(out)} rows -> {out_path}")
    return out


def run_dup_leakage(**overrides):
    args = SimpleNamespace(**{**{'inp': 'data/processed/vn_compphish.csv', 'seeds': 5, 'out': 'data/processed/p2/p2_dup_leakage.csv'}, **overrides})
    df = add_label(pd.read_csv(args.inp)).dropna(subset=COMPPHISH)
    X = df[COMPPHISH].to_numpy(float)
    y = df["y"].to_numpy(int)
    # Exact float tuples, not rounded strings: rounding merges vectors the learner can still
    # tell apart and would overstate the leak.
    fv = np.array([hash(t) for t in map(tuple, X)])
    rdom = df["dom"].astype(str).str.lower().map(registered_domain).to_numpy()
    groups = _group_labels(fv, y)
    mixed = {k for k, s in groups.items() if len(set(s)) > 1}
    mixed_rows = int(np.isin(fv, list(mixed)).sum()) if mixed else 0
    # The schema ceiling: majority label per exact feature vector, fitted and scored on
    # everything. No split, no learner -- the best F1 these 21 features admit on this corpus.
    maj = {k: int(sum(g) * 2 >= len(g)) for k, g in groups.items()}
    ceiling = float(f1_score(y, np.array([maj[k] for k in fv])))
    census = {
        "rows": len(df),
        "unique_regdom": int(len(set(rdom))),
        "unique_featvec": int(len(groups)),
        "mixed_featvec_groups": len(mixed),
        "rows_in_mixed_groups": mixed_rows,
        "positive_rate": float(y.mean()),
        "schema_ceiling_f1": round(ceiling, 4),
    }
    print("census: " + "  ".join(f"{k}={v}" for k, v in census.items()))
    rows = []
    for s in range(args.seeds):
        itr, ite = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y,
                                    random_state=s)
        twin, dom = _rates(fv, rdom, itr, ite)

        acc = _group_labels(fv[itr], y[itr])
        table = {k: int(sum(v) * 2 >= len(v)) for k, v in acc.items()}
        fallback = int(y[itr].mean() >= 0.5)
        pred = np.array([table.get(k, fallback) for k in fv[ite]])

        r = dict(census, seed=s,
                 test_twin_rate=round(twin, 4),
                 test_regdom_shared_rate=round(dom, 4),
                 lookup_oracle_f1=round(float(f1_score(y[ite], pred)), 4),
                 all_positive_f1=round(float(f1_score(y[ite], np.ones(len(ite), int))), 4))
        rows.append(r)
        print(f"  seed={s} twin={twin:.4f} regdom={dom:.4f} "
              f"oracle_F1={r['lookup_oracle_f1']:.4f} allpos_F1={r['all_positive_f1']:.4f}")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} rows -> {args.out}")
    print(out[["test_twin_rate", "test_regdom_shared_rate", "lookup_oracle_f1",
               "all_positive_f1"]].mean().round(4).to_string())
    print("\ndated-row designs (run_p2_temporal_strict's own splits):")
    protocol_rates(args.seeds, OUT_PROTO)


SEEDS = 5


def _save(df, relative):
    path = relative
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"[+] {path}")


def tier_sensitivity(df):
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, _ = split_phishing(ph, 0.70)
    rows = []
    for family in ("CatBoost", "LogReg"):
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            bmask = rng.rand(len(be)) < 0.70
            pool = pd.concat([ph_tr, ph_te]).reset_index(drop=True)
            pmask = rng.rand(len(pool)) < len(ph_tr) / len(pool)
            protocols = {
                "phishing-temporal": (pd.concat([ph_tr, be[bmask]]),
                                       pd.concat([ph_te, be[~bmask]])),
                "random-same-rows": (pd.concat([pool[pmask], be[bmask]]),
                                     pd.concat([pool[~pmask], be[~bmask]])),
            }
            for protocol, (tr, te) in protocols.items():
                model = make_any_model(family, seed, weighting_params(family))
                model.fit(tr[feats].to_numpy(float), tr.y.to_numpy(int))
                score = model.predict_proba(te[feats].to_numpy(float))[:, 1]
                pred = score >= 0.5
                rows.append({"family": family, "protocol": protocol, "seed": seed,
                             "tier": "overall", "n": len(te),
                             "metric": "F1", "value": f1_score(te.y, pred)})
                for tier in ("bronze", "silver", "gold"):
                    mask = (te.y.to_numpy(int) == 1) & te.tier.astype(str).eq(tier).to_numpy()
                    rows.append({"family": family, "protocol": protocol, "seed": seed,
                                 "tier": tier, "n": int(mask.sum()), "metric": "recall",
                                 "value": float(pred[mask].mean()) if mask.any() else np.nan})
    out = pd.DataFrame(rows)
    _save(out, "data/processed/p2/p2_tier_sensitivity.csv")
