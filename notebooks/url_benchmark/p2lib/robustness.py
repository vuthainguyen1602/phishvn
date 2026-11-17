"""Notebook 03: tuning, stacking, whether drift can be forecast, and the label-noise audit."""
import os, time, warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from .common import (COMPPHISH, FAMILIES, _metrics, add_label, corrected_paired_t, load,
                     make_any_model, make_model, pr_curve_row, split_phishing, weighting_params,
                     write_curves)



SPACE = {
    "LogReg": [("C", -3, 3, "logfloat")],
    "RandomForest": [("n_estimators", 100, 600, "int"), ("max_depth", 3, 50, "int"),
                     ("min_samples_leaf", 1, 10, "int"), ("max_features", 0.1, 1.0, "float")],
    "HistGB": [("learning_rate", -2, -0.5, "logfloat"), ("max_leaf_nodes", 15, 255, "int"),
               ("max_depth", 3, 30, "int"), ("l2_regularization", 0.0, 10.0, "float"),
               ("max_iter", 100, 500, "int")],
    "MLP": [("mlpclassifier__alpha", -5, -2, "logfloat"),
            ("mlpclassifier__learning_rate_init", -4, -2, "logfloat")],
    "CatBoost": [("iterations", 100, 600, "int"), ("depth", 4, 10, "int"),
                 ("learning_rate", -2, -0.5, "logfloat"), ("l2_leaf_reg", 1.0, 10.0, "float")],
    "XGBoost": [("n_estimators", 100, 600, "int"), ("max_depth", 3, 10, "int"),
                ("learning_rate", -2, -0.5, "logfloat"), ("subsample", 0.5, 1.0, "float"),
                ("colsample_bytree", 0.5, 1.0, "float")],
}


def _factory(model, seed, params):
    if model in ("CatBoost", "XGBoost", "LightGBM"):
        return make_any_model(model, seed, params)
    return make_model(model, seed, {**(weighting_params(model) or {}), **(params or {})})


def _decode(pos, space):
    params = {}
    for x, (name, lo, hi, kind) in zip(pos, space):
        v = lo + x * (hi - lo)
        if kind == "int":
            params[name] = int(round(v))
        elif kind == "logfloat":
            params[name] = float(10.0 ** v)
        else:
            params[name] = float(v)
    return params


def _fitness(model, params, X, y, seed):
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return float(cross_val_score(_factory(model, seed, params), X, y,
                                     scoring="average_precision", cv=cv, n_jobs=-1).mean())
    except Exception as exc:
        # Invalid candidates may occur near a search-space boundary, but silently
        # swallowing every failure makes a broken estimator indistinguishable from
        # a genuinely poor configuration.
        warnings.warn(
            f"fitness evaluation failed for {model} with {params}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return -1.0


def gwo_budget(n_agents, iters):
    return n_agents * (iters + 1)


def _gwo_step(wolves, leaders, a, rng):
    old = np.asarray(wolves, dtype=float)
    estimates = []
    for lead in np.asarray(leaders, dtype=float):
        r1 = rng.random(old.shape)
        r2 = rng.random(old.shape)
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        estimates.append(lead - A * np.abs(C * lead - old))
    return np.clip(np.mean(estimates, axis=0), 0.0, 1.0)


def _update_leaders(leader_pos, leader_fit, wolves, fit):
    candidate_pos = np.vstack((leader_pos, wolves))
    candidate_fit = np.concatenate((leader_fit, fit))
    order = np.argsort(-candidate_fit, kind="stable")[:3]
    return candidate_pos[order].copy(), candidate_fit[order].copy()


def random_search(model, X, y, seed=0, budget=48):
    space = SPACE.get(model, [])
    if not space:
        return {}, float("nan")
    rng = np.random.default_rng(seed)
    best_p, best_f = {}, -1.0
    for _ in range(budget):
        p = _decode(rng.random(len(space)), space)
        f = _fitness(model, p, X, y, seed)
        if f > best_f:
            best_p, best_f = p, f
    return best_p, best_f


def gwo_search(model, X, y, seed=0, n_agents=6, iters=8):
    space = SPACE.get(model, [])
    if not space:
        return {}, float("nan")
    if n_agents < 3:
        raise ValueError("GWO requires at least three agents (alpha, beta, delta)")
    if iters < 1:
        raise ValueError("GWO requires at least one iteration")
    d = len(space)
    rng = np.random.default_rng(seed)
    wolves = rng.random((n_agents, d))
    fit = np.array([_fitness(model, _decode(w, space), X, y, seed) for w in wolves])
    # alpha/beta/delta = three best (we maximise, so take the largest)
    order = np.argsort(-fit)
    leaders = wolves[order[:3]].copy()
    leader_fit = fit[order[:3]].copy()
    for t in range(iters):
        a = 2 - 2 * t / max(1, iters - 1)  # linearly 2 -> 0
        wolves = _gwo_step(wolves, leaders, a, rng)
        fit = np.array([_fitness(model, _decode(w, space), X, y, seed) for w in wolves])
        leaders, leader_fit = _update_leaders(leaders, leader_fit, wolves, fit)
    return _decode(leaders[0], space), float(leader_fit[0])


HPO_FAMILIES = ["CatBoost", "HistGB", "LogReg"]
N_AGENTS, ITERS = 6, 8


def run_hpo(**overrides):
    args = SimpleNamespace(**{**{'families': None, 'seeds': 3, 'search_seeds': 1, 'search_seed_start': 0, 'out': 'data/processed/p2/p2_hpo.csv'}, **overrides})
    if args.families is None:
        args.families = HPO_FAMILIES
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    rng = np.random.RandomState(0)
    bmask = rng.rand(len(be)) < 0.70
    tr = pd.concat([ph_tr, be[bmask]])
    te = pd.concat([ph_te, be[~bmask]])
    Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
    Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
    print(f"phishing-temporal: train={len(tr)} test={len(te)}")
    def eval_test(name, params):
        vals = []
        for s in range(args.seeds):
            m = _factory(name, s, params).fit(Xtr, ytr)
            vals.append(_metrics(yte, m.predict_proba(Xte)[:, 1]))
        return {k: (float(np.mean([v[k] for v in vals])),
                    float(np.std([v[k] for v in vals]))) for k in vals[0]}
    budget = gwo_budget(N_AGENTS, ITERS)
    rows = []
    def checkpoint():
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    for name in args.families:
        res = eval_test(name, {})
        print(f"[default {name}] " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items()))
        rows.append({"family": name, "config": "default", "cv_prauc": np.nan, "params": "{}",
                     **{k: m for k, (m, s) in res.items()},
                     **{k + "_std": s for k, (m, s) in res.items()}})
        checkpoint()

        for search_seed in range(args.search_seed_start,
                                 args.search_seed_start + args.search_seeds):
            t0 = time.time()
            gp, gf = gwo_search(name, Xtr, ytr, seed=search_seed,
                                n_agents=N_AGENTS, iters=ITERS)
            rp, rf = random_search(name, Xtr, ytr, seed=search_seed, budget=budget)
            adopted = gf >= rf  # pre-specified, equal-budget selection rule
            best_params, src = (gp, "gwo") if adopted else (rp, "random-search")
            res = eval_test(name, best_params)
            print(f"[tuned  {name} seed={search_seed}] CV: gwo={gf:.4f} random={rf:.4f} -> "
                  f"adopted {src} ({time.time()-t0:.0f}s, budget={budget})")
            print("               " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in res.items()))
            rows.append({"family": name, "config": f"tuned({src})", "search_seed": search_seed,
                         "cv_prauc": max(gf, rf), "cv_prauc_gwo": gf,
                         "cv_prauc_random": rf, "budget": budget,
                         "params": str(best_params),
                         **{k: m for k, (m, s) in res.items()},
                         **{k + "_std": s for k, (m, s) in res.items()}})
            checkpoint()
    print(f"[+] -> {args.out}")


def hybrid_select(Xtr: np.ndarray, ytr: np.ndarray, names: list[str], k: int,
                  seed: int) -> list[int]:
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(Xtr, ytr, random_state=seed)
    order = np.argsort(mi)[::-1]
    with np.errstate(invalid="ignore"):  # zero-variance columns yield NaN correlations
        corr = np.nan_to_num(np.corrcoef(Xtr, rowvar=False), nan=0.0)
    kept: list[int] = []
    for i in order:
        if any(abs(corr[i, j]) > 0.90 for j in kept):
            continue
        kept.append(int(i))
        if len(kept) == k:
            break
    return sorted(kept)


ABBREV = {"RandomForest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CB",
          "HistGB": "HGB", "LogReg": "LR", "MLP": "MLP"}


def make_stacker(bases: list[str], seed: int):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    return StackingClassifier(
        # members are configured like the single-family rows they are compared with (uniform
        # weighting unless P2_UNWEIGHTED=0), so the stack adds a combiner and nothing else
        estimators=[(ABBREV[b], make_any_model(b, seed, weighting_params(b))) for b in bases],
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba", cv=5, n_jobs=-1)


def run_stacking(**overrides):
    args = SimpleNamespace(**{**{'seeds': 5, 'k': 12, 'bases': ['RandomForest+XGBoost'], 'cut': 0.7, 'test_after': None, 'out': 'data/processed/p2/p2_stacking_baseline.csv', 'curves': 'data/processed/p2/p2_pr_curves_stacking.csv', 'hfs': False}, **overrides})
    combos = [c.split("+") for c in args.bases]
    for combo in combos:
        for b in combo:
            if b not in ABBREV:
                raise SystemExit(f"unknown family in --bases: {b}")
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, args.test_after)
    if ph_te.empty:
        raise SystemExit(f"no dated phishing after {args.test_after}: the refresh window has "
                         f"not arrived (corpus max date {ph.date.max().date()})")
    print(f"phishing dated: {len(ph)}  train<= {ph_tr.date.max().date()} ({len(ph_tr)})  "
          f"test>= {ph_te.date.min().date()} ({len(ph_te)}; {n_leaked} guard-dropped)  "
          f"benign pool: {len(be)}  features: {len(feats)}")
    rows, curves = [], []
    for proto in ("temporal_strict", "random_same_rows"):
        for s in range(args.seeds):
            rng = np.random.RandomState(s)
            bmask = rng.rand(len(be)) < args.cut
            if proto == "temporal_strict":
                tr = pd.concat([ph_tr, be[bmask]])
                te = pd.concat([ph_te, be[~bmask]])
            else:
                pool = pd.concat([ph_tr, ph_te])
                pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                tr = pd.concat([pool[pmask], be[bmask]])
                te = pd.concat([pool[~pmask], be[~bmask]])
            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)

            sel = hybrid_select(Xtr, ytr, feats, args.k, s) if args.hfs else []
            for combo in combos:
                tag = "Stack[" + "+".join(ABBREV[b] for b in combo) + "]"
                variants = [(tag, list(range(len(feats))))]
                if args.hfs:
                    variants.append((tag + "+HFS", sel))
                for name, cols in variants:
                    t0 = time.time()
                    m = make_stacker(combo, s)
                    m.fit(Xtr[:, cols], ytr)
                    score = m.predict_proba(Xte[:, cols])[:, 1]
                    met = _metrics(yte, score)
                    met.update({"family": name, "seed": s, "protocol": proto,
                                "fit_seconds": round(time.time() - t0, 2),
                                "n_features": len(cols),
                                "features": "" if len(cols) == len(feats) else
                                            "|".join(feats[i] for i in cols)})
                    rows.append(met)
                    curves.append({"family": name, "protocol": proto, "seed": s,
                                   "precision": pr_curve_row(yte, score)})
                    print(f"  {name:<22} {proto:<17} seed={s} F1={met['F1']:.3f} "
                          f"PR-AUC={met['PR-AUC']:.3f} FPR@R0.90={met['FPR@R0.90']:.3f} "
                          f"({met['fit_seconds']}s, {len(cols)} feats)")
            if args.hfs:
                print(f"    HFS kept: {', '.join(feats[i] for i in sel)}")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    write_curves(curves, args.curves)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["family", "protocol"])[["F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"]]
             .mean().round(4).to_string())


OUT_DECAY = "data/processed/p2/p2_forecastability_decay.csv"


OUT_MATRIX = "data/processed/p2/p2_forecastability_shiftmatrix.csv"


OUT_NOVELTY = "data/processed/p2/p2_forecastability_novelty.csv"


N_WIN = 4


def split_canonical(df, cut=0.70):
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    k = int(len(ph) * cut)
    ph_tr, ph_te = ph.iloc[:k], ph.iloc[k:]
    ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
    return feats, ph_tr.reset_index(drop=True), ph_te.reset_index(drop=True), be


def guarded_windows(ph_rest, seen):
    wins, seen = [], set(seen)
    for i in range(N_WIN):
        lo, hi = int(len(ph_rest) * i / N_WIN), int(len(ph_rest) * (i + 1) / N_WIN)
        w = ph_rest.iloc[lo:hi]
        wins.append(w[~w.rdom.isin(seen)])
        seen |= set(w.rdom)
    return wins


def e1_decay(feats, ph_tr, be, seeds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for origin in (0.50, 0.60):
        half = int(len(ph_tr) * origin)
        ph_fit, ph_rest = ph_tr.iloc[:half], ph_tr.iloc[half:].reset_index(drop=True)
        wins = guarded_windows(ph_rest, set(ph_fit.rdom))
        t0 = ph_fit.date.max()
        horiz = [float(w.date.map(lambda d: (d - t0).days).median()) for w in wins]
        print(f"[E1] origin={origin} fit={len(ph_fit)} windows={[len(w) for w in wins]} "
              f"median horizon={[round(h) for h in horiz]}d")
        for s in seeds:
            rng = np.random.RandomState(s)
            bmask = rng.rand(len(be)) < 0.70
            be_tr, be_ev = be[bmask], be[~bmask]
            Xtr = pd.concat([ph_fit, be_tr])[feats].to_numpy(float)
            ytr = np.r_[np.ones(len(ph_fit)), np.zeros(len(be_tr))].astype(int)
            for fam in FAMILIES:
                m = make_any_model(fam, s, weighting_params(fam))
                m.fit(Xtr, ytr)
                sc_be = m.predict_proba(be_ev[feats].to_numpy(float))[:, 1]
                for k, w in enumerate(wins):
                    sc_ph = m.predict_proba(w[feats].to_numpy(float))[:, 1]
                    y = np.r_[np.ones(len(w)), np.zeros(len(be_ev))]
                    sc = np.r_[sc_ph, sc_be]
                    rows.append({"origin": origin, "family": fam, "seed": s, "win": k,
                                 "horizon_d": horiz[k], "n_ph": len(w),
                                 "PR-AUC": average_precision_score(y, sc),
                                 "ROC-AUC": roc_auc_score(y, sc)})
            print(f"[E1] origin={origin} seed={s} done")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DECAY, index=False)

    # verdict: internal decay ranking vs the canonical gap ranking, per origin x metric
    from scipy.stats import spearmanr
    canon = pd.read_csv("data/processed/p2/p2_temporal_strict.csv")
    g = canon.groupby(["family", "protocol"])["PR-AUC"].mean().unstack()
    gap = (g["random_same_rows"] - g["temporal_strict"])
    for origin in (0.50, 0.60):
        for metric in ("PR-AUC", "ROC-AUC"):
            slopes = {}
            sub = out[out.origin == origin]
            for fam in FAMILIES:
                gg = sub[sub.family == fam].groupby("win")[["horizon_d", metric]].mean()
                slopes[fam] = np.polyfit(gg["horizon_d"], gg[metric], 1)[0]
            s_ = pd.Series(slopes)
            rho, p = spearmanr(-s_[gap.index], gap)  # larger decay should mean larger gap
            print(f"[E1] origin={origin} {metric}: Spearman(internal decay vs canonical gap) "
                  f"rho={rho:+.3f} p={p:.3f}")
    print(f"[+] {len(out)} rows -> {OUT_DECAY}")


def e2_matrix(feats, ph_tr, ph_te):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    first = ph_tr.drop_duplicates("rdom", keep="first").reset_index(drop=True)
    blocks, names = [], []
    for i in range(N_WIN):
        lo, hi = int(len(first) * i / N_WIN), int(len(first) * (i + 1) / N_WIN)
        blocks.append(first.iloc[lo:hi])
        names.append(f"W{i + 1}")
    blocks.append(ph_te.drop_duplicates("rdom", keep="first"))
    names.append("TEST")
    med = [b.date.median() for b in blocks]
    rows = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            X = pd.concat([blocks[i], blocks[j]])[feats].to_numpy(float)
            y = np.r_[np.zeros(len(blocks[i])), np.ones(len(blocks[j]))].astype(int)
            oof = cross_val_predict(HistGradientBoostingClassifier(random_state=0), X, y,
                                    cv=5, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, oof)
            dd = abs((med[j] - med[i]).days)
            rows.append({"a": names[i], "b": names[j], "auc": auc, "delta_days": dd,
                         "n_a": len(blocks[i]), "n_b": len(blocks[j])})
            print(f"[E2] {names[i]:>4} vs {names[j]:<4} AUC={auc:.3f} (dt={dd}d, "
                  f"n={len(blocks[i])}/{len(blocks[j])})")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_MATRIX, index=False)
    internal = out[out.a.str.startswith("W") & out.b.str.startswith("W")]
    adj_int = internal[internal.b.str[1].astype(int)
                       - internal.a.str[1].astype(int) == 1]["auc"].mean()
    bound = out[(out.a == "W4") & (out.b == "TEST")]["auc"].iloc[0]
    print(f"[E2] adjacent internal mean AUC={adj_int:.3f}  vs  W4->TEST boundary AUC={bound:.3f}")
    print(f"[+] {len(out)} pairs -> {OUT_MATRIX}")


def e3_novelty(feats, ph_tr, ph_te, be, seeds):
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    rows = []
    for s in seeds:
        rng = np.random.RandomState(s)
        bmask = rng.rand(len(be)) < 0.70
        tr = pd.concat([ph_tr, be[bmask]])
        te = pd.concat([ph_te, be[~bmask]])
        Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
        Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)
        sc = StandardScaler().fit(Xtr)
        nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(Xtr))
        u = nn.kneighbors(sc.transform(Xte))[0].mean(axis=1)
        for fam in ("CatBoost", "LogReg"):
            m = make_any_model(fam, s, weighting_params(fam))
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            wrong = ((p >= 0.5).astype(int) != yte)
            ph_side = yte == 1
            rows.append({"seed": s, "family": fam,
                         "auc_all": roc_auc_score(wrong, u),
                         "auc_phish": roc_auc_score(wrong[ph_side], u[ph_side]),
                         "err_rate": float(wrong.mean())})
            print(f"[E3] seed={s} {fam:<9} AUC(u;wrong) all={rows[-1]['auc_all']:.3f} "
                  f"phish-only={rows[-1]['auc_phish']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_NOVELTY, index=False)
    m = out.groupby("family")[["auc_all", "auc_phish"]].mean().round(3)
    print(f"[E3] means:\n{m.to_string()}")
    print("[E3] pre-set rule: gated backoff stays alive only if CatBoost AUC >= 0.65")
    print(f"[+] {len(out)} rows -> {OUT_NOVELTY}")


def run_forecastability(**overrides):
    args = SimpleNamespace(**{**{'seeds': 5, 'only': None}, **overrides})
    feats, ph_tr, ph_te, be = split_canonical(load())
    print(f"canonical split: ph_tr={len(ph_tr)} ph_te={len(ph_te)} benign={len(be)}")
    seeds = range(args.seeds)
    if args.only in (None, "e2"):
        e2_matrix(feats, ph_tr, ph_te)
    if args.only in (None, "e3"):
        e3_novelty(feats, ph_tr, ph_te, be, seeds)
    if args.only in (None, "e1"):
        e1_decay(feats, ph_tr, be, seeds)


VARIANTS = ("random_fold", "forward_chained", "shuffled_order")


def blocked_folds(order: np.ndarray, n_splits: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    for tr_pos, oo_pos in TimeSeriesSplit(n_splits=n_splits).split(order):
        yield order[tr_pos], order[oo_pos]


def oof_matrix(bases, X, y, folds):
    P = np.full((len(y), len(bases)), np.nan)
    for tr_idx, oo_idx in folds:
        for j, name in enumerate(bases):
            m = make_any_model(name, 0, weighting_params(name))
            m.fit(X[tr_idx], y[tr_idx])
            P[oo_idx, j] = m.predict_proba(X[oo_idx])[:, 1]
    return P, ~np.isnan(P).any(axis=1)


def chain_order(tr: pd.DataFrame, rng: np.random.RandomState) -> np.ndarray:
    n = len(tr)
    pos = np.empty(n, dtype=float)
    dated = tr["date"].notna().to_numpy()
    if dated.any():
        r = tr.loc[dated, "date"].rank(method="first").to_numpy()
        pos[dated] = (r - 0.5) / dated.sum()
    pos[~dated] = rng.rand((~dated).sum())
    return np.argsort(pos, kind="stable")


def run_fcts(**overrides):
    args = SimpleNamespace(**{**{'seeds': 5, 'bases': 'CatBoost+LogReg', 'cut': 0.7, 'splits': 5, 'out': 'data/processed/p2/p2_fcts.csv'}, **overrides})
    bases = args.bases.split("+")
    for b in bases:
        if b not in ABBREV:
            raise SystemExit(f"unknown family in --bases: {b}")
    tag = "Stack[" + "+".join(ABBREV[b] for b in bases) + "]"
    df = load()
    feats = [c for c in COMPPHISH if c in df.columns]
    ph = df[(df.y == 1) & df.date.notna()].sort_values("date").reset_index(drop=True)
    be = df[df.y == 0].reset_index(drop=True)
    ph_tr, ph_te, n_leaked = split_phishing(ph, args.cut, None)
    print(f"phishing dated: {len(ph)}  train {len(ph_tr)}  test {len(ph_te)} "
          f"({n_leaked} guard-dropped)  benign pool: {len(be)}  features: {len(feats)}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for proto in ("temporal_strict", "random_same_rows"):
        for s in range(args.seeds):
            rng = np.random.RandomState(s)
            bmask = rng.rand(len(be)) < args.cut
            if proto == "temporal_strict":
                tr = pd.concat([ph_tr, be[bmask]])
                te = pd.concat([ph_te, be[~bmask]])
            else:
                pool = pd.concat([ph_tr, ph_te])
                pmask = rng.rand(len(pool)) < (len(ph_tr) / (len(ph_tr) + len(ph_te)))
                tr = pd.concat([pool[pmask], be[bmask]])
                te = pd.concat([pool[~pmask], be[~bmask]])
            tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
            Xtr, ytr = tr[feats].to_numpy(float), tr["y"].to_numpy(int)
            Xte, yte = te[feats].to_numpy(float), te["y"].to_numpy(int)

            # Prediction-time bases: fitted once on all of train and shared by the variants, so
            # the only thing that varies is the matrix the meta-learner was fitted on.
            Pte = np.column_stack([make_any_model(b, s, weighting_params(b)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                                   for b in bases])
            order = chain_order(tr, np.random.RandomState(1000 + s))

            for variant in VARIANTS:
                t0 = time.time()
                if variant == "random_fold":
                    folds = list(StratifiedKFold(args.splits, shuffle=True,
                                                 random_state=s).split(Xtr, ytr))
                elif variant == "forward_chained":
                    folds = list(blocked_folds(order, args.splits))
                else:
                    folds = list(blocked_folds(rng.permutation(len(ytr)), args.splits))
                P, cov = oof_matrix(bases, Xtr, ytr, folds)
                meta = LogisticRegression(max_iter=1000).fit(P[cov], ytr[cov])
                met = _metrics(yte, meta.predict_proba(Pte)[:, 1])
                met.update({"family": tag, "variant": variant, "protocol": proto, "seed": s,
                            "meta_rows": int(cov.sum()),
                            "fit_seconds": round(time.time() - t0, 2)})
                met.update({f"coef_{ABBREV[b]}": round(float(c), 3)
                            for b, c in zip(bases, meta.coef_[0])})
                rows.append(met)
                print(f"  {variant:<16} {proto:<17} seed={s} PR-AUC={met['PR-AUC']:.4f} "
                      f"F1={met['F1']:.3f} meta_n={met['meta_rows']} "
                      f"coef={np.round(meta.coef_[0], 2)} ({met['fit_seconds']}s)")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[+] {len(out)} runs -> {args.out}")
    print(out.groupby(["protocol", "variant"])[["PR-AUC", "F1", "FPR@R0.90"]]
             .mean().round(4).to_string())
    t = out[out.protocol == "temporal_strict"].pivot(index="seed", columns="variant",
                                                     values="PR-AUC")
    vs_shuf = corrected_paired_t(t.forward_chained - t.shuffled_order)
    vs_rand = corrected_paired_t(t.forward_chained - t.random_fold)
    print(f"\ntemporal: forward_chained - shuffled_order  {vs_shuf['mean']:+.4f} "
          f"({vs_shuf['wins']}/{vs_shuf['k']} seeds, p={vs_shuf['p']:.3f})")
    print(f"temporal: forward_chained - random_fold     {vs_rand['mean']:+.4f} "
          f"({vs_rand['wins']}/{vs_rand['k']} seeds, p={vs_rand['p']:.3f})")
    beats = vs_shuf["mean"] > 0 and vs_shuf["p"] <= 0.05 and vs_rand["mean"] > 0
    print("E5: " + ("FORWARD CHAINING HELPS — the order carries signal, follow it up"
                    if beats else
                    "NULL — forward-chained OOF is not better than the shuffled-order control; "
                    "the arrow of time inside the training window buys the meta-learner nothing"))


CORPUS = "data/processed/vn_compphish.csv"
NOISE_CSV = "data/processed/p2/label_noise_audit.csv"


def oof_proba(X, y, seed: int, folds: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = CalibratedClassifierCV(
            HistGradientBoostingClassifier(random_state=seed), method="isotonic", cv=3)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def flags(y, p_phish):
    p_benign = 1 - p_phish
    t_phish = p_phish[y == 1].mean()
    t_benign = p_benign[y == 0].mean()
    flagged = np.where(y == 1, p_benign >= t_benign, p_phish >= t_phish)
    return flagged, t_phish, t_benign


def run_label_noise(**overrides):
    args = SimpleNamespace(**{**{'seeds': 5, 'folds': 5}, **overrides})
    df = add_label(pd.read_csv(CORPUS, low_memory=False))
    for c in COMPPHISH:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = df[COMPPHISH].to_numpy(float)
    y = df["y"].to_numpy(int)
    tier = df["tier"].fillna("?").to_numpy()
    print(f"[i] corpus: {len(df)} rows; phishing tiers: "
          + ", ".join(f"{t}={int(((tier == t) & (y == 1)).sum())}"
                      for t in ("gold", "silver", "bronze")))
    # groups audited: phishing per tier, plus all benign as the opposite-direction reference
    groups = {t: (tier == t) & (y == 1) for t in ("gold", "silver", "bronze")}
    groups["benign (all)"] = y == 0
    rates = {g: [] for g in groups}
    per_seed_rows = []
    flag_count = np.zeros(len(df))
    for s in range(args.seeds):
        p = oof_proba(X, y, s, args.folds)
        fl, t_p, t_b = flags(y, p)
        flag_count += fl
        for g, mask in groups.items():
            r = float(fl[mask].mean())
            rates[g].append(r)
            per_seed_rows.append({"group": g, "seed": s, "n": int(mask.sum()),
                                  "flag_rate": round(r, 4)})
        print(f"[i] seed {s}: thresholds t_phish={t_p:.3f} t_benign={t_b:.3f}; "
              + " ".join(f"{g}={np.mean(fl[m]):.3f}" for g, m in groups.items()))
    os.makedirs(os.path.dirname(NOISE_CSV), exist_ok=True)
    pd.DataFrame(per_seed_rows).to_csv(NOISE_CSV, index=False)
    print(f"[+] {NOISE_CSV}")
    # a majority-of-seeds flag is the stable per-row verdict; used for the qualitative sample
    stable = flag_count >= (args.seeds / 2)
    ex = df.loc[stable & groups["bronze"], "dom" if "dom" in df else "url"].astype(str)
    print("[i] example stable-flagged bronze domains:", ", ".join(ex.head(12)))
    for g, r in rates.items():
        print(f"  {g:14s} flagged {100 * np.mean(r):.1f}% +- {100 * np.std(r):.1f} (n={int(groups[g].sum()):,})")
