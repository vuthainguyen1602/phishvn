"""The numbers the paper computes from the result tables (paired tests, bootstrap, counts), split by
the notebook whose tables they read. Each function returns some of the keys of
results/p2_generated.json; together the four cover all of them."""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

from .common import add_label, benign_mask, bh_adjust, corrected_paired_t, split_phishing

RES = "data/processed/p2"
ORDER = ["CatBoost", "MLP", "XGBoost", "LightGBM", "RandomForest", "HistGB", "LogReg"]


def rd(name):
    return pd.read_csv(f"{RES}/{name}")


def mean_m(df, proto, fam, metric):
    return float(df[(df.family == fam) & (df.protocol == proto)][metric].mean())


def dated_test_counts(df_all):
    # phishing rows in the guarded temporal test window, and the seed-mean benign test size
    ph = df_all[(df_all.y == 1) & df_all.date.notna()].sort_values("date")
    cut = int(len(ph) * 0.70)
    ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
    n_ph = int((~ph_te.rdom.isin(set(ph_tr.rdom))).sum())
    n_be_pool = int((df_all.y == 0).sum())
    n_be = int(np.mean([(np.random.RandomState(s).rand(n_be_pool) >= 0.70).sum() for s in range(5)]))
    return n_ph, n_be


def core_numbers(df_all) -> dict:
    """Notebook 01: the benchmark, the protocol decomposition, the guard and the corpus's own split."""
    R = {}
    strict = rd("p2_temporal_strict.csv")
    bench = rd("p2_benchmark.csv")
    ts = strict[strict.protocol == "temporal_strict"]

    # CatBoost minus logistic regression under the phishing-temporal protocol, paired by seed, on all four metrics
    def pair(metric):
        cb = ts[ts.family == "CatBoost"].sort_values("seed")[metric].to_numpy()
        lr = ts[ts.family == "LogReg"].sort_values("seed")[metric].to_numpy()
        return corrected_paired_t(cb - lr)

    f1 = pair("F1")
    R["strict_cb_minus_lr"] = {"F1": f1, "F1_ci95_half": stats.t.ppf(0.975, f1["k"] - 1) * f1["se"] if f1["se"] else 0.0,
                               **{m: pair(m) for m in ("PR-AUC", "ROC-AUC", "FPR@R0.90")}}
    for m in ("F1", "PR-AUC", "ROC-AUC", "FPR@R0.90"):
        t = R["strict_cb_minus_lr"][m]
        print(f"CatBoost minus logistic regression, {m}: {t['mean']:+.4f} (corrected p={t['p']:.3g}, {t['wins']}/{t['k']} seeds positive)")

    # the three same-corpus designs, family by family (F1 at 0.5), and the rows they rest on
    def mean_f1(df, proto, fam):
        return float(df[(df.family == fam) & (df.protocol == proto)]["F1"].mean())

    stage = {"Full": [mean_f1(bench, "random", f) for f in ORDER],
             "Dated": [mean_f1(strict, "random_same_rows", f) for f in ORDER],
             "Temp": [mean_f1(strict, "temporal_strict", f) for f in ORDER]}
    is_ph = df_all["y"].eq(1)
    R["decomposition"] = {"stage_f1": stage, "rows_full": len(df_all),
                          "rows_dated": int((~is_ph).sum() + (is_ph & df_all["date"].notna()).sum())}
    print("best F1 by design (full random, dated random, phishing-temporal):", [round(max(v), 3) for v in stage.values()])

    # the oracle maximum F1 read off each family's seed-mean PR curve
    def max_f1_curve(curves):
        out = {}
        for fam, g in curves.groupby("family"):
            r, p_ = g["recall"].to_numpy(float), g["precision"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(r + p_ > 0, 2 * r * p_ / (r + p_), 0.0)
            out[fam] = float(np.nanmax(f))
        return out

    cf, cs = rd("p2_pr_curves.csv"), rd("p2_pr_curves_strict.csv")
    R["max_f1"] = {"Random (full)": max_f1_curve(cf[cf.protocol == "random"]),
                   "Random (dated rows)": max_f1_curve(cs[cs.protocol == "random_same_rows"]),
                   "Phishing-temporal": max_f1_curve(cs[cs.protocol == "temporal_strict"])}

    # confusion counts at 90% recall for CatBoost, from the guarded test size and the seed-mean FPR
    n_ph, n_be = dated_test_counts(df_all)

    def random_arm_phish_test_mean():
        ph = df_all[(df_all.y == 1) & df_all.date.notna()].sort_values("date")
        cut = int(len(ph) * 0.70)
        ph_tr, ph_te = ph.iloc[:cut], ph.iloc[cut:]
        ph_te = ph_te[~ph_te.rdom.isin(set(ph_tr.rdom))]
        n_be_pool = int((df_all.y == 0).sum())
        n = len(ph_tr) + len(ph_te)
        sizes = []
        for s in range(5):
            rng = np.random.RandomState(s)
            rng.rand(n_be_pool)  # the benign mask is drawn first
            sizes.append(int((~(rng.rand(n) < len(ph_tr) / n)).sum()))
        return int(round(np.mean(sizes)))

    tp = round(0.90 * n_ph)
    cells = {}
    for proto in ("random_same_rows", "temporal_strict"):
        fp = round(strict[(strict.family == "CatBoost") & (strict.protocol == proto)]["FPR@R0.90"].mean() * n_be)
        cells[proto] = (fp, n_be - fp)
    R["confusion"] = {"dated_test_phish": n_ph, "dated_test_benign": n_be, "random_arm_phish_test": random_arm_phish_test_mean(),
                      "tp": tp, "fn": n_ph - tp, "fp_tn": cells}

    # the corpus's own train/val/test split: phishing rows and all rows, by evidence tier
    u = pd.read_csv("data/processed/dataset_url.csv", usecols=["label", "tier", "split"])
    u["is_phish"] = u["label"].astype(str).eq("phishing")
    R["official_split_priors"] = {
        name: {sp: [int(d[d["split"].astype(str).eq(sp)]["is_phish"].sum()), int(d["split"].astype(str).eq(sp).sum())]
               for sp in ("train", "val", "test")}
        for name, d in (("All tiers", u), ("Gold+silver", u[u["tier"].astype(str).isin(["gold", "silver"])]),
                        ("Gold only", u[u["tier"].astype(str).eq("gold")]))}

    # the dated phishing rows: how many come from each feed, where the cut falls, and how the reconstructed dates
    # compare with the national feed's (from the first-seen validation summary)
    src = pd.read_csv("data/processed/dataset_url.csv", usecols=["url_norm", "source"], dtype=str)
    dd = df_all.merge(src.rename(columns={"url_norm": "url"}).drop_duplicates("url"), on="url", how="left")
    ph = dd[(dd.y == 1) & dd.date.notna()].sort_values("date").reset_index(drop=True)
    tr, te, _ = split_phishing(ph, 0.70)
    ncsc = lambda d: 100 * float((d.source == "tinnhiemmang").mean())
    R["dates"] = {"dated_phish": len(ph), "chongluadao": int((ph.source == "chongluadao").sum()), "train": len(tr), "test": len(te),
                  "test_from_2024": int((te.date >= pd.Timestamp("2024-01-01")).sum()), "test_first_day": str(te.date.min().date()),
                  "ncsc_share_train_pct": ncsc(tr), "ncsc_share_test_pct": ncsc(te)}
    print("dated phishing rows:", R["dates"])

    # the one-sided guard: how many benign test rows share a registrable domain with benign training rows, how many phishing
    # rows the guard removes from the random control, and the two test priors
    ph = df_all[(df_all.y == 1) & df_all.date.notna()].sort_values("date").reset_index(drop=True)
    be = df_all[df_all.y == 0].reset_index(drop=True)
    tr, te, _ = split_phishing(ph, 0.70)
    shared, drop_, p_t, p_g = [], [], [], []
    for s in range(5):
        rng = np.random.RandomState(s)
        bm = benign_mask(be, rng, 0.70)
        shared.append(be[~bm].rdom.isin(set(be[bm].rdom)).mean())
        pool = pd.concat([tr, te])
        pm = rng.rand(len(pool)) < len(tr) / (len(tr) + len(te))
        g = pool[~pm].rdom.isin(set(pool[pm].rdom))
        nb = int((~bm).sum())
        drop_.append(g.mean())
        p_t.append(len(te) / (len(te) + nb))
        p_g.append(int((~g).sum()) / (int((~g).sum()) + nb))
    R["guard_macros"] = {"benign_shared": shared, "guard_drop": drop_, "prior_temporal": p_t, "prior_guarded": p_g}
    print(f"benign test rows sharing a domain with training: {100 * np.mean(shared):.0f}%; phishing rows the guard drops: {100 * np.mean(drop_):.1f}%")

    # the protocol step before and after the guard is applied to the control as well
    g_ = rd("p2_temporal_strict_guarded.csv")
    unguarded = {f: mean_m(strict, "random_same_rows", f, "F1") for f in ORDER}
    guarded = {f: mean_m(g_, "random_same_rows_guarded", f, "F1") for f in ORDER}
    temporal = {f: mean_m(strict, "temporal_strict", f, "F1") for f in ORDER}
    d_old = {f: temporal[f] - unguarded[f] for f in ORDER}
    d_new = {f: temporal[f] - guarded[f] for f in ORDER}
    lp = rd("p2_dup_leakage_protocols.csv")
    pm_, pg_ = rd("p2_temporal_strict_guarded_pm.csv"), rd("p2_temporal_strict_guarded_pm_bgroup.csv")
    step_ = lambda df, f, m: mean_m(df, "temporal_strict", f, m) - mean_m(df, "random_same_rows_guarded", f, m)
    d_roc = {f: step_(pm_, f, "ROC-AUC") for f in ORDER}
    d_roc_u = {f: mean_m(strict, "temporal_strict", f, "ROC-AUC") - mean_m(strict, "random_same_rows", f, "ROC-AUC") for f in ORDER}
    boost = [f for f in ORDER if f != "LogReg"]
    R["guard_control"] = {
        "delta_old": d_old, "delta_new": d_new,
        "regdom_shared_random": lp[lp.protocol == "random_same_rows"]["phish_test_regdom_shared_rate"].mean(),
        "regdom_shared_temporal": lp[lp.protocol == "temporal_strict"]["phish_test_regdom_shared_rate"].mean(),
        "guard_shift": float(np.mean([guarded[f] - unguarded[f] for f in ORDER])),
        "catboost_prauc_old": mean_m(strict, "temporal_strict", "CatBoost", "PR-AUC") - mean_m(strict, "random_same_rows", "CatBoost", "PR-AUC"),
        "catboost_prauc_new": mean_m(strict, "temporal_strict", "CatBoost", "PR-AUC") - mean_m(g_, "random_same_rows_guarded", "CatBoost", "PR-AUC"),
        "spearman_old_new": float(spearmanr([d_old[f] for f in ORDER], [d_new[f] for f in ORDER]).statistic),
        "pm_f1": {f: step_(pm_, f, "F1") for f in ORDER}, "pm_roc": d_roc, "pm_fpr": {f: step_(pm_, f, "FPR@R0.90") for f in ORDER},
        "unguarded_roc": d_roc_u, "pm_bgroup_roc": {f: step_(pg_, f, "ROC-AUC") for f in ORDER},
        "roc_share_removed": 1 - np.mean([d_roc[f] for f in boost]) / np.mean([d_roc_u[f] for f in boost])}
    print(f"share of the ROC-AUC protocol step the guard and prior matching remove: {100 * R['guard_control']['roc_share_removed']:.0f}%")
    return R


def transfer_numbers(df_all) -> dict:
    """Notebook 02: the all-positive floor of each test set, and the CORAL cells that sit on it."""
    R = {}
    strict = rd("p2_temporal_strict.csv")
    bench = rd("p2_benchmark.csv")

    # the all-positive F1 floor 2p/(1+p) of each test set, against the best family
    files = {"PhishVN": "vn_compphish.csv", "PhiUSIIL": "external/phiusiil_compphish.csv",
             "ISCXURL2016": "external/iscx_compphish.csv", "PhishStorm": "external/phishstorm_compphish.csv"}
    mat = pd.read_csv(f"{RES}/cross_dataset_F1.csv", index_col=0)
    rates, floor = {}, {}
    for n in mat.columns:
        y_ = add_label(pd.read_csv(os.path.join("data/processed", files[n])))["y"]
        rates[n] = float(y_.mean())
        floor[n] = 2 * rates[n] / (1 + rates[n])
    off = [(i, j) for i in mat.columns for j in mat.columns if i != j]
    n_ph, n_be = dated_test_counts(df_all)
    dated_p = n_ph / (n_ph + n_be)
    best = lambda df, proto: float(df[df.protocol == proto].groupby("family")["F1"].mean().max())
    R["trivial_floor"] = {"positive_rate": rates, "all_positive_f1": floor,
                          "offdiag_f1_mean": float(np.mean([float(mat.loc[i, j]) for i, j in off])),
                          "offdiag_floor_mean": float(np.mean([floor[j] for _, j in off])),
                          "dated_test_phish": n_ph, "dated_test_benign": n_be,
                          "designs": [["Full corpus random", rates["PhishVN"], best(bench, "random")],
                                      ["Dated subset random", dated_p, best(strict, "random_same_rows")],
                                      ["Dated subset temporal", dated_p, best(strict, "temporal_strict")]]}
    print("all-positive F1 floor per corpus:", {k: round(v, 3) for k, v in floor.items()})

    # CORAL cells that sit on the all-positive floor, and how many of those rank below chance
    f_ = pd.read_csv(f"{RES}/cross_dataset_F1_coral.csv", index_col=0)
    r_ = pd.read_csv(f"{RES}/cross_dataset_ROC-AUC_coral.csv", index_col=0)
    const = {k: 2 * v / (1 + v) for k, v in {n: (rates["PhishVN"] if n == "PhishVN" else 0.5) for n in f_.columns}.items()}
    deg, sub = [], 0
    for i in f_.index:
        for j in f_.columns:
            if i != j and abs(f_.loc[i, j] - const[j]) < 0.005:
                deg.append([i, j, f_.loc[i, j], r_.loc[i, j]])
                sub += r_.loc[i, j] < 0.5
    R["coral_degeneracy"] = {"degenerate_cells": deg, "below_chance": int(sub), "offdiag_cells": len(f_.index) * (len(f_.columns) - 1)}
    print(f"CORAL: {len(deg)} of {R['coral_degeneracy']['offdiag_cells']} off-diagonal cells sit on the all-positive F1, {int(sub)} of them below chance")
    return R


def robustness_numbers():
    """Notebook 03: stacking against CatBoost, and whether drift can be forecast. Returns the keys of
    p2_generated.json and, separately, the block shift-matrix summary of p2_shiftmatrix.json."""
    R = {}
    strict = rd("p2_temporal_strict.csv")

    # stacking against CatBoost under the phishing-temporal protocol (BH across the six ensembles), and the 20-seed crossover
    STACKS = [("Stacking", "Stack[RF+XGB]"), ("Stacking+HFS", "Stack[RF+XGB]+HFS"), ("Stack[XGB+LGBM]", "Stack[XGB+LGBM]"),
              ("Stack[CB+HGB]", "Stack[CB+HGB]"), ("Stack[CB+MLP]", "Stack[CB+MLP]"), ("Stack[CB+HGB+LR+MLP]", "Stack[CB+HGB+LR+MLP]"),
              ("Stack[CB+LR]", "Stack[CB+LR]")]
    df5 = pd.concat([rd("p2_stacking_baseline.csv").replace({"family": {v: k for k, v in STACKS[:2]}}), rd("p2_stacking_combos.csv")])

    def paired_stack(a_df, b_df, fam_a, fam_b, proto, k_col="PR-AUC"):
        a = a_df[(a_df.family == fam_a) & (a_df.protocol == proto)].set_index("seed")[k_col]
        b = b_df[(b_df.family == fam_b) & (b_df.protocol == proto)].set_index("seed")[k_col]
        return corrected_paired_t((b.sort_index() - a.sort_index()).dropna().values, test_frac=0.30)

    sfams = [k for k, _ in STACKS if k != "Stacking+HFS"]
    sres = {f: paired_stack(strict, df5, "CatBoost", f, "temporal_strict") for f in sfams}
    sq = dict(zip(sfams, bh_adjust([sres[f]["p"] for f in sfams])[0]))
    cb20, st20 = rd("p2_temporal_strict_cb_k20.csv"), rd("p2_stacking_cblr_k20.csv")
    t20 = paired_stack(cb20, st20, "CatBoost", "Stack[CB+LR]", "temporal_strict")
    r20 = paired_stack(cb20, st20, "CatBoost", "Stack[CB+LR]", "random_same_rows")
    tq, rq = bh_adjust([t20["p"], r20["p"]])[0]
    R["stacking"] = {"k5_vs_catboost": {f: {**sres[f], "q": sq[f]} for f in sfams},
                     "k20_temporal": {**t20, "q": tq}, "k20_random": {**r20, "q": rq}}
    for f in sfams:
        print(f"{f:22s} minus CatBoost, PR-AUC {sres[f]['mean']:+.4f} (BH q={sq[f]:.3f})")

    # can drift be forecast? decay slopes against the protocol gap; the block shift matrix; lexical novelty; FCTS
    dec = rd("p2_forecastability_decay.csv")
    gg_ = strict.groupby(["family", "protocol"])["PR-AUC"].mean().unstack()
    gap = gg_["random_same_rows"] - gg_["temporal_strict"]
    rhos = []
    for origin in sorted(dec.origin.unique()):
        for metric in ("PR-AUC", "ROC-AUC"):
            sub_ = dec[dec.origin == origin]
            slopes = {}
            for fam in sub_.family.unique():
                w = sub_[sub_.family == fam].groupby("win")[["horizon_d", metric]].mean()
                slopes[fam] = np.polyfit(w["horizon_d"], w[metric], 1)[0]
            sl = pd.Series(slopes)
            rho, p_ = spearmanr(-sl[gap.index], gap)
            rhos.append([origin, metric, rho, p_])
    R["forecast_decay_spearman"] = rhos
    win_means = dec[(dec.origin == 0.50) & (dec.family == "CatBoost")].groupby("win")["PR-AUC"].mean()
    mx = rd("p2_forecastability_shiftmatrix.csv")
    internal = mx[mx.a.str.startswith("W") & mx.b.str.startswith("W")]
    adj = internal[internal.b.str[1].astype(int) - internal.a.str[1].astype(int) == 1]
    row_ = lambda a, b: mx[(mx.a == a) & (mx.b == b)].iloc[0]
    nov = rd("p2_forecastability_novelty.csv")
    cbn = nov[nov.family == "CatBoost"]
    R["forecast_other"] = {"catboost_win_swing": win_means.max() - win_means.min(), "adjacent_auc_mean": adj.auc.mean(),
                           "auc_W4_TEST": row_("W4", "TEST").auc, "auc_W1_W4": row_("W1", "W4").auc, "auc_W1_TEST": row_("W1", "TEST").auc,
                           "novelty_auc_all": cbn.auc_all.mean(), "novelty_auc_phish": cbn.auc_phish.mean(),
                           "novelty_seeds_below_half": int((cbn.auc_all < 0.5).sum())}
    fc = rd("p2_fcts.csv")
    fct = fc[fc.protocol == "temporal_strict"]
    piv = fct.pivot(index="seed", columns="variant", values="PR-AUC")
    R["fcts"] = {"forward_minus_shuffled": corrected_paired_t(piv.forward_chained - piv.shuffled_order),
                 "forward_minus_random": corrected_paired_t(piv.forward_chained - piv.random_fold),
                 "coef_ratio_cb_lr": (fct.groupby("variant").coef_CB.mean() / fct.groupby("variant").coef_LR.mean()).to_dict()}
    print("decay slope vs protocol gap, Spearman by origin and metric:", [(o, m, round(r, 2)) for o, m, r, _ in rhos])

    # the block shift matrix summarised as in the paper's figure: elapsed time against separability
    rows_ = rd("p2_forecastability_shiftmatrix.csv").to_dict("records")
    ADJACENT = (("W1", "W2"), ("W2", "W3"), ("W3", "W4"))
    adj_r = [r for r in rows_ if (r["a"], r["b"]) in ADJACENT]
    bnd_r = [r for r in rows_ if r["b"] == "TEST"]
    rho, p_ = spearmanr([r["delta_days"] for r in rows_], [r["auc"] for r in rows_])
    shiftmatrix = {"n": len(rows_), "rho": float(rho), "p": float(p_), "adj_mean": sum(r["auc"] for r in adj_r) / len(adj_r),
                   "adj_lo": min(r["auc"] for r in adj_r), "adj_hi": max(r["auc"] for r in adj_r),
                   "bnd_mean": sum(r["auc"] for r in bnd_r) / len(bnd_r)}
    return R, shiftmatrix


def cnn_dedup_numbers() -> dict:
    """Notebook 04: the character CNN against each tabular family, the benign-source probe, and what
    the deduplicated corpus moves."""
    R = {}
    strict = rd("p2_temporal_strict.csv")

    # the character CNN against each tabular family, paired on identical rows, BH across all 42 comparisons
    cnn = rd("p2_charcnn.csv")

    def paired_cnn(metric, proto, fam):
        c = cnn[cnn.protocol == proto].set_index("seed")[metric]
        t = strict[(strict.protocol == proto) & (strict.family == fam)].set_index("seed")[metric]
        seeds = sorted(set(c.index) & set(t.index))
        d = np.array([c[i] - t[i] for i in seeds], float)
        return d.mean(), corrected_paired_t(d)["p"]

    fams = sorted(strict[strict.protocol == "temporal_strict"].family.unique())
    res, ps = [], []
    for fam in fams:
        for m in ("F1", "PR-AUC", "FPR@R0.90"):
            for proto in ("temporal_strict", "random_same_rows"):
                delta, p_ = paired_cnn(m, proto, fam)
                res.append((fam, m, proto, delta))
                ps.append(p_)
    qs, rej = bh_adjust(ps)
    best_fam = max(fams, key=lambda f: strict[(strict.protocol == "temporal_strict") & (strict.family == f)]["F1"].mean())

    def dproto(fam):
        src_ = cnn if fam == "CharCNN" else strict[strict.family == fam]
        return src_[src_.protocol == "temporal_strict"]["F1"].mean() - src_[src_.protocol == "random_same_rows"]["F1"].mean()

    def spread_gain(proto):
        tab = strict[strict.protocol == proto].groupby("family")["F1"].mean()
        return [cnn[cnn.protocol == proto]["F1"].mean() - tab.max(), tab.max() - tab.min()]

    R["charcnn"] = {"paired": [[f, m, pr, d, p_, q_] for (f, m, pr, d), p_, q_ in zip(res, ps, qs)],
                    "n_significant": int(sum(rej)), "best_tabular": best_fam,
                    "vs_best_temporal": {m: paired_cnn(m, "temporal_strict", best_fam)[0] for m in ("F1", "PR-AUC", "FPR@R0.90")},
                    "protocol_drop_cnn": dproto("CharCNN"), "protocol_drop_tabular": {f: dproto(f) for f in fams},
                    "gain_and_spread": {p_: spread_gain(p_) for p_ in ("temporal_strict", "random_same_rows")}}
    print(f"character CNN vs tabular families: {int(sum(rej))} of {len(ps)} paired comparisons significant after BH")

    # the benign-source probe: the swap's effect against each family's canonical temporal score
    pr_ = rd("p2_source_probe.csv")
    swap = lambda fam, tb, metric="F1": float(pr_[(pr_.family == fam) & (pr_.train_benign == tb)][metric].mean())
    canon = lambda fam, metric="F1": float((cnn[cnn.protocol == "temporal_strict"] if fam == "CharCNN" else
                                            strict[(strict.protocol == "temporal_strict") & (strict.family == fam)])[metric].mean())
    R["source_probe"] = {"cnn_tranco": swap("CharCNN", "tranco") - canon("CharCNN"), "catboost_tranco": swap("CatBoost", "tranco") - canon("CatBoost"),
                         "logreg_tranco": swap("LogReg", "tranco") - canon("LogReg"), "catboost_tranco_prauc": swap("CatBoost", "tranco", "PR-AUC"),
                         "cnn_tinnhiem": swap("CharCNN", "tinnhiem") - canon("CharCNN"), "catboost_tinnhiem": swap("CatBoost", "tinnhiem") - canon("CatBoost")}

    # what the deduplicated corpus moves
    DEDUP = "data/processed/p2/dedup_audit/"
    fam_means = lambda path, proto, met="F1": (lambda d: d[d.protocol == proto].groupby("family")[met].mean())(pd.read_csv(path))

    def dated_of(path):
        d = pd.read_csv(path, usecols=["label", "collected_at"], low_memory=False)
        t = d["collected_at"].astype(str).str.strip().str.slice(0, 10)
        ok = (pd.to_datetime(t, format="%d/%m/%Y", errors="coerce").fillna(pd.to_datetime(t, format="%Y-%m-%d", errors="coerce"))).notna()
        return d, ok

    uu, du = dated_of("data/processed/dataset_url.csv")
    vv, dv = dated_of(DEDUP + "dataset_url_dedup.csv")
    php = vv.label.astype(str).str.lower() == "phishing"
    full1, dr1, t1 = (fam_means(DEDUP + "p2_benchmark.csv", "random"), fam_means(DEDUP + "p2_temporal_strict.csv", "random_same_rows"),
                      fam_means(DEDUP + "p2_temporal_strict.csv", "temporal_strict"))
    stk = pd.concat([full1, dr1, t1], axis=1)
    r0 = pd.read_csv(f"{RES}/cross_dataset_ROC-AUC.csv", index_col=0)
    r1 = pd.read_csv(DEDUP + "cross_dataset_ROC-AUC.csv", index_col=0)
    e = np.eye(len(r0), dtype=bool)
    flips = [[i, j] for i in r0.index for j in r0.columns if i != j and (r0.loc[i, j] < 0.5) != (r1.loc[i, j] < 0.5)]
    gap_of = lambda p_: (lambda m_: m_[e].mean() - m_[~e].mean())(pd.read_csv(p_, index_col=0).values.astype(float))
    R["dedup_sensitivity"] = {
        "rows_released": len(uu), "rows_dedup": len(vv), "dated_dropped": int(du.sum() - dv.sum()), "undated_phish_left": int((php & ~dv).sum()),
        "family_move_max": float((stk.max(axis=1) - stk.min(axis=1)).max()),
        "offdiag_roc_released": r0.values[~e].mean(), "offdiag_roc_dedup": r1.values[~e].mean(),
        "below_chance_released": int((r0.values[~e] < 0.5).sum()), "below_chance_dedup": int((r1.values[~e] < 0.5).sum()),
        "flipped_cells": flips,
        "rf_f1_gap": [gap_of(f"{RES}/cross_dataset_F1.csv"), gap_of(DEDUP + "cross_dataset_F1.csv")],
        "catboost_f1_gap": [gap_of(f"{RES}/cross_dataset_F1_CatBoost.csv"), gap_of(DEDUP + "cross_dataset_F1_CatBoost.csv")]}
    print(f"deduplicated corpus: {len(vv):,} of {len(uu):,} rows; {len(flips)} transfer cells change side of chance")
    return R
