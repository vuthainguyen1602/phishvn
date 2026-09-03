#!/usr/bin/env python3
"""
make_p3_cross_generator_eval.py — Leave-one-LLM-out evaluation (Claude vs. Gemini).

Evaluates the two leave-one-generator-out directions explicitly:
- Vectorizer: char_wb (2, 5) n-grams (exact P3 text branch)
- 20 random seeds (70/30 stratified train/test split)
- Strict source-level role disjunction (Variant A for train augmentation, Variant B for test attack)
- Direction 1: train with Claude A, test held-out Gemini B
- Direction 2: train with Gemini A, test held-out Claude B
- Full paired statistical comparison using Nadeau-Bengio corrected resampled t-tests.

The full within-generator/dual matrix is still written as an audit CSV, but the manuscript table is
the LOO summary only. Dual augmentation is not a leave-one-out contrast.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)

from make_p3_paraphrase_assets import strip_url, SEEDS, vec, miss
from paired_eval import corrected_paired_t
from genfile import write_generated

SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")


def _mean_std(vals):
    return float(np.mean(vals) * 100), float(np.std(vals) * 100)


def _fmt_pm(mean, std):
    return f"{mean:.1f}\\,$\\pm$\\,{std:.1f}"


def _fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _loo_table(rows):
    body = [
        "\\begin{table*}[t]",
        "\\caption{Leave-one-LLM-out paraphrase evaluation under the token-Jaccard band "
        "$[0.20, 0.30]$. Each row trains on clean text plus role-\\texttt{a} rewrites from one "
        "generator and tests on role-\\texttt{b} rewrites from the held-out generator; mean "
        "$\\pm$ std over 20 stratified 70/30 splits.}",
        "\\label{tab:crossgenerator}",
        "\\small",
        "\\centering",
        "\\begin{tabular}{llcccl}",
        "\\toprule",
        "Held-out attack & Train augmentation & D0 miss (\\%) & LOO miss (\\%) & "
        "$\\Delta$ miss (pp) & Corrected test \\\\",
        "\\midrule",
    ]
    for r in rows:
        body.append(
            f"{r['held_out_generator']} & {r['train_augmentation_generator']} A & "
            f"{_fmt_pm(r['baseline_miss_mean'], r['baseline_miss_std'])} & "
            f"{_fmt_pm(r['loo_miss_mean'], r['loo_miss_std'])} & "
            f"${r['delta_miss_pp']:+.1f}$ & ${_fmt_p(r['p'])}$ \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(body)

def load_rewrites(path: str) -> dict[tuple[str, str], str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) == 3:
                out[(parts[0].strip(), parts[1].strip())] = strip_url(parts[2].strip())
    return out

def run_cross_generator_matrix():
    frames = [pd.read_csv(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))
              for t in ("sms", "email")
              if os.path.exists(os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["clean_text"] = df["text"].fillna("").astype(str).map(strip_url)
    df["y"] = (df["label"] == "phishing").astype(int)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    
    claude_map = load_rewrites(os.path.join(ROOT, "data", "raw", "author", "p3_band_rewrites.tsv"))
    gemini_map = load_rewrites(os.path.join(ROOT, "data", "raw", "author", "p3_gemini_rewrites.tsv"))
    
    df["claude_a"] = df["id"].map(lambda i: claude_map.get((i, "a"), ""))
    df["claude_b"] = df["id"].map(lambda i: claude_map.get((i, "b"), ""))
    df["gemini_a"] = df["id"].map(lambda i: gemini_map.get((i, "a"), ""))
    df["gemini_b"] = df["id"].map(lambda i: gemini_map.get((i, "b"), ""))
    
    print("=" * 85)
    print("🛡️ P3 LEAVE-ONE-LLM-OUT EVALUATION (CLAUDE vs. GEMINI)")
    print("=" * 85)
    print(f"Total dataset rows: {len(df)} (Phishing: {df.y.sum()}, Benign: {(df.y == 0).sum()})")
    print(f"Total Claude rewrites: {len(claude_map)}, Total Gemini rewrites: {len(gemini_map)}")
    print(f"Harness: TF-IDF char_wb(2,5), 20 seeds, 70/30 split, Nadeau-Bengio corrected stats")
    print("-" * 85)
    
    txt, y = df["clean_text"].to_numpy(), df["y"].to_numpy()
    c_a, c_b = df["claude_a"].to_numpy(), df["claude_b"].to_numpy()
    g_a, g_b = df["gemini_a"].to_numpy(), df["gemini_b"].to_numpy()
    
    cells = {
        "D0": {"clean": [], "claude_atk": [], "gemini_atk": []},
        "D2_Claude": {"clean": [], "claude_atk": [], "gemini_atk": []},
        "D2_Gemini": {"clean": [], "claude_atk": [], "gemini_atk": []},
        "D2_Dual": {"clean": [], "claude_atk": [], "gemini_atk": []},
    }
    
    for s in range(SEEDS):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=s)
        ytr, yte = y[tr], y[te]
        ph_te = te[yte == 1]
        
        # --- 1. D0 (Clean) ---
        v0 = vec()
        Xtr0 = v0.fit_transform(txt[tr])
        clf0 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr0, ytr)
        
        cells["D0"]["clean"].append(miss(clf0, v0, txt[ph_te], y[ph_te]))
        cells["D0"]["claude_atk"].append(miss(clf0, v0, c_b[ph_te], y[ph_te]))
        cells["D0"]["gemini_atk"].append(miss(clf0, v0, g_b[ph_te], y[ph_te]))
        
        # --- 2. D2_Claude (Clean + Claude Variant A) ---
        ph_tr = tr[ytr == 1]
        
        tr_c_txt = list(txt[tr]) + [t for t in c_a[ph_tr] if t]
        tr_c_y = list(ytr) + [1] * int(sum(1 for t in c_a[ph_tr] if t))
        
        v_c = vec()
        Xtr_c = v_c.fit_transform(tr_c_txt)
        clf_c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr_c, tr_c_y)
        
        cells["D2_Claude"]["clean"].append(miss(clf_c, v_c, txt[ph_te], y[ph_te]))
        cells["D2_Claude"]["claude_atk"].append(miss(clf_c, v_c, c_b[ph_te], y[ph_te]))
        cells["D2_Claude"]["gemini_atk"].append(miss(clf_c, v_c, g_b[ph_te], y[ph_te]))
        
        # --- 3. D2_Gemini (Clean + Gemini Variant A) ---
        tr_g_txt = list(txt[tr]) + [t for t in g_a[ph_tr] if t]
        tr_g_y = list(ytr) + [1] * int(sum(1 for t in g_a[ph_tr] if t))
        
        v_g = vec()
        Xtr_g = v_g.fit_transform(tr_g_txt)
        clf_g = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr_g, tr_g_y)
        
        cells["D2_Gemini"]["clean"].append(miss(clf_g, v_g, txt[ph_te], y[ph_te]))
        cells["D2_Gemini"]["claude_atk"].append(miss(clf_g, v_g, c_b[ph_te], y[ph_te]))
        cells["D2_Gemini"]["gemini_atk"].append(miss(clf_g, v_g, g_b[ph_te], y[ph_te]))
        
        # --- 4. D2_Dual (Clean + Claude A + Gemini A) ---
        tr_dual_txt = list(txt[tr]) + [t for t in c_a[ph_tr] if t] + [t for t in g_a[ph_tr] if t]
        tr_dual_y = list(ytr) + [1] * (int(sum(1 for t in c_a[ph_tr] if t)) + int(sum(1 for t in g_a[ph_tr] if t)))
        
        v_dual = vec()
        Xtr_dual = v_dual.fit_transform(tr_dual_txt)
        clf_dual = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr_dual, tr_dual_y)
        
        cells["D2_Dual"]["clean"].append(miss(clf_dual, v_dual, txt[ph_te], y[ph_te]))
        cells["D2_Dual"]["claude_atk"].append(miss(clf_dual, v_dual, c_b[ph_te], y[ph_te]))
        cells["D2_Dual"]["gemini_atk"].append(miss(clf_dual, v_dual, g_b[ph_te], y[ph_te]))

    # Print Summary Table
    print(f"{'Detector Model':<28} | {'Clean Lures':<14} | {'Claude Attack (b)':<18} | {'Gemini Attack (b)':<18}")
    print("-" * 85)
    
    summary_rows = []
    for d_name, evals in cells.items():
        m_cl, s_cl = _mean_std(evals["clean"])
        m_c, s_c = _mean_std(evals["claude_atk"])
        m_g, s_g = _mean_std(evals["gemini_atk"])
        
        print(f"{d_name:<28} | {m_cl:5.2f}% ± {s_cl:4.2f}% | {m_c:5.2f}% ± {s_c:4.2f}%     | {m_g:5.2f}% ± {s_g:4.2f}%")
        summary_rows.append({
            "detector": d_name,
            "miss_clean_mean": m_cl, "miss_clean_std": s_cl,
            "miss_claude_mean": m_c, "miss_claude_std": s_c,
            "miss_gemini_mean": m_g, "miss_gemini_std": s_g,
        })
    print("-" * 85)
    
    loo_specs = [
        ("Gemini", "Claude", "gemini_atk", "D2_Claude"),
        ("Claude", "Gemini", "claude_atk", "D2_Gemini"),
    ]
    loo_rows = []
    for held_out, train_gen, attack_key, detector in loo_specs:
        baseline = np.array(cells["D0"][attack_key])
        loo = np.array(cells[detector][attack_key])
        res = corrected_paired_t(loo - baseline, test_frac=0.30)
        b_mean, b_std = _mean_std(baseline)
        l_mean, l_std = _mean_std(loo)
        loo_rows.append({
            "held_out_generator": held_out,
            "train_augmentation_generator": train_gen,
            "baseline_miss_mean": b_mean,
            "baseline_miss_std": b_std,
            "loo_miss_mean": l_mean,
            "loo_miss_std": l_std,
            "delta_miss_pp": res["mean"] * 100,
            "t": res.get("t", 0.0),
            "p": res["p"],
        })

    print("\n🔬 LEAVE-ONE-LLM-OUT CONTRASTS (Nadeau-Bengio corrected t-test across 20 splits):")
    print("-" * 85)
    for r in loo_rows:
        print(f"Held out {r['held_out_generator']:<6} | train {r['train_augmentation_generator']:<6} A "
              f"| D0={r['baseline_miss_mean']:.2f}% -> LOO={r['loo_miss_mean']:.2f}% "
              f"| Δ={r['delta_miss_pp']:+.2f} pp, t={r['t']:.2f}, p={r['p']:.4f}")

    # Audit-only diagnostics (not leave-one-out claims).
    print("\n🔬 AUDIT-ONLY WITHIN/DUAL DIAGNOSTICS:")
    print("-" * 85)
    
    # 1. Evasion: D0 clean vs D0 claude / D0 gemini
    res_ev_c = corrected_paired_t(np.array(cells["D0"]["claude_atk"]) - np.array(cells["D0"]["clean"]), test_frac=0.30)
    res_ev_g = corrected_paired_t(np.array(cells["D0"]["gemini_atk"]) - np.array(cells["D0"]["clean"]), test_frac=0.30)
    print(f"1. Claude Evasion against D0: Δ = {res_ev_c['mean']*100:+.2f} pp, t = {res_ev_c.get('t', 0.0):.2f}, p = {res_ev_c['p']:.4f}")
    print(f"2. Gemini Evasion against D0: Δ = {res_ev_g['mean']*100:+.2f} pp, t = {res_ev_g.get('t', 0.0):.2f}, p = {res_ev_g['p']:.4f}")
    
    # 2. Within-Generator Restoration
    res_res_c = corrected_paired_t(np.array(cells["D2_Claude"]["claude_atk"]) - np.array(cells["D0"]["claude_atk"]), test_frac=0.30)
    res_res_g = corrected_paired_t(np.array(cells["D2_Gemini"]["gemini_atk"]) - np.array(cells["D0"]["gemini_atk"]), test_frac=0.30)
    res_dual_c = corrected_paired_t(np.array(cells["D2_Dual"]["claude_atk"]) - np.array(cells["D0"]["claude_atk"]), test_frac=0.30)
    res_dual_g = corrected_paired_t(np.array(cells["D2_Dual"]["gemini_atk"]) - np.array(cells["D0"]["gemini_atk"]), test_frac=0.30)
    print(f"3. Claude Within-Gen Change (D0 -> D2_Claude): Δ = {res_res_c['mean']*100:+.2f} pp, t = {res_res_c.get('t', 0.0):.2f}, p = {res_res_c['p']:.4f}")
    print(f"4. Gemini Within-Gen Change (D0 -> D2_Gemini): Δ = {res_res_g['mean']*100:+.2f} pp, t = {res_res_g.get('t', 0.0):.2f}, p = {res_res_g['p']:.4f}")
    print(f"5. Dual Audit Change on Claude attack (D0 -> D2_Dual): Δ = {res_dual_c['mean']*100:+.2f} pp, t = {res_dual_c.get('t', 0.0):.2f}, p = {res_dual_c['p']:.4f}")
    print(f"6. Dual Audit Change on Gemini attack (D0 -> D2_Dual): Δ = {res_dual_g['mean']*100:+.2f} pp, t = {res_dual_g.get('t', 0.0):.2f}, p = {res_dual_g['p']:.4f}")
    
    print("=" * 85)
    
    out_dir = os.path.join(ROOT, "data", "processed", "p3")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "p3_cross_generator_matrix.csv")
    loo_csv = os.path.join(out_dir, "p3_leave_one_llm_out.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    pd.DataFrame(loo_rows).to_csv(loo_csv, index=False)
    write_generated(os.path.join(SEC, "tab_cross_generator.tex"), _loo_table(loo_rows))
    print(f"[+] Output matrix saved: {out_csv}")
    print(f"[+] LOO summary saved: {loo_csv}")

if __name__ == "__main__":
    run_cross_generator_matrix()
