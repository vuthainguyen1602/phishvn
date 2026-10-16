#!/usr/bin/env python3
"""
p3_login_control.py — page-type-matched benign control (revision 2, R3.2).

Plan, committed before any control page was captured: papers/P3_multimodal/REVISION2/
LOGIN_CONTROL_PLAN.md. Two measurements:

  1. page-type composition of the 207+207 subset (password input; login or payment vocabulary
     in the visible text), per class, and the benign test-fold FPR restricted to the benign
     pages that carry a password field or payment vocabulary;
  2. genuine login/account pages of Vietnamese banks, wallets, telcos, utilities, government
     portals, logistics, e-commerce and game top-up services (data/raw/login_control/), captured
     by urlscan_submit.py --keep-url and passed through build_content_manifest.py's gates, scored
     by the models each of the 20 main-study splits trains (standardised linear head).

RUN:  python scripts/p3_login_control.py [--encoders tfidf,xlm-r]
OUT:  data/processed/p3/results/login_control.json
      papers/P3_multimodal/sections/tab_login_control.tex, gen_login_control.tex
"""
import argparse
import csv
import os
import re
import sys
from urllib.parse import urlparse

import numpy as np
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
import train_content_fusion as tcf
from compphish_features import extract
from train_html_baseline import html_to_text
from vn_filter import is_vietnamese_text, visible_text
from build_content_manifest import PARKED, NOT_CONTENT
from genfile import write_generated, write_results
from make_p3_assets import _balanced_rows, FUSION_SEEDS, ENC_LABEL

SEC = os.path.join(ROOT, "papers", "P3_multimodal", "sections")
CAPTURES = os.path.join(ROOT, "data", "raw", "login_control", "captures.csv")
TARGETS = os.path.join(ROOT, "data", "raw", "login_control", "targets.csv")
CONFIGS = ["content", "url", "content+url"]

PW = re.compile(r'<input[^>]+type\s*=\s*["\']?password', re.I)
PAY = re.compile(r"số thẻ|so the|cvv|ngày hết hạn|mã otp|otp|số tài khoản|card number", re.I)
LOGIN = re.compile(r"đăng nhập|dang nhap|mật khẩu|mat khau|login|sign in|log in", re.I)


def sensitive(html, text):
    """Password input or payment vocabulary: the page asks for something a phisher would."""
    return bool(PW.search(html)) or bool(PAY.search(text))


def _gate(html, title):
    vis = visible_text(html)
    if len(vis) < 40:
        return "empty"
    if PARKED.search((title or "") + " " + vis[:400]) or NOT_CONTENT.search(vis[:600]):
        return "parked"
    if not is_vietnamese_text(vis):
        return "non_vi"
    return ""


def load_controls():
    """Both batches (plan + amendment 1) through the manifest's own gates, with per-batch gate
    losses. A control whose host is a manifest domain is dropped: it could share a host with a
    training page."""
    import pandas as pd
    manifest = set(pd.read_csv(os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv"))
                   .domain.astype(str).str.lower().str.removeprefix("www."))
    cat = {r["url"]: r["category"] for r in csv.DictReader(open(TARGETS, encoding="utf-8"))}
    batches = [("batch1", CAPTURES, lambda r: cat.get(r["domain"], "other")),
               ("batch2", CAPTURES.replace("captures.csv", "captures_batch2.csv"),
                lambda r: r.get("category", "other"))]
    flow, rows, seen = {}, [], set()
    for name, path, category in batches:
        f = flow[name] = {"targets": 0, "no_capture": 0, "in_manifest": 0, "duplicate": 0,
                          "empty": 0, "parked": 0, "non_vi": 0, "kept": 0}
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            f["targets"] += 1
            domf = r.get("dom_file") or ""
            if r.get("found") != "1" or not domf or not os.path.exists(os.path.join(ROOT, domf)):
                f["no_capture"] += 1; continue
            host = (urlparse(r["domain"]).hostname or "").lower().removeprefix("www.")
            if host in manifest:
                f["in_manifest"] += 1; continue
            if host in seen:
                f["duplicate"] += 1; continue
            html = open(os.path.join(ROOT, domf), encoding="utf-8", errors="ignore").read()
            g = _gate(html, r.get("title"))
            if g:
                f[g] += 1; continue
            seen.add(host)
            uf = extract(host)
            text = html_to_text(html)
            rows.append({"text": text, "html": html, "y": 0, "batch": name,
                         "url": [float(uf.get(k, 0)) for k in tcf.URL_NUM],
                         "category": category(r), "target": r["domain"],
                         "sensitive": sensitive(html, text)})
            f["kept"] += 1
    flow["kept"] = len(rows)
    return rows, flow


def page_types(rows):
    out = {}
    for y, name in ((1, "phishing"), (0, "benign")):
        sub = [r for r in rows if r["y"] == y]
        out[name] = {"n": len(sub),
                     "password": sum(bool(PW.search(r["html"])) for r in sub),
                     "payment_words": sum(bool(PAY.search(r["text"])) for r in sub),
                     "login_words": sum(bool(LOGIN.search(r["text"])) for r in sub),
                     "sensitive": sum(sensitive(r["html"], r["text"]) for r in sub)}
    return out


def wilson(x, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run(encoder, rows, ctrl):
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(rows))
    sens = np.array([sensitive(r["html"], r["text"]) for r in rows])
    rec = {c: {"fpr_benign": [], "fpr_benign_sensitive": [], "fpr_control": [],
               "fpr_control_sensitive": [], "recall": [], "score_control": [],
               "score_benign": [], "score_phishing": []} for c in CONFIGS}
    ctrl_sens = np.array([c["sensitive"] for c in ctrl], bool)
    for s in range(FUSION_SEEDS):
        tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=s)
        ytr, yte = y[tr], y[te]
        tr_txt = [rows[i]["text"] for i in tr]
        ev_txt = [rows[i]["text"] for i in te] + [c["text"] for c in ctrl]
        t_tr, t_ev, kind = tcf.text_matrix(tr_txt, ev_txt, encoder)
        u_tr = np.array([rows[i]["url"] for i in tr])
        u_ev = np.vstack([np.array([rows[i]["url"] for i in te]),
                          np.array([c["url"] for c in ctrl])])
        n_te = len(te)
        for cfg in CONFIGS:
            txt = (t_tr, t_ev) if "content" in cfg else (None, None)
            btr, bev = ([u_tr], [u_ev]) if "url" in cfg else ([], [])
            (a, b), btr, bev = tcf._standardise(txt, kind if "content" in cfg else "dense",
                                                btr, bev)
            k = kind if "content" in cfg else "dense"
            clf = tcf.make_head("logreg", s, ytr).fit(tcf.stack(a, btr, k), ytr)
            sc = clf.predict_proba(tcf.stack(b, bev, k))[:, 1]
            st, scc = sc[:n_te], sc[n_te:]
            ben = yte == 0
            r = rec[cfg]
            r["fpr_benign"].append(float(np.mean(st[ben] >= 0.5)))
            sb = ben & sens[te]
            r["fpr_benign_sensitive"].append(float(np.mean(st[sb] >= 0.5)) if sb.any() else np.nan)
            r["fpr_control"].append(float(np.mean(scc >= 0.5)))
            r["fpr_control_sensitive"].append(float(np.mean(scc[ctrl_sens] >= 0.5))
                                              if ctrl_sens.any() else np.nan)
            r["recall"].append(float(np.mean(st[yte == 1] >= 0.5)))
            r["score_control"].append(float(np.mean(scc)))
            r["score_benign"].append(float(np.mean(st[ben])))
            r["score_phishing"].append(float(np.mean(st[yte == 1])))
        print(f"[i] {encoder} split {s + 1}/{FUSION_SEEDS}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="tfidf,phobert,phobert-v2,visobert,xlm-r")
    args = ap.parse_args()
    rows = _balanced_rows()
    ctrl, flow = load_controls()
    types = page_types(rows)
    print("[i] page types:", types)
    print("[i] control flow:", flow)
    encs = args.encoders.split(",")
    res = {e: run(e, rows, ctrl) for e in encs}
    n_c = len(ctrl)
    n_cs = int(sum(c["sensitive"] for c in ctrl))
    summary = {e: {c: {k: float(np.nanmean(v)) for k, v in res[e][c].items()} for c in CONFIGS}
               for e in encs}
    write_results(os.path.join(ROOT, "data", "processed", "p3", "results", "login_control.json"),
                  {"flow": flow, "page_types": types, "n_control_sensitive": n_cs,
                   "categories": {k: sum(1 for c in ctrl if c["category"] == k)
                                  for k in sorted({c["category"] for c in ctrl})},
                   "summary": summary})

    def pct(v):
        return f"{100 * v:.1f}"
    body = []
    for e in encs:
        for c in ("content", "content+url"):
            m = summary[e][c]
            body.append(f"{ENC_LABEL.get(e, e) if c == 'content' else ''} & "
                        f"{'content' if c == 'content' else 'content$+$URL'} & "
                        f"{pct(m['recall'])} & {pct(m['fpr_benign'])} & "
                        f"{pct(m['fpr_benign_sensitive'])} & {pct(m['fpr_control'])} \\\\")
    u = summary[encs[0]]["url"]
    tb = types["benign"]; tp = types["phishing"]
    tex = f"""\\begin{{table}}[t]
\\caption{{Page-type control, mean over the {FUSION_SEEDS} splits at $\\tau=0.5$ (\\%): phishing recall,
FPR on all benign test pages, on the benign test pages that ask for a password or payment data
($n={tb['sensitive']}$ in the subset), and on ${n_c}$ genuine Vietnamese login pages never seen
in training. URL only: {pct(u['recall'])}, {pct(u['fpr_benign'])}, {pct(u['fpr_benign_sensitive'])},
{pct(u['fpr_control'])}.}}
\\label{{tab:logincontrol}}
\\small\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l l r r r r}}
\\toprule
Encoder & Config. & Recall & FPR benign & FPR sens. & FPR login \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    write_generated(os.path.join(SEC, "tab_login_control.tex"), tex, f"(n_control={n_c})")
    lo, hi = wilson(round(summary[encs[0]]["content"]["fpr_control"] * n_c), n_c)
    write_generated(
        os.path.join(SEC, "gen_login_control.tex"),
        f"By a page-function heuristic (a password input, or card, OTP or account-number "
        f"vocabulary in the visible text), ${tp['sensitive']}$ of the ${tp['n']}$ phishing pages "
        f"and ${tb['sensitive']}$ of the ${tb['n']}$ benign pages ask for credentials or payment "
        f"data (password field: ${tp['password']}$ against ${tb['password']}$). Of the "
        f"${sum(flow[b]['targets'] for b in ('batch1', 'batch2'))}$ genuine login pages "
        f"targeted for the control, ${n_c}$ pass the subset's gates and share no host with a "
        f"subset page. The char-$n$-gram content model flags "
        f"${pct(summary[encs[0]]['content']['fpr_control'])}\\%$ of them "
        f"(Wilson $95\\%$ CI ${pct(lo)}$--${pct(hi)}\\%$, counting each page once at the "
        f"split-averaged rate), against ${pct(summary[encs[0]]['content']['fpr_benign'])}\\%$ of "
        f"the benign test pages\n")
    for e in encs:
        print(e, {c: {k: round(v, 3) for k, v in summary[e][c].items()} for c in CONFIGS})


if __name__ == "__main__":
    main()
