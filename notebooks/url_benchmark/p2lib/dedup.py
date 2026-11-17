"""Notebook 04, second half: the corpus with one row per host, and the corpora without `www.`."""
import os, re, shutil
from collections import Counter
from math import log2
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import tldextract
from threadpoolctl import threadpool_limits

from .core import run_benchmark, run_strict, run_dup_leakage
from .transfer import run_cross_dataset, run_combined_training, run_shap_weighting, PRUNE
from .cnn import run_charcnn_xdata

RES = "data/processed/p2"


COPIED = ["p2_benchmark", "p2_temporal_strict", "p2_temporal_strict_guarded", "cross_dataset_F1",
          "cross_dataset_ROC-AUC", "cross_dataset_F1_CatBoost", "p2_dup_leakage",
          "p2_shap_weighting"]


TRANSFER = ["cross_dataset_" + m + t for m in ("F1", "ROC-AUC")
            for t in ("_LogReg", "_HistGB", "_pruned", "_LogReg_pruned", "_HistGB_pruned",
                      "_coral", "_LogReg_coral")] + [
    "cross_dataset_F1_CharCNN", "cross_dataset_ROC-AUC_CharCNN", "p2_charcnn_xdata",
    "combined_training"]


def bare_host(urls: pd.Series) -> pd.Series:
    return (urls.astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
            .str.replace(r"^www\.", "", regex=True))


def dedup() -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv("data/processed/dataset_url.csv", low_memory=False)
    c = pd.read_csv("data/processed/vn_compphish.csv", low_memory=False)
    if len(u) != len(c):
        raise SystemExit("dataset_url.csv and vn_compphish.csv are not row-aligned")
    s = u["collected_at"].astype(str).str.strip().str.slice(0, 10)
    dated = (pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
             .fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce"))).notna()
    host = u["url_norm"].astype(str).str.lower().str.replace(r"^[a-z][a-z0-9+.-]*://", "", regex=True)
    k = pd.DataFrame({"i": np.arange(len(u)), "bare": bare_host(u["url_norm"]),
                      "dated": dated, "www": host.str.startswith("www.")})
    k = k.sort_values(["bare", "dated", "www", "i"], ascending=[True, False, True, True])
    keep = np.sort(k.drop_duplicates("bare")["i"].to_numpy())
    return u.iloc[keep].reset_index(drop=True), c.iloc[keep].reset_index(drop=True)




_EXT = tldextract.TLDExtract(suffix_list_urls=())  # the snapshot bundled with the library, no network


COLUMNS = ["url", "url_len", "dom", "dom_len", "is_ip", "tld", "tld_len", "subdom_cnt",
           "letter_cnt", "digit_cnt", "special_cnt", "eq_cnt", "qm_cnt", "amp_cnt",
           "dot_cnt", "dash_cnt", "under_cnt", "letter_ratio", "digit_ratio", "spec_ratio",
           "is_https", "slash_cnt", "entropy", "path_len", "query_len"]


_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * log2(c / n) for c in Counter(s).values())


def extract(url: str) -> dict:
    url = str(url).strip()
    p = urlparse(url if re.match(r"^[a-z]+://", url, re.I) else "http://" + url)
    host = (p.hostname or "").lower()
    # scheme-stripped URL used for ALL lexical features (scheme-independent)
    core = re.sub(r"^[a-z]+://", "", url, flags=re.I)

    if _EXT is not None:
        ext = _EXT(url if re.match(r"^[a-z]+://", url, re.I) else "http://" + url)
        suffix = ext.suffix
        dom = ext.registered_domain or host
        subdom = ext.subdomain
    else:  # fallback (less accurate for multi-part TLDs)
        parts = host.split(".")
        suffix = parts[-1] if parts else ""
        dom = ".".join(parts[-2:]) if len(parts) >= 2 else host
        subdom = ".".join(parts[:-2]) if len(parts) > 2 else ""

    url_len = len(core)
    letter_cnt = sum(c.isalpha() for c in core)
    digit_cnt = sum(c.isdigit() for c in core)
    special_cnt = url_len - letter_cnt - digit_cnt
    is_ip = 1 if _IP_RE.match(host) else 0
    path_len = len(p.path.strip("/"))
    return {
        "url": url,
        "url_len": url_len,
        "dom": dom,
        "dom_len": len(dom),
        "is_ip": is_ip,
        "tld": suffix,
        "tld_len": len(suffix),
        "subdom_cnt": len([s for s in subdom.split(".") if s]) if subdom else 0,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": core.count("="),
        "qm_cnt": core.count("?"),
        "amp_cnt": core.count("&"),
        "dot_cnt": core.count("."),
        "dash_cnt": core.count("-"),
        "under_cnt": core.count("_"),
        "letter_ratio": letter_cnt / url_len if url_len else 0.0,
        "digit_ratio": digit_cnt / url_len if url_len else 0.0,
        "spec_ratio": special_cnt / url_len if url_len else 0.0,
        "is_https": 1 if url.lower().startswith("https") else 0,
        "slash_cnt": core.count("/"),
        "entropy": _entropy(core),
        "path_len": path_len,
        "query_len": len(p.query),
    }


PROC = "data/processed"
NOWWW_OUT = os.path.join(PROC, "p2", "nowww")
SOURCES = {"phishvn": os.path.join(PROC, "p2", "dedup_audit", "vn_compphish_dedup.csv"),
           "phiusiil": os.path.join(PROC, "external", "phiusiil_compphish.csv"),
           "iscx": os.path.join(PROC, "external", "iscx_compphish.csv"),
           "phishstorm": os.path.join(PROC, "external", "phishstorm_compphish.csv")}


NAMES = {"phishvn": "PhishVN", "phiusiil": "PhiUSIIL", "iscx": "ISCXURL2016",
         "phishstorm": "PhishStorm"}


WWW = re.compile(r"^((?:[A-Za-z][A-Za-z0-9+.-]*://)?)www\.", re.I)


def is_phish(v) -> bool:
    return str(v).strip().lower() in {"1", "phishing", "phish", "malicious", "bad"}


def build():
    os.makedirs(NOWWW_OUT, exist_ok=True)
    share = []
    for key, path in SOURCES.items():
        df = pd.read_csv(path, low_memory=False)
        url = df["url"].astype(str)
        has = url.map(lambda u: bool(WWW.match(u)))
        y = df["label"].map(is_phish)
        for cls, m in (("phishing", y), ("benign", ~y)):
            share.append({"corpus": NAMES[key], "class": cls, "rows": int(m.sum()),
                          "www_share": float(has[m].mean())})
        feats = pd.DataFrame([extract(WWW.sub(r"\1", u)) for u in url])
        for col in df.columns:
            if col not in feats.columns:
                feats[col] = df[col].values
        feats[df.columns].to_csv(os.path.join(NOWWW_OUT, f"{key}_nowww.csv"), index=False)
        print(f"[i] {NAMES[key]}: {len(df)} rows, {int(has.sum())} prefixes removed", flush=True)
    rel = pd.read_csv(os.path.join(PROC, "vn_compphish.csv"), usecols=["url", "label"])
    has, y = rel["url"].astype(str).map(lambda u: bool(WWW.match(u))), rel["label"].map(is_phish)
    for cls, m in (("phishing", y), ("benign", ~y)):
        share.append({"corpus": "PhishVN (released)", "class": cls, "rows": int(m.sum()),
                      "www_share": float(has[m].mean())})
    pd.DataFrame(share).to_csv(os.path.join(NOWWW_OUT, "www_share.csv"), index=False)
    print(pd.DataFrame(share).round(3).to_string(index=False))



def run_dedup_audit(work: Path, tree: Path):
    """Write the one-row-per-host corpus, rebuild the work tree around it in `tree`, rerun there the
    experiments the duplicate rows touch, and copy their results to dedup_audit/ of the main tree."""
    ud, cd = dedup()
    dedup_out = Path(RES) / "dedup_audit"
    dedup_out.mkdir(parents=True, exist_ok=True)
    ud.to_csv(dedup_out / "dataset_url_dedup.csv", index=False)
    cd.to_csv(dedup_out / "vn_compphish_dedup.csv", index=False)
    print(f"kept {len(ud):,} of the corpus's rows, one per host")

    (tree / "data" / "processed" / "p2").mkdir(parents=True, exist_ok=True)
    for name in ("external", "first_seen_validation_summary.json"):
        dst = tree / "data" / "processed" / name
        if not dst.exists():
            os.symlink(work / "data" / "processed" / name, dst)
    ud.to_csv(tree / "data" / "processed" / "dataset_url.csv", index=False)
    cd.to_csv(tree / "data" / "processed" / "vn_compphish.csv", index=False)

    os.chdir(tree)
    try:
        with threadpool_limits(1, "blas"):
            run_benchmark()
            run_strict()
            run_strict(guard_control=True, out=f"{RES}/p2_temporal_strict_guarded.csv", curves="")
            run_cross_dataset(metric="F1", out=f"{RES}/cross_dataset_F1.csv")
            run_cross_dataset(metric="ROC-AUC", out=f"{RES}/cross_dataset_ROC-AUC.csv")
            run_cross_dataset(metric="F1", model="CatBoost", out=f"{RES}/cross_dataset_F1_CatBoost.csv")
            run_dup_leakage()
            run_shap_weighting()
            for metric in ("F1", "ROC-AUC"):
                for model, suf in (("LogReg", "_LogReg"), ("HistGB", "_HistGB")):
                    run_cross_dataset(metric=metric, model=model, out=f"{RES}/cross_dataset_{metric}{suf}.csv")
                for model, suf in (("RandomForest", ""), ("LogReg", "_LogReg"), ("HistGB", "_HistGB")):
                    run_cross_dataset(metric=metric, model=model, drop=PRUNE, out=f"{RES}/cross_dataset_{metric}{suf}_pruned.csv")
                for model, suf in (("RandomForest", ""), ("LogReg", "_LogReg")):
                    run_cross_dataset(metric=metric, model=model, adapt="coral", out=f"{RES}/cross_dataset_{metric}{suf}_coral.csv")
            run_combined_training()
            run_charcnn_xdata()
    finally:
        os.chdir(work)
    for f in COPIED + TRANSFER:
        shutil.copyfile(tree / "data" / "processed" / "p2" / f"{f}.csv", dedup_out / f"{f}.csv")
    return COPIED + TRANSFER


def run_nowww():
    """Strip `www.` from every host of the four corpora, recompute the features, rerun the main matrices."""
    build()
    nowww_corpora = [f"{NAMES[k]}={os.path.join(NOWWW_OUT, k + '_nowww.csv')}" for k in SOURCES]
    with threadpool_limits(1, "blas"):
        for metric, model, suf in (("F1", "RandomForest", ""), ("ROC-AUC", "RandomForest", ""),
                                   ("MCC", "RandomForest", ""), ("F1", "CatBoost", "_CatBoost")):
            run_cross_dataset(corpora=nowww_corpora, metric=metric, model=model,
                              out=os.path.join(NOWWW_OUT, f"cross_dataset_{metric}{suf}.csv"))
