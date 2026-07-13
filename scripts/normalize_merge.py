#!/usr/bin/env python3
"""
normalize_merge.py — Merge & normalize Vietnamese phishing data (P1).

Reads raw CSVs in data/raw/<source>/, maps them to a common SCHEMA (see data/docs/schema.md),
deduplicates, extracts URL features, anonymizes PII, splits train/val/test BY TIME,
then writes data/processed/dataset_{url,sms,email}.csv + splits/.

INSTALL:  pip install pandas
RUN:      python normalize_merge.py --raw data/raw --out data/processed
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import os
import re
import uuid
from urllib.parse import urlparse

import pandas as pd

CORE = ["id", "channel", "label", "source", "is_llm", "lang", "scenario",
        "impersonated_org", "collected_at", "label_source", "split"]

SUSPICIOUS_TLDS = {"top", "xyz", "cc", "online", "info", "click", "shop", "vip", "icu", "buzz"}

# tinnhiemmang handling status -> label-confidence tier (all are blacklisted = phishing).
#   gold   = verified & handled (highest confidence)
#   silver = under processing   (medium confidence)
#   pending= awaiting review     (weak label; excluded from train by default)
STATUS_TIER = {"Đã xử lý": "gold", "Đang xử lý": "silver", "Đang chờ": "pending"}

# Map organization/description/DOMAIN -> impersonation sector. Patterns match both Vietnamese
# free text (impersonated_org, when present) AND brand tokens that appear inside the domain string
# itself, because the tinnhiemmang blacklist ships impersonated_org empty — the brand signal lives
# in the URL (e.g. vietcombank247.com.vn, vtp-vandon.online, chinhphu2025.com). Order matters
# (first match wins); tax before gov so "thue" is not swallowed by the gov pattern.
SCENARIO_MAP = [
    (r"thu[eế]|\btax\b|tongcucthue|etax", "tax"),
    (r"ngân hàng|\bbank\b|vietcom|\bvcb\b|techcom|\btcb\b|mbbank|\bbidv\b|agribank|\bacb\b|"
     r"vpbank|sacombank|shinhan|\bvib\b|tpbank|\bocb\b|hdbank|\bscb\b|eximbank|napas|"
     r"ví điện tử|momo|zalopay|vnpay|viettelpay|\b247\b", "bank"),
    (r"công an|viện kiểm|chính phủ|chinhphu|congan|bocongan|nhà nước|nhanuoc|vneid|"
     r"dịch vụ công|dichvucong|bảo hiểm|baohiem|bhxh|bhyt|hành chính|toà án|toaan|"
     r"\.gov\.vn|\.edu\.vn", "gov"),
    (r"shopee|lazada|\btiki\b|sendo|tiktokshop|tiktok-shop|amazon|amaz|alibaba|"
     r"thương mại|\bsàn\b|\bmall\b|\bshop\b", "ecommerce"),
    (r"viettel|\bvnpt\b|mobifone|vinaphone|\bvtp\b|telecom|viễn thông", "telecom"),
    (r"giao hàng|giaohang|\bghtk\b|\bghn\b|shipper|vận chuyển|vandon|jtexpress|"
     r"ninjavan|\bspx\b|delivery|express", "delivery"),
    (r"nạp thẻ|nap the|muathe|muacard|napgame|game|\bcard\b|nạp tiền", "gaming"),
    (r"facebook|\bfb\b|zalo|tiktok|instagram|telegram|messenger|mạng xã hội|social", "social"),
]

try:
    import tldextract
    _EXT = tldextract.TLDExtract(suffix_list_urls=())  # offline snapshot
except Exception:  # pragma: no cover
    _EXT = None


def reg_domain(host: str) -> str:
    """Registrable domain (eTLD+1), e.g. hanoi3.vietcombank247.com.vn -> vietcombank247.com.vn.
    Used to GROUP same-campaign subdomains so they never span train/val/test (leakage)."""
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    if _EXT is not None:
        rd = _EXT(host).registered_domain
        if rd:
            return rd
    parts = host.split(".")
    # crude multi-part TLD fallback (com.vn, gov.vn, edu.vn, org.vn, net.vn, co.uk ...)
    if len(parts) >= 3 and parts[-2] in {"com", "gov", "edu", "org", "net", "co", "ac", "biz"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")
EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
ACCT_RE = re.compile(r"(?<!\d)\d{9,16}(?!\d)")  # account/card numbers (raw)


def redact_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = ACCT_RE.sub("<NUM>", text)
    return text.strip()


def make_id(channel: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{channel}|{key}"))


def map_scenario(text: str) -> str:
    t = (text or "").lower()
    for pat, sc in SCENARIO_MAP:
        if re.search(pat, t):
            return sc
    return "other"


def url_features(url: str) -> dict:
    u = (url or "").strip()
    if not re.match(r"^https?://", u, re.I):
        u2 = "http://" + u
    else:
        u2 = u
    p = urlparse(u2)
    host = p.hostname or ""
    tld = host.rsplit(".", 1)[-1].lower() if "." in host else ""
    labels = host.split(".")
    has_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0
    return {
        "url": u,
        "url_norm": u2.lower(),
        "domain": host.lower(),
        "tld": tld,
        "url_len": len(u),
        "num_dots": u.count("."),
        "num_subdomains": max(0, len(labels) - 2),
        "has_ip": has_ip,
        "has_at": 1 if "@" in u else 0,
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
    }


def base_row(channel, label, source, is_llm=0, lang="vi", scenario="other",
             org="", collected_at="", label_source="feed"):
    return {"channel": channel, "label": label, "source": source, "is_llm": is_llm,
            "lang": lang, "scenario": scenario, "impersonated_org": org,
            "collected_at": collected_at, "label_source": label_source, "split": ""}


# ---------- Loaders per source ----------
def load_tinnhiemmang(path, keep_status=None) -> list[dict]:
    """keep_status: optional set of allowed status strings (e.g. {"Đã xử lý"} for the gold tier).
    None -> keep all; every row keeps its 'status' column so it can be filtered downstream."""
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        status = str(r.get("status", "") or "").strip()
        if keep_status and status not in keep_status:
            continue
        org = str(r.get("impersonated_org", "") or "")
        dom = str(r.get("domain", "") or "")
        ch = "social" if str(r.get("type", "")).lower() == "social" else "url"
        row = base_row(ch, "phishing", "tinnhiemmang", 0, "vi",
                       map_scenario(org + " " + str(r.get("sector", "")) + " " + dom),
                       org, str(r.get("detected_date", "")), "feed")
        row.update(url_features(str(r.get("domain", ""))))
        row["status"] = status
        row["tier"] = STATUS_TIER.get(status, "silver")   # label-confidence tier
        out.append(row)
    return out


def load_url_feed(path, source) -> list[dict]:
    df = pd.read_csv(path)
    col = "url" if "url" in df.columns else df.columns[0]
    out = []
    for _, r in df.iterrows():
        row = base_row("url", "phishing", source, 0, "mixed", "other", "",
                       str(r.get("scraped_at", "") or ""), "feed")
        row.update(url_features(str(r.get(col, ""))))
        out.append(row)
    return out


def load_text_csv(path, source, channel, is_llm=0) -> list[dict]:
    """SMS/email dataset as CSV with text/label columns (adjust column names if different)."""
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in ("text", "message", "content", "body")), df.columns[0])
    lcol = next((c for c in df.columns if c.lower() in ("label", "class", "y")), None)
    out = []
    for _, r in df.iterrows():
        raw_label = str(r.get(lcol, "phishing")).lower() if lcol else "phishing"
        label = "phishing" if raw_label in ("1", "spam", "smishing", "phishing", "phish") else "benign"
        text = redact_pii(str(r.get(tcol, "")))
        ch = str(r.get("channel") or channel)                 # honor the channel column if present
        collected = str(r.get("collected_at", "") or "")
        row = base_row(ch, label, source, is_llm, "vi",
                       map_scenario(text), str(r.get("brandname", "") or ""), collected,
                       "human" if source == "author" else "auto")
        row["text"] = text
        row["msg_len"] = len(text)
        row["has_url"] = 1 if re.search(
            r"https?://|\b[a-z0-9-]+\.(?:com|net|org|top|xyz|cc|vn|info|online|shop|vip|icu|click|buzz)\b",
            text, re.I) else 0
        out.append(row)
    return out


def load_trusted_orgs(path) -> list[dict]:
    """Trusted organizations (benign). Only produce a benign URL when --enrich ran (has domain/website)."""
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        dom = str(r.get("domain", "") or "").strip()
        web = str(r.get("website", "") or "").strip()
        url = web or dom
        if not url:            # not enriched -> no URL, skip (only the org name is available)
            continue
        row = base_row("url", "benign", "tinnhiem_org", 0, "vi",
                       map_scenario(str(r.get("org_name", "")) + " " + url),
                       str(r.get("org_name", "")), str(r.get("cert_date", "")), "feed")
        row.update(url_features(url))
        out.append(row)
    return out


def load_tinnhiem_web(path) -> list[dict]:
    """Trusted WEBSITES (benign) scraped from filterObj type=web (website-tin-nhiem).
    Each row is a domain; labelled benign, tier gold (curated/verified source)."""
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        dom = str(r.get("domain", "") or "").strip()
        if not dom:
            continue
        row = base_row("url", "benign", "tinnhiem_web", 0, "vi", map_scenario(dom), "",
                       str(r.get("scraped_at", "") or ""), "feed")
        row.update(url_features(dom))
        row["tier"] = "gold"
        out.append(row)
    return out


def load_tranco(path) -> list[dict]:
    """HARD benign domains from the Tranco top-list (see fetch_tranco.py). Labelled benign,
    source 'tranco', tier 'silver' (traffic-based, not a curated whitelist) so it can be sliced
    separately from the easy 'tinnhiem_web' benign when measuring external validity."""
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        dom = str(r.get("domain", "") or "").strip()
        if not dom:
            continue
        row = base_row("url", "benign", "tranco", 0, "mixed", map_scenario(dom), "",
                       str(r.get("scraped_at", "") or ""), "feed")
        row.update(url_features(dom))
        row["tier"] = "silver"
        out.append(row)
    return out


def collect(raw_dir, keep_status=None) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(raw_dir, "tinnhiemmang", "*.csv")):
        rows += load_tinnhiemmang(p, keep_status=keep_status)
    for p in glob.glob(os.path.join(raw_dir, "tinnhiem_web", "*.csv")):
        rows += load_tinnhiem_web(p)
    for p in glob.glob(os.path.join(raw_dir, "tinnhiem_org", "*.csv")):
        rows += load_trusted_orgs(p)
    for p in glob.glob(os.path.join(raw_dir, "tranco", "*.csv")):
        rows += load_tranco(p)
    for p in glob.glob(os.path.join(raw_dir, "openphish", "*.csv")):
        rows += load_url_feed(p, "openphish")
    for p in glob.glob(os.path.join(raw_dir, "urlhaus", "*.csv")):
        rows += load_url_feed(p, "urlhaus")
    for p in glob.glob(os.path.join(raw_dir, "author", "*.csv")):
        rows += load_text_csv(p, "author", "sms")   # change to 'email' if it is email
    for p in glob.glob(os.path.join(raw_dir, "reports", "*.csv")):
        rows += load_text_csv(p, "report", "email")
    for p in glob.glob(os.path.join(raw_dir, "llm", "*.csv")):
        rows += load_text_csv(p, "llm", "sms", is_llm=1)
    df = pd.DataFrame(rows)
    return df


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """GROUP-AWARE time-based split. Rows are grouped by registrable domain (or text hash for
    message channels) so that every URL of one campaign/domain lands in the SAME split — a plain
    per-row split leaks near-duplicate subdomains (hanoi.x.com.vn, hanoi3.x.com.vn, ...) across
    train and test and inflates metrics. Dated groups split by their EARLIEST date (train=oldest,
    test=newest); undated groups are assigned 70/15/15 at random so both classes appear everywhere."""
    import random
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["collected_at"], errors="coerce", dayfirst=True)

    # group key: registrable domain for URL-ish rows, else text/id (message rows)
    def _grp(r):
        host = str(r.get("domain", "") or "")
        rd = reg_domain(host)
        if rd:
            return "d:" + rd
        t = r.get("text")
        return "t:" + (str(t) if isinstance(t, str) and t else str(r.get("id", "")))
    df["_grp"] = df.apply(_grp, axis=1)

    # one representative datetime per group = earliest observed (NaT if the whole group is undated)
    gmin = df.groupby("_grp")["_dt"].min()
    dated_groups = gmin.dropna().sort_values()
    n = len(dated_groups)
    tr, va = int(n * 0.70), int(n * 0.85)
    split_of = {}
    for i, g in enumerate(dated_groups.index):
        split_of[g] = "train" if i < tr else ("val" if i < va else "test")

    rng = random.Random(42)
    for g in gmin.index[gmin.isna()]:
        r = rng.random()
        split_of[g] = "train" if r < 0.70 else ("val" if r < 0.85 else "test")

    df["split"] = df["_grp"].map(split_of)
    return df.drop(columns=["_dt", "_grp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--status", default="",
                    help='Keep only these tinnhiemmang statuses, comma-separated '
                         '(e.g. "Đã xử lý" for the gold tier). Empty = keep all.')
    ap.add_argument("--include-pending", action="store_true",
                    help='Keep weak-label "Đang chờ" (pending) rows; default drops them.')
    args = ap.parse_args()

    keep_status = {s.strip() for s in args.status.split(",") if s.strip()} or None
    df = collect(args.raw, keep_status=keep_status)
    if df.empty:
        print("No data found in", args.raw, "— place CSVs into data/raw/<source>/")
        return

    # tier column: fill non-tinnhiemmang rows (benign, etc.) as 'gold' (curated sources)
    if "tier" not in df.columns:
        df["tier"] = "gold"
    df["tier"] = df["tier"].fillna("gold")
    if not args.include_pending:
        n0 = len(df)
        df = df[df["tier"] != "pending"].reset_index(drop=True)
        if len(df) < n0:
            print(f"[i] dropped {n0 - len(df)} weak-label 'pending' rows (use --include-pending to keep)")

    # id + deduplication
    def keyof(r):
        u = r.get("url_norm"); t = r.get("text")
        u = "" if pd.isna(u) else str(u)
        t = "" if pd.isna(t) else str(t)
        return u or t
    df["id"] = df.apply(lambda r: make_id(r["channel"], keyof(r)), axis=1)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    df = temporal_split(df)

    os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
    groups = {
        "url": df[df["channel"].isin(["url", "qr", "social"])],
        "sms": df[df["channel"] == "sms"],
        "email": df[df["channel"] == "email"],
    }
    for name, g in groups.items():
        if g.empty:
            continue
        path = os.path.join(args.out, f"dataset_{name}.csv")
        g.to_csv(path, index=False, encoding="utf-8")
        print(f"[+] {path}: {len(g)} records "
              f"(phishing={sum(g.label=='phishing')}, benign={sum(g.label=='benign')})")
        for sp in ("train", "val", "test"):
            gs = g[g.split == sp]
            if not gs.empty:
                gs.to_csv(os.path.join(args.out, "splits", f"{name}_{sp}.csv"),
                          index=False, encoding="utf-8")
    print(f"Total: {len(df)} records. Remember to verify PII redaction & update the datasheet.")


if __name__ == "__main__":
    main()
