#!/usr/bin/env python3
"""
build_brand_tokens.py — Generate VN brand tokens from the Tin Nhiem Mang trusted-org registry.

Turns data/raw/tinnhiem_org/orgs.csv (scrape_trusted_orgs.py output) into
data/processed/brand_tokens.json: one token per certified org, extracted from its real
domain (vietcombank.com.vn -> "vietcombank") and its unaccented name ("The Gioi Di Dong"
-> "thegioididong"). vn_filter.is_vn_target() picks the file up automatically, so the
brand list grows with the registry instead of being maintained by hand in VN_TOKENS.

Tokens already matched by the static VN_TOKENS regex are excluded (reported under
"covered_by_static") — the JSON only adds coverage the hand-written list lacks.
Tokens of 3-4 chars are matched with word boundaries ("word" mode) to limit false
positives; note this misses fused forms ("tikivn.com" does not match \\btiki\\b).

Three guards keep global noise out (each drop is reported for audit):
  * FOREIGN-ORG rule: the registry certifies VN branches of global brands (PayPal, JPMorgan,
    DHL...) under their global domains; a token whose source orgs have no .vn domain is dropped
    ("dropped_foreign") — its phishing is worldwide, not Vietnamese-targeting.
  * GLOBAL-AMBIGUITY probe: a surviving token matching >= --tranco-max popular non-.vn domains
    (real Tranco top list) is dropped ("dropped_global").
  * Substring matching needs >= 6 chars; 3-5 char tokens are word-boundary matched (a 5-char
    substring like "shost" otherwise fires inside unrelated words: "...themeshosting...").

RUN:
  python scripts/build_brand_tokens.py                       # default in/out paths
  python scripts/build_brand_tokens.py --in data/raw/tinnhiem_org/orgs.csv \
      --out data/processed/brand_tokens.json --min-len 3 \
      --tranco data/raw/tranco/benign.csv --tranco-max 2
"""
from __future__ import annotations
import argparse, csv, json, re, sys, unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vn_filter import VN_TOKENS  # noqa: E402

# Multi-label public suffixes seen on VN org sites; single-label TLDs are stripped generically.
SUFFIXES = (
    ".com.vn", ".net.vn", ".org.vn", ".gov.vn", ".edu.vn", ".ac.vn", ".int.vn",
    ".biz.vn", ".info.vn", ".pro.vn", ".name.vn", ".health.vn", ".id.vn",
)
# Legal/structural words that carry no brand signal in org names (unaccented).
GENERIC_NAME_WORDS = {
    "cong", "ty", "co", "phan", "tnhh", "mtv", "tap", "doan", "tong", "chi", "nhanh",
    "ngan", "hang", "tmcp", "thuong", "mai", "dich", "vu", "bo", "so", "uy", "ban",
    "nhan", "dan", "truong", "dai", "hoc", "vien", "trung", "tam", "cuc", "tt",
    "viet", "nam", "vietnam", "vn", "joint", "stock", "company", "corporation",
    "group", "bank", "cp",
}
# Tokens too generic to ever use as a phishing signal, whatever their origin.
DENY = {
    "vietnam", "online", "store", "shop", "group", "admin", "portal", "home",
    "www", "web", "app", "mail", "news", "info", "gov", "com", "net", "org", "edu",
    "airlines", "airline", "hotel", "travel", "media", "digital", "global", "service",
    "whatsapp", "telegram", "mega", "smart", "cloud", "solution", "solutions",
}


def unaccent(s: str) -> str:
    s = (s or "").lower().replace("đ", "d")  # NFD does not decompose đ
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def domain_token(domain: str) -> str:
    """Registrable label: vietcombank.com.vn -> vietcombank, tiki.vn -> tiki."""
    d = (domain or "").lower().strip(".")
    if not d or "." not in d:
        return ""
    for suf in SUFFIXES:
        if d.endswith(suf):
            d = d[: -len(suf)]
            break
    else:
        d = d.rsplit(".", 1)[0]
    return d.rsplit(".", 1)[-1]  # drop remaining subdomains (www.foo -> foo)


def name_token(org_name: str) -> str:
    """Unaccented org name minus legal boilerplate, fused: The Gioi Di Dong -> thegioididong."""
    words = re.findall(r"[a-z0-9]+", unaccent(org_name))
    return "".join(w for w in words if w not in GENERIC_NAME_WORDS)


def token_mode(token: str, min_len: int) -> str | None:
    """'substring' (len>=6), 'word' (short, boundary-matched), or None (rejected)."""
    if len(token) < min_len or token in DENY or token.isdigit():
        return None
    return "substring" if len(token) >= 6 else "word"


def load_tranco_names(path: str) -> list[str]:
    """Registrable names of globally popular NON-.vn domains — the global-ambiguity probe set."""
    if not path or not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r["domain"].rsplit(".", 1)[0].lower() for r in csv.DictReader(f)
                if r.get("domain") and not r["domain"].endswith(".vn")]


def global_hits(token: str, mode: str, names: list[str]) -> int:
    pat = re.compile(re.escape(token) if mode == "substring" else rf"\b{re.escape(token)}\b", re.I)
    return sum(1 for n in names if pat.search(n))


def build(rows: list[dict], min_len: int, tranco_names: list[str] | None = None,
          tranco_max: int = 2) -> dict:
    tokens: dict[str, dict] = {}
    covered: dict[str, list[str]] = {}
    for r in rows:
        org = (r.get("org_name") or "").strip()
        for origin, tok in (("domain", domain_token(r.get("domain", ""))),
                            ("org_name", name_token(org))):
            mode = token_mode(tok, min_len)
            if not mode:
                continue
            if VN_TOKENS.search(tok):
                covered.setdefault(tok, []).append(org)
                continue
            e = tokens.setdefault(tok, {"token": tok, "mode": mode, "origins": set(),
                                        "orgs": set(), "domains": set(), "_vn_backed": False})
            e["origins"].add(origin)
            e["orgs"].add(org)
            dom = (r.get("domain") or "").lower()
            if dom:
                e["domains"].add(dom)
            # a token is VN-backed when some source org is certified under .vn (or has no
            # website at all — a domain-less org in the VN registry is a VN entity)
            if not dom or dom.endswith(".vn"):
                e["_vn_backed"] = True

    # Drop tokens with no .vn-certified source org: their brand's phishing is global, not VN.
    dropped_foreign = {}
    for tok in list(tokens):
        if not tokens[tok]["_vn_backed"]:
            dropped_foreign[tok] = sorted(tokens[tok]["orgs"])[:3]
            del tokens[tok]

    # Drop globally ambiguous tokens: ones that also match popular non-.vn domains (Tranco).
    dropped_global = {}
    if tranco_names:
        for tok in list(tokens):
            n = global_hits(tok, tokens[tok]["mode"], tranco_names)
            if n >= tranco_max:
                dropped_global[tok] = n
                del tokens[tok]

    # Prune substring tokens subsumed by a shorter substring token (vietinbanksc ⊃ vietinbank).
    subs = sorted((t for t in tokens.values() if t["mode"] == "substring"),
                  key=lambda t: len(t["token"]))
    kept_subs: list[str] = []
    for t in subs:
        if any(k in t["token"] for k in kept_subs):
            del tokens[t["token"]]
        else:
            kept_subs.append(t["token"])

    out = [{k: v for k, v in {**e, "origins": sorted(e["origins"]), "orgs": sorted(e["orgs"]),
            "domains": sorted(e["domains"])}.items() if k != "_vn_backed"}
           for e in tokens.values()]
    out.sort(key=lambda e: e["token"])
    return {
        "n_orgs": len(rows),
        "tokens": out,
        "dropped_foreign": dict(sorted(dropped_foreign.items())),
        "dropped_global": dict(sorted(dropped_global.items(), key=lambda kv: -kv[1])),
        "covered_by_static": {t: sorted(set(v)) for t, v in sorted(covered.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/tinnhiem_org/orgs.csv")
    ap.add_argument("--out", default="data/processed/brand_tokens.json")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--tranco", default="data/external/tranco_top100k.csv",
                    help="Tranco top-list CSV for the global-ambiguity filter ('' to disable). "
                         "NOT data/raw/tranco/ — that dir is a dataset input (normalize_merge "
                         "globs it as the benign class); refresh this probe file with "
                         "fetch_tranco.py --n 100000 pointed at a path outside data/raw/.")
    ap.add_argument("--tranco-max", type=int, default=2,
                    help="Drop a token matching >= this many non-.vn Tranco domains")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = build(rows, args.min_len, load_tranco_names(args.tranco), args.tranco_max)
    result = {"generated_from": args.inp,
              "generated_at": datetime.now(timezone.utc).isoformat(), **result}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[+] {len(result['tokens'])} new tokens "
          f"({len(result['covered_by_static'])} already covered by VN_TOKENS; "
          f"{len(result['dropped_foreign'])} foreign-org, "
          f"{len(result['dropped_global'])} globally ambiguous dropped) -> {args.out}")


if __name__ == "__main__":
    main()
