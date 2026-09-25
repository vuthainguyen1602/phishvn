#!/usr/bin/env python3
"""Ingest the Vietnamese SMS corpus, audit it, and build a split that does not leak.

SOURCE. trannguyenthaituan/vietnamese_sms_dataset (CC-BY-4.0), full_dataset.csv: 2,991 messages
labelled 0/1 with a message_id and a date. The publisher also ships train.csv and test.csv; this
script does NOT use them, for a reason it measures rather than assumes -- 27 message texts appear
on both sides, touching 47 of the 597 test rows. Its own split is by message text, so a message
that appears twice cannot straddle the boundary.

WHAT IT DOES NOT DO. It does not fit anything, and it does not relabel. The labels are the
publisher's; where this script disagrees with the dataset card it says so in the snapshot rather
than correcting the data.

Reads   data/raw/sms_hf_full/full_dataset.csv
Writes  data/processed/sms/sms_messages.csv    one row per message, with URL and split columns
        data/processed/sms/sms_urls.csv        one row per extracted URL
        data/processed/sms/sms_snapshot.json   the counts the paper's macros read

RUN:  python3 scripts/sms_corpus_import.py
"""
from __future__ import annotations
import collections, csv, hashlib, json, os, re, sys, unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
from psl import registered_domain  # noqa: E402

SRC = os.path.join("data", "raw", "sms_hf_full", "full_dataset.csv")
PUB_TRAIN = os.path.join("data", "raw", "sms_hf_full", "train.csv")
PUB_TEST = os.path.join("data", "raw", "sms_hf_full", "test.csv")
OUTDIR = os.path.join("data", "processed", "sms")
MSG_OUT = os.path.join(OUTDIR, "sms_messages.csv")
URL_OUT = os.path.join(OUTDIR, "sms_urls.csv")
SNAP = os.path.join(OUTDIR, "sms_snapshot.json")

# A bare host is a URL in an SMS: "truy cap vietcombank.vn-gll.top" carries no scheme and is still
# the click target. Requiring https:// would have missed most of the phishing side.
CAND = re.compile(r"(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})(?:/[^\s,;)\]]*)?", re.I)
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "rb.gy", "shorturl.at", "cutt.ly",
              "is.gd", "ow.ly", "s.id", "zalo.me", "page.link", "me.qr", "link.vn",
              "shorturl.asia", "buff.ly", "rebrand.ly"}
PII_TOKEN = re.compile(r"\[[A-Z_]+\]")
RESIDUAL = {"phone_vn": re.compile(r"(?<![0-9])0[0-9]{8,10}(?![0-9])"),
            "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")}
HOLDOUT = 0.20
VI_MARKS = set("\u0300\u0301\u0303\u0309\u0323\u0302\u0306\u031b")


def has_vietnamese_diacritic(text: str) -> bool:
    """Detect the complete Vietnamese tone/shape inventory, including non-decomposing đ."""
    nfd = unicodedata.normalize("NFD", text or "")
    return "đ" in (text or "").lower() or any(ch in VI_MARKS for ch in nfd)


def sha(t: str) -> str:
    return hashlib.sha1((t or "").strip().encode("utf-8")).hexdigest()[:16]


def split_of(text_sha: str) -> str:
    """Keyed on the message text, so duplicates land together. Stable across runs."""
    h = int(hashlib.sha1(("sms-split:" + text_sha).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < HOLDOUT else "train"


def hosts_in(msg: str):
    out = []
    for m in CAND.finditer(msg):
        host = m.group(1).lower()
        apex = registered_domain(host)
        if apex and "." in apex:
            out.append((host, apex))
    return out


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"[!] {SRC} missing", file=sys.stderr)
        return 1
    rows = read(SRC)

    msgs, urls = [], []
    for r in rows:
        text, lab = r.get("message") or "", r.get("label", "")
        h = sha(text)
        hh = hosts_in(text)
        short = sum(1 for host, apex in hh
                    if apex in SHORTENERS or host.endswith(".page.link"))
        msgs.append({
            "message_id": r.get("message_id", ""), "date": r.get("date", ""),
            "text_sha1": h, "label": lab, "split": split_of(h),
            "chars": len(text), "accented": int(has_vietnamese_diacritic(text)),
            "pii_tokens": " ".join(sorted(set(PII_TOKEN.findall(text)))),
            "residual_pii": " ".join(sorted(k for k, rx in RESIDUAL.items() if rx.search(text))),
            "n_urls": len(hh), "n_shortened": short,
            "apexes": " ".join(sorted({a for _, a in hh})),
        })
        for host, apex in hh:
            urls.append({"message_id": r.get("message_id", ""), "label": lab,
                         "host": host, "apex": apex, "tld": apex.rsplit(".", 1)[-1],
                         "shortener": int(apex in SHORTENERS or host.endswith(".page.link"))})

    os.makedirs(OUTDIR, exist_ok=True)
    for path, data in ((MSG_OUT, msgs), (URL_OUT, urls)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    # --- the publisher's split, measured rather than trusted
    pub_leak_texts = pub_leak_rows = pub_test_rows = 0
    if os.path.isfile(PUB_TRAIN) and os.path.isfile(PUB_TEST):
        tr = collections.Counter(sha(x["message"]) for x in read(PUB_TRAIN))
        te = collections.Counter(sha(x["message"]) for x in read(PUB_TEST))
        both = set(tr) & set(te)
        pub_leak_texts, pub_test_rows = len(both), sum(te.values())
        pub_leak_rows = sum(te[k] for k in both)

    # --- our split must not leak at all
    by_split: dict = collections.defaultdict(set)
    for m in msgs:
        by_split[m["split"]].add(m["text_sha1"])
    own_leak = len(by_split["train"] & by_split["test"])

    def per_label(pred):
        return {lab: sum(1 for m in msgs if m["label"] == lab and pred(m)) for lab in ("0", "1")}

    n_lab = {lab: sum(1 for m in msgs if m["label"] == lab) for lab in ("0", "1")}
    url_rows = collections.defaultdict(list)
    for u in urls:
        url_rows[u["label"]].append(u)
    snap = {
        "source": "trannguyenthaituan/vietnamese_sms_dataset full_dataset.csv (CC-BY-4.0)",
        "messages": len(msgs), "unique_texts": len({m["text_sha1"] for m in msgs}),
        "exact_duplicates": len(msgs) - len({m["text_sha1"] for m in msgs}),
        "ham": n_lab["0"], "phishing": n_lab["1"],
        "with_url": per_label(lambda m: m["n_urls"] > 0),
        "urls": {k: len(v) for k, v in url_rows.items()},
        "unique_apexes": {k: len({u["apex"] for u in v}) for k, v in url_rows.items()},
        "shortened_urls": {k: sum(u["shortener"] for u in v) for k, v in url_rows.items()},
        "top_tlds": {k: collections.Counter(u["tld"] for u in v).most_common(6)
                     for k, v in url_rows.items()},
        "residual_pii_messages": sum(1 for m in msgs if m["residual_pii"]),
        "pii_token_kinds": len({t for m in msgs for t in m["pii_tokens"].split()}),
        "own_split": {s: len(v) for s, v in sorted(by_split.items())},
        "own_split_leak_texts": own_leak,
        "published_split_leak_texts": pub_leak_texts,
        "published_split_leak_rows": pub_leak_rows,
        "published_test_rows": pub_test_rows,
    }
    json.dump(snap, open(SNAP, "w", encoding="utf-8"), indent=2, ensure_ascii=False, sort_keys=True)

    # macros, so no collection figure in the paper is typed by a person
    def mc(name, val):
        # {,} not "," — every one of these is used inside math mode, where a bare comma is a
        # separator and TeX sets a space after it: 2,991 came out as "2, 991" on the page.
        return "\\newcommand{\\Sms%s}{%s}\n" % (name, str(val).replace(",", "{,}"))
    pct = lambda a, b: f"{100*a/b:.1f}" if b else "0.0"
    tex = (mc("Messages", f"{len(msgs):,}") + mc("Ham", f"{n_lab['0']:,}")
           + mc("Phish", f"{n_lab['1']:,}")
           + mc("UniqueTexts", f"{snap['unique_texts']:,}")
           + mc("Duplicates", f"{snap['exact_duplicates']:,}")
           + mc("HamWithUrl", f"{snap['with_url']['0']:,}")
           + mc("PhishWithUrl", f"{snap['with_url']['1']:,}")
           + mc("HamUrls", f"{snap['urls'].get('0', 0):,}")
           + mc("PhishUrls", f"{snap['urls'].get('1', 0):,}")
           + mc("HamApexes", f"{snap['unique_apexes'].get('0', 0):,}")
           + mc("PhishApexes", f"{snap['unique_apexes'].get('1', 0):,}")
           + mc("HamShort", f"{snap['shortened_urls'].get('0', 0):,}")
           + mc("PhishShort", f"{snap['shortened_urls'].get('1', 0):,}")
           + mc("HamShortPct", pct(snap['shortened_urls'].get('0', 0), snap['urls'].get('0', 1)))
           + mc("PhishShortPct", pct(snap['shortened_urls'].get('1', 0), snap['urls'].get('1', 1)))
           + mc("PubLeakRows", f"{pub_leak_rows:,}") + mc("PubTestRows", f"{pub_test_rows:,}")
           + mc("PubLeakTexts", f"{pub_leak_texts:,}")
           + mc("PubLeakPct", pct(pub_leak_rows, pub_test_rows or 1))
           + mc("OwnTrain", f"{snap['own_split'].get('train', 0):,}")
           + mc("OwnTest", f"{snap['own_split'].get('test', 0):,}")
           + mc("OwnLeak", f"{own_leak}")
           + mc("ResidualEmail", f"{snap['residual_pii_messages']}"))
    tex_out = os.path.join("papers", "future_smishing", "sections", "gen_corpus.tex")
    if os.path.isdir(os.path.dirname(tex_out)):
        with open(tex_out, "w", encoding="utf-8") as f:
            f.write("% generated by scripts/sms_corpus_import.py; do not edit\n" + tex)
        print(f"  [+] {tex_out}")

    print(f"  {len(msgs)} messages ({snap['ham']} ham / {snap['phishing']} phishing), "
          f"{snap['unique_texts']} unique texts, {snap['exact_duplicates']} exact duplicates")
    for lab, name in (("0", "ham"), ("1", "phishing")):
        print(f"  {name:<9} with URL {snap['with_url'][lab]:<5} urls {snap['urls'].get(lab,0):<5} "
              f"apexes {snap['unique_apexes'].get(lab,0):<4} shortened {snap['shortened_urls'].get(lab,0)}")
    print(f"  published split leaks {pub_leak_rows}/{pub_test_rows} test rows "
          f"({pub_leak_texts} texts); ours leaks {own_leak}")
    print(f"  residual phone/e-mail: {snap['residual_pii_messages']} message(s)")
    print(f"  [+] {MSG_OUT}\n  [+] {URL_OUT}\n  [+] {SNAP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
