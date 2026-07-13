#!/usr/bin/env python3
"""
sms_import.py — Normalize collected SMS (SIM honeypot / Android export) into the SMS schema (PII-redacted).

Supports 2 input sources:
  1) --gammu-inbox DIR : gammu-smsd inbox folder (FILES backend), file names of the form
     IN<YYYYMMDD>_<HHMMSS>_<seq>_<sender>_...txt ; file content = message body.
  2) --csv FILE        : CSV exported from an Android app (SMS Backup & Restore, etc.). Specify columns.

RUN:
  python sms_import.py --gammu-inbox /var/spool/gammu/inbox --label phishing --out data/raw/reports/sms.csv
  python sms_import.py --csv sms_backup.csv --text-col body --sender-col address --date-col date \
         --label phishing --out data/raw/reports/sms.csv
"""
from __future__ import annotations
import argparse, csv, glob, os, re

PHONE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
ACCT = re.compile(r"(?<!\d)\d{9,16}(?!\d)")
URL = re.compile(r"https?://|\b[a-z0-9-]+\.(?:com|net|org|top|xyz|cc|vn|info|online|shop|vip|icu|click|buzz)\b", re.I)


def redact(t: str) -> str:
    if not t:
        return ""
    t = EMAIL.sub("<EMAIL>", t)
    t = PHONE.sub("<PHONE>", t)
    t = ACCT.sub("<NUM>", t)
    return t.strip()


def row_of(text, sender, date, label):
    text_r = redact(str(text))
    return {
        "channel": "sms", "label": label, "source": "honeypot",
        "brandname": str(sender or "")[:40], "collected_at": str(date or ""),
        "text": text_r, "has_url": 1 if URL.search(text_r) else 0, "msg_len": len(text_r),
    }


def from_gammu(dir_):
    rows = []
    for p in sorted(glob.glob(os.path.join(dir_, "IN*"))):
        base = os.path.basename(p)
        parts = base.split("_")
        sender = parts[3] if len(parts) > 3 else ""
        date = parts[0][2:] if parts and parts[0].startswith("IN") else ""
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                body = f.read()
        except OSError:
            continue
        rows.append(row_of(body, sender, date, ARGS_LABEL))
    return rows


def from_csv(path, tcol, scol, dcol):
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            rows.append(row_of(r.get(tcol, ""), r.get(scol, ""), r.get(dcol, ""), ARGS_LABEL))
    return rows


ARGS_LABEL = "phishing"


def main():
    global ARGS_LABEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammu-inbox")
    ap.add_argument("--csv")
    ap.add_argument("--text-col", default="body")
    ap.add_argument("--sender-col", default="address")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--label", default="phishing", choices=["phishing", "benign"])
    ap.add_argument("--out", default="data/raw/reports/sms.csv")
    args = ap.parse_args()
    ARGS_LABEL = args.label

    rows = []
    if args.gammu_inbox:
        rows += from_gammu(args.gammu_inbox)
    if args.csv:
        rows += from_csv(args.csv, args.text_col, args.sender_col, args.date_col)
    if not rows:
        raise SystemExit("No data. Need --gammu-inbox or --csv.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["channel", "label", "source", "brandname", "collected_at", "text", "has_url", "msg_len"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} SMS -> {args.out} (PII redacted)")


if __name__ == "__main__":
    main()
