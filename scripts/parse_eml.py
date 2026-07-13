#!/usr/bin/env python3
"""
parse_eml.py — Parse .eml emails (user-reported / honeypot) into CSV following the email schema.

Extracts: subject, body (text, PII-REDACTED), URLs in the body, has_attachment, sender domain (type).
Does not store raw PII; phone numbers/emails/account numbers are replaced with tokens.

RUN:  python parse_eml.py --dir data/raw/reports --out data/raw/reports/emails.csv --label phishing
"""
from __future__ import annotations
import argparse, csv, glob, os, re
from email import policy
from email.parser import BytesParser

PHONE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
ACCT = re.compile(r"(?<!\d)\d{9,16}(?!\d)")
URL = re.compile(r"https?://[^\s\"'<>]+", re.I)


def redact(t: str) -> str:
    if not t:
        return ""
    t = EMAIL.sub("<EMAIL>", t)
    t = PHONE.sub("<PHONE>", t)
    t = ACCT.sub("<NUM>", t)
    return t.strip()


def body_text(msg) -> str:
    """Get the text/plain part; if only HTML is available, strip the tags."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                try:
                    return part.get_content()
                except Exception:
                    pass
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    return re.sub(r"<[^>]+>", " ", part.get_content())
                except Exception:
                    pass
        return ""
    else:
        try:
            c = msg.get_content()
        except Exception:
            return ""
        return re.sub(r"<[^>]+>", " ", c) if msg.get_content_type() == "text/html" else c


def has_attach(msg) -> int:
    if not msg.is_multipart():
        return 0
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")).lower():
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing .eml files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="phishing", choices=["phishing", "benign"])
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.dir, "**", "*.eml"), recursive=True)
    fields = ["source_file", "channel", "label", "source", "subject", "text",
              "sender_domain", "num_urls", "has_url", "has_attachment", "text_len"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for path in files:
            with open(path, "rb") as fp:
                msg = BytesParser(policy=policy.default).parse(fp)
            subject = redact(str(msg.get("subject", "")))
            frm = str(msg.get("from", ""))
            dom = ""
            m = EMAIL.search(frm)
            if m:
                dom = m.group(0).split("@")[-1].lower()
            raw_body = body_text(msg) or ""
            urls = URL.findall(raw_body)
            text = redact(raw_body)
            w.writerow({
                "source_file": os.path.basename(path),
                "channel": "email", "label": args.label, "source": "report",
                "subject": subject, "text": text, "sender_domain": dom,
                "num_urls": len(urls), "has_url": 1 if urls else 0,
                "has_attachment": has_attach(msg), "text_len": len(text),
            })
    print(f"Parsed {len(files)} emails -> {args.out}")


if __name__ == "__main__":
    main()
