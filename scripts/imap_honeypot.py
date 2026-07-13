#!/usr/bin/env python3
"""
imap_honeypot.py — Pull emails from a honeypot / catch-all mailbox into .eml files (then use parse_eml.py).

Idea: register a domain + enable a catch-all mailbox, seed addresses to "lure" phishing; this script
logs into IMAP and downloads messages to data/raw/reports/*.eml. You OWN the mailbox, so legal risk is low.

INSTALL: only standard library (imaplib, email).
CONFIG (env or arguments):
  export IMAP_HOST=imap.tenmien.com  IMAP_USER=catchall@tenmien.com  IMAP_PASS=****
RUN:
  python imap_honeypot.py --out-dir data/raw/reports --unseen-only --mark-seen
  # then:
  python parse_eml.py --dir data/raw/reports --out data/raw/reports/emails.csv --label phishing
"""
from __future__ import annotations
import argparse, imaplib, os, email
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("IMAP_HOST", ""))
    ap.add_argument("--user", default=os.environ.get("IMAP_USER", ""))
    ap.add_argument("--password", default=os.environ.get("IMAP_PASS", ""))
    ap.add_argument("--folder", default="INBOX")
    ap.add_argument("--out-dir", default="data/raw/reports")
    ap.add_argument("--unseen-only", action="store_true", help="Only unread messages")
    ap.add_argument("--mark-seen", action="store_true", help="Mark as read after downloading")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    if not (args.host and args.user and args.password):
        raise SystemExit("Missing IMAP_HOST/IMAP_USER/IMAP_PASS (env or --host/--user/--password).")

    os.makedirs(args.out_dir, exist_ok=True)
    M = imaplib.IMAP4_SSL(args.host)
    M.login(args.user, args.password)
    M.select(args.folder)

    crit = "(UNSEEN)" if args.unseen_only else "ALL"
    typ, data = M.search(None, crit)
    ids = data[0].split()
    if args.limit:
        ids = ids[-args.limit:]

    n = 0
    for i in ids:
        typ, msg_data = M.fetch(i, "(RFC822)")
        if typ != "OK" or not msg_data or not msg_data[0]:
            continue
        raw = msg_data[0][1]
        uid = i.decode()
        fname = os.path.join(args.out_dir, f"hp_{datetime.now():%Y%m%d}_{uid}.eml")
        with open(fname, "wb") as f:
            f.write(raw)
        if args.mark_seen:
            M.store(i, "+FLAGS", "\\Seen")
        n += 1
    M.logout()
    print(f"Downloaded {n} emails -> {args.out_dir}")


if __name__ == "__main__":
    main()
