#!/usr/bin/env python3
"""
telegram_sms_bot.py — Telegram bot to collect scam SMS/content that users FORWARD (with consent).

Only standard library + requests (long-polling getUpdates). No python-telegram-bot needed.
Ethical workflow:
  * /start shows info + asks for CONSENT (press /dongy). Only store data from users who agreed.
  * User pastes/forwards scam message content -> bot stores it (PII-REDACTED), user_id is HASHED/anonymized.
  * /rutlui to withdraw consent (stop collecting).

CONFIG:  export TELEGRAM_BOT_TOKEN=123:ABC   (create the bot via @BotFather)
RUN:     python telegram_sms_bot.py --out data/raw/reports/tg_sms.csv
"""
from __future__ import annotations
import argparse, csv, hashlib, os, re, time
import requests

PHONE = re.compile(r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)")
EMAIL = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
ACCT = re.compile(r"(?<!\d)\d{9,16}(?!\d)")
URL = re.compile(r"https?://|\b[a-z0-9-]+\.(?:com|net|org|top|xyz|cc|vn|info|online|shop|vip|icu|click|buzz)\b", re.I)

CONSENT_MSG = (
    "Xin chào! Đây là bot THU THẬP MẪU tin nhắn lừa đảo phục vụ NGHIÊN CỨU.\n"
    "• Dữ liệu chỉ dùng cho nghiên cứu, được ẩn danh (không lưu danh tính bạn).\n"
    "• Vui lòng CHỈ gửi tin nhắn lừa đảo (không gửi thông tin cá nhân của bạn).\n\n"
    "Gõ /dongy để đồng ý tham gia, /rutlui để rút lui bất kỳ lúc nào."
)


def redact(t: str) -> str:
    if not t:
        return ""
    t = EMAIL.sub("<EMAIL>", t)
    t = PHONE.sub("<PHONE>", t)
    t = ACCT.sub("<NUM>", t)
    return t.strip()


def anon(uid: int) -> str:
    return hashlib.sha256(f"tg{uid}".encode()).hexdigest()[:16]


class Store:
    def __init__(self, out, consent_file):
        self.out, self.cf = out, consent_file
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        self.consented = set()
        if os.path.exists(consent_file):
            with open(consent_file, encoding="utf-8") as f:
                self.consented = {l.strip() for l in f if l.strip()}
        if not os.path.exists(out):
            with open(out, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["channel", "label", "source", "user_hash", "ts", "text", "has_url", "msg_len"])

    def set_consent(self, uid, on=True):
        h = anon(uid)
        if on:
            self.consented.add(h)
        else:
            self.consented.discard(h)
        with open(self.cf, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(self.consented)))

    def has_consent(self, uid):
        return anon(uid) in self.consented

    def save(self, uid, text, ts):
        t = redact(text)
        with open(self.out, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["sms", "phishing", "telegram", anon(uid), ts, t,
                 1 if URL.search(t) else 0, len(t)])


def api(token, method, **params):
    try:
        return requests.get(f"https://api.telegram.org/bot{token}/{method}",
                            params=params, timeout=60).json()
    except requests.RequestException:
        return {}


def send(token, chat, text):
    api(token, "sendMessage", chat_id=chat, text=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/reports/tg_sms.csv")
    ap.add_argument("--consent-file", default="data/private/tg_consent.txt")
    args = ap.parse_args()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN (create the bot via @BotFather).")
    os.makedirs(os.path.dirname(args.consent_file) or ".", exist_ok=True)
    st = Store(args.out, args.consent_file)

    print("Bot is running (Ctrl+C to stop)...")
    offset = None
    while True:
        upd = api(token, "getUpdates", timeout=30, offset=offset)
        for u in upd.get("result", []):
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("channel_post")
            if not msg:
                continue
            chat = msg["chat"]["id"]
            uid = msg.get("from", {}).get("id", chat)
            text = msg.get("text", "") or msg.get("caption", "")
            low = text.strip().lower()
            if low in ("/start", "/help"):
                send(token, chat, CONSENT_MSG)
            elif low == "/dongy":
                st.set_consent(uid, True)
                send(token, chat, "Cảm ơn! Bạn có thể forward/dán tin nhắn lừa đảo để đóng góp.")
            elif low == "/rutlui":
                st.set_consent(uid, False)
                send(token, chat, "Đã ghi nhận rút lui. Bot sẽ không lưu dữ liệu của bạn nữa.")
            elif text:
                if st.has_consent(uid):
                    st.save(uid, text, msg.get("date", ""))
                    send(token, chat, "Đã nhận, xin cảm ơn! (dữ liệu đã ẩn danh)")
                else:
                    send(token, chat, "Bạn cần gõ /dongy để đồng ý tham gia trước khi gửi.")
        time.sleep(1)


if __name__ == "__main__":
    main()
