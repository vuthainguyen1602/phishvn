#!/usr/bin/env python3
"""
gen_llm_adversarial.py — Generate SIMULATED Vietnamese phishing SAMPLES with an LLM (for P2 + the P1 adversarial set).

DEFENSIVE PURPOSE: create data to (a) measure the LLM threat (P2) and (b) train/evaluate a detector
that is ROBUST to LLM-generated content (P1). Each sample is tagged with 'gen_model' for leave-one-LLM-out evaluation.

GUARDRAILS (ethics):
  * Research use only; DO NOT distribute the samples externally.
  * The prompt requires SIMULATED LINKS (http://sim.example.vn/...), NO real active brands/URLs,
    and NO targeting of specific individuals.
  * Comply with the LLM provider's terms.

API: OpenAI-compatible (/chat/completions). Run with OpenAI, or local (Ollama/vLLM) via LLM_API_BASE.
  export LLM_API_KEY=...   export LLM_API_BASE=https://api.openai.com/v1   # or http://localhost:11434/v1
RUN:
  python gen_llm_adversarial.py --models gpt-4o-mini --channels sms email \
     --scenarios bank gov tax ecommerce delivery --per 20 --out data/raw/llm/llm_adv.csv
"""
from __future__ import annotations
import argparse, csv, os, time
import requests

BASE = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")

SYS = (
    "Bạn là công cụ tạo MẪU MÔ PHỎNG phục vụ nghiên cứu xây dựng bộ phát hiện phishing (mục đích "
    "PHÒNG THỦ). Hãy tạo các tin nhắn lừa đảo MẪU bằng tiếng Việt, sát thực tế Việt Nam, để huấn "
    "luyện mô hình nhận diện. RÀNG BUỘC: chỉ dùng liên kết GIẢ LẬP dạng http://sim.example.vn/xxx; "
    "KHÔNG dùng thương hiệu/đường link thật đang hoạt động; KHÔNG nhắm tới cá nhân có thật; KHÔNG "
    "kèm hướng dẫn thực hiện tấn công. Chỉ trả về NỘI DUNG tin nhắn, mỗi mẫu một dòng, không đánh số."
)

PROMPT = ("Tạo {n} tin nhắn {channel} lừa đảo MẪU theo kịch bản '{scenario}' (bối cảnh Việt Nam), "
          "đa dạng văn phong, có yếu tố hối thúc/thuyết phục khác nhau. Mỗi mẫu một dòng.")


def chat(model, prompt):
    r = requests.post(f"{BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                      json={"model": model, "temperature": 0.9,
                            "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": prompt}]},
                      timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Model names (multiple for leave-one-LLM-out)")
    ap.add_argument("--channels", nargs="+", default=["sms", "email"])
    ap.add_argument("--scenarios", nargs="+", default=["bank", "gov", "tax", "ecommerce", "delivery"])
    ap.add_argument("--per", type=int, default=20, help="Number of samples per (model,channel,scenario)")
    ap.add_argument("--out", default="data/raw/llm/llm_adv.csv")
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()
    if not KEY:
        raise SystemExit("Missing LLM_API_KEY.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["channel", "label", "source", "is_llm", "scenario", "gen_model", "text"]
    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for model in args.models:
            for ch in args.channels:
                for sc in args.scenarios:
                    try:
                        txt = chat(model, PROMPT.format(n=args.per, channel=ch, scenario=sc))
                    except Exception as e:
                        print(f"[!] {model}/{ch}/{sc}: {e}"); continue
                    for line in txt.splitlines():
                        line = line.strip(" -•\t")
                        if len(line) < 8:
                            continue
                        w.writerow({"channel": ch, "label": "phishing", "source": "llm",
                                    "is_llm": 1, "scenario": sc, "gen_model": model, "text": line})
                        n += 1
                    f.flush()
                    print(f"[+] {model}/{ch}/{sc} (total {n})")
                    time.sleep(args.delay)
    print(f"Done: {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
