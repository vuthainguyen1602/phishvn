#!/usr/bin/env python3
"""
p3_login_control_batch2.py — second batch of the genuine-login control (R3.2), by the rule of
amendment 1 in papers/P3_multimodal/REVISION2/LOGIN_CONTROL_PLAN.md: per organisation domain, the
most recent existing urlscan.io scan of a login/account URL on a customer-facing host, HTTP 200.

RUN:  URLSCAN_API_KEY=... python scripts/p3_login_control_batch2.py
OUT:  data/raw/login_control/captures_batch2.csv (+ dom/<uuid>.html)
"""
import csv
import os
import re
import time
from urllib.parse import urlparse

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LC = os.path.join(ROOT, "data", "raw", "login_control")
H = {"API-Key": os.environ.get("URLSCAN_API_KEY", ""),
     "User-Agent": "research (contact: thaivn_ph@utc.edu.vn)"}
PATH = re.compile(r"login|dang-nhap|dangnhap|signin|sign-in|ebank|ibank|account", re.I)
INTERNAL = re.compile(r"dev|test|uat|staging|api|itsm|webmail|mail|mkt|crm|vpn|admin", re.I)


def main():
    batch1 = {(urlparse(r["url"]).hostname or "").removeprefix("www.")
              for r in csv.DictReader(open(os.path.join(LC, "targets.csv"), encoding="utf-8"))}
    out = open(os.path.join(LC, "captures_batch2.csv"), "w", newline="", encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=["domain", "org", "category", "scan_uuid", "scan_time",
                                        "title", "dom_file", "found"])
    w.writeheader()
    for org in csv.DictReader(open(os.path.join(LC, "orgs_batch2.csv"), encoding="utf-8")):
        d = org["domain"]
        q = (f"(page.domain:{d} OR page.domain:*.{d}) AND page.status:200 AND (page.url:*login* "
             f"OR page.url:*dang-nhap* OR page.url:*dangnhap* OR page.url:*signin* OR "
             f"page.url:*sign-in* OR page.url:*ebank* OR page.url:*ibank* OR page.url:*account*)")
        try:
            res = requests.get("https://urlscan.io/api/v1/search/", headers=H, timeout=30,
                               params={"q": q, "size": 100}).json().get("results", [])
        except (requests.RequestException, ValueError):
            res = []
        pick = None
        for r in sorted(res, key=lambda r: r["task"]["time"], reverse=True):
            u = r["page"]["url"]
            host = (urlparse(u).hostname or "").removeprefix("www.")
            if not (host == d or host.endswith("." + d)) or INTERNAL.search(host):
                continue
            if host in batch1 or not PATH.search(urlparse(u).path + "?" + urlparse(u).query
                                                  + host):
                continue
            pick = r
            break
        row = {"org": d, "category": org["category"], "found": 0}
        if pick:
            uuid = pick["_id"]
            try:
                dom = requests.get(f"https://urlscan.io/dom/{uuid}/", headers=H, timeout=30)
            except requests.RequestException:
                dom = None
            if dom is not None and dom.status_code == 200 and dom.text:
                f = os.path.join("data", "raw", "login_control", "dom", f"{uuid}.html")
                open(os.path.join(ROOT, f), "w", encoding="utf-8").write(dom.text)
                row.update({"domain": pick["page"]["url"], "scan_uuid": uuid,
                            "scan_time": pick["task"]["time"],
                            "title": pick["page"].get("title", ""), "dom_file": f, "found": 1})
        w.writerow(row); out.flush()
        print(d, "->", row.get("domain", "none"), flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
