#!/usr/bin/env python3
"""
vn_filter.py — Shared Vietnamese-targeting detection for phishing domains and page content.

Two signals, used across the collection scripts (watch_chongluadao.py, fetch_phishing_feeds.py,
build_content_manifest.py):
  * is_vn_target(domain): the domain is .vn OR its NAME contains a Vietnamese unaccented token
    (vietcombank247, 247-napas, giaohang..., chinhphu, dichvucong, ...). Catches VN phishing on
    international TLDs, which the .vn filter alone misses.
  * is_vietnamese_text(text): the rendered page text carries Vietnamese diacritics above a small
    density threshold — used after crawling to confirm the content is actually Vietnamese.

is_vn_target() additionally matches brand tokens generated from the Tin Nhiem Mang
trusted-org registry (data/processed/brand_tokens.json, built by build_brand_tokens.py)
when that file exists; VN_TOKENS stays as the hand-curated base.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

BRAND_TOKENS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "brand_tokens.json"

VN_TOKENS = re.compile(
    r"(vietcom|vcb|techcom|\btcb\b|agribank|sacombank|mbbank|vpbank|tpbank|bidv|\bacb\b|\bocb\b|"
    r"hdbank|shinhan|eximbank|napas|momo|zalopay|vnpay|viettelpay|nganhang|taikhoan|thanhtoan|"
    r"chuyentien|nhantien|naptien|ruttien|vaytien|tietkiem|tindung|chinhphu|congan|\bthue\b|"
    r"thuedientu|\bgdt\b|"  # e-tax portal thuedientu.gdt.gov.vn — top impersonation target; \bthue\b alone misses the fused form
    r"baohiem|bhxh|bhyt|dichvucong|vneid|cccd|canhcuoc|shopee|lazada|sendo|tiktokshop|muahang|"
    r"khuyenmai|trungthuong|nhanqua|tichdiem|giaohang|vanchuyen|buukien|donhang|ghtk|\bghn\b|"
    r"vietnampost|viettel|vinaphone|mobifone|\bvnpt\b|napthe|muathe|thecao|khachhang|dangnhap|"
    r"dangky|xacminh|xacnhan|capnhat|kichhoat|baomat|the-visa|thevisa|luadao|vietnam|\bvn-|-vn\b)",
    re.I)

VN_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")


def load_brand_regex(path: Path = BRAND_TOKENS_PATH) -> re.Pattern | None:
    """Compile the registry-generated brand tokens; None if the file is absent/invalid."""
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))["tokens"]
    except (OSError, ValueError, KeyError):
        return None
    parts = [re.escape(e["token"]) if e["mode"] == "substring"
             else rf"\b{re.escape(e['token'])}\b" for e in entries if e.get("token")]
    return re.compile("(" + "|".join(parts) + ")", re.I) if parts else None


BRAND_TOKENS = load_brand_regex()


def host_of(url: str) -> str:
    return re.sub(r"^https?://", "", (url or "").strip()).split("/")[0].split(":")[0].lower().strip(".")


def is_vn_target(domain: str) -> bool:
    d = (domain or "").lower()
    return (d.endswith(".vn") or bool(VN_TOKENS.search(d))
            or bool(BRAND_TOKENS and BRAND_TOKENS.search(d)))


def is_vietnamese_text(text: str, threshold: float = 0.008) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    return sum(1 for c in low if c in VN_CHARS) / len(low) > threshold
