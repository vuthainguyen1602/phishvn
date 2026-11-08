#!/usr/bin/env python3
"""
pii_mask.py — Mask personal identifiers in extracted page text before any model or release sees it.

Three kinds of identifier are replaced by a fixed token, in this order:
  * e-mail addresses                          -> [EMAIL]
  * Vietnamese phone numbers (mobile, landline, 1800/1900 hotlines, with or without +84 and
    the usual space/dot/dash grouping)       -> [PHONE]
  * any other standalone run of 9-19 digits (bank account, card, citizen-ID, tax numbers)
                                              -> [NUMBER]
  * Vietnamese personal names (name_mask: surname + title-case syllables, places, streets and
    historical figures kept)                  -> [NAME]
Names are masked last, so a name inside an e-mail address is already gone. The name rule
over-masks some title-case headings and brands, and misses foreign names written without an
honorific (see name_mask.py). Added 2026-11-08 after the first two public packages shipped names.

The function is deterministic and idempotent (masking masked text changes nothing), so a text
masked here equals the text the release builder masks from the same HTML.
"""
from __future__ import annotations
import re

try:
    from name_mask import mask_names
except ImportError:  # imported as a package module
    from .name_mask import mask_names  # type: ignore

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# +84 / 84 / 0 prefix, then 9-10 more digits optionally grouped by space, dot or dash; or a
# 1800/1900 hotline of 4-6 trailing digits.
PHONE_RE = re.compile(r"(?<![\w])(?:(?:\+?84|0)[ .-]?[1-9](?:[ .-]?\d){7,9}|1[89]00[ .-]?\d(?:[ .-]?\d){3,5})(?![\w])")
LONGNUM_RE = re.compile(r"(?<![\w])\d{9,19}(?![\w])")


def mask_pii(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = LONGNUM_RE.sub("[NUMBER]", text)
    return mask_names(text)


if __name__ == "__main__":
    for s in ["Liên hệ 0324561411 hoặc lienhe@aramex.vn", "CSKH: 1900.63.66.97, hotline +84 912 345 678",
              "STK 1234567890123 Vietcombank", "Năm 2026, giá 150.000đ, mã 12345"]:
        print(mask_pii(s))
