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
  * a year or date of birth after "sinh năm", "sinh ngày" or "SN"            -> [BIRTH]
  * a home address after "thường trú", "tạm trú", "cư trú", "trú tại", "ngụ tại", "HKTT"
                                              -> [ADDRESS]
The last two were added 2026-11-10: police and court news name a suspect with a birth year and a
home address, and the two together identify the person even after the name is masked.
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


# "sinh năm 1974", "sinh ngày 12/05/1990", "SN 1988", "SN1976". Years 1920-2019 only, and not after
# "tuyển" or "học": "tuyển sinh năm 2018" is an admission year.
BIRTH_RE = re.compile(r"(?i)(?<![\w])(?<!tuyển )(?<!học )(sinh năm|sinh ngày|sn)([ .:]{0,2})"
                      r"(?:\d{1,2}[/-]\d{1,2}[/-])?(?:19[2-9]\d|20[01]\d)(?!\d)")
# A residence phrase followed by the start of an address. "nơi cư trú đối với ..." has no address
# after it and is left alone.
_PLACE_HEAD = (r"(?:(?:số|tổ|ấp|thôn|xóm|khu|khóm|xã|phường|thị trấn|thị xã|huyện|quận|tỉnh|thành phố|"
               r"tp\.?|đường|bản|buôn|làng)(?![\w])|\d)")
RESIDENCE_RE = re.compile(r"(?i)(?<![\w])((?:hộ khẩu thường trú|thường trú|tạm trú|cư trú|trú tại|ngụ tại|"
                          r"hktt|(?<=[(,)] )trú|(?<=[(,)] )ngụ|(?<=\()trú|(?<=\()ngụ)(?:\s+(?:tại|ở))?\s*:?\s*)"
                          r"(?=" + _PLACE_HEAD + r")")
# "trú" and "ngụ" alone count only after a bracket or a comma ("(SN 1979, trú tổ ..."): "nội trú"
# is a boarding school.
# the address ends at a sentence mark, a bracket, a mask token, or the verb that resumes the sentence
_ADDRESS_END = re.compile(r"[.;()\[\]\n]|,?\s+(?:để|bị|về|có|đã|đang|là|cùng|và|do|thì|khi|sau|vì|"
                          r"nhằm|với|hiện|đến|ra|tới|theo|trong|trên|nhưng|nên|được)(?![\w])")
# lower-case words that belong to an address; any other lower-case word ends it
_ADDRESS_WORDS = set("""số tổ ấp thôn xóm khu khóm xã phường thị trấn huyện quận tỉnh thành phố tp tp.
đường bản buôn làng dân đặc ngõ hẻm ngách nhà khối kp kp. tdp""".split())


def _address_end(body: str) -> int:
    stop = _ADDRESS_END.search(body)
    body = body[:stop.start()] if stop else body
    end = 0
    for m in re.finditer(r"\S+", body):
        w = m.group().strip(",")
        if w and w[0].islower() and w.lower() not in _ADDRESS_WORDS:
            break
        end = m.end()
    return len(body[:end].rstrip(", "))


def _mask_residence(text: str) -> str:
    out, last = [], 0
    for m in RESIDENCE_RE.finditer(text):
        if m.end() < last:
            continue
        n = _address_end(text[m.end():m.end() + 160])
        if n < 3 or not re.search(r"[^\W\d_]{2}", text[m.end():m.end() + n]):
            continue                                  # a blank form line, not an address
        out.append(text[last:m.end()])
        out.append("[ADDRESS]")
        last = m.end() + n
    out.append(text[last:])
    return "".join(out)


def mask_pii(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = LONGNUM_RE.sub("[NUMBER]", text)
    text = BIRTH_RE.sub(lambda m: m.group(1) + (m.group(2) or " ") + "[BIRTH]", text)
    text = _mask_residence(text)
    return mask_names(text)


if __name__ == "__main__":
    for s in ["Liên hệ 0324561411 hoặc lienhe@aramex.vn", "CSKH: 1900.63.66.97, hotline +84 912 345 678",
              "STK 1234567890123 Vietcombank", "Năm 2026, giá 150.000đ, mã 12345"]:
        print(mask_pii(s))
