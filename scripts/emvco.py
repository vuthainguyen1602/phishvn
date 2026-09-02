#!/usr/bin/env python3
"""
emvco.py — read the EMVCo merchant-presented QR payload that VietQR uses.

WHY THIS EXISTS. A quishing code in Vietnam is often not a URL at all: it is a payment string, and
the fraud is that the account it names is not the one the victim thinks they are paying. Everything
else in this repository reads URLs, so such a payload arrived at the collector as `non_url` and
stopped there -- recorded, but not read. This reads it.

WHAT IT IS. EMVCo MPM is nested TLV: two ASCII digits of tag, two of length, then that many
characters of value, repeated. Tags 26-51 are payment-network templates, and VietQR (NAPAS) lives
in one of them, itself TLV, carrying the acquirer BIN and the account number. Tag 63 is a CRC-16
over everything preceding it, which is what lets a parser say the string is well-formed rather than
merely plausible.

WHAT IT DELIBERATELY DOES NOT DO. It does not resolve a BIN to a bank name from a bundled table:
those tables go stale, and a wrong bank name in a fraud report is worse than none. It returns the
BIN and leaves the lookup to whoever has a current list.
"""
from __future__ import annotations

# Tags that carry a payment-network template. 38 is where NAPAS/VietQR sits in practice, but the
# spec allocates the whole 26-51 range and issuers do move, so the range is walked rather than one
# tag being assumed.
TEMPLATE_TAGS = {f"{i:02d}" for i in range(26, 52)}
NAPAS_GUID_HINTS = ("A000000727", "napas", "VIETQR")


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE, the checksum EMVCo tag 63 carries."""
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def parse_tlv(s: str) -> dict:
    """Flat TLV at one level. Malformed input returns what was read before the break rather than
    raising: a truncated payload is a real thing a camera produces, and half a reading is still
    evidence."""
    out, i = {}, 0
    while i + 4 <= len(s):
        tag, ln = s[i:i + 2], s[i + 2:i + 4]
        if not (tag.isdigit() and ln.isdigit()):
            break
        n = int(ln)
        val = s[i + 4:i + 4 + n]
        if len(val) < n:
            break
        out[tag] = val
        i += 4 + n
    return out


def is_emvco(payload: str) -> bool:
    """Tag 00 is the payload-format indicator and is '01' for every EMVCo MPM code."""
    t = parse_tlv(payload)
    return t.get("00") == "01"


def parse(payload: str) -> dict:
    """Returns what could be read. `crc_ok` is the honest signal: a payload that parses but fails
    its own checksum is either corrupt or hand-made, and both matter to a fraud analysis."""
    out = {"is_emvco": False, "crc_ok": False, "bin": "", "account": "",
           "amount": "", "currency": "", "country": "", "merchant": "", "network": ""}
    if not payload or not is_emvco(payload):
        return out
    out["is_emvco"] = True
    top = parse_tlv(payload)

    # CRC covers everything up to and including the "6304" header of the checksum field itself.
    idx = payload.rfind("6304")
    if idx != -1 and len(payload) >= idx + 8:
        want = payload[idx + 4:idx + 8].upper()
        got = f"{crc16_ccitt(payload[:idx + 4].encode('ascii', 'replace')):04X}"
        out["crc_ok"] = (want == got)

    out["currency"] = top.get("53", "")
    out["amount"] = top.get("54", "")
    out["country"] = top.get("58", "")
    out["merchant"] = top.get("59", "")

    for tag in sorted(TEMPLATE_TAGS):
        if tag not in top:
            continue
        inner = parse_tlv(top[tag])
        guid = inner.get("00", "")
        if any(h.lower() in guid.lower() for h in NAPAS_GUID_HINTS):
            out["network"] = guid
        # The beneficiary sits one level deeper again, in the template's own sub-template.
        for sub in inner.values():
            deep = parse_tlv(sub)
            if deep.get("00", "").isdigit() and len(deep.get("00", "")) == 6:
                out["bin"] = deep["00"]          # 6-digit acquirer BIN
                out["account"] = deep.get("01", "")
        if out["bin"]:
            break
    return out
