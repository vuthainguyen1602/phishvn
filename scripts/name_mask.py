#!/usr/bin/env python3
"""
name_mask.py — Mask Vietnamese personal names in extracted page text.

A name is a run of 2-4 title-case syllables separated by single spaces whose first syllable is a
Vietnamese surname. The run is replaced by [NAME] unless the context says it is a place:

  * the run (or its first two syllables) is in PLACES (provinces, cities, districts, regions);
  * the word before it is an address or institution cue (đường, phố, quận, trường, số ...) or a
    house number, because Vietnamese streets, schools and wards carry historical persons' names.

A run that starts with a WEAK surname (also a common word or place element: Hà, Cao, Thái, Châu,
Lâm, Mai, Tô, Đoàn, ...) is masked only when it has three or more syllables or a middle name
Văn/Thị. Any title-case run of 1-4 syllables after a person honorific (ông, bà, anh, chị, Mr, Ms,
Mrs) is masked whatever its first syllable, which covers names without a Vietnamese surname.

The rule over-masks some brands named after a person (better for privacy than the reverse) and
leaves foreign names without an honorific unmasked. Deterministic and idempotent.
"""
from __future__ import annotations
import re
import unicodedata

STRONG = set("""Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Trịnh Trương
Đinh Lương Nguyen Tran Pham Hoang Huynh Dang Bui Ngo Duong Truong Dinh Luong Trinh""".split())
WEAK = set("""Hà Cao Thái Châu Lâm Mai Tô Đoàn Lại Tạ Kiều Tăng Lưu Quách Vương Đào Hứa Mạc
Triệu Từ Thân Doãn Khúc Phùng Âu Tôn Thạch Lò Lục
Chu Cấn Quàng Khuất Nông Hầu Giáp Đồng Cà Lã Ninh Phí Ông Bế""".split())
MIDDLE = {"Văn", "Thị", "Van", "Thi"}

# Places whose name begins with a surname-like syllable. Matched on the first two syllables.
PLACES = set("""Hà Nội|Hồ Chí|Lâm Đồng|Hà Tĩnh|Cao Bằng|Thái Nguyên|Thái Bình|Thái Lan|Phan Thiết|Lý Sơn|
Châu Đốc|Hà Giang|Hà Đông|Cao Lãnh|Hà Nam|Châu Âu|Châu Á|Châu Phi|Châu Mỹ|Châu Úc|Châu Phú|Châu Thành|
Văn Giang|Dương Kinh|Lê Chân|Ngô Quyền|Hoàng Mai|Hoàng Kiếm|Phan Rang|Phan Rí|Hồ Tây|Hồ Gươm|Hà Tiên|
Hà Trung|Hà Quảng|Hà Hòa|Mai Châu|Mai Sơn|Thái Thụy|Thái Hòa|Lâm Hà|Lâm Thao|Cao Phong|Văn Lâm|Văn Yên|
Văn Chấn|Văn Quan|Văn Lãng|Văn Bàn|Đông Anh|Vũng Tàu|Tô Châu|Lý Nhân|Phan Thiết|Bùi Viện|Đỗ Sơn|
Hồ Tràm|Hồ Núi|Lương Sơn|Lương Tài|Trần Văn Thời|Châu Đức|Hoàng Su|Hoàng Hóa|Hà Lan|Hà Quang|
Mai Dịch|Dương Nội|Cao Xà|Đào Duy|Văn Miếu|Văn Điển""".replace("\n", "").split("|"))
# Historical figures whose names are also the commonest street, school and ward names. A public
# historical person is not personal data, and masking them would erase most addresses.
HISTORIC = set("""Trần Hưng Đạo|Lê Lợi|Nguyễn Trãi|Lê Quý Đôn|Phan Đăng Lưu|Lý Tự Trọng|Võ Thị Sáu|
Nguyễn Du|Nguyễn Huệ|Hoàng Văn Thụ|Phạm Văn Đồng|Trần Phú|Lê Hồng Phong|Hồ Chí Minh|Nguyễn Văn Linh|
Võ Văn Kiệt|Phan Bội Châu|Phan Chu Trinh|Lê Duẩn|Nguyễn Thị Minh Khai|Lý Thường Kiệt|Trần Nhân Tông|
Lê Thánh Tông|Nguyễn Văn Cừ|Nguyễn Đình Chiểu|Phan Đình Phùng|Lê Văn Sỹ|Nguyễn Văn Trỗi|Trần Quốc Toản|
Hoàng Hoa Thám|Nguyễn Chí Thanh|Lê Duẩn|Trường Chinh|Lý Thái Tổ|Ngô Quyền|Đinh Tiên Hoàng|Lê Đại Hành|
Trần Quang Khải|Nguyễn Thái Học|Phạm Ngũ Lão|Phạm Hùng|Lê Thánh Tôn|Hai Bà Trưng|Nguyễn Tất Thành|
Võ Nguyên Giáp|Tôn Đức Thắng|Nguyễn Khuyến|Lê Văn Lương|Nguyễn Văn Huyên|Trần Duy Hưng|Lê Văn Việt|
Nguyễn Hữu Thọ|Nguyễn Lương Bằng|Huỳnh Thúc Kháng|Phan Huy Chú|Lê Trọng Tấn|Nguyễn Xiển|Trần Khát Chân|
Lê Thanh Nghị|Đào Duy Từ|Đào Duy Anh|Nguyễn Bỉnh Khiêm|Chu Văn An|Lương Thế Vinh|Nguyễn Hiền|
Trần Đại Nghĩa|Lý Chính Thắng|Võ Thị Sáu|Nguyễn Thượng Hiền|Hồ Tùng Mậu|Trần Cao Vân|Phan Xích Long|
Lê Lai|Bùi Thị Xuân|Đoàn Thị Điểm|Lê Thị Riêng|Lê Thị Hồng Gấm|Nhữ Văn Lan|Chu Văn An|Trần Văn Thời|Phan Văn Trị|Hà Huy Tập|Ngô Gia Tự|Nguyễn Công Trứ|Nguyễn Tri Phương|Nguyễn Đức Cảnh|Nguyễn Văn Tố|Lê Văn Lương|Nguyễn Thị Định|Nguyễn Văn Bé|Hoàng Diệu|Trần Bình Trọng|Lê Văn Tám|Nguyễn Sinh Sắc""".replace("\n", "").split("|"))
# Provinces, major cities and words that only occur in place names. A run holding any of these
# (compared without diacritics, so drop-down lists typed unaccented count too) is a place list.
PROVINCES = """An Giang|Bà Rịa|Bắc Giang|Bắc Kạn|Bạc Liêu|Bắc Ninh|Bến Tre|Bình Định|Bình Dương|Bình Phước|
Bình Thuận|Cà Mau|Cần Thơ|Cao Bằng|Đà Nẵng|Đắk Lắk|Đắk Nông|Điện Biên|Đồng Nai|Đồng Tháp|Gia Lai|
Hà Giang|Hà Nam|Hà Nội|Hà Tĩnh|Hải Dương|Hải Phòng|Hậu Giang|Hòa Bình|Hưng Yên|Khánh Hòa|Kiên Giang|
Kon Tum|Lai Châu|Lâm Đồng|Lạng Sơn|Lào Cai|Long An|Nam Định|Nghệ An|Ninh Bình|Ninh Thuận|Phú Thọ|
Phú Yên|Quảng Bình|Quảng Nam|Quảng Ngãi|Quảng Ninh|Quảng Trị|Sóc Trăng|Sơn La|Tây Ninh|Thái Bình|
Thái Nguyên|Thanh Hóa|Thừa Thiên|Tiền Giang|Trà Vinh|Tuyên Quang|Vĩnh Long|Vĩnh Phúc|Yên Bái|Huế|
Biên Hòa|Nha Trang|Đà Lạt|Vũng Tàu|Phan Thiết|Quy Nhơn|Buôn Ma|Pleiku|Cần Giờ|Cao Lãnh|Thủ Đức|
Hoàng Sa|Trường Sa|Phú Quốc|Côn Đảo|Hạ Long|Sài Gòn|Gò Vấp|Bình Thạnh|Tân Bình|Hóc Môn|Củ Chi"""
# compared WITH diacritics: unaccented "phuong" is also the given name Phương, "huyen" Huyền
PLACE_WORDS = {"Xã", "Phường", "Quận", "Huyện", "Tỉnh", "Thôn", "Ấp", "Thị Xã", "Thị Trấn"}
TITLES = {"Lương Y", "Văn Hóa", "Hà Huy Tập", "Lương Y Như"}
# Cue words: the name that follows is a street, school, ward, bridge, market or similar.
# Administrative words (tỉnh, xã, huyện, quận, phường) are NOT cues: on government pages they
# precede an official's name ("Chủ tịch HĐND tỉnh Nguyễn Văn Sơn") more often than a place.
CUES = set("""đường phố ngõ hẻm kiệt ngách số đ. Đ. tp tp. thành phố q. p.
cầu chợ trường thpt thcs th mầm non tiểu học đại lộ khu kđt chung cư tòa bến cảng lăng tượng
công viên bảo tàng giải quỹ ấp thôn tổ thị trấn street st. st road rd.""".split())
HONORIFIC = {"ông", "bà", "anh", "chị", "cô", "chú", "mr", "mr.", "ms", "ms.", "mrs", "mrs.",
             "bapak", "ibu", "pak"}  # last three: the Indonesian-language feed pages

WORD_RE = re.compile(r"[^\W\d_]+")
TOKEN = "[NAME]"


def _plain(w: str) -> str:
    w = w.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn").lower()


_PROV = {_plain(x.strip()) for x in PROVINCES.replace("\n", "").split("|")} | \
        {_plain(x) for x in PLACES}


def _looks_like_place_list(words: list[str]) -> bool:
    p = [_plain(w) for w in words]
    if any(w in PLACE_WORDS for w in words[1:]):
        return True
    hits = [k for k in range(len(p) - 1) if " ".join(p[k:k + 2]) in _PROV]
    # a place at the head of the run, or two places in one run (a drop-down list); one place
    # pair later in a run is usually a given name ("Nguyễn Thanh Hoa" is not Thanh Hóa)
    if 0 in hits or len(hits) >= 2:
        return True
    return " ".join(words[:2]) in TITLES or " ".join(words[:3]) in TITLES


def _title(w: str) -> bool:
    return w[:1].isupper() and (len(w) == 1 or w[1:].islower())


def _prev_words(text: str, start: int, n: int = 2) -> list[str]:
    return text[max(0, start - 40):start].lower().split()[-n:]


def _keep_as_place(words: list[str], text: str, start: int) -> bool:
    if " ".join(words[:2]) in PLACES or " ".join(words) in PLACES:
        return True
    if _looks_like_place_list(words):
        return True
    if any(" ".join(words[:k]) in HISTORIC for k in range(2, len(words) + 1)):
        return True
    prev = _prev_words(text, start)
    if not prev:
        return False
    if prev[-1] in CUES or (len(prev) == 2 and " ".join(prev) in CUES):
        return True
    # house number (12, 12a, 464/60/8, 12-14), but not a date such as 19/06/2025
    w = prev[-1].rstrip(",")
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", w):
        return False
    return bool(re.fullmatch(r"\d{1,4}[a-z]?(?:[/-]\d{1,4}[a-z]?)*", w))


def _name_spans(text: str) -> list[tuple[int, int]]:
    words = list(WORD_RE.finditer(text))
    spans, i = [], 0
    while i < len(words):
        w = words[i].group()
        if (w in STRONG or w in WEAK) and not text[max(0, words[i].start() - 1):words[i].start()] == "[":
            tail = [words[i]]
            j = i + 1
            # extend over title-case syllables joined by exactly one space, up to four in all
            while (j < len(words) and len(tail) < 4 and _title(words[j].group())
                   and text[tail[-1].end():words[j].start()] == " " and len(words[j].group()) <= 7):
                tail.append(words[j]); j += 1
            names = [t.group() for t in tail]
            ok = len(tail) >= 2 and not (w in WEAK and len(tail) < 3 and names[1] not in MIDDLE)
            if ok and not _keep_as_place(names, text, tail[0].start()):
                spans.append((tail[0].start(), tail[-1].end()))
                i = j
                continue
            # a place or a historical name: move one word on, so a person's name that follows
            # the place in the same title-case run ("Thái Lan Mai Thị Huyền Trâm") is still read
        i += 1
    return spans


def _mask_honorific(text: str) -> str:
    def rep(m: re.Match) -> str:
        return m.group(1) + TOKEN
    hon = "|".join(re.escape(h) for h in sorted(HONORIFIC, key=len, reverse=True))
    pat = re.compile(rf"(?i:(?<![\w])((?:{hon}) ))((?-i:[^\W\d_a-zà-ỹ][^\W\d_]{{0,6}}(?: [^\W\d_a-zà-ỹ][^\W\d_]{{0,6}}){{0,3}}))(?![\w])")
    out, last = [], 0
    for m in pat.finditer(text):
        words = m.group(2).split(" ")
        if not all(_title(w) for w in words):
            continue
        out.append(text[last:m.start()])
        out.append(rep(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def mask_names(text: str) -> str:
    if not text:
        return text
    text = _mask_honorific(text)
    out, last = [], 0
    for a, b in _name_spans(text):
        out.append(text[last:a]); out.append(TOKEN); last = b
    out.append(text[last:])
    return "".join(out)


if __name__ == "__main__":
    for s in ["Liên hệ anh Nguyễn Văn Nam để nhận quà", "Số 12 Lê Lợi, Quận 1, TP. Hồ Chí Minh",
              "Trường THPT Lê Quý Đôn, Hà Nội", "Giám đốc: Phạm Thị Thanh Trà", "Tổng Bí thư Tô Lâm",
              "Đoàn Thanh niên phường", "Mr John Smith", "Chị Hoa đã nhận 5 triệu"]:
        print(mask_names(s))
