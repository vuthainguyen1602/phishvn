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

Since 2026-11-08 (v1.3 of the content package, after an audit of the published v1.2 found names
that had escaped): forms of address and academic or professional titles (đồng chí, PGS.TS.,
giảng viên, kỹ thuật viên ...) count as honorifics, and the name after one stops at the office
that follows it; a Western given name before a Vietnamese surname (Emily Hoang) is masked; a
two-syllable historical name followed by a further syllable is a person; a bare number before a
run is a house number only at the start of an address, never a year; and two-word cues
("công viên", "tiểu học") are matched as phrases, not as their single words.

Since 2026-11-10 (v1.4, after a scan of the published v1.3 found suspects named in police news):
"tượng" is a monument cue except in "đối tượng", which had kept every name after it as a place;
and the words that introduce a person in crime reporting (đối tượng, bị can, bị cáo, nghi phạm,
nạn nhân ...) count as honorifics, so a name after one is masked whatever its first syllable.

The rule over-masks some brands named after a person (better for privacy than the reverse) and
leaves foreign names without an honorific or a Vietnamese surname unmasked. Deterministic and
idempotent.
"""
from __future__ import annotations
import re
import unicodedata

STRONG = set("""Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Trịnh Trương
Đinh Lương Nguyen Tran Pham Hoang Huynh Dang Bui Ngo Duong Truong Dinh Luong Trinh""".split())
WEAK = set("""Hà Cao Thái Châu Lâm Mai Tô Đoàn Lại Tạ Kiều Tăng Lưu Quách Vương Đào Hứa Mạc
Triệu Từ Thân Doãn Khúc Phùng Âu Tôn Thạch Lò Lục
Chu Cấn Quàng Khuất Nông Hầu Giáp Đồng Cà Lã Ninh Phí Ông Bế Phù Kha Diệp Thiều""".split())
MIDDLE = {"Văn", "Thị", "Van", "Thi"}

# Places whose name begins with a surname-like syllable. Matched on the first two syllables.
PLACES = set("""Hà Nội|Hồ Chí|Lâm Đồng|Hà Tĩnh|Cao Bằng|Thái Nguyên|Thái Bình|Thái Lan|Phan Thiết|Lý Sơn|
Châu Đốc|Hà Giang|Hà Đông|Cao Lãnh|Hà Nam|Châu Âu|Châu Á|Châu Phi|Châu Mỹ|Châu Úc|Châu Phú|Châu Thành|
Văn Giang|Dương Kinh|Lê Chân|Ngô Quyền|Hoàng Mai|Hoàng Kiếm|Phan Rang|Phan Rí|Hồ Tây|Hồ Gươm|Hà Tiên|
Hà Trung|Hà Quảng|Hà Hòa|Mai Châu|Mai Sơn|Thái Thụy|Thái Hòa|Lâm Hà|Lâm Thao|Cao Phong|Văn Lâm|Văn Yên|
Văn Chấn|Văn Quan|Văn Lãng|Văn Bàn|Đông Anh|Vũng Tàu|Tô Châu|Lý Nhân|Phan Thiết|Bùi Viện|
Hồ Tràm|Hồ Núi|Lương Sơn|Lương Tài|Trần Văn Thời|Châu Đức|Hoàng Su|Hoàng Hóa|Hà Lan|Hà Quang|
Mai Dịch|Dương Nội|Cao Xà|Đào Duy|Văn Miếu|Văn Điển|Phù Đổng|Phù Ninh|Phù Cát|Phù Mỹ|Phù Yên|Phù Cừ|Phù Lưu""".replace("\n", "").split("|"))
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
TITLES = {"Lương Y", "Văn Hóa", "Hà Huy Tập", "Lương Y Như", "Ông Già", "Ông Táo", "Bà Rịa", "Bà Triệu",
          # Title Case headings and common words that begin with a surname-like syllable
          "Lý Do", "Lý Thuyết", "Lý Lịch", "Phù Hợp", "Tăng Cường", "Tăng Trưởng", "Từ Vựng", "Từ Điển",
          "Cao Ốc", "Cao Cấp", "Cao Điểm", "Đoàn Kết", "Đoàn Viên", "Mai Táng", "Lưu Ý", "Lưu Trữ",
          "Dương Lịch", "Đào Tạo", "Hà Lan", "Thái Độ", "Lâm Nghiệp", "Lâm Sàng", "Châu Báu"}
# the word before a surname-like syllable that turns it into an ordinary compound
PREV_COMPOUND = {("đại", "Lý"), ("thành", "Đoàn"), ("công", "Đoàn"), ("trấn", "Dương"), ("đại", "Dương"),
                 ("thái", "Bình"), ("ấn", "Độ"), ("quản", "Lý"), ("xử", "Lý"), ("vật", "Lý"), ("địa", "Lý"),
                 ("tâm", "Lý"), ("trợ", "Lý"), ("hợp", "Lý"), ("chân", "Lý"), ("nguyên", "Lý"), ("bảo", "Vũ")}
# Cue words: the name that follows is a street, school, ward, bridge, market or similar.
# Administrative words (tỉnh, xã, huyện, quận, phường) are NOT cues: on government pages they
# precede an official's name ("Chủ tịch HĐND tỉnh Nguyễn Văn Sơn") more often than a place.
CUES = set("""đường phố ngõ hẻm kiệt ngách số đ. Đ. tp tp. q. p.
cầu chợ trường thpt thcs th khu kđt tòa bến cảng lăng tượng giải quỹ ấp thôn tổ
street st. st road rd.""".split())
# Two-word cues are matched as phrases only: split into single words, "viên" (công viên) and "học"
# (tiểu học) made every "kỹ thuật viên Võ Văn Tài" and "giảng viên ..." read as an address.
CUE_PHRASES = {"thành phố", "mầm non", "tiểu học", "đại lộ", "chung cư", "công viên", "bảo tàng",
               "thị trấn", "trung học", "phổ thông", "cơ sở", "đại học"}
# a cue word that is the tail of another word: "đối tượng" (the suspect) is not "tượng" (a statue)
NOT_CUE_PHRASES = {"đối tượng"}
# words that introduce a person in crime and court reporting; two or more syllables must follow
CRIME_ROLES = {"đối tượng", "bị can", "bị cáo", "nghi phạm", "nghi can", "bị hại", "nạn nhân",
               "phạm nhân", "bị án", "tử tù"}
HONORIFIC = CRIME_ROLES | {"ông", "bà", "anh", "chị", "cô", "chú", "mr", "mr.", "ms", "ms.", "mrs", "mrs.",
             "bapak", "ibu", "pak",  # the Indonesian-language feed pages
             # forms of address and academic or professional titles written before a name
             "đồng chí", "đồng chí:", "các đồng chí:", "đ/c", "thầy", "gs.", "pgs.", "ts.", "ths.", "bs.", "ks.", "ls.", "cn.",
             "gs.ts.", "pgs.ts.", "gs.tskh.", "ths.bs.", "bsckii.", "bscki.", "ncs.",
             "tiến sĩ", "thạc sĩ", "bác sĩ", "luật sư", "dược sĩ", "kiến trúc sư",
             "kỹ thuật viên", "giảng viên", "giáo viên", "tác giả", "chủ biên:", "tác giả:",
             "đại tá", "thượng tá", "trung tá", "thiếu tá", "đại úy", "thượng úy", "trung úy", "thiếu úy",
             "đại tướng", "thượng tướng", "trung tướng", "thiếu tướng"}
# after these the name must begin with a surname: Title Case headings follow them as well
# ("Giáo viên Có Học Sinh Giỏi", "Giáo viên Tiếng Anh")
NEEDS_SURNAME = {"tiến sĩ", "thạc sĩ", "bác sĩ", "luật sư", "dược sĩ", "kiến trúc sư", "kỹ thuật viên",
                 "giảng viên", "giáo viên", "tác giả", "chủ biên:", "tác giả:", "ts.", "ths.", "bs.",
                 "ks.", "ls.", "cn.", "ncs."}
# a name after an honorific stops at these: they begin the title or office that follows it
# (not Thành, Trung, Công, Tiến, Khoa, Đại, Trường: those are common given names as well)
NAME_STOP = {"Phó", "Bí", "Chủ", "Tỉnh", "Giám", "Trưởng", "Ủy", "Uỷ", "Tổng", "Kỹ", "Giảng",
             "Phường", "Xã", "Huyện", "Quận", "Sở", "Ban", "Viện", "Hội", "Đảng", "Bộ",
             "Cục", "Vụ", "Tòa", "Nhà", "Giáo", "Thạc", "Thường", "Cô", "Thầy", "Thuốc", "Ý", "Môn",
             "Quyền", "Hiệu", "Bs", "Ths", "Ts"}
# the first word of a rank or office written after "đồng chí": the name, if any, follows the rank
RANK_HEAD = {"Đại", "Thượng", "Trung", "Thiếu"}

WORD_RE = re.compile(r"[^\W\d_]+")
TOKEN = "[NAME]"


def _plain(w: str) -> str:
    w = w.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn").lower()


_PROV = {_plain(x.strip()) for x in PROVINCES.replace("\n", "").split("|")} | \
        {_plain(x) for x in PLACES}


# with diacritics, for the honorific rule: unaccented, "Lâm Đông" (a name) is Lâm Đồng (a province)
_PROV_ACCENTED = {x.strip() for x in PROVINCES.replace("\n", "").split("|")} | PLACES | {"Việt Nam"}


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
    # a historic name is kept when it is the whole run, or when it has three or more syllables
    # ("Hoàng Văn Thụ Ban"); a two-syllable one followed by another syllable is a person
    # ("Nguyễn Hiền Phương" is not the scholar Nguyễn Hiền)
    if " ".join(words) in HISTORIC or any(" ".join(words[:k]) in HISTORIC for k in range(3, len(words) + 1)):
        return True
    prev = _prev_words(text, start)
    if not prev:
        return False
    if " ".join(prev) in NOT_CUE_PHRASES:
        pass                                          # "đối tượng X" is a person, not a statue
    elif prev[-1] in CUES or (len(prev) == 2 and " ".join(prev) in CUE_PHRASES):
        return True
    # house number (12, 12a, 464/60/8, 12-14), but not a date such as 19/06/2025, not a year
    # ("năm 1945, Nguyễn Lan Dung") and not a bare number after an ordinary word ("Win 11
    # Nguyễn Đức Lợi"). A bare number counts only at the start of an address: after "số", "địa
    # chỉ", a colon or the start of the text, or when the run is followed by a ward or district.
    w = prev[-1].rstrip(",")
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?", w):
        return False
    if re.fullmatch(r"\d{1,4}[a-z]?(?:[/-]\d{1,4}[a-z]?)+|\d{1,4}[a-z]", w):
        return True                                   # 464/60/8, 12a: only addresses look like this
    if not re.fullmatch(r"\d{1,4}", w) or re.fullmatch(r"(19|20)\d\d", w):
        return False
    before = prev[-2] if len(prev) == 2 else ""
    if before in ("số", "chỉ:", "chỉ", "sở", "address:", "add:", "đc:", "") or before.endswith(":") or before in CUES:
        return True
    after = text[start:start + 90]
    if re.match(r"(?i)[^\n,]{0,40},?\s*(p\.|q\.|phường|quận|huyện|xã|tp\.?|thành phố|ward|district)", after):
        return True
    # an address list: two more comma- or dash-separated place names after the street
    return bool(re.match(r"[^\n,–-]{0,40}(?:,|\s[-–])\s*[A-ZÀ-ỸĐ][^\n,–-]{1,30}(?:,|\s[-–])\s*[A-ZÀ-ỸĐ]", after))


def _name_spans(text: str) -> list[tuple[int, int]]:
    words = list(WORD_RE.finditer(text))
    spans, i = [], 0
    while i < len(words):
        w = words[i].group()
        prev_w = words[i - 1].group().lower() if i else ""
        if (w in STRONG or w in WEAK) and (prev_w, w) not in PREV_COMPOUND \
                and not text[max(0, words[i].start() - 1):words[i].start()] == "[":
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


# honorifics after which a single given name is still a person ("chị Hoa", "anh Nam")
_SHORT_OK = {"ông", "bà", "anh", "chị", "cô", "chú", "mr", "mr.", "ms", "ms.", "mrs", "mrs.", "bapak",
             "ibu", "pak", "thầy", "đồng chí", "đồng chí:", "các đồng chí:", "đ/c"}


# honorific + word sequences that are places or set phrases, not a person
# places and set phrases, kept whatever follows them ("Bà Chúa Xứ Núi Sam", "Bà Nà Hills")
HON_FIXED = {"Bà Rịa", "Ông Già", "Bà Chúa", "Bà Đen", "Bà Triệu", "Bà Trưng", "Ông Táo", "Bà Nà",
             "Bà Huyện", "Bà Chiểu", "Bà Điểm", "Bà Hom", "Cô Tô", "Ông Lãnh", "Ông Ích", "Bà Ký",
             "Bà Quẹo", "Bà Hạt"}
# kept only when nothing name-like follows: "Anh Quốc" is England, "anh Quốc Bảo" a person
HON_FIXED_ALONE = {"Anh Quốc", "Anh Đào", "Cô Ba", "Chị Hằng"}


def _mask_honorific(text: str) -> str:
    def rep(m: re.Match) -> str:
        return m.group(1) + TOKEN
    hon = "|".join(re.escape(h) for h in sorted(HONORIFIC, key=len, reverse=True))
    pat = re.compile(rf"(?i:(?<![\w])((?:{hon}) ))([^\W\d_]{{1,7}}(?: [^\W\d_]{{1,7}}){{0,3}})(?![\w])")
    # case is checked word by word below (_title): a character class for "upper case" written as
    # [^...a-zà-ỹ] also excluded Đ (U+0110 lies inside à-ỹ), so no name starting with Đ was masked
    out, last, pos = [], 0, 0
    # searched by hand rather than with finditer: a match reads up to four words whatever their
    # case, and "ông Park Jin và bà Kim Narae" must resume after "Jin", not after "bà"
    while True:
        m = pat.search(text, pos)
        if not m:
            break
        pos = m.start(2)
        words = m.group(2).split(" ")
        keep = []
        for w in words:
            if w in NAME_STOP or not _title(w):
                break
            keep.append(w)
        if not keep or (len(keep) == 1 and m.group(1).strip().lower() not in _SHORT_OK):
            continue
        hon_word = m.group(1).strip()
        low = hon_word.lower()
        nxt = text[m.start(2) + len(keep[0]):m.start(2) + len(keep[0]) + 6].lower()
        if keep[0] in RANK_HEAD and re.match(r" (tá|tướng|úy|uý)\b", nxt):
            pos = m.start(2)                           # "đồng chí Đại tá X": the rank's own honorific reads X
            continue
        if keep[-1] == "Thành" and re.match(r" (ủy|uỷ|phố|viên)\b", text[m.start(2) + len(" ".join(keep)):][:7].lower()):
            keep = keep[:-1]                           # "Thành ủy viên", "Thành phố"
            if not keep:
                continue
        if low in NEEDS_SURNAME and keep[0] not in STRONG and keep[0] not in WEAK:
            continue
        if low in ("anh", "chị", "cô", "bà", "ông") and text[max(0, m.start() - 6):m.start()].lower().endswith("tiếng "):
            continue                                   # "Tiếng Anh", not the honorific anh
        if " ".join(keep) in HISTORIC:
            continue                                   # "đồng chí Phạm Văn Đồng", a historical figure
        pair = f"{hon_word[:1].upper()}{hon_word[1:]} {keep[0]}"
        if pair in HON_FIXED or (pair in HON_FIXED_ALONE and len(keep) == 1) or \
                _plain(f"{hon_word} {keep[0]}") in _PROV or " ".join(keep[:2]) in _PROV_ACCENTED:
            continue
        end = m.start(2) + len(" ".join(keep))
        out.append(text[last:m.start(2)])
        out.append(TOKEN)
        last = pos = end
    out.append(text[last:])
    return "".join(out)


_ROMAN = re.compile(r"(?<![\w\[])([A-Z][a-z]{2,11}) (" + "|".join(sorted(
    STRONG | {"Hoang", "Pham", "Nguyen", "Tran", "Le", "Vu", "Vo", "Do", "Ho", "Ly", "Dao", "Mai", "Cao"},
    key=len, reverse=True)) + r")(?![\w])")
_ROMAN_STOP = {"Saigon", "Hanoi", "Viet", "Vietnam", "Tower", "Group", "Hotel", "Plaza", "Street",
               "The", "Ben", "Cho", "Pho", "Bun", "Banh", "Com", "Cafe", "Coffee", "Spa", "Mall", "Center",
               "Park", "Bank", "Land", "Home", "House", "Holdings", "Bay"}


# one Vietnamese syllable written without diacritics: initial consonants, a vowel cluster and
# an allowed final. A given name that fits it ("Lan", "Long", "Hue") is Vietnamese text, not
# a Western given name, and "Lan Mai" or "Long Phan" stay as they are.
_VN_SYLLABLE = re.compile(r"(?i)(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
                          r"[aeiouy]{1,3}(?:ch|ng|nh|[cmnpt])?")


# common Western given names that Vietnamese people use with their surname (staff pages, authors,
# streamers). A list, not a shape: "Media Do", "Melamine Cao" and "Tour Phan" are not people.
WESTERN_GIVEN = set("""Emily Anna Anne John David Michael Peter Charlie Rosie Kevin Tony Jenny Linda Lisa
Mary James Robert William Daniel Henry Jack Kelly Lily Lucy Sophie Amy Helen Vivian Tina Jimmy Jason
Ryan Eric Alex Andy Steven Steve Thomas Richard Paul Mark Sarah Jessica Emma Olivia Grace Chloe Rachel
Hannah Jennifer Julia Julie Kate Katie Laura Leo Lena Lina Louis Luna Maria Mia Mike Nancy Nick Nina
Oliver Patrick Rose Ruby Sam Sandy Sean Selina Simon Sophia Stella Sunny Susan Ted Tom Tommy Victor
Vincent Vicky Wendy Winnie Yuki Ken Ben Bella Bill Bob Cindy Daisy Dan Dennis Diana Elena Ella Elsa
Eva Frank George Harry Hellen Ivy Jacky Jane Jasmine Jay Jenna Jerry Joe Joey Jolie Josh Joy Judy
Kai Kathy Kenny Kris Kyle Lana Lee Lilly Liz Lola Lynn Max Megan Mina Molly Nathan Noah Rita Robin
Roy Sabrina Sally Shawn Sky Sofia Stephen Teresa Tiffany Tracy Tyler Vanessa Vera Zoe Ivan Ian Ethan
Adam Aaron Ann Annie Ashley Brian Bryan Carol Cathy Chris Christina Christine""".split())


def _mask_roman(text: str) -> str:
    """A Western given name before a Vietnamese surname, as staff pages write them."""
    return _ROMAN.sub(lambda m: TOKEN if m.group(1) in WESTERN_GIVEN else m.group(0), text)


def mask_names(text: str) -> str:
    if not text:
        return text
    text = _mask_honorific(text)
    text = _mask_roman(text)
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
