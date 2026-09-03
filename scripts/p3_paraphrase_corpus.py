#!/usr/bin/env python3
"""
p3_paraphrase_corpus.py — the SEMANTIC (paraphrase) attack set for P3's evasion study.

A char-n-gram detector is near-constitutionally robust to character perturbation, so that result
cannot separate "understands the lure" from "attack too shallow". This supplies the attack that
can: an intent-preserving REWRITE of each simulated lure, two per source message, variant 'a' for
TRAINING augmentation only and 'b' for the TEST attack only. Protocol fixed in advance:
papers/P3_multimodal/protocols/PARAPHRASE_PROTOCOL.md. Guardrails (simulated non-resolving link,
generic sector references, no real person, gated tier) are CHECKED not asserted — see validate().

RUN:  python scripts/p3_paraphrase_corpus.py          # writes data/processed/p3/p3_paraphrase.csv
What is preserved vs changed, the orthography choice and the authorship bias:
kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
OUT = os.path.join(ROOT, "data", "processed", "p3", "p3_paraphrase.csv")
GEN_MODEL = "claude-fable-5"
L = "http://sim.example.vn/x"

# id -> (variant 'a' — training augmentation, variant 'b' — test attack)
PARA: dict[str, tuple[str, str]] = {
 # ---- sms / bank ----
 "ce5d3153-97de-5303-b957-e91894eaba18": (
  f"Chung toi vua ghi nhan mot luot truy cap la va da han che tam thoi cac chuc nang tren tai khoan cua ban. Hoan tat kiem tra danh tinh tai {L} trong 120 phut toi, neu khong quyen su dung se bi thu hoi.",
  f"Vi ly do an toan, mot phan dich vu tren tai khoan ban da bi dung sau lan dang nhap dang ngo hom nay. Tu xac nhan o {L} trong hai tieng nua de khong mat quyen vao tai khoan."),
 "fe08ab08-8fca-5826-b69f-bee7f9ddeaff": (
  f"Vua co lenh chuyen 12.500.000d tu tai khoan cua ban. Khong phai ban thuc hien? Chan lenh nay tai {L}",
  f"He thong vua ghi nhan mot khoan chi 12.500.000d. Neu ban khong nhan ra khoan nay, vao {L} de tu choi ngay."),
 "31982fc1-8e84-5084-bd03-1ee0bb0fa8df": (
  f"Mot lenh giao dich cua ban van dang treo va can duoc duyet. Vao {L} de xem lai; qua mot ngay ma chua xu ly thi tai khoan se bi dung.",
  f"Ban con mot yeu cau chuyen tien chua duoc chap thuan. Truy cap {L} de giai quyet, neu de qua 24 tieng he thong tu dong phong toa tai khoan."),
 "1ea6d560-2584-53cd-90d3-02f40deac79d": (
  f"Ho so tin dung cua ban vua duoc xep len muc cao hon. Mo goi lai suat 0% danh cho hang moi tai {L}, han dang ky la 18h chieu nay.",
  f"Ban da du dieu kien len hang tin dung. Nhan quyen loi lai suat 0 phan tram bang cach kich hoat o {L} truoc 6 gio toi."),
 "4c7cc155-210a-5f34-94de-c22a3810ec8f": (
  f"The cua ban da qua han su dung theo ghi nhan cua he thong. Khai bao lai thong tin the tai {L} de dich vu khong bi ngat.",
  f"Chung toi thay the ban dang dung khong con hieu luc. Bo sung du lieu the o {L} neu muon tiep tuc thanh toan binh thuong."),
 "640b37c2-5b9a-5009-9bd6-87de6680c839": (
  f"Luc 2 gio 14 sang co mot may khong quen dang nhap vao tai khoan ngan hang cua ban. Dung khan cap tai {L}",
  f"Ghi nhan truy cap tu thiet bi chua tung dung, thoi diem 02:14. Neu khong phai ban, chan tai khoan ngay o {L}"),
 "f9509629-2ecf-5824-a0f5-1e11d2c3aed2": (
  f"Ban thuoc nhom duoc tra lai 500.000d phi duy tri nam. Lam thu tuc nhan tai {L}, chuong trinh ket thuc trong ngay.",
  f"Khoan phi thuong nien 500.000d cua ban du dieu kien hoan lai. Dang ky nhan o {L}, chi con hieu luc hom nay."),
 "970cede4-d744-5c8c-9a2c-6bd930548efe": (
  f"So tiet kiem cua ban vua den han. Chon tai tuc tai {L} de tiep tuc huong muc lai cao.",
  f"Khoan gui cua ban da toi ngay dao han. Xac nhan gia han o {L} neu muon giu lai suat uu dai."),
 "2e59bd8c-3fd6-5f15-aa02-2df06f089a23": (
  f"Ma xac thuc cua ban sap khong con dung duoc. Vao {L} de lay lai quyen xac thuc, tranh bi gian doan.",
  f"He thong bao ma OTP gan het thoi han. Thiet lap lai tai {L} de dich vu khong bi ngung giua chung."),
 "c5fa9c5c-0a99-59e6-a558-8719d5630a3c": (
  f"Chung toi dang doi he thong an ninh nen moi tai khoan phai khai bao lai. Hoan tat dinh danh o {L} truoc 8 gio toi.",
  f"Dot nang cap bao mat yeu cau ban xac nhan lai thong tin chu tai khoan. Thuc hien tai {L}, han chot 20h hom nay."),
 # ---- sms / gov ----
 "08a11685-da13-5a0d-adae-e4bb38b1b26a": (
  f"Ho so ban nop con thieu muc bat buoc nen chua duoc tiep nhan. Nop bo sung tai {L} truoc 5 gio chieu de khong bi phat.",
  f"Phan khai bao hanh chinh cua ban chua day du. Hoan thien o {L} truoc 17h, qua han se bi xu ly theo quy dinh."),
 "a89d893b-7f88-5790-aaec-6529fd93f3bd": (
  f"Tai khoan dinh danh dien tu cua ban da cu, can khai lai. Lam tai {L} trong mot ngay.",
  f"Thong tin dinh danh dien tu cua ban chua dong bo voi du lieu moi. Cap nhat o {L} trong vong 24 tieng."),
 "327a9cf8-d6ac-5d9e-b9e2-c385114acc55": (
  f"Ban dang co mot loi phat nguoi chua thanh toan. Tra cuu va nop tai {L}",
  f"He thong ghi nhan bien ban phat giao thong cua ban con treo. Xem chi tiet va xu ly o {L}"),
 "3a202f76-0fb3-581e-b796-1716a81f8156": (
  f"Khoan tro cap bao hiem cua ban dang cho duyet. Bo sung thong tin nhan tien tai {L}",
  f"Ho so huong ho tro xa hoi cua ban con thieu du lieu tai khoan. Khai bao o {L} de duoc chi tra."),
 "7864ce37-8130-59f5-afc0-37268eaef60f": (
  f"Du lieu cu tru cua ban can duoc xac nhan lai. Vao {L} hoan tat, han la trong ngay.",
  f"Thong tin noi o cua cong dan phai duoc kiem tra lai theo dot ra soat. Thuc hien tai {L} truoc khi het ngay hom nay."),
 "6848f7f3-1392-5d87-92f5-8410c0e36904": (
  f"Giay to dien tu cua ban gan het hieu luc. Gia han tai {L} de con dung duoc dich vu cong.",
  f"Can cuoc dien tu cua ban sap toi ngay het han su dung. Lam thu tuc keo dai o {L}"),
 "35e804fd-c648-5b93-8c42-5f6d8aa8ee6c": (
  f"Ban co mot khoan hoan tra an sinh chua nhan. Dang ky tai {L} truoc khi qua han.",
  f"Danh sach hoan tien an sinh co ten ban. Khai thong tin nhan tai {L}, han nhan sap het."),
 "88596340-0709-5dcb-9fb8-bd7b6d17e66e": (
  f"Co quan quan ly can ban khai lai thong tin ho khau. Nop tai {L}",
  f"Du lieu ho khau cua ban chua khop voi he thong. Dieu chinh o {L}"),
 # ---- sms / tax ----
 "37317cbe-52c7-5510-a38d-8bee245104e5": (
  f"Ban duoc tra lai 3.200.000d tien thue. Khai tai khoan nhan tien tai {L}",
  f"Ket qua ra soat cho thay ban thua 3.200.000d thue. Cung cap so tai khoan o {L} de nhan lai."),
 "4d61f0ab-3de2-5bb6-8acb-78be1c18b43d": (
  f"Ho so quyet toan cua ban con thieu giay to. Nop them tai {L} truoc 5 gio chieu neu khong se bi truy thu.",
  f"Phan quyet toan thue cua ban chua du chung tu hop le. Bo sung o {L} truoc 17h de tranh bi thu hoi."),
 "fdaba926-bba8-57f2-945c-dc4e3f99667e": (
  f"Quy dinh moi yeu cau xac nhan lai ma so thue. Cap nhat tai {L}",
  f"Ma so thue cua ban can duoc kiem tra lai theo huong dan vua ban hanh. Thuc hien o {L}"),
 "326a4f2f-1e3c-5900-b035-604120bf68ce": (
  f"Ban dang con khoan thue chua nop. Doi soat va tra tai {L} de khong bi cuong che.",
  f"He thong ghi nhan no thue dung ten ban. Kiem tra va thanh toan o {L} truoc khi bi ap dung bien phap manh."),
 "fc8ba5cf-d122-5cf3-8ca3-660b93c2667f": (
  f"Ban thuoc dien duoc giam tru thue. Dang ky huong tai {L} truoc han.",
  f"Ho so cua ban dat dieu kien mien giam. Lam thu tuc o {L}, han dang ky sap het."),
 "279c4d10-73f2-59d7-9924-5ecdbd8dc5f7": (
  f"He thong thue vua doi phien ban. Dang nhap lai tai {L} de du lieu duoc dong bo.",
  f"Sau dot nang cap, tai khoan thue dien tu cua ban can vao lai mot lan tai {L} de khop du lieu."),
 # ---- sms / ecommerce ----
 "78ef9e77-6ad7-525c-b1b5-d388b219c9b3": (
  f"Don cua ban dang dung lai vi thieu thong tin nguoi nhan. Bo sung tai {L} de duoc giao tiep.",
  f"Kien hang cua ban chua di duoc do ho so don khong day du. Khai lai o {L}"),
 "88ed2f06-605d-5ceb-82ec-9bab6af2747e": (
  f"Ban trung phan thuong 2.000.000d trong dot tri an. Nhan tai {L} trong 3 tieng.",
  f"Chuong trinh tri an vua quay trung ten ban voi voucher 2 trieu dong. Lay ma o {L}, han 3 gio."),
 "225be388-1334-5d2e-a914-b06a6781d4a1": (
  f"Don ban huy du dieu kien duoc tra lai tien. Xac nhan tai khoan nhan tai {L}",
  f"San da duyet hoan tien cho don bi huy cua ban. Khai so nhan tien o {L}"),
 "738c771b-e410-5a16-bb27-f413a87b3a18": (
  f"Diem tich luy cua ban het han trong hom nay. Doi qua tai {L} truoc nua dem.",
  f"So diem thuong ban dang co sap bi xoa. Dung de doi qua o {L} truoc 23h59."),
 "9dff634c-cd96-5f92-8141-7b404e56071f": (
  f"Tai khoan mua sam cua ban vua bi dang nhap la. Khoa lai tai {L}",
  f"Co truy cap dang ngo vao tai khoan mua hang cua ban. Bao ve ngay o {L}"),
 "88713d72-7d28-519a-b2f4-5bc0492ff691": (
  f"Ban co ma giam 50% rieng trong dot flash sale. Kich hoat tai {L}",
  f"Uu dai giam mot nua danh cho tai khoan cua ban. Nhan ma o {L}"),
 "74282801-ec6d-522d-8bf4-8def1fb3bd8f": (
  f"Don A83920 giao khong thanh cong. Chon lich giao lai va xac nhan dia chi tai {L}",
  f"Chuyen giao don #A83920 that bai. Dat lai thoi gian nhan va kiem tra dia chi o {L}"),
 # ---- sms / delivery ----
 "706cad3a-8b3b-5cb5-9a5d-946583a3ede1": (
  f"Buu kien cua ban dang nam o kho vi dia chi khong dung. Sua lai tai {L} de nhan trong hom nay.",
  f"Hang cua ban chua giao duoc do thong tin dia chi sai. Chinh o {L} de duoc phat trong ngay."),
 "f0371896-9b9c-562b-bed0-13eb21658199": (
  f"Don cua ban phat sinh 15.000d tien luu kho. Nop tai {L} de khong bi tra hang.",
  f"Kien hang cua ban dang bi tinh phi kho 15.000d. Thanh toan o {L} truoc khi hang bi hoan."),
 "788752de-f1c9-56e4-9aa3-fe198780a21f": (
  f"Shipper goi ban nhieu lan khong duoc. Chot lai gio giao tai {L}",
  f"Nguoi giao hang khong ket noi duoc voi ban. Xac nhan lich nhan o {L}"),
 "cd6ef3d8-865f-53a8-9e00-71b91520be1b": (
  f"Hang gui tu nuoc ngoai cua ban dang bi giu o hai quan. Khai bo sung tai {L}",
  f"Kien quoc te dung ten ban chua thong quan. Hoan tat khai bao o {L}"),
 "3f773d86-bf82-5c1a-96ef-06e5f9ab3041": (
  f"Van don cua ban can xac nhan dung nguoi nhan. Vao {L} hoan tat.",
  f"He thong yeu cau kiem tra danh tinh nguoi nhan cho ma van don cua ban. Thuc hien tai {L}"),
 "ab59524e-40ca-55cd-a5b2-3fc1ee138691": (
  f"Ban co buu pham uu tien dang cho. Chon gio nhan va xac nhan tai {L}",
  f"Mot kien uu tien sap giao cho ban. Dat khung gio o {L}"),
 # ---- sms / telecom ----
 "4354f30e-28ba-5798-8ed3-d32a530798e2": (
  f"SIM cua ban chua chuan hoa nen sap bi khoa ca hai chieu. Cap nhat tai {L}",
  f"Thong tin thue bao cua ban chua dung quy dinh, so se bi ngat lien lac hai chieu. Khai lai o {L}"),
 "66a248ff-4a7e-5002-936e-23483dba95e0": (
  f"Ban duoc cong 50GB data mien phi. Kich hoat tai {L} trong mot ngay.",
  f"Goi 50GB tang them dang cho tai khoan cua ban. Nhan o {L} truoc 24 tieng."),
 "81298c97-c944-5a20-bf98-f7f32a06bac4": (
  f"Diem tich luy cua ban sap bi xoa. Doi qua tai {L}",
  f"So diem thuong tren thue bao cua ban het han den noi. Dung de doi qua o {L}"),
 "ccead8ff-12d3-56c5-b8ae-863746fc0061": (
  f"Thue bao cua ban con no 120.000d cuoc. Tra tai {L} de khong bi cat lien lac.",
  f"Ban dang co khoan cuoc 120.000d chua thanh toan. Nop o {L} truoc khi so bi khoa."),
 "b1b57c87-95a1-566a-aedf-75c70a487dfe": (
  f"Chuong trinh doi SIM 4G/5G khong mat phi dang mo. Dang ky tai {L}",
  f"Ban duoc nang doi SIM len 4G/5G mien phi. Ghi danh o {L}"),
 # ---- sms / social ----
 "e73849ec-4a87-5d56-b8ce-7dd2cddcdaf5": (
  f"Tai khoan mang xa hoi cua ban vua bi dang nhap tu may khac. Bao ve ngay tai {L}",
  f"Co phien dang nhap la vao trang ca nhan cua ban. Khoa lai o {L}"),
 "15d49bcd-586b-5ee1-863e-cce42a435628": (
  f"Trang cua ban bi danh gia vi pham quy tac cong dong va sap bi an. Gui khang nghi tai {L} trong 12 tieng.",
  f"Chung toi nhan bao cao ve noi dung tren trang cua ban; trang co the bi khoa. Phan hoi o {L} trong vong 12 gio."),
 "824c1a61-0bf0-5d54-af97-32c33bebc4ef": (
  f"Ban duoc bat tinh nang kiem tien tren nen tang. Dang ky tai {L}",
  f"Tai khoan cua ban da du dieu kien bat kiem tien. Kich hoat o {L}"),
 "fdddaa0f-a091-5dc3-bad7-1ad6abe4374c": (
  f"Ban vua bi nhac ten trong mot bai dang nhay cam. Xem tai {L}",
  f"Co tai khoan gan the ban vao noi dung khong phu hop. Kiem tra o {L}"),
 "2f700985-35b2-5dc3-a2f5-ca1d17aae857": (
  f"Hoan tat xac minh danh tinh de duoc cap huy hieu chinh chu tai {L}",
  f"De lay lai dau tich chinh chu, ban can xac thuc danh tinh o {L}"),
 # ---- email / bank ----
 "0ce69f65-b964-5df1-8918-7384033ad53b": (
  f"Thua Quy khach, he thong an ninh cua chung toi ghi nhan mot phien truy cap khong binh thuong vao tai khoan cua Quy khach. Kinh de nghi Quy khach hoan tat buoc xac thuc danh tinh trong 02 tieng tai {L}; qua khoang thoi gian nay mot so chuc nang se tam bi dung.",
  f"Kinh thua Quy khach, chung toi phat hien hoat dong dang nhap dang ngo tren tai khoan cua Quy khach. De bao dam an toan tai san, Quy khach vui long xac nhan danh tinh tai {L} trong vong hai gio, neu khong tai khoan se bi gioi han giao dich."),
 "689ed820-7304-5f87-b5f0-7556a41d8552": (
  f"Kinh gui Quy khach, he thong ghi nhan mot lenh chi 15.750.000 VND co dau hieu bat thuong. Neu day khong phai giao dich do Quy khach thuc hien, de nghi huy lenh va khai bao lai thong tin an toan tai {L}.",
  f"Thong bao an ninh: mot khoan giao dich tri gia 15.750.000 VND vua duoc khoi tao tu tai khoan cua Quy khach. Truong hop Quy khach khong nhan ra, vui long chan lenh nay va cap nhat bao mat tai {L}."),
 "cfe81eb9-41ae-5783-8a96-8dba8e145d65": (
  f"Kinh gui Quy khach, the tin dung Quy khach dang su dung sap den ngay het hieu luc. De dich vu khong bi gian doan, kinh de nghi Quy khach khai bao lai thong tin the tai {L} truoc thoi han.",
  f"Quy khach kinh men, thoi han su dung the tin dung cua Quy khach sap ket thuc. Vui long bo sung du lieu the tai {L} de tiep tuc giao dich binh thuong."),
 "bdb20e4a-16bc-5e6c-b8e2-4ef1d9358612": (
  f"Kinh gui Quy khach, tai khoan cua Quy khach nam trong dien duoc hoan phi thuong nien dot nay. De nghi Quy khach xac nhan thong tin nhan hoan tai {L} truoc 18 gio hom nay.",
  f"Thong bao: chinh sach hoan phi duy tri the dang ap dung cho tai khoan cua Quy khach. Vui long khai bao tai khoan nhan tien tai {L}, han chot 18h00 trong ngay."),
 "a99f8fa3-333e-53eb-82b0-773364f8172b": (
  f"Kinh gui Quy khach, he thong ngan hang dang trien khai chuan an ninh moi. Kinh de nghi Quy khach thuc hien lai buoc dinh danh tai khoan tai {L} de tiep tuc su dung kenh truc tuyen.",
  f"Thong bao: chung toi dang cap nhat tieu chuan bao mat cho toan bo khach hang. Quy khach vui long xac thuc lai thong tin chu tai khoan tai {L} de khong bi gian doan dich vu ngan hang so."),
 # ---- email / gov ----
 "2dcda735-4ae8-56d3-aee0-7957af80a277": (
  f"Kinh gui Cong dan, qua ra soat he thong nhan thay ho so dich vu cong cua Quy vi con thieu mot so noi dung bat buoc. Kinh de nghi Quy vi bo sung tai {L} truoc 17 gio de ho so duoc thu ly.",
  f"Thua Quy vi, ho so truc tuyen Quy vi da nop chua dat yeu cau ve thanh phan giay to. Vui long hoan thien tai {L} truoc 17h00, qua han ho so se bi tra lai."),
 "b36ad21c-e6b7-5d5e-ba3f-106996bdd665": (
  f"Kinh gui Quy vi, theo quy dinh moi ban hanh, tai khoan dinh danh dien tu cua Quy vi phai duoc xac thuc lai. De nghi Quy vi thuc hien tai {L} trong vong mot ngay.",
  f"Thong bao: du lieu dinh danh dien tu cua Quy vi chua dong bo voi co so du lieu moi. Vui long hoan tat xac thuc tai {L} trong 24 gio toi."),
 "43eb63e6-477e-5ab3-a4c8-9a0496d5fe1d": (
  f"Kinh gui Quy vi, Quy vi co mot khoan tro cap an sinh dang cho buoc xac nhan cuoi. De duoc chi tra, de nghi Quy vi khai bao thong tin nhan tien tai {L}.",
  f"Thong bao: danh sach huong ho tro an sinh co ten Quy vi nhung con thieu thong tin tai khoan. Vui long bo sung tai {L} de duoc giai ngan."),
 "7447b6bd-ab62-5db0-9aba-4b7b387c2e11": (
  f"Kinh gui Quy vi, he thong ghi nhan Quy vi con nghia vu hanh chinh chua thuc hien xong. De nghi Quy vi tra cuu va xu ly tai {L} nham tranh phat sinh che tai.",
  f"Thong bao: Quy vi dang co ho so hanh chinh o trang thai chua hoan tat. Vui long kiem tra va giai quyet tai {L} de khong bi xu ly theo quy dinh."),
 # ---- email / tax ----
 "89e679f4-eb52-52b1-be6e-dbd6535594b5": (
  f"Kinh gui Nguoi nop thue, ket qua ra soat cho thay Quy vi thuoc dien duoc hoan mot khoan thue. De nghi Quy vi cung cap thong tin tai khoan tiep nhan tai {L}.",
  f"Thong bao: he thong xac dinh Quy vi co so thue nop thua va du dieu kien nhan lai. Vui long khai bao so tai khoan nhan hoan tai {L}."),
 "68ca2cdd-9aee-5c07-ba64-e9dcac8e6bbf": (
  f"Kinh gui Nguoi nop thue, ho so quyet toan cua Quy vi con thieu chung tu bat buoc. De nghi bo sung tai {L} truoc 17 gio de tranh bi truy thu va xu phat.",
  f"Thong bao: phan quyet toan thue cua Quy vi chua day du giay to hop le. Vui long nop bo sung tai {L} truoc 17h00, neu khong co quan se ap dung bien phap truy thu."),
 "6b925a0b-f6bc-522b-9be0-136b13e537a5": (
  f"Kinh gui Quy vi, theo huong dan moi cua co quan thue, ma so thue cua Quy vi can duoc xac thuc lai. De nghi Quy vi cap nhat tai {L}.",
  f"Thong bao: du lieu ma so thue cua Quy vi phai duoc kiem tra lai theo quy trinh vua ban hanh. Vui long thuc hien tai {L}."),
 "5b80e5af-11bb-566c-b0f3-3ce4154f4396": (
  f"Kinh gui Quy vi, he thong thue dien tu phat hien sai lech giua du lieu khai bao va du lieu luu tru tren tai khoan cua Quy vi. De nghi Quy vi doi chieu tai {L} de dong bo.",
  f"Thong bao: co su khong khop trong so lieu thue dien tu cua Quy vi. Vui long kiem tra va cap nhat tai {L} de he thong hop nhat du lieu."),
 # ---- email / ecommerce ----
 "3f3f8b5d-500b-5a43-8378-c98b3b3b0a0b": (
  f"Kinh gui Quy khach, don hang cua Quy khach hien dung o khau xu ly do chua co thong tin xac nhan. De nghi Quy khach bo sung tai {L} de don duoc giao tiep.",
  f"Thong bao: kien hang cua Quy khach chua the chuyen di vi thieu du lieu nguoi nhan. Vui long khai bao tai {L} de tiep tuc quy trinh."),
 "326683ad-4448-5f1e-8230-9639c084a54a": (
  f"Kinh gui Quy khach, Quy khach nam trong danh sach khach hang than thiet duoc tang phan qua tri gia 2.000.000 VND. De nghi Quy khach xac nhan nhan qua tai {L} trong 3 gio.",
  f"Chuc mung Quy khach da duoc chon trong dot uu dai danh cho khach hang lau nam voi giai thuong 2 trieu dong. Vui long hoan tat thu tuc nhan tai {L} trong vong 03 tieng."),
 "b9043dd7-f426-5a4a-8037-3e0c4f9ca63b": (
  f"Kinh gui Quy khach, don hang da huy cua Quy khach du dieu kien duoc hoan tien. De nghi Quy khach xac nhan tai khoan tiep nhan tai {L}.",
  f"Thong bao hoan tien: he thong da duyet khoan tra lai cho don bi huy cua Quy khach. Vui long khai bao so tai khoan nhan tien tai {L}."),
 "ca5d1fc7-cbed-5091-8bab-44293d5c26af": (
  f"Kinh gui Quy khach, tai khoan mua sam cua Quy khach vua ghi nhan mot luot dang nhap tu thiet bi chua tung su dung. De bao ve tai khoan, de nghi Quy khach xac minh tai {L}.",
  f"Thong bao an ninh: co truy cap la vao tai khoan mua hang cua Quy khach. Vui long xac nhan danh tinh tai {L} de giu an toan cho tai khoan."),
 # ---- email / delivery ----
 "59d04bee-7ad4-54e7-844d-c12b60653362": (
  f"Kinh gui Quy khach, buu kien cua Quy khach dang duoc giu tai kho do thong tin dia chi khong khop. De nghi Quy khach cap nhat tai {L} de duoc giao trong ngay.",
  f"Thong bao: kien hang cua Quy khach chua phat duoc vi dia chi nhan chua chinh xac. Vui long chinh sua tai {L} de don duoc chuyen di hom nay."),
 "d01a36f7-efa8-55b2-837e-df8766d99326": (
  f"Kinh gui Quy khach, kien hang cua Quy khach phat sinh khoan phi luu kho 15.000 VND. De nghi Quy khach thanh toan tai {L} de hang khong bi tra ve nguoi gui.",
  f"Thong bao: don cua Quy khach dang chiu phi luu giu 15.000 VND tai kho. Vui long nop tai {L}, neu khong hang se duoc hoan cho ben gui."),
 "ecf8f784-1160-53fd-94ce-3b87fd20dc98": (
  f"Kinh gui Quy khach, buu pham gui tu nuoc ngoai cua Quy khach dang duoc luu de hoan tat thu tuc khai bao. De nghi Quy khach bo sung thong tin tai {L}.",
  f"Thong bao: kien hang quoc te dung ten Quy khach chua du ho so thong quan. Vui long khai bao them tai {L}."),
 # ---- email / telecom ----
 "95e5b92c-e0c8-5503-be98-d24838595b5e": (
  f"Kinh gui Quy khach, thue bao cua Quy khach chua hoan thanh viec chuan hoa thong tin va co nguy co bi tam dung. De nghi Quy khach cap nhat tai {L} de tiep tuc su dung.",
  f"Thong bao: thong tin dang ky cua thue bao Quy khach chua dung chuan, dich vu co the bi ngung. Vui long khai bao lai tai {L}."),
 "5e49f815-6fc0-5d85-b301-374ea8bd15d4": (
  f"Kinh gui Quy khach, trong chuong trinh tri an, Quy khach duoc tang mot goi data uu dai. De nghi Quy khach kich hoat tai {L} trong 24 gio.",
  f"Thong bao: Quy khach nhan duoc goi cuoc data khuyen mai tu chuong trinh danh cho khach hang lau nam. Vui long dang ky su dung tai {L} truoc 24 tieng."),
 "84961294-c4ad-5186-a577-13fda99fb83d": (
  f"Kinh gui Quy khach, thue bao cua Quy khach dang ton mot khoan cuoc chua duoc thanh toan. De nghi Quy khach doi chieu tai {L}.",
  f"Thong bao cuoc phi: he thong ghi nhan khoan no cuoc tren thue bao cua Quy khach. Vui long kiem tra va xu ly tai {L}."),
 # ---- email / social ----
 "87de156d-8376-5293-9f14-83e23faf2ccb": (
  f"Kinh gui Quy khach, chung toi ghi nhan mot luot dang nhap khong binh thuong vao tai khoan cua Quy khach. Neu day khong phai Quy khach, de nghi bao ve tai khoan ngay tai {L}.",
  f"Thong bao an ninh: co phien truy cap dang ngo vao tai khoan cua Quy khach. Truong hop khong phai Quy khach thuc hien, vui long khoa bao ve tai {L}."),
 "813260de-8e7a-53be-a2c4-a444624727d8": (
  f"Kinh gui Quy khach, trang cua Quy khach bi bao cao vi pham tieu chuan cong dong va co the bi gioi han hien thi. De nghi Quy khach gui khang nghi tai {L} trong 12 gio.",
  f"Thong bao: chung toi nhan duoc phan anh ve noi dung tren trang cua Quy khach; trang co nguy co bi han che. Vui long phan hoi tai {L} trong vong 12 tieng."),
 "bfc26a93-f9fb-5cee-8904-ce0fb7ee48c7": (
  f"Kinh gui Quy khach, tai khoan cua Quy khach du dieu kien duoc cap huy hieu xac minh chinh chu. De nghi Quy khach hoan tat buoc xac thuc danh tinh tai {L}.",
  f"Thong bao: Quy khach co the dang ky nhan dau tich xac minh cho tai khoan. Vui long thuc hien xac thuc danh tinh tai {L}."),
}

# Guardrails, checked rather than asserted.
URL_ANY = re.compile(r"https?://\S+", re.I)
BRAND_STOP = ("vietcom", "techcom", "vietin", "agribank", "bidv", "mbbank", "sacombank",
              "acb", "tpbank", "vpbank", "momo", "zalopay", "vnpay", "shopee", "lazada",
              "tiki", "sendo", "grab", "viettel", "vinaphone", "mobifone", "facebook",
              "zalo", "tiktok", "instagram", "ghtk", "ghn", "vnpost", "jtexpress")


def validate(text: str, src_id: str, variant: str) -> None:
    """Raise on any guardrail violation. A violating row must never reach the CSV."""
    for u in URL_ANY.findall(text):
        if u.rstrip(".,;") != L:
            raise SystemExit(f"[guardrail] {src_id}/{variant}: non-simulated URL {u!r}")
    low = text.lower()
    for b in BRAND_STOP:
        if b in low:
            raise SystemExit(f"[guardrail] {src_id}/{variant}: real brand token {b!r}")
    if any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text)):
        raise SystemExit(f"[guardrail] {src_id}/{variant}: diacritics present — the orthography "
                         "control of papers/P3_multimodal/protocols/PARAPHRASE_PROTOCOL.md "
                         "requires unaccented text")
    if len(text) < 20:
        raise SystemExit(f"[guardrail] {src_id}/{variant}: implausibly short")


def load_source() -> dict[str, dict]:
    import pandas as pd
    frames = []
    for t in ("sms", "email"):
        p = os.path.join(ROOT, "data", "processed", f"dataset_{t}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No dataset_sms/email.csv — run p2_generate_corpus.py + normalize_merge.")
    df = pd.concat(frames, ignore_index=True)
    ph = df[df["label"] == "phishing"]
    return {r["id"]: r for _, r in ph.iterrows()}


def main() -> None:
    from p3_paraphrase_ext import PARA_EXT      # rewrites for the 2026-08-07 corpus extension
    overlap = set(PARA) & set(PARA_EXT)
    if overlap:
        raise SystemExit(f"[!] {len(overlap)} id(s) paraphrased twice: {sorted(overlap)[:3]}")
    para = {**PARA, **PARA_EXT}
    src = load_source()
    missing = sorted(set(src) - set(para))
    extra = sorted(set(para) - set(src))
    if missing or extra:
        raise SystemExit(f"[!] paraphrase set out of sync with the corpus: "
                         f"{len(missing)} source messages unparaphrased, {len(extra)} orphans.\n"
                         f"    missing: {missing[:3]}\n    orphan:  {extra[:3]}")
    rows = []
    for sid, (pa, pb) in para.items():
        s = src[sid]
        for variant, text in (("a", pa), ("b", pb)):
            validate(text, sid, variant)
            rows.append({"src_id": sid, "variant": variant, "role": "train" if variant == "a" else "test",
                         "channel": s["channel"], "scenario": s["scenario"], "label": "phishing",
                         "is_llm": 1, "attack": "paraphrase", "gen_model": GEN_MODEL, "text": text})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[+] {OUT}: {len(rows)} paraphrases over {len(para)} source messages "
          f"({sum(r['role'] == 'train' for r in rows)} train-role, "
          f"{sum(r['role'] == 'test' for r in rows)} test-role); all guardrails passed")


if __name__ == "__main__":
    main()
