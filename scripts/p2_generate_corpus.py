#!/usr/bin/env python3
"""
p2_generate_corpus.py — Write the P2 SIMULATED Vietnamese phishing corpus + a benign control set.

DEFENSIVE PURPOSE (P2 + P1b adversarial set): a labelled corpus to (a) characterise LLM-generated
Vietnamese phishing and (b) train/evaluate a detector robust to it. Every phishing sample is a
SIMULATION, watermarked is_llm=1 with a gen_model tag, and follows the P2 guardrails:
  * placeholder links only (http://sim.example.vn/...), never a real or reachable target,
  * GENERIC sector references (a bank / the tax office / a carrier), NOT real brand names or URLs,
  * no targeting of real people, no operational attack instructions.

This file ships in the public code mirror as of 2026-09-05. It said "keep in the gated tier, do
not disseminate" until then, and the line is gone rather than quietly contradicted: the lures it
writes are already published with the corpus, the paper argues in its Ethics Statement that
releasing inert rewrites is the safe direction because an evasion result nobody can reproduce is
one nobody can check, and what stays withheld is the generation PROMPTING, which is in
gen_llm_adversarial.py and is not exported. Rewrites are artefacts; the procedure that
manufactures fresh ones at a chosen attack strength is not.

This file's samples were authored by the model 'claude-fable-5' acting as ONE generator under the
P2 guardrail prompt. For the leave-one-LLM-out experiment, add >=2 more generators by running
gen_llm_adversarial.py with an API key (or a local Ollama model); their rows merge by gen_model.

RUN:  python scripts/p2_generate_corpus.py
"""
from __future__ import annotations
import csv
import os

GEN_MODEL = "claude-fable-5"
LINK = "http://sim.example.vn/x"   # simulated, non-resolving

# ---- SIMULATED phishing lures (generic entities, urgency/persuasion variety) ----
# keyed by (channel, scenario) -> list of texts
PHISH = {
 ("sms", "bank"): [
  f"Ngan hang thong bao: tai khoan cua quy khach dang bi tam khoa do dang nhap bat thuong. Xac minh trong 2 gio tai {LINK} de tranh mat quyen truy cap.",
  f"[Thong bao] Giao dich 12.500.000d vua duoc thuc hien. Neu khong phai ban, huy ngay tai {LINK}",
  f"Quy khach co 1 giao dich cho xac nhan. Dang nhap {LINK} de kiem tra, sau 24h tai khoan se bi khoa.",
  f"Diem tin dung cua ban duoc nang hang. Kich hoat uu dai lai suat 0% tai {LINK} truoc 18h hom nay.",
  f"He thong ghi nhan the cua ban het han. Cap nhat thong tin the tai {LINK} de tiep tuc su dung dich vu.",
  f"Canh bao: co thiet bi la truy cap tai khoan ngan hang cua ban luc 02:14. Khoa khan cap tai {LINK}",
  f"Ban duoc hoan 500.000d phi thuong nien. Nhan tai {LINK}, chi ap dung trong hom nay.",
  f"Tai khoan tiet kiem dao han. Xac nhan tai tuc tai {LINK} de huong lai suat uu dai.",
  f"OTP cua ban sap het hieu luc. Vao {LINK} de xac thuc lai, tranh gian doan dich vu.",
  f"Ngan hang nang cap he thong bao mat. Dinh danh lai tai khoan tai {LINK} truoc 20h.",
 ],
 ("sms", "gov"): [
  f"Co quan chuc nang thong bao ban co ho so hanh chinh chua hoan tat. Bo sung tai {LINK} truoc 17h de tranh xu phat.",
  f"Dich vu cong: dinh danh dien tu cua ban can cap nhat. Xac minh tai {LINK} trong 24h.",
  f"Thong bao phat nguoi giao thong chua nop. Kiem tra va xu ly tai {LINK}",
  f"Bao hiem xa hoi: ho tro cua ban dang cho duyet. Cap nhat thong tin nhan tro cap tai {LINK}",
  f"Ho so cu tru cua cong dan can xac thuc lai. Truy cap {LINK} de hoan tat, han chot hom nay.",
  f"Can cuoc dien tu cua ban sap het han. Gia han tai {LINK} de tiep tuc su dung dich vu cong.",
  f"Thong bao: ban duoc hoan thue an sinh. Dang ky nhan tai {LINK} truoc khi het han.",
  f"Co quan quan ly yeu cau cap nhat thong tin ho khau. Thuc hien tai {LINK}",
 ],
 ("sms", "tax"): [
  f"Co quan thue thong bao ban duoc hoan 3.200.000d. Cung cap thong tin nhan hoan thue tai {LINK}",
  f"Ho so quyet toan thue cua ban thieu chung tu. Bo sung tai {LINK} truoc 17h de tranh truy thu.",
  f"Ma so thue cua ban can xac thuc lai theo quy dinh moi. Cap nhat tai {LINK}",
  f"Thong bao no thue: vui long doi chieu va thanh toan tai {LINK} de tranh cuong che.",
  f"Ban du dieu kien giam tru thue. Dang ky nhan uu dai tai {LINK} truoc han.",
  f"He thong thue dien tu nang cap. Dang nhap lai tai {LINK} de dong bo du lieu.",
 ],
 ("sms", "ecommerce"): [
  f"Don hang cua ban dang bi giu do thieu thong tin. Cap nhat tai {LINK} de tiep tuc giao.",
  f"Chuc mung! Ban trung voucher 2.000.000d tu chuong trinh tri an. Nhan tai {LINK} trong 3 gio.",
  f"San thuong mai thong bao hoan tien don hang huy. Xac nhan tai khoan nhan tien tai {LINK}",
  f"Diem thuong cua ban sap het han. Doi qua tai {LINK} truoc 23h59 hom nay.",
  f"Tai khoan mua sam cua ban dang nhap bat thuong. Bao mat ngay tai {LINK}",
  f"Uu dai flash sale danh rieng cho ban. Kich hoat ma giam 50% tai {LINK}",
  f"Don hang #A83920 giao that bai. Dat lich giao lai va xac nhan dia chi tai {LINK}",
 ],
 ("sms", "delivery"): [
  f"Buu kien cua ban dang cho tai kho do sai dia chi. Cap nhat tai {LINK} de nhan trong hom nay.",
  f"Don ship cua ban can dong phi luu kho 15.000d. Thanh toan tai {LINK} de tranh hoan hang.",
  f"Nhan vien giao hang khong lien he duoc ban. Xac nhan lich giao tai {LINK}",
  f"Kien hang quoc te cua ban bi giu tai hai quan. Khai bao bo sung tai {LINK}",
  f"Ma van don cua ban can xac thuc nguoi nhan. Vao {LINK} de hoan tat.",
  f"Buu pham uu tien dang cho: chon khung gio giao va xac nhan tai {LINK}",
 ],
 ("sms", "telecom"): [
  f"Nha mang thong bao SIM cua ban sap bi khoa 2 chieu do chua chuan hoa. Cap nhat tai {LINK}",
  f"Ban duoc tang 50GB data mien phi. Kich hoat tai {LINK} truoc 24h.",
  f"Diem tich luy nha mang cua ban sap het han. Doi qua tai {LINK}",
  f"Thue bao cua ban no cuoc 120.000d. Thanh toan tai {LINK} de tranh cat lien lac.",
  f"Chuong trinh nang cap SIM 4G/5G mien phi. Dang ky tai {LINK}",
 ],
 ("sms", "social"): [
  f"Tai khoan mang xa hoi cua ban co dang nhap la tu thiet bi khac. Bao mat ngay tai {LINK}",
  f"Trang cua ban vi pham tieu chuan cong dong va sap bi khoa. Khang nghi tai {LINK} trong 12h.",
  f"Ban duoc mo tinh nang kiem tien tren nen tang. Dang ky tai {LINK}",
  f"Co nguoi gan the ban trong mot bai viet nhay cam. Kiem tra tai {LINK}",
  f"Xac minh danh tinh de lay lai huy hieu chinh chu tai {LINK}",
 ],
 ("email", "bank"): [
  f"Kinh gui Quy khach, he thong ghi nhan hoat dong dang nhap bat thuong tren tai khoan cua Quy khach. De dam bao an toan, vui long xac minh danh tinh trong vong 02 gio tai {LINK}. Sau thoi gian nay tai khoan se tam thoi bi han che.",
  f"Thong bao bao mat: chung toi phat hien mot giao dich co dau hieu rui ro tri gia 15.750.000 VND. Neu Quy khach khong thuc hien, vui long huy giao dich va cap nhat thong tin bao mat tai {LINK}.",
  f"Quy khach than men, the tin dung cua Quy khach sap het han. De tranh gian doan dich vu, vui long cap nhat thong tin the tai {LINK} truoc ngay het han.",
  f"Chuong trinh hoan phi thuong nien dang duoc ap dung cho tai khoan cua Quy khach. Vui long xac nhan thong tin nhan hoan tai {LINK} truoc 18h00 hom nay.",
  f"He thong ngan hang dang nang cap tieu chuan bao mat. Quy khach vui long dinh danh lai tai khoan tai {LINK} de tiep tuc su dung dich vu truc tuyen.",
 ],
 ("email", "gov"): [
  f"Kinh gui Cong dan, ho so dich vu cong cua Quy vi hien chua hoan tat mot so truong thong tin bat buoc. De nghi bo sung tai {LINK} truoc 17h00 de ho so duoc tiep nhan.",
  f"Thong bao: tai khoan dinh danh dien tu cua Quy vi can duoc xac thuc lai theo quy dinh moi. Vui long thuc hien tai {LINK} trong vong 24 gio.",
  f"Quy vi co mot khoan ho tro an sinh dang cho xac nhan. De nhan tro cap, vui long cap nhat thong tin tai {LINK}.",
  f"Co quan quan ly thong bao Quy vi co nghia vu hanh chinh chua hoan thanh. Vui long kiem tra va xu ly tai {LINK} de tranh phat sinh xu ly.",
 ],
 ("email", "tax"): [
  f"Kinh gui Nguoi nop thue, qua ra soat he thong ghi nhan Quy vi du dieu kien duoc hoan mot khoan thue. Vui long cung cap thong tin tai khoan nhan hoan tai {LINK}.",
  f"Thong bao: ho so quyet toan thue cua Quy vi con thieu chung tu. De nghi bo sung tai {LINK} truoc 17h00 de tranh bi truy thu va xu phat.",
  f"Ma so thue cua Quy vi can xac thuc lai theo huong dan moi cua co quan thue. Vui long cap nhat tai {LINK}.",
  f"He thong thue dien tu ghi nhan chenh lech du lieu tren tai khoan cua Quy vi. Vui long doi chieu tai {LINK} de dong bo.",
 ],
 ("email", "ecommerce"): [
  f"Kinh gui Quy khach, don hang cua Quy khach hien dang bi tam giu do thieu thong tin xac nhan. Vui long cap nhat tai {LINK} de don hang tiep tuc duoc xu ly.",
  f"Chuc mung Quy khach da duoc chon trong chuong trinh tri an khach hang than thiet voi mot phan qua tri gia 2.000.000 VND. Vui long xac nhan nhan qua tai {LINK} trong 03 gio.",
  f"Thong bao hoan tien: don hang huy cua Quy khach du dieu kien hoan. Vui long xac nhan tai khoan nhan tien tai {LINK}.",
  f"Tai khoan mua sam cua Quy khach vua co dang nhap tu thiet bi la. De bao ve tai khoan, vui long xac minh tai {LINK}.",
 ],
 ("email", "delivery"): [
  f"Kinh gui Quy khach, buu kien cua Quy khach hien dang luu tai kho do thong tin dia chi chua chinh xac. Vui long cap nhat tai {LINK} de duoc giao trong hom nay.",
  f"Thong bao: kien hang cua Quy khach phat sinh phi luu kho 15.000 VND. Vui long thanh toan tai {LINK} de tranh hoan tra ve nguoi gui.",
  f"Buu pham quoc te cua Quy khach dang duoc giu de hoan tat thu tuc khai bao. Vui long bo sung thong tin tai {LINK}.",
 ],
 ("email", "telecom"): [
  f"Kinh gui Quy khach, thue bao cua Quy khach chua hoan tat chuan hoa thong tin va co the bi tam ngung dich vu. Vui long cap nhat tai {LINK} de tiep tuc su dung.",
  f"Chuong trinh tri an: Quy khach duoc tang goi data uu dai. Vui long kich hoat tai {LINK} truoc 24 gio.",
  f"Thong bao cuoc: thue bao cua Quy khach dang co khoan cuoc chua thanh toan. Vui long doi chieu tai {LINK}.",
 ],
 ("email", "social"): [
  f"Kinh gui Quy khach, chung toi ghi nhan mot lan dang nhap bat thuong vao tai khoan cua Quy khach. Neu khong phai Quy khach, vui long bao mat ngay tai {LINK}.",
  f"Thong bao: trang cua Quy khach bi bao cao vi pham tieu chuan cong dong va co the bi han che. Vui long khang nghi tai {LINK} trong vong 12 gio.",
  f"Tai khoan cua Quy khach du dieu kien nhan huy hieu xac minh chinh chu. Vui long hoan tat xac minh danh tinh tai {LINK}.",
 ],
}

# ---- BENIGN control: legitimate service-style Vietnamese messages (no links or plain info) ----
BENIGN = {
 ("sms", "bank"): [
  "Quy khach vua thanh toan 250.000d tai cua hang tien loi luc 19:20. So du kha dung: 4.120.000d.",
  "Cam on Quy khach da su dung dich vu. Diem thuong thang nay cua Quy khach la 1.250 diem.",
  "Tai khoan cua Quy khach nhan 8.000.000d luong thang. So du hien tai: 12.480.000d.",
  "Nhac lich: sao ke tai khoan thang cua Quy khach da san sang trong ung dung ngan hang.",
  "Quy khach da dang ky thanh cong dich vu thong bao bien dong so du qua tin nhan.",
  "Ky nghi le sap toi, chi nhanh lam viec den 11h30 ngay thu Bay. Xin cam on.",
 ],
 ("sms", "gov"): [
  "Nhac lich: cuoc hen lam thu tuc hanh chinh cua ban vao 09:00 thu Nam tai bo phan mot cua.",
  "Ho so cua ban da duoc tiep nhan va dang xu ly. Thoi gian tra ket qua du kien 5 ngay lam viec.",
  "Thong bao lich cat dien bao tri khu vuc cua ban tu 8h-11h ngay Chu Nhat.",
  "Ket qua ho so dich vu cong cua ban da co. Vui long den nhan tai bo phan tra ket qua trong gio hanh chinh.",
  "Nhac dong bao hiem y te dinh ky truoc ngay 05 hang thang de duy tri quyen loi.",
 ],
 ("sms", "tax"): [
  "Ban da nop to khai thue thang thanh cong. Ma giao dich duoc luu trong tai khoan thue dien tu.",
  "Nhac han: thoi han nop to khai thue GTGT thang nay la ngay 20. Vui long hoan tat dung han.",
  "Bien lai dien tu cho khoan nop cua ban da duoc phat hanh va luu tren he thong.",
  "Thong bao: he thong thue dien tu bao tri tu 22h-23h hom nay, mong ban thong cam.",
 ],
 ("sms", "ecommerce"): [
  "Don hang cua ban da duoc xac nhan va dang chuan bi. Du kien giao trong 3-5 ngay.",
  "Cam on ban da mua sam. Danh gia san pham de nhan 100 diem thuong nhe!",
  "Don hang #A83920 da giao thanh cong luc 15:40. Cam on ban da tin tuong.",
  "Vi cua ban nhan hoan 30.000d tu don hang huy. So du vi hien tai da cap nhat.",
  "Chuong trinh sale cuoi tuan bat dau tu 0h thu Bay. Xem trong ung dung nhe!",
 ],
 ("sms", "delivery"): [
  "Buu kien cua ban da duoc lay va dang tren duong giao. Ma van don da cap nhat trong ung dung.",
  "Shipper se giao hang cua ban trong khung 14h-16h hom nay. Vui long de y dien thoai.",
  "Buu kien cua ban da giao thanh cong. Cam on ban da su dung dich vu.",
  "Don cua ban dang o buu cuc gan nhat, du kien phat trong ngay mai.",
 ],
 ("sms", "telecom"): [
  "Ban da nap thanh cong 100.000d vao thue bao. Han su dung den ngay cuoi thang sau.",
  "Goi cuoc data cua ban con 3.2GB, gia han tu dong vao ngay 01 thang sau.",
  "Cam on ban da su dung dich vu. Diem tich luy thang nay: 320 diem.",
  "Thong bao bao tri he thong ngan han tu 2h-3h sang, dich vu co the gian doan nhe.",
 ],
 ("sms", "social"): [
  "Ban co 3 loi moi ket ban moi va 5 thong bao chua doc trong ung dung.",
  "Ban vua dang nhap tren thiet bi moi. Neu la ban thi khong can lam gi them.",
  "Trang cua ban dat 1.000 luot theo doi trong thang nay. Chuc mung!",
  "Ban be cua ban vua chia se mot ky niem cu. Xem trong dong thoi gian nhe.",
 ],
 ("email", "bank"): [
  "Kinh gui Quy khach, sao ke tai khoan thang cua Quy khach da san sang trong ung dung. Cam on Quy khach da su dung dich vu.",
  "Kinh gui Quy khach, giao dich thanh toan hoa don dien nuoc cua Quy khach da hoan tat. Chi tiet duoc luu trong lich su giao dich.",
  "Kinh gui Quy khach, chuong trinh tich diem quy nay da cap nhat. Quy khach co the xem chi tiet trong muc uu dai cua ung dung.",
  "Kinh gui Quy khach, chi nhanh se dieu chinh gio lam viec trong dip le. Xin thong bao de Quy khach chu dong sap xep.",
 ],
 ("email", "gov"): [
  "Kinh gui Cong dan, ho so cua Quy vi da duoc tiep nhan va dang trong qua trinh xu ly. Ket qua du kien tra sau 5 ngay lam viec.",
  "Kinh gui Cong dan, xin thong bao lich lam viec cua bo phan mot cua trong dip le sap toi de Quy vi chu dong.",
  "Kinh gui Cong dan, ket qua thu tuc hanh chinh cua Quy vi da co. Vui long den nhan trong gio hanh chinh.",
 ],
 ("email", "tax"): [
  "Kinh gui Nguoi nop thue, to khai thue thang cua Quy vi da duoc ghi nhan thanh cong tren he thong. Cam on Quy vi da tuan thu.",
  "Kinh gui Nguoi nop thue, xin nhac thoi han nop to khai dinh ky sap toi de Quy vi chu dong hoan tat dung han.",
  "Kinh gui Nguoi nop thue, bien lai dien tu cho khoan nop cua Quy vi da duoc phat hanh va luu tren he thong.",
 ],
 ("email", "ecommerce"): [
  "Kinh gui Quy khach, don hang cua Quy khach da duoc xac nhan va dang chuan bi giao. Cam on Quy khach da mua sam.",
  "Kinh gui Quy khach, don hang cua Quy khach da giao thanh cong. Rat mong nhan duoc danh gia cua Quy khach.",
  "Kinh gui Quy khach, chuong trinh uu dai cuoi tuan sap bat dau. Quy khach co the xem chi tiet trong ung dung.",
 ],
 ("email", "delivery"): [
  "Kinh gui Quy khach, buu kien cua Quy khach da duoc lay va dang tren duong van chuyen. Ma van don da cap nhat.",
  "Kinh gui Quy khach, buu kien cua Quy khach du kien duoc giao trong ngay mai. Cam on Quy khach da su dung dich vu.",
  "Kinh gui Quy khach, buu kien cua Quy khach da duoc giao thanh cong. Cam on Quy khach.",
 ],
 ("email", "telecom"): [
  "Kinh gui Quy khach, goi cuoc cua Quy khach da duoc gia han thanh cong. Cam on Quy khach da tin dung dich vu.",
  "Kinh gui Quy khach, thong bao lich bao tri he thong ngan han de nang cao chat luong dich vu.",
  "Kinh gui Quy khach, hoa don cuoc thang cua Quy khach da san sang trong ung dung.",
 ],
 ("email", "social"): [
  "Kinh gui Quy khach, ban tin tong hop hoat dong trong tuan cua Quy khach da san sang trong ung dung.",
  "Kinh gui Quy khach, chung toi vua cap nhat mot so tinh nang moi. Quy khach co the tim hieu trong phan tro giup.",
  "Kinh gui Quy khach, cam on Quy khach da la thanh vien. Chuc Quy khach mot tuan tot lanh.",
 ],
}


def merged(base: dict, ext: dict) -> dict:
    """Pilot corpus first, extension appended per cell.

    The pilot's messages stay byte-identical and in order, so their content-derived uuid5 ids do
    not move and the paraphrase set keyed by those ids stays valid. See scripts/p2_corpus_ext.py
    for why the extension exists and what was fixed before it was written.
    """
    out = {k: list(v) for k, v in base.items()}
    for k, v in ext.items():
        out.setdefault(k, []).extend(v)
    return out


def main():
    from p2_corpus_ext import PHISH_EXT, BENIGN_EXT
    phish, benign = merged(PHISH, PHISH_EXT), merged(BENIGN, BENIGN_EXT)
    llm_path = "data/raw/llm/llm_adv.csv"
    ben_path = "data/raw/author/benign_text.csv"
    os.makedirs(os.path.dirname(llm_path), exist_ok=True)
    os.makedirs(os.path.dirname(ben_path), exist_ok=True)

    with open(llm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "label", "source", "is_llm",
                                          "scenario", "gen_model", "text"])
        w.writeheader()
        n = 0
        for (ch, sc), texts in phish.items():
            for t in texts:
                w.writerow({"channel": ch, "label": "phishing", "source": "llm", "is_llm": 1,
                            "scenario": sc, "gen_model": GEN_MODEL, "text": t})
                n += 1
    print(f"[+] {llm_path}: {n} simulated phishing (gen_model={GEN_MODEL})")

    with open(ben_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "label", "scenario", "collected_at", "text"])
        w.writeheader()
        m = 0
        for (ch, sc), texts in benign.items():
            for t in texts:
                w.writerow({"channel": ch, "label": "benign", "scenario": sc,
                            "collected_at": "", "text": t})
                m += 1
    print(f"[+] {ben_path}: {m} benign control messages")


if __name__ == "__main__":
    main()
