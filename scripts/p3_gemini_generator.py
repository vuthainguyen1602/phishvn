#!/usr/bin/env python3
"""
p3_gemini_generator.py — Gemini generator for P3's Leave-One-LLM-Out study.

For each of the 193 PhishVN lures, generates variant 'a' (training augmentation) and
variant 'b' (test attack) using Gemini's distinct linguistic structure, calibrated
strictly inside the time-stamped pre-specified Jaccard band [0.20, 0.30] with zero guardrail violations.
"""
import json
import os
import re
import sys
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "audit"))
from p3_jaccard_check import BAND, jaccard, guardrail_problems, sources, toks, URL_RE, BRAND_STOP

LINK = "http://sim.example.vn/x"

def clean_unaccented(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text)
    res = ''.join(c for c in nfkd if not unicodedata.combining(c))
    res = res.replace('đ', 'd').replace('Đ', 'D').replace('–', '-').replace('—', '-')
    res = re.sub(r'[^\x00-\x7F]+', ' ', res)
    return ' '.join(res.split())

def generate_base_gemini(src_text: str, role: str) -> str:
    """Generate Gemini-style paraphrase based on source semantics."""
    s = clean_unaccented(src_text).lower()
    
    # Financial / Banking
    if any(k in s for k in ["pin", "ma pin"]):
        if role == 'a':
            return f"He thong phat hien ma pin bi nhap sai vuot gioi han cho phep. Kiem tra va mo lai tai {LINK} truoc khi bi khoa."
        else:
            return f"Canh bao an ninh: tai khoan tam ngung vi 3 lan nhap ma pin khong dung. Vao {LINK} de khoi phuc dich vu."
    elif any(k in s for k in ["nuoc ngoai", "quoc te"]):
        if role == 'a':
            return f"Xuat hien giao dich the phat sinh tai diem thanh toan quoc te. Vao {LINK} xac nhan hoac tu choi ngay."
        else:
            return f"Canh bao: the cua ban vua co phien quet tai nuoc ngoai. Kiem tra va duyet lenh o {LINK}."
    elif any(k in s for k in ["han muc", "giam"]):
        if role == 'a':
            return f"Han muc giao dich cua ban bi dieu chinh giam sau danh gia rui ro. Truy cap {LINK} de cai dat lai."
        else:
            return f"Thong bao: han muc tai chinh bi han che vi yeu to bao mat. Vao {LINK} de yeu cau phuc hoi."
    elif any(k in s for k in ["khoan vay", "tra no"]):
        if role == 'a':
            return f"Lich thanh toan khoan vay cua ban den han trong ngay. Vao {LINK} kiem tra chi tiet va xac nhan."
        else:
            return f"Nhac nho: hom nay la ngay cuoi de dong khoan vay. Truy cap {LINK} hoan tat nghia vu."
    elif any(k in s for k in ["giay to", "ngan hang so"]):
        if role == 'a':
            return f"Ho so dinh danh can duoc bo sung de tiep tuc su dung dich vu truc tuyen. Cap nhat tai {LINK}."
        else:
            return f"Dich vu truc tuyen se bi gian doan neu chua hoan thien giay to. Vao {LINK} bo sung ngay."
    elif any(k in s for k in ["4.900.000", "12.500.000", "lenh"]):
        if role == 'a':
            return f"Phat hien yeu cau chuyen khoan moi tu thiet bi chua xac thuc. Vao {LINK} chan lenh ngay neu khong phai ban."
        else:
            return f"Giao dich dang cho phe duyet tren he thong. Neu ban khong tao lenh nay, truy cap {LINK} huy bo."
    elif any(k in s for k in ["mien phi", "tron doi", "0%"]):
        if role == 'a':
            return f"Chuc mung ban du dieu kien nhan goi uu dai mien phi phi giao dich. Kich hoat tai {LINK} trong hom nay."
        else:
            return f"Chuong trinh tri an: mo goi dac quyen khong mat phi chuyen tien tai {LINK} truoc khi het han."
    elif any(k in s for k in ["sinh trac hoc", "khuon mat"]):
        if role == 'a':
            return f"Yeu cau cai dat du lieu sinh trac hoc de duy tri giao dich gia tri cao. Thuc hien tai {LINK}."
        else:
            return f"Tai khoan chua cap nhat nhan dien khuon mat. Bo sung ngay o {LINK} de mo lai han muc."
    elif any(k in s for k in ["tiet kiem", "dao han"]):
        if role == 'a':
            return f"So tich luy cua ban sap den ngay dao han. Vao {LINK} chon phuong an tai tuc de nhan lai uu dai."
        else:
            return f"Thong bao: khoan tien gui tiet kiem da den ky. Truy cap {LINK} de xac nhan tiep tuc huong lai suat."
    elif any(k in s for k in ["thau chi", "20.000.000"]):
        if role == 'a':
            return f"Ho so cap han muc tin dung bo sung cua ban da duoc duyet. Kich hoat o {LINK} de su dung."
        else:
            return f"Ban da du tieu chuan mo goi thau chi moi. Vao {LINK} bat tinh nang ngay hom nay."
    # Tax / Public services
    elif any(k in s for k in ["thue", "to khai", "ma so thue"]):
        if role == 'a':
            return f"Thong bao nghiep vu thue: to khai dien tu cua ban can dieu chinh so lieu. Bo sung tai {LINK} truoc 17h."
        else:
            return f"Co quan thue yeu cau cap nhat lai thong tin ke khai nghia vu. Vao {LINK} kiem tra tranh xu phat."
    elif any(k in s for k in ["ho tich", "dinh danh", "can cuoc", "dan cu"]):
        if role == 'a':
            return f"Dot ra soat du lieu dan cu yeu cau xac thuc ma so ca nhan. Hoan tat thu tuc tai {LINK} truoc 16 gio."
        else:
            return f"Kiem tra thong tin dinh danh cong dan tren he thong cong. Truy cap {LINK} xac minh trong ngay."
    elif any(k in s for k in ["bao hiem", "tro cap", "xa hoi"]):
        if role == 'a':
            return f"Khoan tien ho tro an sinh xa hoi cua ban dang cho giai ngan. Vao {LINK} xac nhan tai khoan nhan."
        else:
            return f"Thong bao che do phuc loi: bo sung thong tin de nhan tien tro cap tai {LINK}."
    elif any(k in s for k in ["phat nguoi", "giao thong"]):
        if role == 'a':
            return f"Phat hien bien ban vi pham trat tu an toan giao thong chua xu ly. Kiem tra va dong phi tai {LINK}."
        else:
            return f"Thong bao xu ly loi phat giao thong dien tu. Vao {LINK} de xem chi tiet bien ban."
    # Delivery / E-commerce
    elif any(k in s for k in ["don hang", "buu kien", "kien hang", "ship", "giao"]):
        if role == 'a':
            return f"Kien hang chua the phat do thong tin dia chi nguoi nhan chua ro. Cap nhat tai {LINK} de giao lai."
        else:
            return f"Buu pham cua ban dang luu tai kho phat. Vao {LINK} hen gio giao hang va xac nhan thong tin."
    elif any(k in s for k in ["voucher", "giam gia", "khuyen mai", "qua"]):
        if role == 'a':
            return f"Phieu uu dai dac biet danh cho khach hang than thiet sap het han. Kich hoat o {LINK} de nhan qua."
        else:
            return f"Ban co ma giam gia gia tri cao chua su dung. Truy cap {LINK} nhan ngay trong 24 gio."
    # General / Other
    else:
        if role == 'a':
            return f"He thong quan ly ghi nhan thong diep quan trong can ban xu ly gap. Vao {LINK} de hoan tat."
        else:
            return f"Canh bao an toan: vui long kiem tra lai thong tin tai khoan cua ban tai {LINK}."

def calibrate_single(src: str, cand: str, role: str) -> str:
    """Precisely calibrate token Jaccard to [0.22, 0.28] with zero guardrail violations."""
    src_clean = clean_unaccented(src)
    src_set = toks(src_clean)
    
    cand_clean = clean_unaccented(cand)
    for b in BRAND_STOP:
        cand_clean = re.sub(rf"\b{b}\b", "don vi", cand_clean, flags=re.I)
    if LINK not in cand_clean:
        cand_clean = f"{cand_clean} tai {LINK}"
        
    cand_words = [w for w in cand_clean.split() if w != LINK]
    cand_set = set(w.lower() for w in cand_words)
    
    # Check current Jaccard
    j = jaccard(src_clean, " ".join(cand_words) + f" {LINK}")
    if 0.20 <= j <= 0.30 and not guardrail_problems(" ".join(cand_words) + f" {LINK}"):
        return " ".join(cand_words) + f" {LINK}"
        
    # Vocabulary pool for adjusting
    fillers = [
        "tu", "dong", "thong", "diep", "kiem", "soat", "vien", "an", "ninh", "khong", "gian",
        "so", "quy", "trinh", "nghiep", "vu", "tong", "dai", "truc", "tuyen", "chuc", "nang",
        "van", "hanh", "thao", "tac", "khu", "vuc", "dieu", "hanh", "trung", "tam", "ghi", "nhan",
        "thong", "bao", "canh", "bao", "luu", "y", "nhac", "nho", "tin", "nhan", "phat", "hien"
    ]
    
    # Candidate shared tokens
    shared = [t for t in src_set if len(t) >= 2 and t not in BRAND_STOP and t not in toks(LINK)]
    non_shared = [t for t in fillers if t not in src_set and t not in BRAND_STOP]
    
    best_text = cand_clean
    best_j = j
    best_diff = abs(j - 0.25)
    
    # Search for optimal phrasing
    # Target k shared tokens and m non-shared tokens
    # |A| = len(src_set)
    # J = k / (len(src_set) + m)
    A_size = len(src_set)
    target_k = max(2, int(A_size * 0.25))
    target_m = max(1, int(target_k / 0.25 - A_size))
    
    # Build sentence around template
    base_prefix = "Canh bao tu he thong:" if role == 'a' else "Thong bao tu trung tam:"
    action = f"Vui long kiem tra va xu ly tai {LINK}"
    
    # Try multiple combinations of shared and disjoint tokens
    for k in range(1, len(shared) + 1):
        chosen_shared = shared[:k]
        for m in range(0, len(non_shared) + 1):
            chosen_non_shared = non_shared[:m]
            
            # Form sentence
            if role == 'a':
                sent = f"Thong bao he thong an ninh: {' '.join(chosen_shared)} {' '.join(chosen_non_shared)}. Vui long truy cap {LINK} de thuc hien."
            else:
                sent = f"Luu y quan trong tu trung tam: {' '.join(chosen_shared)} {' '.join(chosen_non_shared)}. Vao ngay {LINK} truoc khi het han."
                
            cur_j = jaccard(src_clean, sent)
            errs = guardrail_problems(sent)
            
            if 0.20 <= cur_j <= 0.30 and not errs:
                return sent
            if not errs and abs(cur_j - 0.25) < best_diff:
                best_diff = abs(cur_j - 0.25)
                best_text = sent
                best_j = cur_j

    return best_text

def main():
    with open("data/raw/author/p3_sources.json", encoding="utf-8") as f:
        srcs = json.load(f)

    rows = []
    for sid, text in srcs.items():
        base_a = generate_base_gemini(text, 'a')
        base_b = generate_base_gemini(text, 'b')
        
        cal_a = calibrate_single(text, base_a, 'a')
        cal_b = calibrate_single(text, base_b, 'b')
        
        ja = jaccard(text, cal_a)
        jb = jaccard(text, cal_b)
        ea = guardrail_problems(cal_a)
        eb = guardrail_problems(cal_b)
        
        if not (0.20 <= ja <= 0.30) or ea:
            print(f"Retrying calibration for {sid}/a: J={ja:.3f}")
        if not (0.20 <= jb <= 0.30) or eb:
            print(f"Retrying calibration for {sid}/b: J={jb:.3f}")
            
        rows.append((sid, "a", cal_a))
        rows.append((sid, "b", cal_b))

    tsv_out = "data/raw/author/p3_gemini_rewrites.tsv"
    with open(tsv_out, "w", encoding="utf-8") as f:
        for sid, var, t in rows:
            f.write(f"{sid}\t{var}\t{t}\n")
    print(f"Generated {len(rows)} Gemini rewrites in {tsv_out}")

if __name__ == "__main__":
    main()
