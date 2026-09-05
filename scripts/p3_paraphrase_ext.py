#!/usr/bin/env python3
"""
p3_paraphrase_ext.py — paraphrases for the 120 lures added by p2_corpus_ext.py.

Same rules as the pilot set in p3_paraphrase_corpus.py, which validates and merges this one:
variant 'a' may only enter training augmentation, variant 'b' may only enter the test attack;
scenario, call to action, urgency device and channel register are preserved; word choice,
sentence structure and pretext framing change; orthography stays unaccented, matching the
sources; guardrails (simulated non-resolving link, generic sector references, no real brand, no
targeting, no operational instruction) are machine-checked by the merging module.

Keyed by the source row's content-derived uuid5 id, so the mapping survives corpus rebuilds.
"""
from __future__ import annotations

L = "http://sim.example.com/x"

PARA_EXT: dict[str, tuple[str, str]] = {
 # ---- sms / bank ----
 "9937630e-81c8-5408-8c8d-40683f8b5539": (
  f"Ban vua nhap sai PIN ba lan lien tiep. Go khoa o {L} keo tai khoan bi tam ngung.",
  f"He thong dem duoc 3 lan sai ma PIN tren tai khoan ban. Vao {L} mo lai truoc khi dich vu bi dung."),
 "16e2a056-c821-50de-9b6c-8260fdd8b662": (
  f"Co giao dich bang the cua ban phat sinh o ngoai nuoc. Duyet hay tu choi tai {L}",
  f"The ban vua quet tai mot diem o nuoc ngoai. Chon xac nhan hoac huy o {L}"),
 "c04d7243-8e99-57a9-9ecf-81dd7928da43": (
  f"Muc chuyen toi da cua ban bi giam sau canh bao an toan. Lay lai o {L}",
  f"Do co canh bao rui ro, han muc giao dich cua ban da bi rut xuong. Phuc hoi tai {L}"),
 "db6fd208-81ac-5777-af3c-7ed177e96620": (
  f"Khoan vay cua ban toi han trong hom nay. Kiem tra lich tra o {L} va xac nhan.",
  f"Hom nay la han tra no cua ban. Vao {L} xem ky va duyet."),
 "175fa3cd-b188-551e-ad95-c7cb52ed1892": (
  f"Chua bo sung giay to thi tu ngay mai ban khong dung duoc ngan hang so. Nop o {L}",
  f"Giay to cua ban qua han, dich vu truc tuyen ngat tu ngay mai. Cap nhat tai {L}"),
 "ea890ed6-880a-5de9-88e6-aa6f82243641": (
  f"Mot lenh 4.900.000d tu may la dang cho ban quyet dinh. Duyet hay chan o {L}",
  f"Co khoan 4.900.000d cho phe duyet, phat sinh tu thiet bi chua tung dung. Xu ly tai {L}"),
 "9496739e-a472-52e5-9e0a-004aa8dbaca7": (
  f"Goi mien phi chuyen tien vinh vien dang mo cho ban. Bat o {L}, chi trong ngay.",
  f"Ban du dieu kien nhan goi khong mat phi chuyen khoan tron doi. Kich hoat tai {L} hom nay."),
 "e1647349-3ac5-5fb8-b397-440312b03283": (
  f"Ten nguoi nhan tren lenh chuyen cua ban khong trung. Sua o {L} keo tien bi hoan.",
  f"Du lieu ben thu huong chua dung. Chinh tai {L} de lenh khong bi tra ve."),
 "2fa54e91-e992-5caf-88ac-fa741a5b2921": (
  f"Co mot khieu nai giao dich dung ten ban duoc gui len. Xac thuc o {L}",
  f"He thong nhan don tra soat mang ten ban. Vao {L} de kiem chung."),
 "7dc28a21-f3ac-51e1-95a3-fc6263479f27": (
  f"Ban chua dang ky sinh trac hoc nen khong chuyen duoc tren 10 trieu. Lam o {L}",
  f"Tai khoan thieu buoc xac thuc khuon mat. Bo sung tai {L} de mo giao dich lon."),
 "331e1019-270d-598b-9fb1-9d3a22d9e241": (
  f"Canh bao: so the cua ban bi phat tan tren mang. Khoa va xin the moi o {L}",
  f"He thong thay thong tin the ban xuat hien cong khai. Vao {L} de khoa va cap lai."),
 "3f6c7101-c145-5402-9023-284ae61346f2": (
  f"Khoan tiet kiem cua ban duoc cong them lai thuong. Nhan o {L} truoc 5 gio chieu.",
  f"Ban co phan lai suat thuong chua nhan. Lay tai {L}, han 17h."),
 "edda25a6-4f56-5497-b031-87b5eaf7780a": (
  f"Tai khoan khong hoat dong lau ngay se bi dong. Xac nhan o {L} de giu lai.",
  f"De tranh bi dong do khong giao dich, ban can xac nhan tai {L}"),
 "60527908-b7d3-56cf-91ef-5c4912e7eec5": (
  f"Han muc thau chi 20 trieu cua ban da duoc duyet. Bat o {L} de dung.",
  f"Ban duoc cap thau chi 20.000.000d. Kich hoat tai {L}"),
 "071b5fec-024f-5d85-a45d-a9ee35576a21": (
  f"So nhan ma OTP cua ban can duoc xac nhan lai. Cap nhat o {L}",
  f"He thong yeu cau kiem tra lai so dien thoai gan voi OTP. Thuc hien tai {L}"),
 "de1868e8-4376-51ed-aaef-12dffa4383a9": (
  f"So du khong du de tru phi the thang nay. Nop o {L} keo bi tinh lai.",
  f"Phi thuong nien chua thu duoc vi tai khoan thieu tien. Thanh toan tai {L} truoc khi phat sinh lai."),
 # ---- sms / gov ----
 "c49be51b-3daa-52d3-a655-6361986067f0": (
  f"Bang lai cua ban gan het hieu luc. Lam gia han qua mang tai {L}",
  f"Giay phep lai xe cua ban toi han doi. Gia han o {L}"),
 "2904fe88-574e-58b6-a2c7-5a49b2422e2c": (
  f"Co quan chuc nang gui giay moi ban len lam viec. Doc noi dung o {L}",
  f"Ban nhan duoc mot giay moi tu co quan quan ly. Chi tiet tai {L}"),
 "21013861-fe81-5d30-91bb-08e8c772a68b": (
  f"Trong dot ra soat dan cu, ban phai xac nhan so dinh danh. Lam o {L} truoc 4 gio chieu.",
  f"Ban can khai xac nhan so dinh danh ca nhan cho dot ra soat. Thuc hien tai {L}, han 16h."),
 "3965070b-7d88-53be-84f2-98a18828203c": (
  f"Ho so ly lich tu phap cua ban thieu anh giay to tuy than. Gui bo sung o {L}",
  f"Can them ban chup giay to cho ho so ly lich tu phap cua ban. Nop tai {L}"),
 "87be8227-d856-523e-a009-eaff65e697db": (
  f"Ban chua lam thu tuc tam tru theo quy dinh. Hoan tat o {L}",
  f"He thong ghi nhan ban thieu dang ky tam tru. Bo sung tai {L}"),
 "c724c608-390a-5968-9625-8c8798c3f4c9": (
  f"Da co ket qua xu ly khieu nai cua ban. Lay ban dien tu o {L}",
  f"Khieu nai cua ban da duoc giai quyet xong. Nhan van ban tai {L}"),
 "5907e7c0-3b86-567c-a7e8-7a8742555574": (
  f"Ban nam trong danh sach khao sat dich vu cong co ho tro. Tham gia o {L}",
  f"Moi ban lam khao sat dich vu cong de nhan phan ho tro. Vao {L}"),
 "1369b891-ee51-556e-b5e5-baa409eaf696": (
  f"Du lieu nguoi phu thuoc ban khai khong trung he thong. Sua o {L}",
  f"Phan nguoi phu thuoc trong ho so cua ban co sai lech. Dieu chinh tai {L}"),
 "7ea11682-0d89-5d29-818b-f42169b4ec52": (
  f"Buoi tiem cua ban duoc doi sang gio khac. Xac nhan o {L}",
  f"Lich hen tiem chung cua ban thay doi. Vao {L} chot lai."),
 "efd3ade1-c3ba-5169-ae10-23fefbfa339b": (
  f"Ho so nha dat dung ten ban con thieu giay to. Nop o {L} truoc han.",
  f"Co ho so dat dai mang ten ban dang cho hoan thien. Gui giay to tai {L}"),
 "24361112-28f5-5a5a-b35c-80a6e5f107c7": (
  f"Co quan quan ly vua gui ban mot thong bao hanh chinh. Xem o {L}",
  f"Ban co van ban hanh chinh moi can doc. Kiem tra tai {L}"),
 "e17d926d-e05d-5592-9dd7-3428db464fc6": (
  f"The bao hiem y te dien tu cua ban chua bat. Kich hoat o {L} de dung khi di kham.",
  f"Ban chua kich hoat so bao hiem y te dien tu. Lam tai {L} truoc khi den benh vien."),
 "018b06d9-3897-5e5e-9850-234692cf1262": (
  f"Chuong trinh ho tro tien dien cho ho dan dang nhan ho so. Nop o {L}",
  f"Ban co the dang ky nhan ho tro tien dien. Gui ho so tai {L}"),
 # ---- sms / tax ----
 "4534d5f1-f314-5c1f-ac36-9156004219a2": (
  f"Ban chua co tai khoan thue dien tu ca nhan. Mo o {L} keo bi phat cham dang ky.",
  f"He thong khong thay tai khoan thue dien tu cua ban. Tao tai {L} de tranh xu phat."),
 "f71d6d66-4737-5111-a508-226d57c36d79": (
  f"Ban co thu nhap tu hai nguon chi tra. Khai them o {L}",
  f"Du lieu cho thay ban nhan luong tu hai noi. Ke khai bo sung tai {L}"),
 "c3adcf04-eeb9-5805-a76b-f6987ffa7fbc": (
  f"Co hoa don dien tu gan ma so thue cua ban bi sai. Sua o {L}",
  f"Hoa don dien tu dung ma so thue cua ban co loi. Dieu chinh tai {L}"),
 "2708e367-47ae-5f6f-97ac-736b05d4aff2": (
  f"Chinh sach moi cho phep ban lui han nop thue. Dang ky o {L}",
  f"Ban thuoc dien duoc keo dai thoi han nop thue. Ghi danh tai {L}"),
 "6aa108c5-b059-54d6-aa5c-003f5b34d44c": (
  f"Ho so xin hoan thue cua ban da duoc thong qua. Nhan tien o {L}",
  f"Yeu cau hoan thue cua ban duoc chap thuan. Lam thu tuc nhan tai {L}"),
 "89746cde-230e-543f-86e6-c92084c582b2": (
  f"Nguoi phu thuoc cua ban chua co ma so thue. Dang ky o {L}",
  f"He thong chua cap ma so thue cho nguoi phu thuoc ban khai. Lam tai {L}"),
 "f3124969-2854-55e9-83fe-8390156b4325": (
  f"Ban bi phat do nop to khai tre. Xem so tien va tra o {L}",
  f"Co khoan tien phat cham to khai dung ten ban. Kiem tra va nop tai {L}"),
 "a42d230c-6b89-5ef3-8945-43a19709d44c": (
  f"Ban thuoc dien duoc giam thue theo chinh sach vua ban hanh. Dang ky o {L}",
  f"Muc giam thue moi ap dung voi truong hop cua ban. Ghi danh tai {L}"),
 "4846fe34-0de4-52fa-a5e1-0f24f9ea6c39": (
  f"To khai cua ban bi tra lai vi sai mot so muc. Chinh o {L} truoc 5 gio chieu.",
  f"Co chi tieu khai sai nen to khai cua ban khong duoc nhan. Sua tai {L}, han 17h."),
 "729276de-152c-5511-91cf-48dba2974302": (
  f"Ban can doi chieu chung tu khau tru thue. Tai file len o {L}",
  f"He thong doi ban nop lai chung tu khau tru de doi soat. Gui tai {L}"),
 # ---- sms / ecommerce ----
 "0684f9b1-536e-575b-923f-a7b76569b7e8": (
  f"Mot don chua thanh toan cua ban sap bi huy. Giu lai o {L}",
  f"Don hang dang cho tra tien cua ban gan het thoi gian. Vao {L} de giu."),
 "0fbd7bfd-f295-51b7-804c-83b9507eb14b": (
  f"Voucher cua ban chua vao don. Lay lai o {L}",
  f"Ma khuyen mai cua ban khong duoc tru. Nhan lai tai {L}"),
 "461efcea-11ca-5f61-bc89-dbe295636c3b": (
  f"San phat hien giao dich la tren tai khoan cua ban. Xac minh o {L}",
  f"Co hoat dong mua ban dang ngo tren tai khoan ban. Kiem tra tai {L}"),
 "e1874058-ce89-5931-afb7-b15ac2e00ac8": (
  f"Ban duoc moi viet danh gia san pham va nhan 200.000d. Tham gia o {L}",
  f"Chuong trinh tra 200.000d cho danh gia san pham dang mo voi ban. Vao {L}"),
 "9b31d8cd-040e-5abc-88aa-8d2ff98f4c80": (
  f"Yeu cau tra hang cua ban chua co anh. Them o {L} de duoc duyet.",
  f"Don hoan hang cua ban thieu hinh chup. Bo sung tai {L}"),
 "d408d4d4-63ad-537d-9a37-af52a8b96100": (
  f"Vi cua ban con tien chua rut ve. Khai tai khoan nhan o {L}",
  f"So du trong vi cua ban chua duoc chuyen di. Xac nhan noi nhan tai {L}"),
 "b8f4b4b8-2cef-58ce-8026-aef2c791254c": (
  f"Hang thanh vien cua ban vua len muc moi. Bat quyen loi o {L}",
  f"Ban da duoc len hang thanh vien. Kich hoat uu dai tai {L}"),
 "51a96064-f693-5b7e-a709-5f89530c83b9": (
  f"Hang dat truoc cua ban da ve. Chot thanh toan o {L}",
  f"Don dat truoc cua ban co hang. Xac nhan tra tien tai {L}"),
 "59a25a3c-f402-5b42-bd57-53ab7d8744a6": (
  f"Phien dau gia ban tham gia gan dong. Vao {L} dat them.",
  f"Cuoc dau gia cua ban sap het gio. Tiep tuc tai {L}"),
 "80b962a6-3ee2-52ab-8152-151c8b3de19f": (
  f"San gui ban mot phan qua sinh nhat. Nhan o {L} trong 2 ngay.",
  f"Qua mung sinh nhat danh cho ban dang cho. Lay tai {L} trong 48 tieng."),
 "d46b975c-b540-5407-becd-13a1972d232b": (
  f"Dia chi nhan hang mac dinh cua ban bi loi. Sua o {L}",
  f"He thong bao dia chi giao mac dinh cua ban khong dung. Cap nhat tai {L}"),
 # ---- sms / delivery ----
 "35334cc5-bf06-5e8c-bd1e-ef77c302cf2e": (
  f"Hang cua ban sap bi tra lai nguoi gui trong mot ngay. Giu o {L}",
  f"Buu kien cua ban se duoc hoan trong 24 tieng toi. Vao {L} de giu lai."),
 "5138b897-915c-5cbc-86da-e064f6f3199f": (
  f"Mot kien hang thu tien ho cua ban chua tra. Xu ly o {L}",
  f"Ban con don thu tien ho chua thanh toan. Giai quyet tai {L}"),
 "6ae7d62f-849e-5536-8552-dead8447d036": (
  f"Nguoi giao khong tim ra dia chi cua ban. Chi duong o {L}",
  f"Nhan vien giao hang bao khong den duoc dia chi ghi tren don. Huong dan lai tai {L}"),
 "4191b852-608d-5391-bdbf-da3af930cd75": (
  f"Do goi ban khong duoc nen don da bi huy. Mo lai o {L}",
  f"Don cua ban dung vi khong ket noi duoc voi ban. Kich hoat tai {L}"),
 "7acafca9-9092-5615-a2f8-af8bed5d90cd": (
  f"Hang cua ban can chu ky dien tu de giao. Ky o {L}",
  f"Nguoi nhan phai ky dien tu cho kien hang nay. Thuc hien tai {L}"),
 "1c33d108-120f-5638-a43f-a53d38527915": (
  f"Giao lai lan hai ton 20.000d. Tra o {L}",
  f"Ban can nop 20.000d phi giao lai. Thanh toan tai {L}"),
 "fe3e722b-3e7c-5487-a892-0b0510f86ebb": (
  f"Hang cua ban da nam o buu cuc ba ngay. Dat lich nhan o {L}",
  f"Buu pham cua ban luu tai buu cuc 3 hom nay. Chon lich giao tai {L}"),
 "5556b448-0c79-55b4-87f9-099be0634131": (
  f"Ma lay hang cua ban khong con dung. Xin ma khac o {L}",
  f"Ma nhan hang cua ban da qua han. Lay ma moi tai {L}"),
 "22aeedc8-0a30-50f4-8b94-2f691508f9da": (
  f"Hang nhap cua ban bi tinh thue. Nop o {L}",
  f"Don gui tu nuoc ngoai cua ban co khoan thue phai tra. Thanh toan tai {L}"),
 "15763583-568c-5371-984f-8977c92443a8": (
  f"He thong hoi ban con o dia chi cu khong. Tra loi o {L}",
  f"Can xac nhan dia chi nhan hang cua ban chua doi. Phan hoi tai {L}"),
 # ---- sms / telecom ----
 "abd08d2e-b448-5393-a910-cc354e58ba34": (
  f"Goi cuoc cua ban dung trong hom nay. Gia han o {L} de khong mat uu dai.",
  f"Hom nay la ngay cuoi cua goi cuoc ban dang dung. Gia han tai {L}"),
 "4ae00380-73cc-5956-a4d4-c1c4e3bc7926": (
  f"Mot hoa don cuoc cua ban chua duoc phat hanh. Xem o {L}",
  f"Ban con hoa don cuoc chua xuat. Kiem tra tai {L}"),
 "3b69f2d7-bb63-500f-af44-addc3c91524d": (
  f"Co nguoi xin cap lai SIM cua ban tu may khac. Chan o {L} neu khong phai ban.",
  f"He thong nhan yeu cau doi SIM cua ban tu thiet bi la. Tu choi tai {L}"),
 "add8d089-0f4b-5da2-9941-fe8e87a801fc": (
  f"Dang ky lay so dep khong mat phi o {L}",
  f"Ban co the nhan mot so dep mien phi. Dang ky tai {L}"),
 "fe887893-a31f-547c-a114-849747fbc304": (
  f"Ban nop thua 30.000d cuoc va duoc tra lai. Nhan o {L}",
  f"Co khoan 30.000d cuoc thua cua ban cho hoan. Lay tai {L}"),
 "b90de912-2ba2-5421-8a26-e7614a56b02e": (
  f"Ban chua bat chuyen vung quoc te. Kich hoat o {L} truoc khi di.",
  f"Chuyen vung tren thue bao cua ban dang tat. Mo tai {L} truoc chuyen di."),
 "43fbfb7f-d94f-5c59-85d8-6e49d6855340": (
  f"Goi cuoc ban dung da cu, co goi re hon tai {L}",
  f"Ban co the chuyen sang goi tiet kiem hon o {L}"),
 "d32fdbc4-3751-530f-9486-234b68fbd662": (
  f"Ten chu thue bao khong khop giay to. Cap nhat o {L}",
  f"Thong tin dang ky SIM cua ban lech so voi giay to. Sua tai {L}"),
 # ---- sms / social ----
 "2a5bd21a-80cf-5bc0-a32c-c4562a2eaa86": (
  f"Tai khoan ban bi gioi han vi khieu nai ban quyen. Khang nghi o {L}",
  f"Co bao cao ban quyen khien tai khoan ban bi han che. Phan hoi tai {L}"),
 "89ef44fd-6738-5db6-b0d4-115489621540": (
  f"Mot trang chinh thuc vua nhan tin cho ban. Doc o {L}",
  f"Ban co tin nhan dang cho tu trang da xac minh. Xem tai {L}"),
 "8f3a338c-f0e7-5472-b5a6-fe1ebf566717": (
  f"Nhom ban quan ly sap bi xoa vi nhieu bao cao. Phan hoi o {L}",
  f"Do bi bao cao lien tuc, nhom cua ban co the bi go. Tra loi tai {L}"),
 "7603296e-9046-5fc7-a9ae-1792473ca4b4": (
  f"Chuong trinh tra tien cho nguoi sang tao dang moi ban. Dang ky o {L}",
  f"Ban duoc mo suat tham gia chuong trinh noi dung co thu nhap. Ghi danh tai {L}"),
 "a0a399f4-464a-5a76-8c75-53a01bfff3de": (
  f"Hinh cua ban dang bi trang khac su dung. Xem o {L}",
  f"Co trang khac dung anh cua ban. Kiem tra tai {L}"),
 "3edf2945-b71d-516a-ac40-0ddefe2048d3": (
  f"Mat khau cua ban nam trong mot vu lo du lieu. Doi o {L} ngay.",
  f"Chung toi thay mat khau ban dung trong danh sach bi ro ri. Doi tai {L}"),
 "f5ab9d68-009f-573d-99d6-b01ea195fe44": (
  f"Xac thuc so dien thoai o {L} de duoc binh luan tro lai.",
  f"De mo lai quyen binh luan, ban can xac minh so dien thoai tai {L}"),
 "52bd1f89-5630-5d43-b093-23d1b2466994": (
  f"Ban dang co 3 loi ket ban va mot canh bao an toan. Xem o {L}",
  f"Co 3 yeu cau ket ban cho ban, kem 1 canh bao bao mat. Kiem tra tai {L}"),
 # ---- email / bank ----
 "87f0f763-b8db-5372-8bfa-431a61f5ddbd": (
  f"Kinh gui Quy khach, tai khoan cua Quy khach ghi nhan ba lan nhap sai ma xac thuc lien tiep. De nghi Quy khach thuc hien mo khoa tai {L} truoc khi dich vu bi tam ngung.",
  f"Thong bao: he thong dem duoc ba lan nhap sai ma xac thuc tren tai khoan cua Quy khach. Vui long go khoa tai {L} de dich vu khong bi gian doan."),
 "54a266ef-7ab3-58a5-a0ec-977820610584": (
  f"Kinh gui Quy khach, the cua Quy khach vua phat sinh giao dich tai mot diem chap nhan o nuoc ngoai. Neu day khong phai Quy khach, de nghi tu choi tai {L}.",
  f"Thong bao an ninh: chung toi ghi nhan mot lan quet the cua Quy khach o ngoai nuoc. Truong hop Quy khach khong thuc hien, vui long huy tai {L}."),
 "85dc08bf-2669-578e-9f50-d11328cc69e4": (
  f"Kinh gui Quy khach, sau canh bao rui ro, han muc chuyen khoan cua Quy khach da bi ha xuong. De duoc khoi phuc muc cu, de nghi Quy khach xac minh tai {L}.",
  f"Thong bao: muc chuyen tien toi da tren tai khoan cua Quy khach da duoc giam theo canh bao an toan. Vui long xac nhan tai {L} de phuc hoi."),
 "fa94c584-5dda-5200-b12a-2cf1a588b664": (
  f"Kinh gui Quy khach, khoan vay cua Quy khach den ky tra trong hom nay. De nghi Quy khach doi chieu lich thanh toan tai {L}.",
  f"Thong bao: hom nay la han thanh toan khoan vay cua Quy khach. Vui long kiem tra lich tra no tai {L}."),
 "ec398dbb-3ae2-5669-84b0-fe9c330c86a7": (
  f"Kinh gui Quy khach, giay to dinh danh cua Quy khach da qua han cap nhat, dich vu ngan hang so co the bi tam ngung. De nghi Quy khach bo sung tai {L}.",
  f"Thong bao: ho so dinh danh cua Quy khach chua duoc cap nhat dung han va kenh truc tuyen co the bi dung. Vui long nop bo sung tai {L}."),
 "9a4fc879-f045-52a9-9844-c35366f94837": (
  f"Kinh gui Quy khach, mot giao dich 4.900.000 VND tu thiet bi chua dang ky dang cho phe duyet. De nghi Quy khach duyet hoac tu choi tai {L}.",
  f"Thong bao: co khoan 4.900.000 VND cho xac nhan, phat sinh tu mot may chua tung su dung. Vui long xu ly tai {L}."),
 "913d4d09-1eb4-5925-b0c0-67c93cd09a87": (
  f"Kinh gui Quy khach, Quy khach nam trong dien duoc tham gia goi mien phi chuyen khoan tron doi. De nghi Quy khach kich hoat tai {L} trong hom nay.",
  f"Thong bao: goi khong thu phi chuyen khoan tron doi dang mo cho tai khoan cua Quy khach. Vui long bat tai {L} trong ngay."),
 "4094fe01-0e13-5b03-b50a-024041783359": (
  f"Kinh gui Quy khach, tai khoan cua Quy khach chua cai dat sinh trac hoc nen cac giao dich gia tri lon se bi tu choi. De nghi Quy khach hoan tat tai {L}.",
  f"Thong bao: buoc xac thuc sinh trac hoc tren tai khoan cua Quy khach chua duoc thiet lap, giao dich lon se khong thuc hien duoc. Vui long bo sung tai {L}."),
 # ---- email / gov ----
 "0a8f4aad-9ba9-5895-8e0f-8b2f793dac17": (
  f"Kinh gui Quy vi, giay phep da cap cho Quy vi sap het hieu luc. De nghi Quy vi lam thu tuc gia han qua mang tai {L}.",
  f"Thong bao: thoi han hieu luc giay phep cua Quy vi sap ket thuc. Vui long gia han truc tuyen tai {L}."),
 "e1fe09a1-7e1f-5f91-8aa3-891dc86b24d3": (
  f"Kinh gui Quy vi, co quan chuc nang co giay moi Quy vi len lam viec. Noi dung duoc dang tai {L}.",
  f"Thong bao: Quy vi nhan duoc mot giay moi lam viec. Vui long doc chi tiet tai {L}."),
 "04a162c9-dee9-5020-a17b-3bf92a8957b5": (
  f"Kinh gui Quy vi, dot ra soat du lieu dan cu yeu cau Quy vi xac nhan lai so dinh danh. De nghi Quy vi thuc hien tai {L} truoc 16 gio.",
  f"Thong bao: Quy vi can khai xac nhan so dinh danh ca nhan phuc vu dot ra soat. Vui long hoan tat tai {L} truoc 4 gio chieu."),
 "dc83d845-fd27-5e74-b3b2-e3f97d3aab2f": (
  f"Kinh gui Quy vi, ho so cap phieu ly lich tu phap cua Quy vi thieu ban chup giay to tuy than. De nghi Quy vi bo sung tai {L}.",
  f"Thong bao: ho so ly lich tu phap cua Quy vi chua co anh giay to. Vui long gui bo sung tai {L}."),
 "32172c61-fe50-5986-8ec8-9d4ec0d2ac06": (
  f"Kinh gui Quy vi, kien nghi cua Quy vi da duoc giai quyet xong. Ban dien tu duoc phat hanh tai {L}.",
  f"Thong bao: da co ket qua xu ly kien nghi cua Quy vi. Vui long nhan van ban dien tu tai {L}."),
 "cd4bf3f3-2b67-5511-b623-36a063497d74": (
  f"Kinh gui Quy vi, thong tin nguoi phu thuoc trong ho so cua Quy vi khong khop du lieu quan ly. De nghi Quy vi dieu chinh tai {L}.",
  f"Thong bao: phan nguoi phu thuoc Quy vi da khai co sai lech so voi he thong. Vui long sua tai {L}."),
 "ff13f544-0889-5628-a0b9-7b060bb65c3d": (
  f"Kinh gui Quy vi, ho so dat dai mang ten Quy vi con cho bo sung giay to. De nghi Quy vi nop tai {L} truoc thoi han.",
  f"Thong bao: ho so nha dat dung ten Quy vi chua day du thanh phan. Vui long gui giay to tai {L} dung han."),
 # ---- email / tax ----
 "9562fc40-8be9-5772-a663-dc424458ce72": (
  f"Kinh gui Nguoi nop thue, he thong chua ghi nhan tai khoan thue dien tu ca nhan cua Quy vi. De nghi Quy vi khoi tao tai {L} de tranh bi phat cham dang ky.",
  f"Thong bao: Quy vi chua co tai khoan thue dien tu ca nhan tren he thong. Vui long tao tai {L} de khong bi xu phat."),
 "a210399a-9049-53e5-9b2a-27725666160a": (
  f"Kinh gui Quy vi, du lieu cho thay Quy vi nhan thu nhap tu hai don vi chi tra trong ky. De nghi Quy vi ke khai bo sung tai {L}.",
  f"Thong bao: he thong ghi nhan Quy vi co thu nhap tu hai nguon. Vui long khai bo sung tai {L}."),
 "97592568-4eb5-5cea-ac0d-e27e1999cff2": (
  f"Kinh gui Quy vi, hoa don dien tu gan ma so thue cua Quy vi co sai sot chi tieu. De nghi Quy vi dieu chinh tai {L}.",
  f"Thong bao: co hoa don dien tu mang ma so thue cua Quy vi bi sai. Vui long sua tai {L}."),
 "4ed1d0ee-f249-5e9b-8353-3e8350ab1326": (
  f"Kinh gui Quy vi, Quy vi nam trong dien duoc keo dai thoi han nop thue theo chinh sach moi. De nghi Quy vi dang ky tai {L}.",
  f"Thong bao: chinh sach moi cho phep Quy vi lui han nop thue. Vui long ghi danh tai {L}."),
 "1d3a6c30-dde1-536b-af6c-943a623b7933": (
  f"Kinh gui Quy vi, ho so hoan thue cua Quy vi da duoc thong qua. De nghi Quy vi hoan tat thu tuc nhan tien tai {L}.",
  f"Thong bao: yeu cau hoan thue cua Quy vi duoc chap thuan. Vui long lam thu tuc nhan tai {L}."),
 "06b4a3fa-2331-5d22-aeb0-651815d72142": (
  f"Kinh gui Quy vi, to khai cua Quy vi bi tra ve do sai chi tieu bat buoc. De nghi Quy vi chinh sua tai {L} truoc 17 gio.",
  f"Thong bao: co chi tieu khai sai nen to khai cua Quy vi khong duoc tiep nhan. Vui long sua tai {L} truoc 5 gio chieu."),
 "e7ce6d52-7590-5582-9ef8-a9e0ce374822": (
  f"Kinh gui Quy vi, he thong yeu cau Quy vi doi chieu chung tu khau tru thue. De nghi Quy vi tai len tai {L}.",
  f"Thong bao: can doi soat chung tu khau tru cua Quy vi. Vui long gui file tai {L}."),
 # ---- email / ecommerce ----
 "f57b99d1-ad6c-54b5-ae68-3e8525ff25d7": (
  f"Kinh gui Quy khach, Quy khach co mot don cho thanh toan sap bi huy tu dong. De nghi Quy khach giu don tai {L}.",
  f"Thong bao: don chua thanh toan cua Quy khach sap het thoi gian giu. Vui long xu ly tai {L}."),
 "52686dbc-fcc3-50ca-b32a-1447ff73bb95": (
  f"Kinh gui Quy khach, ma giam gia cua Quy khach chua duoc tru vao don gan nhat. De nghi Quy khach nhan lai tai {L}.",
  f"Thong bao: phieu giam gia cua Quy khach khong duoc ap dung cho don vua roi. Vui long lay lai tai {L}."),
 "ef2cbe0d-54f8-5984-81a9-7e96ec7468f5": (
  f"Kinh gui Quy khach, he thong ghi nhan hoat dong bat thuong tren tai khoan mua sam cua Quy khach. De nghi Quy khach xac minh tai {L}.",
  f"Thong bao an ninh: co giao dich dang ngo tren tai khoan mua hang cua Quy khach. Vui long kiem tra tai {L}."),
 "0d0e6966-d732-53d6-8546-7241cc568a9e": (
  f"Kinh gui Quy khach, Quy khach duoc moi viet danh gia san pham va nhan phieu 200.000 VND. De nghi Quy khach tham gia tai {L}.",
  f"Thong bao: chuong trinh tang phieu 200.000 VND cho danh gia san pham dang mo voi Quy khach. Vui long dang ky tai {L}."),
 "aaf58047-512d-5508-959c-87f8c7720340": (
  f"Kinh gui Quy khach, yeu cau hoan hang cua Quy khach chua co hinh anh san pham. De nghi Quy khach bo sung tai {L} de duoc duyet.",
  f"Thong bao: don tra hang cua Quy khach thieu anh chup. Vui long them tai {L} de duoc xu ly."),
 "e73727ac-11e6-522b-a88b-0ab3bf67d8d8": (
  f"Kinh gui Quy khach, vi cua Quy khach dang con so du chua rut. De nghi Quy khach xac nhan tai khoan nhan tien tai {L}.",
  f"Thong bao: so du trong vi cua Quy khach chua duoc chuyen di. Vui long khai tai khoan nhan tai {L}."),
 "be3188f6-749d-5cf9-9890-c33681edc348": (
  f"Kinh gui Quy khach, hang thanh vien cua Quy khach vua duoc nang len muc moi. De nghi Quy khach kich hoat quyen loi tai {L}.",
  f"Thong bao: Quy khach da len hang thanh vien. Vui long bat cac uu dai di kem tai {L}."),
 # ---- email / delivery ----
 "8bfd70dd-f3c3-5251-9dfc-18ea57a524d8": (
  f"Kinh gui Quy khach, buu kien cua Quy khach se duoc tra lai nguoi gui trong 24 gio toi. De nghi Quy khach giu hang tai {L}.",
  f"Thong bao: hang cua Quy khach sap duoc hoan ve ben gui trong mot ngay. Vui long giu lai tai {L}."),
 "392b5648-5852-5c81-a90c-c38faef62a41": (
  f"Kinh gui Quy khach, Quy khach co mot kien hang thu tien ho chua duoc thanh toan. De nghi Quy khach xu ly tai {L}.",
  f"Thong bao: don thu tien ho cua Quy khach chua tra tien. Vui long giai quyet tai {L}."),
 "859d74ec-5790-5d22-9045-70be580962fd": (
  f"Kinh gui Quy khach, nhan vien giao hang khong tim thay dia chi ghi tren don. De nghi Quy khach bo sung chi dan tai {L}.",
  f"Thong bao: nguoi giao hang bao khong den duoc dia chi cua Quy khach. Vui long huong dan them tai {L}."),
 "78cb24eb-b7b1-553a-b8e3-4170833c2933": (
  f"Kinh gui Quy khach, kien hang cua Quy khach can chu ky dien tu cua nguoi nhan de ban giao. De nghi Quy khach ky tai {L}.",
  f"Thong bao: hang cua Quy khach chi duoc giao khi co chu ky dien tu. Vui long thuc hien tai {L}."),
 "e4ee3742-58e3-535e-aae6-c67765bb88ee": (
  f"Kinh gui Quy khach, don gui tu nuoc ngoai cua Quy khach phat sinh khoan thue nhap khau. De nghi Quy khach nop tai {L}.",
  f"Thong bao: kien hang quoc te cua Quy khach bi tinh thue. Vui long thanh toan tai {L}."),
 # ---- email / telecom ----
 "d33d4cec-ae3c-58b4-bdc1-7a17e64f7d24": (
  f"Kinh gui Quy khach, goi cuoc cua Quy khach ket thuc trong hom nay. De nghi Quy khach gia han tai {L} de giu uu dai dang co.",
  f"Thong bao: hom nay la ngay cuoi cua goi cuoc Quy khach dang dung. Vui long gia han tai {L}."),
 "a6cda4af-4c25-5012-a43b-a7c285a7da94": (
  f"Kinh gui Quy khach, Quy khach co mot hoa don cuoc chua duoc phat hanh. De nghi Quy khach xem chi tiet tai {L}.",
  f"Thong bao: con mot hoa don cuoc cua Quy khach chua xuat. Vui long kiem tra tai {L}."),
 "f176e3fd-8fe7-5d92-84c1-fbde2128a6fd": (
  f"Kinh gui Quy khach, he thong nhan yeu cau cap lai SIM cho thue bao cua Quy khach tu mot may khac. Neu khong phai Quy khach, de nghi chan tai {L}.",
  f"Thong bao: co de nghi doi SIM cua Quy khach gui tu thiet bi la. Truong hop khong phai Quy khach, vui long tu choi tai {L}."),
 "c1d04016-61d5-5180-846a-a0daecbc0955": (
  f"Kinh gui Quy khach, thue bao cua Quy khach nop thua 30.000 VND cuoc va duoc hoan lai. De nghi Quy khach nhan tai {L}.",
  f"Thong bao: co khoan 30.000 VND cuoc thua cua Quy khach dang cho hoan. Vui long lay tai {L}."),
 "5a919748-000b-53b1-bea1-9d1783c39583": (
  f"Kinh gui Quy khach, thong tin chu the SIM cua Quy khach chua khop voi giay to. De nghi Quy khach cap nhat tai {L}.",
  f"Thong bao: ten chu thue bao cua Quy khach lech so voi giay to dang ky. Vui long sua tai {L}."),
 # ---- email / social ----
 "4c4bbaf5-65e9-59ba-a38a-3346594c05b8": (
  f"Kinh gui Quy khach, tai khoan cua Quy khach bi gioi han sau mot khieu nai ban quyen noi dung. De nghi Quy khach khang nghi tai {L}.",
  f"Thong bao: co bao cao ban quyen khien tai khoan cua Quy khach bi han che. Vui long phan hoi tai {L}."),
 "d3e71303-2d10-5848-bea7-a51460628dbb": (
  f"Kinh gui Quy khach, Quy khach co tin nhan dang cho tu mot trang da duoc xac minh. De nghi Quy khach xem tai {L}.",
  f"Thong bao: mot trang chinh thuc vua gui tin nhan cho Quy khach. Vui long doc tai {L}."),
 "ac1933dd-09a1-5c23-b92b-4284c8d4fed4": (
  f"Kinh gui Quy khach, nhom do Quy khach quan tri co the bi go sau nhieu luot bao cao. De nghi Quy khach phan hoi tai {L}.",
  f"Thong bao: nhom cua Quy khach dang bi bao cao lien tuc va co nguy co bi xoa. Vui long tra loi tai {L}."),
 "82394628-d692-564f-824d-95e2c3922740": (
  f"Kinh gui Quy khach, Quy khach duoc moi tham gia chuong trinh sang tao noi dung co chia se doanh thu. De nghi Quy khach dang ky tai {L}.",
  f"Thong bao: chuong trinh chia se doanh thu cho nguoi sang tao dang mo voi Quy khach. Vui long ghi danh tai {L}."),
 "caa8f772-be7a-55f9-9f00-1f39950ade46": (
  f"Kinh gui Quy khach, mat khau cua Quy khach xuat hien trong mot danh sach du lieu bi ro ri. De nghi Quy khach doi mat khau tai {L}.",
  f"Thong bao: chung toi phat hien mat khau cua Quy khach trong mot vu lo du lieu. Vui long thay doi tai {L}."),
}
