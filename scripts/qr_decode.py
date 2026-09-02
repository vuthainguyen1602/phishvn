#!/usr/bin/env python3
"""
qr_decode.py — Decode QR-code images (quishing) into URLs -> CSV, ready for the URL pipeline.

INSTALL (either backend):  pip install opencv-python   OR   pip install pyzbar pillow
RUN:
  python qr_decode.py --dir data/raw/qr_images --out data/raw/tinnhiemmang/qr_urls.csv --label phishing
"""
from __future__ import annotations
import argparse, csv, glob, os


def decode_opencv(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return []
    det = cv2.QRCodeDetector()
    try:
        ok, infos, _, _ = det.detectAndDecodeMulti(img)
        if ok:
            return [t for t in infos if t]
    except Exception:
        pass
    t, _, _ = det.detectAndDecode(img)
    return [t] if t else []


def decode_pyzbar(path):
    from pyzbar.pyzbar import decode
    from PIL import Image
    return [d.data.decode("utf-8", "ignore") for d in decode(Image.open(path))]


def get_decoder():
    try:
        import cv2
        return decode_opencv, "opencv"
    except Exception:
        pass
    try:
        import pyzbar
        return decode_pyzbar, "pyzbar"
    except Exception:
        raise SystemExit("Install opencv-python or pyzbar+pillow to decode QR images.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder of QR images (png/jpg)")
    ap.add_argument("--out", default="data/raw/tinnhiemmang/qr_urls.csv")
    ap.add_argument("--label", default="phishing", choices=["phishing", "benign"])
    args = ap.parse_args()

    decode, backend = get_decoder()
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files += glob.glob(os.path.join(args.dir, "**", ext), recursive=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fields = ["domain", "detected_date", "impersonated_org", "status", "type", "qr_decoded_url",
              "source_image", "label", "page", "scraped_at"]
    n = 0
    from datetime import date
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p in files:
            for url in decode(p):
                url = url.strip()
                if not url:
                    continue
                dom = url.split("//")[-1].split("/")[0]
                w.writerow({"domain": dom, "detected_date": date.today().strftime("%d/%m/%Y"),
                            "impersonated_org": "", "status": "", "type": "qr",
                            "qr_decoded_url": url, "source_image": os.path.basename(p),
                            "label": args.label, "page": "", "scraped_at": ""})
                n += 1
    print(f"Decoded {n} QR URLs ({backend}) from {len(files)} images -> {args.out}")


if __name__ == "__main__":
    main()
