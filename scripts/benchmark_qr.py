#!/usr/bin/env python3
"""
benchmark_qr.py — Decode Failure Rate of QR decoders on the synthetic adversarial set.

Reads the ground truth gen_synthetic_qr.py wrote and asks each decoder to read each image. A
decode COUNTS ONLY IF THE PAYLOAD MATCHES: a decoder that returns something else has failed, and
counting "returned a string" as success would flatter every library here. The URL is not stored in
the index -- only its SHA1 -- so the check is against the hash, which also means the index can be
published without republishing the corpus.

Three decoders, because they fail differently and a defender picks one:
  opencv    cv2.QRCodeDetector, the default in most pipelines
  pyzbar    the zbar binding, what the collectors on the Jetsons use
  wechat    cv2.wechat_qrcode, a CNN-assisted detector from opencv-contrib

RUN
    python3 scripts/benchmark_qr.py --dir data/raw/qr_synth
    python3 scripts/benchmark_qr.py --dir data/raw/qr_synth --decoders pyzbar,wechat
"""
from __future__ import annotations
import argparse, csv, hashlib, os, sys, time

import numpy as np


def load_decoders(names: list) -> dict:
    out = {}
    if "opencv" in names:
        import cv2
        det = cv2.QRCodeDetector()

        def _cv(path):
            img = cv2.imread(path)
            if img is None:
                return []
            try:
                ok, infos, _, _ = det.detectAndDecodeMulti(img)
                if ok:
                    return [t for t in infos if t]
            except cv2.error:
                pass
            try:
                t, _, _ = det.detectAndDecode(img)
                return [t] if t else []
            except cv2.error:
                return []
        out["opencv"] = _cv
    if "pyzbar" in names:
        # zbar is a C library with a Python binding, so this import fails on any machine without
        # the shared object -- the workstation, for one, where the restoration network trains. It
        # used to raise and take the whole run with it, which meant a two-decoder run was
        # impossible on a machine that has two decoders. Skip and say so, exactly as wechat does.
        try:
            from pyzbar.pyzbar import decode as zdec
            from PIL import Image
        except (ImportError, OSError) as e:
            print(f"[!] pyzbar requested but not usable ({e}) — skipping", file=sys.stderr)
        else:
            def _pz(path):
                try:
                    return [d.data.decode("utf-8", "replace") for d in zdec(Image.open(path))]
                except Exception:
                    return []
            out["pyzbar"] = _pz
    if "wechat" in names:
        import cv2
        if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            print("[!] wechat requested but opencv-contrib is not installed — skipping",
                  file=sys.stderr)
        else:
            wd = cv2.wechat_qrcode_WeChatQRCode()

            def _wx(path):
                img = cv2.imread(path)
                if img is None:
                    return []
                try:
                    res, _ = wd.detectAndDecode(img)
                    return [t for t in res if t]
                except cv2.error:
                    return []
            out["wechat"] = _wx
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("data", "raw", "qr_synth"))
    ap.add_argument("--decoders", default="opencv,pyzbar,wechat")
    ap.add_argument("--out", default=os.path.join("data", "processed", "qr", "qr_dfr.csv"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (a smoke test)")
    a = ap.parse_args()

    idx = os.path.join(a.dir, "index.csv")
    if not os.path.isfile(idx):
        print(f"[!] {idx} not found — run gen_synthetic_qr.py first", file=sys.stderr)
        return 1
    with open(idx, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[:a.limit]

    decs = load_decoders([x.strip() for x in a.decoders.split(",") if x.strip()])
    if not decs:
        print("[!] no decoder available", file=sys.stderr)
        return 1
    print(f"[*] {len(rows):,} images x {len(decs)} decoder(s): {', '.join(decs)}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        # `calibration` rides along so a recalibrated set can never be pooled with the frozen
        # one by a reader who only has the CSV. Blank on any index written before it existed.
        w = csv.DictWriter(fh, fieldnames=["sample_id", "label", "tld", "url_len", "ec_level",
                                           "box_size", "transform", "strength", "calibration",
                                           "qr_version", "modules", "decoder", "decoded",
                                           "correct"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            path = os.path.join(a.dir, r["image"])
            want = r["url_sha1"]
            for name, fn in decs.items():
                got = fn(path)
                # correct = at least one payload whose SHA1 prefix matches the source URL's
                ok = any(hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()[:16] == want
                         for t in got)
                w.writerow({**{k: r.get(k, "") for k in ("sample_id", "label", "tld", "url_len",
                                                        "ec_level", "box_size", "transform",
                                                        "strength", "calibration", "qr_version",
                                                        "modules")},
                            "decoder": name, "decoded": int(bool(got)), "correct": int(ok)})
            if i % 2000 == 0:
                el = time.time() - t0
                print(f"    [{i:,}/{len(rows):,}] {el:.0f}s, {i/max(el,1):.0f} img/s")
    print(f"[+] {a.out}  ({time.time()-t0:.0f}s)")

    # A summary on stdout, because the point of the run is the shape of the failure and a CSV
    # nobody reads is not that.
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    with open(a.out, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = (r["decoder"], r["transform"])
            agg[k][0] += 1
            agg[k][1] += int(r["correct"])
    print(f"\n  {'decoder':<9} {'transform':<12} {'n':>7}  DFR")
    for (dec, tr), (n, ok) in sorted(agg.items()):
        print(f"  {dec:<9} {tr:<12} {n:>7}  {100*(1-ok/n):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
