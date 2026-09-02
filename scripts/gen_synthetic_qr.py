#!/usr/bin/env python3
"""
gen_synthetic_qr.py — the adversarial half of the quishing study: synthesise QR codes from the
PhishVN corpus, degrade them the way an attacker or a bad photograph would, and measure what the
decoders can still read.

WHY SYNTHETIC. The in-the-wild half cannot answer this. Measured 2026-08-30 over 1,047 pages, a
Vietnamese phishing landing page carries a decodable QR essentially never (one page, and its code
was a legitimate Zalo contact link), because quishing travels on posters and e-mails and POINTS AT
the landing page rather than living on it. Decoder robustness, by contrast, is a property of the
image and the library, so it can be measured exactly -- every label is known by construction,
because we made the code.

WHAT IT MEASURES. Decode Failure Rate per (transformation, strength, error-correction level,
decoder). The interesting quantity is not "can a decoder read a clean QR" -- they all can -- but
the band where a human phone still scans and an automated pipeline no longer does, because that is
the gap an attacker lives in. This script produces the images and the ground truth; benchmark_qr.py
reads them.

SCALE, and why the default is not the whole corpus. 53,116 URLs x 4 error-correction levels is
212,464 base images before a single transformation, and x9 degradations is 1.9M. What actually
drives decode failure is the transformation and the module density -- and density is a function of
URL LENGTH, which is why the sample is stratified by length rather than drawn at random. A
stratified 2,000 covers the length range as faithfully as 53,116 and costs 1% of the compute; the
full corpus is available with --all for a final run, and the sampling is seeded so either is
reproducible.

TWO PRODUCTS, and they want opposite things from this script. The BENCHMARK wants the full
factorial and no pixels kept (--stream); the RESTORATION set wants pixels kept, one module size,
and only the transforms an image-to-image network could possibly invert. Both are below.

RUN (on the second Jetson: it is idle, and it has all three decoders)
    python3 scripts/gen_synthetic_qr.py --n 2000 --out data/raw/qr_synth
    python3 scripts/gen_synthetic_qr.py --all --out data/raw/qr_synth   # 212k base images

    # the benchmark, as registered and as run to completion on 2026-08-30 (792,000 renders)
    python3 scripts/gen_synthetic_qr.py --n 2000 --box 2,3,4 --stream

    # the restoration training set: pixels kept, wild render scale, photometric transforms only.
    # 200 URLs x 4 EC x 25 variants = 20,000 renders and 94 MB, which is the pilot that ran.
    python3 scripts/gen_synthetic_qr.py --n 200 --box 2 --ec L,M,Q,H \
        --transforms blur,motion,saltpepper,contrast,invert,logo --calibration modules \
        --out data/raw/qr_restore
"""
from __future__ import annotations
import argparse, csv, hashlib, io, os, random, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import qrcode
from qrcode.constants import (ERROR_CORRECT_L, ERROR_CORRECT_M,
                              ERROR_CORRECT_Q, ERROR_CORRECT_H)

CORPUS = os.path.join("data", "processed", "dataset_url.csv")
EC = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}
# label/tld/tier ride along from the corpus. The decode question is about the image, but the
# study's question is about a PHISHING pipeline, and phishing URLs are longer than benign ones in
# this corpus -- longer URL, higher QR version, denser modules, and density is what a degradation
# has to overwhelm. Without the label the run could not tell a decoder weakness from a length
# effect, and re-deriving it later would mean re-joining on a hash that deliberately does not carry
# the URL.
FIELDS = ["sample_id", "url_sha1", "url_len", "label", "tld", "tier", "ec_level", "box_size",
          "px_per_module", "transform", "strength", "calibration", "qr_version", "modules",
          "image"]


# ---------------------------------------------------------------- transformations
# Each returns a NEW image. Strength is 0..1 within a band chosen so that the LOW end is plainly
# readable and the HIGH end is plainly not: a sweep that never fails measures nothing, and one that
# always fails measures nothing either. The bands were set by eye on a pilot, and the sweep reports
# where the break actually falls rather than assuming these are the right numbers.
#
# EVERY TRANSFORM TAKES `px`, THE RENDER SCALE, AND MOST IGNORE IT. That is the point of passing it:
# the ones that ignore it are the ones whose magnitude is already relative -- `saltpepper` is a
# fraction of area, `rotate` and `perspective` are geometric, `logo` is a fraction of the symbol --
# and they behave gradedly across the sweep. The two that were written in absolute pixels, `blur`
# and `motion`, saturated at 100% for all three decoders at every strength and every module size,
# which is a property of the calibration and not of the decoders (deviation record, PREREG,
# 2026-08-30). `cal="modules"` expresses those two in module widths instead. It is OPT-IN and the
# `px` default is byte-identical to the frozen grid, because the completed 792,000-render sweep is
# bound to that grid and a recalibrated run is a different study, not a correction folded into it.

def t_clean(img, s, rng, px=2, cal="px"):
    return img


def t_logo(img, s, rng, px=2, cal="px"):
    """Brand overlay in the middle, the tactic a VietQR or bank-spoof code uses. Covers s x 30% of
    the module area, which at the high end exceeds what even level-H redundancy can reconstruct."""
    w, h = img.size
    side = int(w * (0.08 + 0.22 * s))
    box = Image.new("RGB", (side, side), "white")
    d = ImageDraw.Draw(box)
    d.rectangle([2, 2, side - 3, side - 3], outline=(200, 30, 40), width=max(2, side // 12))
    d.ellipse([side // 4, side // 4, 3 * side // 4, 3 * side // 4], fill=(0, 90, 170))
    img = img.copy()
    img.paste(box, ((w - side) // 2, (h - side) // 2))
    return img


def t_blur(img, s, rng, px=2, cal="px"):
    """Gaussian: a camera that did not focus.

    `px` band: 0.5 to 8.0 px. At 2 px per module the weakest setting is already 1.2 modules wide,
    which is why this transform has no strength axis in the frozen sweep.
    `modules` band: 0.10 to 0.65 module widths, so the weakest setting blurs within a module and
    the strongest across one."""
    r = (0.5 + 7.5 * s) if cal == "px" else px * (0.10 + 0.55 * s)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def t_motion(img, s, rng, px=2, cal="px"):
    """Motion: a camera that moved. A directional kernel, not a symmetric one -- they break a QR
    differently, and a symmetric blur would be the previous transformation twice.

    `px` band: 3 to 25 px, i.e. 1.5 to 12.5 modules at the wild render scale -- destroyed at every
    setting. `modules` band: 0.25 to 3.0 module widths.

    A FLOOR THE RECALIBRATION CANNOT REMOVE: a convolution kernel is at least 3 px, which at 2 px
    per module is 1.5 modules however the band is expressed. At box 2 the weakest motion setting is
    therefore not weak, and any gradient this transform shows at that scale is bounded by the
    raster and not by the calibration. Stated here because it is the kind of limit that otherwise
    gets read as a decoder result."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    kpx = (3 + 22 * s) if cal == "px" else px * (0.25 + 2.75 * s)
    k = max(3, int(kpx)) | 1
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i in range(k):
        out += ker[k // 2, i] * p[pad:pad + a.shape[0], i:i + a.shape[1]]
    return Image.fromarray(out.astype(np.uint8)).convert("RGB")


def t_saltpepper(img, s, rng, px=2, cal="px"):
    """Print or sensor noise. Applied to the binary image, so it flips modules outright."""
    a = np.asarray(img.convert("L")).copy()
    n = int(a.size * 0.30 * s)
    if n:
        ys = rng.integers(0, a.shape[0], n)
        xs = rng.integers(0, a.shape[1], n)
        a[ys, xs] = np.where(rng.random(n) < 0.5, 0, 255)
    return Image.fromarray(a).convert("RGB")


def t_rotate(img, s, rng, px=2, cal="px"):
    """In-plane rotation. Decoders correct small angles; the question is where they stop."""
    return img.rotate(45.0 * s, resample=Image.BICUBIC, fillcolor=(255, 255, 255), expand=True)


def t_perspective(img, s, rng, px=2, cal="px"):
    """3D tilt: a code photographed off-axis, off a poster or a screen."""
    w, h = img.size
    d = int(min(w, h) * 0.35 * s)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(d, int(d * 0.4)), (w - d, 0), (w, h), (0, h - int(d * 0.3))]
    # PIL wants the INVERSE map, solved as an 8-parameter least squares
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    coeffs = np.linalg.lstsq(np.asarray(A, np.float64), np.asarray(B, np.float64), rcond=None)[0]
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def t_invert(img, s, rng, px=2, cal="px"):
    """Light modules on dark. Some decoders handle it, some do not, and it costs an attacker
    nothing -- so which decoders do is worth knowing. Binary, so strength only gates whether it
    is applied at all."""
    a = 255 - np.asarray(img.convert("L"))
    return Image.fromarray(a).convert("RGB")


def t_contrast(img, s, rng, px=2, cal="px"):
    """Grey on grey: a faded print, or a screen photographed at an angle."""
    a = np.asarray(img.convert("L"), np.float32)
    lo, hi = 128 - 127 * (1 - 0.95 * s), 128 + 127 * (1 - 0.95 * s)
    a = lo + (a / 255.0) * (hi - lo)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


TRANSFORMS = {"clean": t_clean, "logo": t_logo, "blur": t_blur, "motion": t_motion,
              "saltpepper": t_saltpepper, "rotate": t_rotate, "perspective": t_perspective,
              "invert": t_invert, "contrast": t_contrast}
STRENGTHS = [0.25, 0.50, 0.75, 1.00]


def stratified(urls: list, n: int, seed: int) -> list:
    """By URL LENGTH, because length sets the QR version and the module density, and density is
    what a degradation has to overwhelm. A random draw would over-sample the modal length and say
    little about the long, dense codes that fail first."""
    rng = random.Random(seed)
    if n >= len(urls):
        return list(urls)
    buckets: dict = {}
    for u in urls:
        buckets.setdefault(min(len(u) // 25, 11), []).append(u)
    out, per = [], max(1, n // max(1, len(buckets)))
    for k in sorted(buckets):
        b = buckets[k]
        out += rng.sample(b, min(per, len(b)))
    chosen = set(out)
    rest = [u for u in urls if u not in chosen]
    rng.shuffle(rest)
    return (out + rest)[:n]



def stream_run(a, picked, levels, boxes, meta, total) -> int:
    """Generate, degrade, decode, discard. Writes DFR rows and no images."""
    import hashlib as _h
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from benchmark_qr import load_decoders
    decs = load_decoders([x.strip() for x in a.decoders.split(",") if x.strip()])
    if not decs:
        print("[!] no decoder available", file=sys.stderr)
        return 1
    print(f"[*] streaming through {len(decs)} decoder(s): {', '.join(decs)} — no images written")

    import numpy as _np
    import tempfile
    os.makedirs(os.path.dirname(a.dfr_out) or ".", exist_ok=True)
    rng = np.random.default_rng(a.seed)
    done = 0
    t0 = __import__("time").time()
    # The decoders read a path, not an array. A single reused temp file keeps the write to one
    # page of cache rather than 792,000 separate files, and the OS never has to allocate a new
    # inode; measured, this is where the saving actually comes from.
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    with open(a.dfr_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "label", "tld", "url_len", "ec_level",
                                           "box_size", "transform", "strength", "calibration",
                                           "qr_version", "modules", "decoder", "decoded",
                                           "correct"])
        w.writeheader()
        for url in picked:
            sha = _h.sha1(url.encode("utf-8", "replace")).hexdigest()[:16]
            m = meta.get(url, {})
            for lvl, box in [(l, b) for l in levels for b in boxes]:
                q = qrcode.QRCode(error_correction=EC[lvl], box_size=box, border=4)
                q.add_data(url)
                try:
                    q.make(fit=True)
                except Exception:
                    continue
                base = q.make_image(fill_color="black", back_color="white").convert("RGB")
                for tname in a._transforms:
                    fn = TRANSFORMS[tname]
                    for st in ([1.0] if tname == "clean" else STRENGTHS):
                        img = fn(base, st, rng, box, a.calibration)
                        img.save(tmp.name)
                        row = {"sample_id": f"{sha}_{lvl}_b{box}_{tname}_{int(st*100):03d}",
                               "label": m.get("label", ""), "tld": m.get("tld", ""),
                               "url_len": len(url), "ec_level": lvl, "box_size": box,
                               "transform": tname, "strength": f"{st:.2f}",
                               "calibration": a.calibration,
                               "qr_version": q.version, "modules": q.modules_count}
                        for dname, dfn in decs.items():
                            got = dfn(tmp.name)
                            ok = any(_h.sha1(t.encode("utf-8", "replace")).hexdigest()[:16] == sha
                                     for t in got)
                            w.writerow({**row, "decoder": dname,
                                        "decoded": int(bool(got)), "correct": int(ok)})
                        done += 1
                        if done % 5000 == 0:
                            el = __import__("time").time() - t0
                            print(f"    [{done:,}/{total:,}] {el:.0f}s, {done/max(el,1):.0f} img/s")
    os.unlink(tmp.name)
    print(f"[+] {done:,} images streamed, 0 written to disk -> {a.dfr_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="URLs to sample (stratified by length)")
    ap.add_argument("--all", action="store_true", help="every URL in the corpus")
    ap.add_argument("--out", default=os.path.join("data", "raw", "qr_synth"))
    ap.add_argument("--seed", type=int, default=20260830)
    # Two modes, because two things want this data and they want opposite storage.
    #
    #   --stream : generate, degrade, decode and DISCARD, writing only the DFR row. The benchmark
    #              never needs the pixels again, and at 2,000 URLs x 4 EC x 3 module sizes x 33
    #              variants that is 792,000 images it would otherwise write and read once each.
    #              No disk, and no I/O between generating a frame and decoding it.
    #
    #   default  : write the PNGs. The restoration network in Section 3 trains on (degraded, clean)
    #              PAIRS, and pairs cannot be reconstructed from a CSV of ones and zeroes -- so the
    #              training set is built by a separate, much smaller run that keeps its images.
    #
    # Streaming the benchmark and saving a few hundred URLs for training is the combination that
    # costs least; saving everything, which is what this script did first, costs the most and buys
    # nothing the benchmark uses.
    ap.add_argument("--stream", action="store_true",
                    help="decode in RAM and write only results; saves no images")
    ap.add_argument("--decoders", default="opencv,pyzbar,wechat")
    ap.add_argument("--dfr-out", default=os.path.join("data", "processed", "qr", "qr_dfr.csv"))
    ap.add_argument("--ec", default="L,M,Q,H")
    # MEASURED, not chosen. The one QR this study found in the wild occupies 103 px on a 1600 px
    # page -- about 2.3 pixels per module once the quiet zone is counted -- while the first version
    # of this script used a flat 4, which is roughly twice the real scale. Blur and noise are far
    # more destructive when a module is two pixels wide than when it is four, so a benchmark run at
    # 4 measures an easier world than the one the attack happens in. Module size is an axis now,
    # bracketing the measurement: 2 is the wild scale, 4 the generous one, 3 between them.
    ap.add_argument("--box", default="2,3,4",
                    help="pixels per module, comma-separated; 2 matches the wild measurement")
    # Opt-in, and the default reproduces the frozen grid exactly. `blur` and `motion` are the only
    # two transforms this touches; see their docstrings and the PREREG deviation record for why
    # they saturated, and why a recalibrated run is a SEPARATE study rather than a fix applied to
    # the completed one. Anything generated with --calibration modules carries `calibration` in
    # every row, so the two can never be pooled by accident.
    ap.add_argument("--calibration", choices=["px", "modules"], default="px",
                    help="magnitude of blur/motion in absolute pixels (frozen grid) or in module "
                         "widths (a separate, declared study)")
    # The restoration set does not want the whole grid. A conv net cannot invert a rotation or a
    # homography -- those need a rectifier that finds the finder patterns, which is a different
    # algorithm and not this one -- so training it on `rotate` and `perspective` teaches it to
    # output the blurred average of every rotation it saw. Selecting the photometric transforms is
    # a data decision, made here rather than silently inside the trainer.
    ap.add_argument("--transforms", default="",
                    help="comma-separated subset (default all); 'clean' is added automatically "
                         "because it is the restoration target")
    a = ap.parse_args()

    want = [t.strip() for t in a.transforms.split(",") if t.strip()] or list(TRANSFORMS)
    unknown = [t for t in want if t not in TRANSFORMS]
    if unknown:
        print(f"[!] unknown transform(s): {', '.join(unknown)}; "
              f"have {', '.join(TRANSFORMS)}", file=sys.stderr)
        return 1
    a._transforms = (["clean"] if "clean" not in want else []) + want

    if not os.path.isfile(CORPUS):
        print(f"[!] {CORPUS} not found", file=sys.stderr)
        return 1
    with open(CORPUS, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        col = "url" if "url" in cols else (cols or ["url"])[0]
        rows = [r for r in rd if r.get(col)]
    urls = [r[col] for r in rows]
    meta = {r[col]: {"label": r.get("label", ""), "tld": r.get("tld", ""),
                     "tier": r.get("tier", "")} for r in rows}
    import collections as _c
    print(f"[*] corpus {len(urls):,} URLs from {CORPUS} (column '{col}'); "
          f"labels {dict(_c.Counter(r.get('label','') for r in rows).most_common(3))}")

    picked = urls if a.all else stratified(urls, a.n, a.seed)
    print(f"[*] sample labels: {dict(_c.Counter(meta[u]['label'] for u in picked).most_common(3))}")
    levels = [x.strip().upper() for x in a.ec.split(",") if x.strip()]
    boxes = [int(b) for b in str(a.box).split(",") if str(b).strip()]
    n_deg = len([t for t in a._transforms if t != "clean"])
    total = len(picked) * len(levels) * len(boxes) * (1 + n_deg * len(STRENGTHS))
    print(f"[*] {len(picked):,} URLs x {len(levels)} EC x {len(boxes)} module size(s) x "
          f"(1 clean + {n_deg} transforms x {len(STRENGTHS)} strengths) "
          f"= {total:,} images, calibration={a.calibration}")
    if a.calibration == "modules":
        print("[!] --calibration modules is NOT the frozen grid: blur and motion are expressed in "
              "module widths.\n    Rows carry calibration='modules'; do not pool them with the "
              "792,000-render sweep.")

    if a.stream:
        return stream_run(a, picked, levels, boxes, meta, total)

    os.makedirs(a.out, exist_ok=True)
    idx_path = os.path.join(a.out, "index.csv")
    rng = np.random.default_rng(a.seed)
    written = 0
    with open(idx_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for ui, url in enumerate(picked):
            sha = hashlib.sha1(url.encode("utf-8", "replace")).hexdigest()[:16]
            for lvl, box in [(l, b) for l in levels for b in boxes]:
                q = qrcode.QRCode(error_correction=EC[lvl], box_size=box, border=4)
                q.add_data(url)
                try:
                    q.make(fit=True)
                except Exception:
                    continue          # a URL too long for any version at this EC level
                base = q.make_image(fill_color="black", back_color="white").convert("RGB")
                mods = q.modules_count
                sub = os.path.join(a.out, sha[:2])
                os.makedirs(sub, exist_ok=True)
                for tname in a._transforms:
                    fn = TRANSFORMS[tname]
                    for s in ([1.0] if tname == "clean" else STRENGTHS):
                        img = fn(base, s, rng, box, a.calibration)
                        sid = f"{sha}_{lvl}_b{box}_{tname}_{int(s*100):03d}"
                        rel = os.path.join(sha[:2], sid + ".png")
                        img.save(os.path.join(a.out, rel), optimize=True)
                        m = meta.get(url, {})
                        w.writerow({"sample_id": sid, "url_sha1": sha, "url_len": len(url),
                                    "label": m.get("label", ""), "tld": m.get("tld", ""),
                                    "tier": m.get("tier", ""),
                                    "ec_level": lvl, "box_size": box,
                                    "px_per_module": box, "transform": tname,
                                    "strength": f"{s:.2f}", "calibration": a.calibration,
                                    "qr_version": q.version,
                                    "modules": mods, "image": rel})
                        written += 1
            if (ui + 1) % 200 == 0:
                print(f"    [{ui+1}/{len(picked)}] {written:,} images")
    print(f"[+] {written:,} images -> {a.out}")
    print(f"[+] ground truth -> {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
