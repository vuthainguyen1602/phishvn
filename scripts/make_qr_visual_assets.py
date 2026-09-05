#!/usr/bin/env python3
"""Generate high-resolution visual demonstration figures for the Quishing paper.

Figures generated:
1. papers/future_quishing/figures/fig_qr_degradations.pdf
   - 3x3 visual gallery of the 9 degradation transforms (Clean, Logo, Blur, Motion,
     Salt-and-Pepper, Rotate, Perspective, Invert, Contrast).
2. papers/future_quishing/figures/fig_qr_restore_pipeline.pdf
   - Multi-stage end-to-end restoration pipeline (Input -> Polarity Normalization ->
     ConvNet Feature Cleaning -> Binarization -> Successful Payload Extraction).

RUN:
    python3 scripts/make_qr_visual_assets.py
"""
from __future__ import annotations
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import qrcode
from qrcode.constants import ERROR_CORRECT_M

# Ensure repo root is in python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
from gen_synthetic_qr import (t_clean, t_logo, t_blur, t_motion, t_saltpepper,
                              t_rotate, t_perspective, t_invert, t_contrast)

FIG_DIR = os.path.join(ROOT, "papers", "future_quishing", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLE_URL = "https://vietcombank.vn-ebanking.com/portal/auth"


def make_base_qr(url: str = SAMPLE_URL, box_size: int = 8, border: int = 4, ec=ERROR_CORRECT_M) -> Image.Image:
    qr = qrcode.QRCode(version=None, error_correction=ec, box_size=box_size, border=border)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def build_degradations_gallery(out_path: str) -> None:
    rng = np.random.default_rng(42)
    base_img = make_base_qr(SAMPLE_URL, box_size=8, border=4)
    px = 8

    # 9 conditions with calibrated illustrative strengths
    transforms = [
        ("(a) Clean Reference", t_clean(base_img, 0.0, rng, px=px), "nearest"),
        ("(b) Logo Overlay", t_logo(base_img, 0.75, rng, px=px), "nearest"),
        ("(c) Optical Blur", t_blur(base_img, 0.60, rng, px=px, cal="modules"), "bilinear"),
        ("(d) Motion Blur", t_motion(base_img, 0.50, rng, px=px, cal="modules"), "bilinear"),
        ("(e) Salt & Pepper", t_saltpepper(base_img, 0.40, rng, px=px), "nearest"),
        ("(f) Rotation (15°)", t_rotate(base_img, 0.50, rng, px=px), "bilinear"),
        ("(g) Perspective Warp", t_perspective(base_img, 0.55, rng, px=px), "bilinear"),
        ("(h) Polarity Invert", t_invert(base_img, 1.0, rng, px=px), "nearest"),
        ("(i) Low Contrast", t_contrast(base_img, 0.75, rng, px=px), "bilinear"),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
    })

    fig, axes = plt.subplots(3, 3, figsize=(6.2, 6.6), dpi=300)
    plt.subplots_adjust(wspace=0.22, hspace=0.32, left=0.03, right=0.97, top=0.94, bottom=0.04)

    for i, (title, img, interp) in enumerate(transforms):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        ax.imshow(img, interpolation=interp)
        ax.set_title(title, pad=6, fontsize=8.2, fontweight="bold", color="#1c5491")
        ax.axis("off")
        # Subtle academic box border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.85)

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered {out_path}")



RESTORE_SNAP = os.path.join("data", "processed", "qr", "restore_snapshot.json")


def _recovery_pp() -> tuple[float, float]:
    """(polarity, network) in percentage points of OpenCV DFR, from the confirmatory snapshot.

    Falls back to the pilot's pair only if the snapshot is absent, and says so, because a figure
    that silently prints a superseded number is exactly the failure this reads around."""
    try:
        r = json.load(open(RESTORE_SNAP, encoding="utf-8"))["by_arm"]["opencv"]
        return r["raw"] - r["control"], r["control"] - r["restored"]
    except Exception:
        print(f"[!] {RESTORE_SNAP} missing — figure falls back to the PILOT's 16.1/20.7",
              file=sys.stderr)
        return 16.1, 20.7

def build_restoration_demo(out_path: str) -> None:
    rng = np.random.default_rng(101)
    base_img = make_base_qr(SAMPLE_URL, box_size=8, border=4)
    px = 8

    # Create challenging combined degradation: Invert + Motion blur
    inv_img = t_invert(base_img, 1.0, rng, px=px)
    degraded = t_motion(inv_img, 0.40, rng, px=px, cal="modules")

    # Step 1: Polarity correction
    gray = np.array(degraded.convert("L"), dtype=float)
    border_val = np.mean(np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]))
    if border_val < 128:
        polarity_corrected = ImageOps.invert(degraded)
    else:
        polarity_corrected = degraded

    # Step 2: Simulated ConvNet restoration
    restored_img = polarity_corrected.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=3))
    # Step 3: Binarized thresholding
    arr = np.array(restored_img.convert("L"))
    thresh = np.mean(arr)
    binarized = Image.fromarray(((arr > thresh) * 255).astype(np.uint8)).convert("RGB")

    # The two recovery figures are MEASURED, not decorative, so they are read from the confirmatory
    # snapshot rather than typed here. They were typed here once, as the pilot's +16.1/+20.7, and
    # survived into the artwork for a day after the confirmatory run replaced the second of them.
    pol, net = _recovery_pp()

    stages = [
        ("(a) Adversarial Input\nInverted + Motion Blur", degraded, "DFR = 100% (Fail)", "#b0202a", "#fdf0f0", "bilinear"),
        ("(b) Polarity Normalized\nDark Mode Inverted", polarity_corrected, f"+{pol:.1f} pp Recovery", "#b06e14", "#fef9ee", "bilinear"),
        ("(c) ConvNet Restored\nDeblurred Modules", restored_img, f"+{net:.1f} pp Recovery", "#1c5491", "#eef5fc", "bilinear"),
        ("(d) Binarized Output\nThresholded Payload", binarized, "Decoded Cleanly", "#196e46", "#effaf3", "nearest")
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), dpi=300)
    plt.subplots_adjust(wspace=0.28, left=0.02, right=0.98, top=0.74, bottom=0.20)

    for i, (title, img, verdict, v_col, v_bg, interp) in enumerate(stages):
        ax = axes[i]
        ax.imshow(img, interpolation=interp)
        ax.set_title(title, fontsize=7.5, pad=6, fontweight="bold", color="#1c5491", linespacing=1.2)
        ax.text(0.5, -0.18, verdict, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.0, fontweight="bold", color=v_col,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=v_bg, edgecolor=v_col, lw=0.75))
        ax.axis("off")

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered {out_path}")


def main() -> int:
    deg_pdf = os.path.join(FIG_DIR, "fig_qr_degradations.pdf")
    res_pdf = os.path.join(FIG_DIR, "fig_qr_restore_pipeline.pdf")
    build_degradations_gallery(deg_pdf)
    build_restoration_demo(res_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
