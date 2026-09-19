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
# Three levels, not two: this file moved to scripts/ in the September
# reorganisation, which added a directory and left ROOT pointing at scripts/. The two figures
# then went to scripts/papers/future_quishing/figures/, a path nothing reads, while the copies
# the manuscript includes sat untouched. Same cause as abstract_text.py's parents[2].
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "core", "dataset"))
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



def build_restoration_demo(out_path: str) -> None:
    """Schematic only: no simulated network output or unmeasured decode verdicts."""
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.16, "Raw image", "Try decoder cascade"),
        (0.50, "Polarity normalisation", "If whole-image mean < 128:\ninvert, then retry cascade"),
        (0.84, "Neural restoration", "Decode grayscale output\n(no binarisation)")]
    for x, title, detail in stages:
        ax.text(x, .68, title + "\n\n" + detail, ha="center", va="center",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.8", facecolor="#eef5fc",
                                     edgecolor="#1c5491"))
        ax.text(x, .22, "Any payload returned: stop", ha="center", fontsize=7)
    for x1, x2 in ((.29, .36), (.65, .71)):
        ax.annotate("", xy=(x2, .68), xytext=(x1, .68),
                    arrowprops=dict(arrowstyle="->", color="#1c5491"))
        ax.text((x1+x2)/2, .36, "No payload", ha="center", fontsize=6)
    fig.text(.5, .04, "Processing schematic; not a restored-image example or a latency measurement.",
             ha="center", fontsize=8)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Rendered {out_path}")


def main() -> int:
    deg_pdf = os.path.join(FIG_DIR, "fig_qr_degradations.pdf")
    res_pdf = os.path.join(FIG_DIR, "fig_qr_restore_pipeline.pdf")
    build_degradations_gallery(deg_pdf)
    build_restoration_demo(res_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
