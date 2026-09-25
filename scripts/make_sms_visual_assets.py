#!/usr/bin/env python3
"""Generate high-resolution visual demonstration figures for the Smishing paper.

Figures generated:
1. papers/future_smishing/figures/fig_sms_examples.pdf
   - Multi-panel visual comparison of real Vietnamese SMS messages (phishing smishing
     lures with malicious disposable domains vs. legitimate brand/telecom messages).

RUN:
    python3 scripts/make_sms_visual_assets.py
"""
from __future__ import annotations
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:
    ROOT = os.path.dirname(_HERE)
from figstyle import ORANGE, BLUE  # noqa: E402
FIG_DIR = os.path.join(ROOT, "papers", "future_smishing", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def build_sms_examples_figure(out_path: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.6), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.28, left=0.03, right=0.97, top=0.94, bottom=0.03)

    # The class colours come from scripts/figstyle.py, which every other figure in this paper
    # already uses: ORANGE is the phishing side, BLUE the legitimate one. They were a hand-rolled
    # red/green pair, described in a comment as the repo's standard tokens although the repo has
    # no such tokens. That cost two things. It taught the reader a second encoding for one binary,
    # in the FIRST figure of the paper, after four others had taught orange/blue; and red against
    # green is the one pair in this paper that a red-green colour blindness cannot separate, which
    # is roughly one man in twelve.
    COL_PHISH = ORANGE
    COL_HAM = BLUE
    # Navy and its pale box stay: they carry the card title and the URL on ALL FOUR cards, so they
    # encode no class, and a URL that is not link-coloured stops reading as a URL.
    COL_NAVY = "#1c5491"
    COL_URL = "#1c5491"
    COL_MUTED = "#5f6368"

    cards = [
        {
            "ax_idx": (0, 0),
            "label": "SMISHING",
            "accent_col": COL_PHISH,
            "badge_bg": "#fdf0eb",
            "sender": "Sender: +8491... (Spoofed ID)",
            "title": "(a) Account Lockout Alert",
            "body": "Tai khoan cua ban dang duoc dang nhap\ntren thiet bi khac, neu khong phai ban\nvui long truy cap:",
            "link": "https://vietcombank.vn-gll.top",
            "tail": "de doi mat khau hoac thoat thiet bi.",
            "meta": "Target: Vietcombank  |  Suffix: .top (Disposable)"
        },
        {
            "ax_idx": (0, 1),
            "label": "SMISHING",
            "accent_col": COL_PHISH,
            "badge_bg": "#fdf0eb",
            "sender": "Sender: (ACB) Security",
            "title": "(b) Foreign Transaction Panic",
            "body": "(ACB) Phat hien tai khoan cua ban dang\ntieu dung o nuoc ngoai, neu khong phai\nban vui long vao:",
            "link": "https://acb.i-pay.vip",
            "tail": "de huy giao dich tranh mat tien.",
            "meta": "Target: ACB  |  Suffix: .vip (Combosquatting)"
        },
        {
            "ax_idx": (1, 0),
            "label": "SMISHING",
            "accent_col": COL_PHISH,
            "badge_bg": "#fdf0eb",
            "sender": "Sender: Techcombank_CSKH",
            "title": "(c) Points / Card Expiry",
            "body": "[Techcombank] Uu dai co thoi han!\n12.000 diem thuong het han hom nay.\nNhan qua tai:",
            "link": "https://techcombank.huy-the-visa-vn.com",
            "tail": "qua han diem se bi huy bo.",
            "meta": "Target: Techcombank  |  Suffix: .com (Hyphen Squat)"
        },
        {
            "ax_idx": (1, 1),
            "label": "HAM (LEGIT)",
            "accent_col": COL_HAM,
            "badge_bg": "#eaf1fb",
            "sender": "Sender: VIETTEL_TB",
            "title": "(d) Official Promotion",
            "body": "[TB] Viettel khuyen mai 20% gia tri\nthe nap ngay 31/08. Giam them 2.5%\nkhi nap the tai:",
            "link": "https://viettel.vn/pay/tt",
            "tail": "hoac https://myvt.page.link/km . LH: 198.",
            "meta": "Sender: Viettel Telecom  |  Suffix: .vn / page.link"
        }
    ]

    for item in cards:
        r, c = item["ax_idx"]
        ax = axes[r, c]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        accent = item["accent_col"]

        # Outer card container
        card_rect = patches.FancyBboxPatch((1, 2), 98, 96, boxstyle="round,pad=0.5,rounding_size=3",
                                          facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.85)
        ax.add_patch(card_rect)

        # Header bar with top accent stroke
        header_rect = patches.FancyBboxPatch((1, 74), 98, 24, boxstyle="round,pad=0.5,rounding_size=3",
                                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.75)
        ax.add_patch(header_rect)

        # Accent line at very top
        ax.plot([3, 97], [97.5, 97.5], color=accent, lw=2.2, solid_capstyle="round")

        # Header Title & Badge
        ax.text(4.5, 86.5, item["title"], fontsize=8.0, fontweight="bold", color=COL_NAVY, va="center")
        ax.text(95.5, 86.5, item["label"], fontsize=6.6, fontweight="bold", color=accent, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=item["badge_bg"], edgecolor=accent, lw=0.6))
        ax.text(4.5, 78, item["sender"], fontsize=6.8, color=COL_MUTED, style="italic", va="center")

        # Body Message Bubble
        bubble_rect = patches.FancyBboxPatch((3.5, 17), 93, 54, boxstyle="round,pad=0.5,rounding_size=2.5",
                                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.75)
        ax.add_patch(bubble_rect)

        # Body text & Link
        ax.text(5.5, 66, item["body"], fontsize=7.1, color="#1e293b", va="top", linespacing=1.25)
        ax.text(5.5, 36, item["link"], fontsize=7.1, fontweight="bold", color=COL_URL, va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#eff6ff", edgecolor="#bfdbfe", lw=0.5))
        ax.text(5.5, 27, item["tail"], fontsize=6.8, color="#334155", va="top")

        # Meta footer
        ax.text(4.5, 8.5, item["meta"], fontsize=6.4, fontweight="bold", color="#475569", va="center")

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered {out_path}")


def main() -> int:
    out_pdf = os.path.join(FIG_DIR, "fig_sms_examples.pdf")
    build_sms_examples_figure(out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
