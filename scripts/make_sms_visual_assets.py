#!/usr/bin/env python3
"""Figure 1 of the smishing paper: four rows quoted from the corpus, as message cards.

Every message on the figure is a row of data/raw/sms_hf_full/full_dataset.csv, quoted verbatim
(wrapped, and cut with an ellipsis when it runs past the card). Nothing is paraphrased and no
sender field is drawn: the published file carries no sender, so a card that printed one would be
inventing it. Rows are chosen by the host they carry, three hosts not used in Table 1, plus one
ham-labelled telecom template that opens with the [TB] notice prefix.

Writes papers/future_smishing/figures/fig_sms_examples.pdf

RUN:
    python3 scripts/make_sms_visual_assets.py
"""
from __future__ import annotations
import csv, os, sys, textwrap
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
SRC = os.path.join(ROOT, "data", "raw", "sms_hf_full", "full_dataset.csv")
os.makedirs(FIG_DIR, exist_ok=True)

# (panel letter, host the row must carry, label the row must have, text the row must start with)
PANELS = (("a", "vietcombank.vn-gll.top", "1", ""),
          ("b", "acb.i-pay.vip", "1", ""),
          ("c", "techcombank.huy-the-visa-vn.com", "1", ""),
          ("d", "viettel.vn", "0", "[TB]"))
WRAP = 48      # characters per line on a card
LINES = 6      # lines a card can hold before the ellipsis


def pick_rows():
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for letter, host, label, prefix in PANELS:
        hit = next((r for r in rows if host in (r["message"] or "").lower()
                    and r["label"] == label and (r["message"] or "").lstrip().startswith(prefix)),
                   None)
        if hit is None:
            raise SystemExit(f"[!] no label-{label} row carries {host}; the figure must change")
        out.append((letter, host, hit))
    return out


def card_lines(text: str):
    lines = textwrap.wrap(" ".join((text or "").split()), WRAP)
    if len(lines) > LINES:
        lines = lines[:LINES]
        lines[-1] = lines[-1].rstrip(" ,.;:-") + " …"
    return "\n".join(lines)


def build_sms_examples_figure(out_path: str) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
    })
    # The class colours come from scripts/figstyle.py, which every other figure in this
    # paper uses: ORANGE is the positive (spam/scam) side, BLUE the legitimate one.
    COL = {"1": ORANGE, "0": BLUE}
    BADGE_BG = {"1": "#fdf0eb", "0": "#eaf1fb"}
    BADGE = {"1": "SPAM/SCAM (1)", "0": "LEGITIMATE (0)"}
    COL_NAVY = "#1c5491"

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.6), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.28, left=0.03, right=0.97, top=0.94, bottom=0.03)

    for ax, (letter, host, r) in zip(axes.flat, pick_rows()):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        accent = COL[r["label"]]

        ax.add_patch(patches.FancyBboxPatch((1, 2), 98, 96, boxstyle="round,pad=0.5,rounding_size=3",
                                            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.85))
        ax.add_patch(patches.FancyBboxPatch((1, 78), 98, 20, boxstyle="round,pad=0.5,rounding_size=3",
                                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.75))
        ax.plot([3, 97], [97.5, 97.5], color=accent, lw=2.2, solid_capstyle="round")

        d_ = r["date"]
        try:
            import datetime as _dt
            d_ = _dt.datetime.strptime(d_, "%d/%m/%Y").date().isoformat()
        except (TypeError, ValueError):
            d_ = d_ or "undated"
        ax.text(4.5, 88, f"({letter}) {r['message_id']}, {d_}",
                fontsize=7.2, fontweight="bold", color=COL_NAVY, va="center")
        ax.text(95.5, 88, BADGE[r["label"]], fontsize=6.0, fontweight="bold", color=accent,
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BADGE_BG[r["label"]],
                          edgecolor=accent, lw=0.6))

        ax.add_patch(patches.FancyBboxPatch((3.5, 21), 93, 53, boxstyle="round,pad=0.5,rounding_size=2.5",
                                            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=0.75))
        ax.text(5.5, 70, card_lines(r["message"]), fontsize=6.9, color="#1e293b", va="top",
                linespacing=1.3)

        # Host and suffix sit on separate lines: on one line the longest host
        # (techcombank.huy-the-visa-vn.com) pushed the suffix past the card edge.
        suffix = host.rsplit(".", 1)[-1]
        ax.text(4.5, 14.5, f"embedded host: {host}", fontsize=6.4,
                fontweight="bold", color="#475569", va="center")
        ax.text(4.5, 8.0, f"suffix: .{suffix}", fontsize=6.4,
                fontweight="bold", color="#475569", va="center")

    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Rendered {out_path}")


def main() -> int:
    build_sms_examples_figure(os.path.join(FIG_DIR, "fig_sms_examples.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
