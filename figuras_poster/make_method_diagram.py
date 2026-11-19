"""Methodology diagram for the abstract and the poster (300 dpi PNG + vector PDF)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "."); os.makedirs(OUT, exist_ok=True)
FROZEN, TRAIN, NEUT = "#d6eaf8", "#f5b7b1", "#eaecee"
EDGE = "#34495e"
fig, ax = plt.subplots(figsize=(15, 3.9)); ax.set_xlim(0, 150); ax.set_ylim(0, 39); ax.axis("off")

def box(x, y, w, h, text, fc, fs=13, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6", fc=fc, ec=EDGE, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x0, y0, x1, y1, color=EDGE, lw=2.2, style="-|>", cs="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=22, lw=lw, color=color, connectionstyle=cs))

box(1, 12, 20, 16, "Martian NavCam\nimage\n512 × 512 px", NEUT)
arrow(21.6, 20, 27.4, 20)
# frozen backbone with 12 blocks and taps
ax.add_patch(FancyBboxPatch((28, 6), 58, 28, boxstyle="round,pad=0.4,rounding_size=1.6", fc=FROZEN, ec=EDGE, lw=1.8))
ax.text(57, 31.2, "DINOv3 ViT-B/16 — FROZEN (86 M parameters)", ha="center", va="center", fontsize=13.5, fontweight="bold")
taps = [2, 4, 6, 8, 10, 12]
for i in range(12):
    x = 30.2 + i * 4.6; tap = (i + 1) in taps
    ax.add_patch(FancyBboxPatch((x, 17), 3.7, 8.5, boxstyle="round,pad=0.1,rounding_size=0.6", fc="white", ec=EDGE, lw=1.2))
    ax.text(x + 1.85, 21.2, str(i + 1), ha="center", va="center", fontsize=9.5)
    if tap: arrow(x + 1.85, 16.8, x + 1.85, 11.3, color="#c0392b", lw=1.6)
ax.text(57, 8.6, "features read at blocks 2, 4, 6, 8, 10, 12 and at the final norm", ha="center", va="center", fontsize=11, color="#c0392b")
arrow(86.6, 20, 92.4, 20)
box(93, 12, 24, 16, "Linear probe\nBatchNorm + 1×1 conv\n4,612 trained parameters\n(one per layer)", TRAIN, fs=11.5, bold=False)
arrow(117.6, 20, 123.4, 20)
box(124, 12, 25, 16, "Per-pixel class\nsoil · bedrock\nsand · big rock", NEUT)
ax.add_patch(FancyBboxPatch((1, 0.6), 3.2, 2.6, boxstyle="round,pad=0.1", fc=FROZEN, ec=EDGE)); ax.text(5.2, 1.9, "frozen", va="center", fontsize=11)
ax.add_patch(FancyBboxPatch((17, 0.6), 3.2, 2.6, boxstyle="round,pad=0.1", fc=TRAIN, ec=EDGE)); ax.text(21.2, 1.9, "trained (same data, schedule and 3 seeds for every layer)", va="center", fontsize=11)
plt.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "fig0_method_diagram.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig0_method_diagram.pdf"), bbox_inches="tight")
print("ok")
