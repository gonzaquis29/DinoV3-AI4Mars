"""Validation mIoU per epoch for every main run (P1 probes, P2 LoRA zones, DeepLabV3 baseline), read from results/estudio_principal/raw.
Only validation mIoU is logged per epoch (no training loss). Usage: python src/make_training_curves.py"""
import json, glob, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path(__file__).resolve().parent.parent / "results" / "estudio_principal"
load = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(str(RAW / "raw" / pat)))]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)

ax = axs[0]
cols = {"original": "#444444", "hist_eq": "#e07b39", "clahe": "#2a6fdb", "standardize": "#3a9d5d", "blur": "#9b59b6"}
for t, c in cols.items():
    runs = load(f"a/P1_{t}_s*.json")
    for i, r in enumerate(runs):
        ax.plot(range(1, len(r["val_hist"]) + 1), r["val_hist"], "-o", ms=3, lw=1, color=c, alpha=0.8, label=t if i == 0 else None)
ax.set_title("P1: probe, appearance transforms", fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("validation mIoU")

ax = axs[1]
cols = {"early": "#2a6fdb", "mid": "#3a9d5d", "late": "#e07b39"}
for z, c in cols.items():
    for i, r in enumerate(load(f"b/P2_{z}_s*.json")):
        ax.plot(range(1, len(r["val_hist"]) + 1), r["val_hist"], "-o", ms=4, lw=1.2, color=c, label=z if i == 0 else None)
ax.set_title("P2: LoRA rank 8 by zone", fontsize=9); ax.set_xlabel("epoch"); ax.set_xticks([1, 2])

ax = axs[2]
for key, lab, c in (("pre", "DeepLabV3-R50, ImageNet", "#2a6fdb"), ("scratch", "DeepLabV3-R50, no pretraining", "#e07b39")):
    for r in load(f"d/BL_{key}_full_s*.json"):
        ax.plot(range(1, len(r["val_hist"]) + 1), r["val_hist"], "-o", ms=4, lw=1.2, color=c, label=lab)
ax.set_title("Baseline, 3 epochs (2,916 steps)", fontsize=9); ax.set_xlabel("epoch"); ax.set_xticks([1, 2, 3])

for ax in axs:
    ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); out = RAW / "figures" / "training_curves.png"; fig.savefig(out, dpi=200); print("wrote", out)
