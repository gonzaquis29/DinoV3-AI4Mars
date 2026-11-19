"""English figures for the v3 poster. Every number comes from the round-2 verdicts (results/estudio_principal/veredictos_ronda2.csv) and the report."""
import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
# --- Fig A: models on the three expert sets ---
models = ["Linear probe\n(4.6 k)", "Decoder\n(3.5 M)", "DeepLabV3\n(39.6 M)", "SegFormer-B3\n(published)", "UPerNet-Swin-S\n(published)"]
M1 = [0.674, 0.682, 0.675, 0.69, 0.70]; M2 = [0.723, 0.732, 0.727, 0.74, 0.74]; M3 = [0.862, 0.867, 0.884, 0.93, 0.94]
fig, ax = plt.subplots(figsize=(11, 5)); w = 0.26
for k, (v, c, n) in enumerate(((M1, "#4b2a82", "M1"), (M2, "#8e6fc7", "M2"), (M3, "#c9b8e8", "M3"))):
    b = ax.bar([i + (k - 1) * w for i in range(5)], v, w, color=c, label=n)
    for r, y in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, y + 0.004, f"{y:.2f}" if y in (0.69, 0.70, 0.74, 0.93, 0.94) else f"{y:.3f}", ha="center", fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(models, fontsize=12); ax.set_ylim(0.55, 1.0); ax.set_ylabel("test mIoU"); ax.legend(ncol=3, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/figA_models.png", dpi=200); plt.close()
# --- Fig B: paired differences on T (forest plot) ---
rows = [("H1  histogram equalisation", -0.008, -0.016, -0.001, "inconclusive"), ("H1  CLAHE", -0.010, -0.018, -0.004, "real drop"),
        ("H1  standardisation", -0.001, -0.005, 0.002, "equivalent"), ("H1  Gaussian blur", 0.000, -0.009, 0.009, "equivalent"),
        ("H2  late − early", 0.033, 0.023, 0.044, "real"), ("H2  middle − early", 0.030, 0.019, 0.041, "real"), ("H2  late − middle", 0.004, -0.005, 0.013, "inconclusive"),
        ("C1  rare-class loss, big-rock IoU", 0.024, 0.010, 0.039, "real"), ("C2  LoRA over decoder", 0.006, -0.002, 0.014, "inconclusive"), ("C3  probe − DeepLabV3", -0.005, -0.019, 0.010, "inconclusive")]
col = {"real": "#4b2a82", "real drop": "#b3261e", "equivalent": "#2e7d32", "inconclusive": "#8a8a93"}
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.axvspan(-0.01, 0.01, color="#e6e1f0", zorder=0); ax.axvline(0, color="k", lw=1)
for i, (n, d, lo, hi, v) in enumerate(rows[::-1]):
    ax.plot([lo, hi], [i, i], color=col[v], lw=3); ax.plot(d, i, "o", color=col[v], ms=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=12); ax.set_xlabel("difference in mIoU on T (95 % CI); shaded band = ±0.01")
for v, c in col.items(): ax.plot([], [], "o-", color=c, label=v)
ax.legend(frameon=False, loc="lower right", fontsize=11); plt.tight_layout(); plt.savefig(f"{OUT}/figB_differences.png", dpi=200); plt.close()
print("ok")

# --- Figs C and D: from the round-2 raw results (results/estudio_principal/raw/e), English labels ---
import numpy as np
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "estudio_principal", "raw", "e")
V = {"dec_ce": ("Decoder + CE", "#4b2a82"), "dec_lov": ("Decoder + Lovász", "#8e6fc7"), "lora_ce": ("LoRA + CE", "#e08a1e"), "lora_lov": ("LoRA + Lovász", "#b3261e")}
def load(v): return [np.load(f"{RAW}/R2_{v}_s{s}.npz") for s in (0, 1, 2)]
def iou(c): return np.diag(c) / (c.sum(0) + c.sum(1) - np.diag(c))
Z = {v: load(v) for v in V}
fig, ax = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [3, 2]}); w = 0.2
for j, (v, (lab, c)) in enumerate(V.items()):
    vals = np.mean([iou(z["T__conf"][:, 0].sum(0)) for z in Z[v]], 0); ax[0].bar(np.arange(4) + (j - 1.5) * w, vals, w, color=c, label=lab)
    fp = np.mean([z["M3__conf"][:, 0].sum(0)[:3, 3].sum() for z in Z[v]]) / 1e3; ax[1].bar(j, fp, 0.7, color=c)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["soil", "bedrock", "sand", "big rock"]); ax[0].set_ylabel("IoU on T"); ax[0].set_title("Per-class IoU on T"); ax[0].set_ylim(0, 1.0)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["Dec\nCE", "Dec\nLov.", "LoRA\nCE", "LoRA\nLov."], fontsize=12); ax[1].set_ylabel("thousands of pixels"); ax[1].set_title("Big-rock false positives on M3")
fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.91)); plt.savefig(f"{OUT}/figC_perclass.png", dpi=200); plt.close()
def ece(cal): cnt, sc, cc = cal[:, 0], cal[:, 1], cal[:, 2]; return float(np.abs(cc - sc).sum() / cnt.sum())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
for v, (lab, c) in V.items():
    cal = np.mean([z["T__calib"].sum(0) for z in Z[v]], 0)
    for a, g in ((ax[0], 0), (ax[1], 1)):
        cnt, sc, cc = cal[g, :, 0], cal[g, :, 1], cal[g, :, 2]; ok = cnt > 0
        a.plot(sc[ok] / cnt[ok], cc[ok] / cnt[ok], "o-", ms=4, color=c, label=f"{lab} (ECE {ece(cal[g]):.3f})")
for a, t in zip(ax, ("All labelled pixels", "Pixels predicted as big rock")):
    a.plot([0, 1], [0, 1], "k--", lw=1); a.set_title(t); a.set_xlabel("confidence"); a.set_ylabel("accuracy"); a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.legend(frameon=False, fontsize=11.5, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/figD_calibration.png", dpi=200); plt.close()
print("ok CD")
