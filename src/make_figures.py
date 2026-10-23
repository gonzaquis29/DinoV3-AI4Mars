"""Figures and key numbers of the FINAL run (Kaggle, 512 px, 3 seeds, 15,564 train / 500 val / 322 test images).
Everything is computed from the per-image confusion matrices (*.npz) and metadata (*.json) saved by the run; nothing is typed by hand
except the published baseline numbers (Mohammad, Li & Gao, iSpaRo 2025, Table I).
Usage (from the repo root): python src/make_figures.py   ->  results/figures/*.png|pdf + numbers.json
Bootstrap CIs here use 1,000 resamples of the images (the notebook's stage E uses 2,000)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KEEP = os.path.expanduser(os.environ.get("A4M_KEEP", os.path.join(ROOT, "results", "final_run", "results")))
OUT = os.path.join(ROOT, "results", "figures"); os.makedirs(OUT, exist_ok=True)
CL = ["soil", "bedrock", "sand", "big rock"]; SEEDS = (0, 1, 2); B = 1000
plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15, "legend.fontsize": 11, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False})
GREEN, BLUE, PURPLE, RED, GRAY, ORANGE = "#1e8449", "#2e86c1", "#8e44ad", "#c0392b", "#7f8c8d", "#e67e22"

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); plt.close(fig)

_cache = {}
def load(n):
    if n not in _cache: _cache[n] = np.load(f"{KEEP}/{n}.npz")
    return _cache[n]
def meta(n): return json.load(open(f"{KEEP}/{n}.json"))

def iou(conf):                                    # (...,4,4) confusion (rows = truth) -> per-class IoU; classes with empty union -> nan
    c = conf.astype(np.float64); i = np.diagonal(c, axis1=-2, axis2=-1); u = c.sum(-1) + c.sum(-2) - i
    with np.errstate(invalid="ignore", divide="ignore"): return np.where(u > 0, i / u, np.nan)
def miou(conf): return np.nanmean(iou(conf), -1)
def conf_of(arrs, r=0): return [a[:, r] for a in arrs]            # per-image confusion at boundary radius index r

def score(name, key, r=0):                         # mIoU per seed
    return np.array([miou(load(name.format(s=s))[key][:, r].sum(0)) for s in SEEDS])
def ci(name, key, B=B, seed=0):                    # 95 % bootstrap over images, mean over seeds (same resample for every seed)
    arrs = [load(name.format(s=s))[key][:, 0] for s in SEEDS]; n = len(arrs[0]); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n); out.append(np.mean([miou(a[idx].sum(0)) for a in arrs]))
    return np.percentile(out, [2.5, 97.5])
def ms(x): return float(np.mean(x)), float(np.std(x))

LAYERS = ["2", "4", "6", "8", "10", "12", "last_norm"]; LABEL = ["2", "4", "6", "8", "10", "12", "final"]
NUM = {}

# ------------------------------------------------------------------ Fig 1: layer-wise probe
val = {l: score("A_layers_s{s}", f"val__{l}") for l in LAYERS}
M = {s: {l: score("A_layers_s{s}", f"{s}__{l}") for l in LAYERS} for s in ("M1", "M2", "M3")}
cis = {s: np.array([ci("A_layers_s{s}", f"{s}__{l}") for l in LAYERS]) for s in ("M1", "M2")}
V = np.array([val[l].mean() for l in LAYERS]); VSD = np.array([val[l].std() for l in LAYERS])
M1m = np.array([M["M1"][l].mean() for l in LAYERS]); M2m = np.array([M["M2"][l].mean() for l in LAYERS])
IOU_VAL = np.array([np.mean([iou(load(f"A_layers_s{s}")[f"val__{l}"][:, 0].sum(0)) for s in SEEDS], 0) for l in LAYERS])
fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1])); x = np.arange(7)
ax[0].errorbar(x, V, yerr=VSD, c=GREEN, marker="o", capsize=3, label="validation (± sd over 3 seeds)")
ax[0].errorbar(x, M1m, yerr=[M1m - cis["M1"][:, 0], cis["M1"][:, 1] - M1m], c=PURPLE, marker="^", capsize=3, label="test M1 (95 % bootstrap CI)")
ax[0].errorbar(x, M2m, yerr=[M2m - cis["M2"][:, 0], cis["M2"][:, 1] - M2m], c=BLUE, marker="s", capsize=3, alpha=0.8, label="test M2 (95 % bootstrap CI)")
ax[0].set_xticks(x); ax[0].set_xticklabels(LABEL); ax[0].set_xlabel("layer (block k or final)"); ax[0].set_ylabel("mIoU (4 classes)"); ax[0].set_ylim(0.3, 0.85)
ax[0].set_title("mIoU per layer of frozen DINOv3 (linear probe)"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10, frameon=False)
im = ax[1].imshow(IOU_VAL, cmap="viridis", vmin=0, vmax=1)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(CL, rotation=25); ax[1].set_yticks(range(7)); ax[1].set_yticklabels(LABEL); ax[1].set_title("Per-class IoU (validation)")
for i in range(7):
    for j in range(4): ax[1].text(j, i, f"{IOU_VAL[i, j]:.2f}", ha="center", va="center", color="w", fontsize=10)
plt.colorbar(im, ax=ax[1]); plt.tight_layout(); save(fig, "fig1_layers")
NUM["layers"] = {l: dict(val=ms(val[l]), M1=ms(M["M1"][l]), M2=ms(M["M2"][l]), M3=ms(M["M3"][l])) for l in LAYERS}
NUM["val_per_class_last_norm"] = dict(zip(CL, map(float, IOU_VAL[-1])))
# H1 as defined in the proposal (block 12 is NOT an intermediate layer): best intermediate by validation vs final norm, paired bootstrap on M1/M2
inter = ["2", "4", "6", "8", "10"]; best = max(inter, key=lambda l: val[l].mean()); NUM["best_intermediate_by_val"] = best
for s in ("M1", "M2"):
    arrs_a = [load(f"A_layers_s{sd}")[f"{s}__{best}"][:, 0] for sd in SEEDS]; arrs_b = [load(f"A_layers_s{sd}")[f"{s}__last_norm"][:, 0] for sd in SEEDS]
    rng = np.random.default_rng(1); n = len(arrs_a[0]); d = []
    for _ in range(B):
        idx = rng.integers(0, n, n); d.append(np.mean([miou(a[idx].sum(0)) - miou(b[idx].sum(0)) for a, b in zip(arrs_a, arrs_b)]))
    NUM[f"H1_best_intermediate_minus_final_{s}"] = dict(delta=float(np.mean([miou(a.sum(0)) - miou(b.sum(0)) for a, b in zip(arrs_a, arrs_b)])), ci=list(map(float, np.percentile(d, [2.5, 97.5]))))
# majority-class predictor on validation (same convention as the models: classes with GT but no prediction score 0)
g = load("A_layers_s0")["val__last_norm"][:, 0].sum(0).sum(1).astype(float); NUM["majority_val_mIoU"] = float(g.max() / g.sum() / 4)

# ------------------------------------------------------------------ Fig 2: backbone control (stage D, N = 2000)
DK = [("dinov3_b16_lvd", "DINOv3 ViT-B/16", GREEN), ("dinov2_b14", "DINOv2 ViT-B/14", BLUE), ("dinov3_b16_rand", "DINOv3 arch., random init", GRAY)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True); NUM["D"] = {}
for a, s in zip(axes, ("M1", "M2")):
    for j, (k, lab, col) in enumerate(DK):
        v = score("D_" + k + "_s{s}", f"{s}__probe"); NUM["D"].setdefault(k, {})[s] = ms(v)
        a.bar(j, v.mean(), 0.7, yerr=v.std(), capsize=3, color=col, label=lab)
        a.text(j, v.mean() + 0.02, f"{v.mean():.3f}", ha="center", fontsize=11)
    a.set_xticks([]); a.set_title(f"test {s}", fontsize=13); a.set_ylim(0, 1.0)
axes[0].set_ylabel("mIoU (4 classes)"); axes[0].legend(fontsize=10, loc="upper right"); fig.suptitle("Frozen backbone control (same probe, 2,000 images, 3 seeds)", fontsize=14)
plt.tight_layout(); save(fig, "fig2_backbones")

# ------------------------------------------------------------------ Fig 3: label efficiency (LP / FT2 / FT4)
NS = [250, 500, 1000, 2000, 4000]; fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False); NUM["C"] = {}
for a, s in zip(axes, ("M1", "M2")):
    for meth, col, lab in (("LP", BLUE, "linear probe"), ("FT2", ORANGE, "fine-tune last 2 blocks"), ("FT4", RED, "fine-tune last 4 blocks")):
        vals = []
        for n in NS:
            z0 = load(f"C_{meth}_n{n}_s0"); key = [k for k in z0.files if k.startswith(s + "__")][0]
            vals.append(ms(score(f"C_{meth}_n{n}" + "_s{s}", key)))
        NUM["C"].setdefault(meth, {})[s] = {n: v for n, v in zip(NS, vals)}
        a.errorbar(NS, [v[0] for v in vals], yerr=[v[1] for v in vals], c=col, marker="o", capsize=3, label=lab)
    a.set_xscale("log"); a.set_xticks(NS); a.set_xticklabels([str(n) for n in NS]); a.minorticks_off(); a.set_xlabel("labelled training images (log scale)"); a.set_title(f"test {s}", fontsize=13)
axes[0].set_ylabel("mIoU (4 classes)"); axes[0].legend(fontsize=9, loc="lower right"); fig.suptitle("Label efficiency (± sd over 3 seeds)", fontsize=14)
plt.tight_layout(); save(fig, "fig3_label_efficiency")

# ------------------------------------------------------------------ Fig 4: published baselines + trainable parameters
PUB = {"DeepLabV3+ R101 (512)": ((.67, .72, .82), 58.75), "SegFormer-B3 (512)": ((.67, .72, .83), 47.23),
       "SegFormer-B3 (1024)": ((.69, .74, .93), 47.23), "UPerNet Swin-S (1024)": ((.70, .74, .94), 81.15)}
tr = lambda n: meta(n)["trainable"] / 1e6
OURS = {"Frozen DINOv3 + linear probe": (tuple(float(score("A_layers_s{s}", f"{s}__last_norm").mean()) for s in ("M1", "M2", "M3")), meta("A_layers_s0")["trainable_per_head"] / 1e6, GREEN),
        "Frozen DINOv3 + decoder": (tuple(float(score("B_decoder512_s{s}", f"{s}__decoder").mean()) for s in ("M1", "M2", "M3")), tr("B_decoder512_s0"), PURPLE),
        "DINOv3, last 4 blocks fine-tuned": (tuple(float(score("C_FT4_n4000_s{s}", [k for k in load("C_FT4_n4000_s0").files if k.startswith(m + "__")][0]).mean()) for m in ("M1", "M2", "M3")), tr("C_FT4_n4000_s0"), RED)}
NUM["ours"] = {k: dict(M1M2M3=list(v[0]), trainable_M=v[1]) for k, v in OURS.items()}
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
names = list(PUB) + list(OURS); vals = [v[0] for v in PUB.values()] + [v[0] for v in OURS.values()]
cm = plt.cm.Blues(np.linspace(0.35, 0.85, len(PUB))); cols = list(cm) + [v[2] for v in OURS.values()]
x = np.arange(3); w = 0.82 / len(names)
for j, (nm, v) in enumerate(zip(names, vals)): ax[0].bar(x + (j - (len(names) - 1) / 2) * w, v, w, color=cols[j], label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(["M1 (min1)", "M2 (min2)", "M3 (min3)"]); ax[0].set_ylim(0.4, 1.0); ax[0].set_ylabel("mIoU (4 classes)")
ax[0].set_title("Comparison with published baselines"); ax[0].legend(fontsize=8, loc="upper left", ncol=2)
ax[1].scatter([v[1] for v in PUB.values()], [(v[0][0] + v[0][1]) / 2 for v in PUB.values()], s=100, c=BLUE, label="published baselines (end-to-end)")
for k, v in OURS.items(): ax[1].scatter([v[1]], [(v[0][0] + v[0][1]) / 2], s=200, c=v[2], marker="*", label=k)
ax[1].set_xscale("log"); ax[1].set_ylim(0.66, 0.74); ax[1].set_xlabel("trainable parameters (millions, log scale)"); ax[1].set_ylabel("mean mIoU, M1 and M2")
ax[1].set_title("Trainable parameters vs mIoU"); ax[1].legend(fontsize=8, loc="lower right")
plt.tight_layout(); save(fig, "fig4_baselines")

# ------------------------------------------------------------------ Fig 5: boundary band excluded (label-noise probe)
RAD = meta("A_layers_s0")["radii"]; fig, a = plt.subplots(figsize=(7.2, 4.4)); NUM["border"] = {}
for name, key, col, lab in (("A_layers_s{s}", "M1__last_norm", BLUE, "linear probe"), ("B_decoder512_s{s}", "M1__decoder", PURPLE, "decoder")):
    arr = np.array([[miou(load(name.format(s=s))[key][:, i].sum(0)) for i in range(len(RAD))] for s in SEEDS])
    a.errorbar(RAD, arr.mean(0), yerr=arr.std(0), c=col, marker="o", capsize=3, label=lab); NUM["border"][lab] = dict(zip(map(str, RAD), map(float, arr.mean(0))))
a.set_xlabel("band around label edges excluded (px, native 1,024)"); a.set_ylabel("test M1 mIoU"); a.set_title("Excluding label borders"); a.legend(fontsize=10, loc="lower right")
plt.tight_layout(); save(fig, "fig5_border_band")

# ------------------------------------------------------------------ Fig 6: confusion (validation) + Fig 7: where M3 differs
def vconf(name, key): return sum(load(name.format(s=s))[key][:, 0].sum(0).astype(float) for s in SEEDS)
CONF = {"linear probe (final)": vconf("A_layers_s{s}", "val__last_norm"), "decoder": vconf("B_decoder512_s{s}", "val__decoder")}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2)); NUM["confusion_val"] = {}
for a, (k, c) in zip(axes, CONF.items()):
    r = c / c.sum(1, keepdims=True); NUM["confusion_val"][k] = r.round(4).tolist()
    a.imshow(r, cmap="Blues", vmin=0, vmax=1); a.set_xticks(range(4)); a.set_xticklabels(CL, rotation=30, fontsize=10); a.set_yticks(range(4)); a.set_yticklabels(CL, fontsize=10)
    a.set_xlabel("predicted"); a.set_title(k, fontsize=12)
    for i in range(4):
        for j in range(4): a.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", color="w" if r[i, j] > 0.5 else "k", fontsize=10)
axes[0].set_ylabel("true"); plt.tight_layout(); save(fig, "fig6_confusion")

M3KEYS = [("Linear probe", "A_layers_s{s}", "M3__last_norm", GREEN), ("Decoder", "B_decoder512_s{s}", "M3__decoder", PURPLE),
          ("Fine-tune 4 blocks", "C_FT4_n4000_s{s}", [k for k in load("C_FT4_n4000_s0").files if k.startswith("M3__")][0], RED)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); NUM["M3"] = {}
gt_px = tp = None
for j, (lab, name, key, col) in enumerate(M3KEYS):
    cs = [load(name.format(s=s))[key][:, 0].sum(0).astype(float) for s in SEEDS]
    tpm = np.mean([c[3, 3] for c in cs]); fpm = np.mean([c[:, 3].sum() - c[3, 3] for c in cs]); gt_px = cs[0][3].sum()
    m4 = np.mean([np.nanmean(iou(c)) for c in cs]); m3 = np.mean([np.nanmean(iou(c)[:3]) for c in cs])
    NUM["M3"][lab] = dict(mIoU4=float(m4), mIoU3_without_big_rock=float(m3), big_rock_TP_px=float(tpm), big_rock_FP_px=float(fpm), big_rock_GT_px=float(gt_px),
                          recall=float(tpm / gt_px), big_rock_IoU=float(np.mean([iou(c)[3] for c in cs])))
    axes[0].bar(j - 0.2, tpm / 1e3, 0.38, color=col, label="true positive" if j == 0 else None, alpha=0.95)
    axes[0].bar(j + 0.2, fpm / 1e3, 0.38, color=col, alpha=0.45, hatch="//", label="false positive" if j == 0 else None)
    axes[1].bar(j - 0.2, m4, 0.38, color=col); axes[1].bar(j + 0.2, m3, 0.38, color=col, alpha=0.45, hatch="//")
    axes[1].text(j - 0.2, m4 + 0.01, f"{m4:.2f}", ha="center", fontsize=10); axes[1].text(j + 0.2, m3 + 0.01, f"{m3:.2f}", ha="center", fontsize=10)
axes[0].axhline(gt_px / 1e3, ls="--", c="k", lw=1); axes[0].text(2.45, gt_px / 1e3 + 4, "ground truth (5 images)", ha="right", fontsize=10)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels([m[0] for m in M3KEYS], fontsize=11); axes[0].set_ylabel("big-rock pixels in M3 (thousands)"); axes[0].set_ylim(0, 135); axes[0].legend(fontsize=10, loc="upper center", ncol=2)
axes[0].set_title("Big rock in M3: found by all, false alarms differ", fontsize=12)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels([m[0] for m in M3KEYS], fontsize=11); axes[1].set_ylim(0.6, 1.02); axes[1].set_ylabel("test M3 mIoU")
axes[1].set_title("M3 mIoU: 4 classes (solid) vs without big rock (hatched)", fontsize=12)
plt.tight_layout(); save(fig, "fig7_m3_bigrock")

json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print("saved:", sorted(os.listdir(OUT)))
