"""Figuras del informe de la ronda 2 (curvas de entrenamiento y veredictos). Solo lee `results/estudio_principal/`; no entrena nada.

Uso (CPU, segundos):  python src/make_report_figures.py
Escribe en docs/informe_figuras/. Requiere numpy y matplotlib.

Cada serie sale de los campos `val_hist` (mIoU de validación por época) y `loss_hist` (pérdida de entrenamiento por época, solo en E) de los .json de cada corrida.
Los intervalos del gráfico de veredictos se copian de `results/estudio_principal/veredictos_ronda2.csv` y se comprueba que el texto exista en ese archivo.
"""
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "estudio_principal" / "raw"
OUT = ROOT / "docs" / "informe_figuras"; OUT.mkdir(parents=True, exist_ok=True)
load = lambda pat: [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(RAW / pat)))]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"dec_ce": "#1f77b4", "dec_lov": "#ff7f0e", "lora_ce": "#2ca02c", "lora_lov": "#d62728"}


def curves(ax, runs, color, label, key="val_hist", mark_best=True):
    for i, r in enumerate(runs):
        y = r[key]; x = np.arange(1, len(y) + 1)
        ax.plot(x, y, "-", lw=0.8, alpha=0.35, color=color)
        if mark_best and "best_epoch" in r and key == "val_hist":
            ax.plot(r["best_epoch"], y[r["best_epoch"] - 1], "o", ms=4, color=color, mec="k", mew=0.5, zorder=3)
    n = min(len(r[key]) for r in runs)
    m = np.mean([r[key][:n] for r in runs], 0)
    ax.plot(np.arange(1, n + 1), m, "-", lw=2, color=color, label=label)


# ---- Fig 1: ronda 2 (E): validación y pérdida --------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for v, c in C.items():
    rs = load(f"e/R2_{v}_s*.json"); curves(ax[0], rs, c, v); curves(ax[1], rs, c, v, key="loss_hist", mark_best=False)
ax[0].set(title="Validación (gruesa = media; fina = semilla; punto = mejor época)", xlabel="época", ylabel="mIoU de validación")
ax[1].set(title="Pérdida de entrenamiento (Lovász ≠ CE: no comparar)", xlabel="época", ylabel="pérdida")
ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "fig1_ronda2_E_curvas.png", dpi=180); plt.close(fig)

# ---- Fig 2: dónde adaptar (B_T) ----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for z, c in {"early": "#1f77b4", "mid": "#2ca02c", "late": "#ff7f0e"}.items():
    curves(ax, load(f"b_T/P2T_{z}_s*.json"), c, f"{z} (LR {load(f'b_T/P2T_{z}_s0.json')[0]['lora_lr']:g})")
ax.set(title="LoRA rango 8 por zona (2 semillas)", xlabel="época", ylabel="mIoU de validación"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig2_HRW_zonas_curvas.png", dpi=180); plt.close(fig)

# ---- Fig 3: apariencia (A_T) -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for t, c in {"original": "#444444", "hist_eq": "#e07b39", "clahe": "#1f77b4", "standardize": "#2ca02c", "blur": "#9467bd"}.items():
    curves(ax, load(f"a_T/P1_{t}_s*.json"), c, t)
ax.set(title="Sonda lineal con 5 versiones de imagen (3 semillas)", xlabel="época", ylabel="mIoU de validación"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig3_HRA_apariencia_curvas.png", dpi=180); plt.close(fig)

# ---- Fig 4: baseline DeepLabV3 (D_T) -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for k, c, lab in (("pre", "#1f77b4", "DeepLabV3-R50, ImageNet"), ("scratch", "#ff7f0e", "DeepLabV3-R50, sin preentrenar")):
    r = load(f"d_T/BL_{k}_full_s0.json")[0]; y = r["val_hist"]
    ax.plot(np.arange(1, len(y) + 1), y, "-o", ms=3, color=c, label=f"{lab} (mejor época {r['best_epoch']}, paró en {r['epochs_run']})")
    ax.plot(r["best_epoch"], y[r["best_epoch"] - 1], "o", ms=7, color=c, mec="k")
ax.set(title="Baseline de punta a punta (1 semilla)", xlabel="época", ylabel="mIoU de validación"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig4_HRB_baseline_curvas.png", dpi=180); plt.close(fig)

# ---- Fig 5: veredictos en T (forest plot) ------------------------------------------------------------------------------------
csv = (ROOT / "results" / "estudio_principal" / "veredictos_ronda2.csv").read_text(encoding="utf-8")
ROWS = [  # (etiqueta, delta, lo, hi, margen, texto exacto que debe aparecer en el CSV)
    ("HRW  late − early", +0.033, +0.023, +0.044, 0.01, "late−early T +0.033 [+0.023, +0.044]"),
    ("HRW  mid − early", +0.030, +0.019, +0.041, 0.01, "mid−early T +0.030 [+0.019, +0.041]"),
    ("HRW  late − mid", +0.004, -0.005, +0.013, 0.01, "late−mid T +0.004 [-0.005, +0.013]"),
    ("HRA  hist_eq − original", -0.008, -0.016, -0.001, 0.01, "hist_eq T -0.008 [-0.016, -0.001]"),
    ("HRA  clahe − original", -0.010, -0.018, -0.004, 0.01, "clahe T -0.010 [-0.018, -0.004]"),
    ("HRA  standardize − original", -0.001, -0.005, +0.002, 0.01, "standardize T -0.001 [-0.005, +0.002]"),
    ("HRAb blur − original", +0.000, -0.009, +0.009, 0.01, "blur − original T +0.000 [-0.009, +0.009]"),
    ("HR2  LoRA − decoder (mIoU)", +0.006, -0.002, +0.014, 0.01, "T: +0.006 [-0.002, +0.014]"),
    ("HRB  sonda − DeepLabV3", -0.005, -0.019, +0.010, 0.01, "T: -0.005 [-0.019, +0.010]"),
    ("HR1  Lovász − CE (IoU big_rock)", +0.024, +0.010, +0.039, 0.02, "T: +0.024 [+0.010, +0.039]"),
]
for r in ROWS:
    assert r[5] in csv, f"no está en veredictos_ronda2.csv: {r[5]}"
fig, ax = plt.subplots(figsize=(7.4, 4.3)); y = np.arange(len(ROWS))[::-1]
ax.axvspan(-0.01, 0.01, color="#cccccc", alpha=0.5, lw=0)
for yi, (lab, d, lo, hi, tol, _) in zip(y, ROWS):
    real = (lo > 0 or hi < 0) and abs(d) >= tol; eq = lo > -tol and hi < tol
    col = "#d62728" if real else ("#2ca02c" if eq else "#7f7f7f")
    ax.plot([lo, hi], [yi, yi], "-", color=col, lw=2.2); ax.plot(d, yi, "o", color=col)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS])
ax.set(xlabel="diferencia (mIoU; HR1: IoU de big_rock, margen 0.02) e IC 95 %", title="Veredictos en T: rojo real · verde equivalente · gris inconcluso")
fig.tight_layout(); fig.savefig(OUT / "fig5_veredictos_forest.png", dpi=180); plt.close(fig)
print("figuras en", OUT, sorted(p.name for p in OUT.glob("*.png")))
