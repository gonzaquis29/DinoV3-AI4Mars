# %% [markdown]
# # AI4Mars × DINOv3 — v2 · Cuaderno C: resultados y veredictos
#
# Solo lee resultados (corre en CPU, sin dataset ni GPU). `A4M_RUN=full` (por defecto) o `A4M_RUN=pilot`. Junta las salidas de los cuadernos A y B, calcula
# intervalos de confianza y aplica las reglas escritas **antes** de correr (`hypotheses_v2.json`).
#
# | Sección | Pregunta |
# |---|---|
# | 1 | Estadística: cómo decidimos si una diferencia es real |
# | 2 | **P1 — datos:** ¿la brecha Web→Marte es de apariencia? |
# | 3 | **P2 — modelo:** ¿dónde conviene adaptar? |
# | 4 | **P3 — errores:** ¿cuánto es ambigüedad de etiquetas y sabe el modelo cuándo falla? |
# | 5 | Teardown (preliminar) |
# | 6 | Relectura post-hoc de la corrida final con la estadística nueva |
# | 6b | Baseline supervisado (DeepLabV3 ResNet-50) frente a la sonda lineal |

# %% 0. Configuración
import os, sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(globals().get("__file__", "notebooks/v2/C_resultados.py")).resolve().parent
sys.path.insert(0, str(HERE))
from a4m_lib import miou_np, iou_np, ece, CLASS_NAMES

# Qué corrida se analiza: A4M_RUN=full (por defecto, las 15,564 imágenes) o A4M_RUN=pilot (2,000 imágenes).
# Lee los resultados crudos de results/v2_<run>/raw/{a,b,d} y escribe tablas y figuras en results/v2_<run>/.
RUN = os.environ.get("A4M_RUN", "full")
BASE = HERE.parents[1] / f"results/v2_{RUN}"
V2 = [Path(p) for p in os.environ.get("A4M_V2", ":".join(str(BASE / "raw" / d) for d in ("a", "b", "d"))).split(":")]
V1 = Path(os.environ.get("A4M_V1", HERE.parents[1] / "results/final_run/results"))
OUT = Path(os.environ.get("A4M_OUT_C", BASE)); (OUT / "figures").mkdir(parents=True, exist_ok=True)
H = json.load(open(HERE / "hypotheses_v2.json"))
TOL, NB, BLOCK, SEEDS = H["common"]["tol"], H["common"]["n_boot"], 10, (0, 1, 2)
print("hipótesis registradas:", H["registered_at"], "| resultados v2 en", [str(p) for p in V2])


def load(name, folders=V2):
    if name.startswith("P2_probe_"):                                 # versión final: el control de P2 es la sonda de P1 (misma receta)
        name = name.replace("P2_probe_", "P1_original_")
    for f in folders:
        j, z = f / f"{name}.json", f / f"{name}.npz"
        if j.exists() and z.exists():
            return json.load(open(j)), dict(np.load(z))
    return None


def group(prefix, key, folders=V2, seeds=SEEDS):
    """{set: [array por semilla]} para una configuración."""
    res = [load(f"{prefix}_s{s}", folders) for s in seeds]
    res = [r for r in res if r]
    return {k: [r[1][f"{k}__{key}"] for r in res if f"{k}__{key}" in r[1]] for k in ("val", "M1", "M2", "M3")}, [r[0] for r in res]

# %% [markdown]
# ## 1. Estadística
# * **Bootstrap por bloques:** las imágenes de test vienen ordenadas por el reloj de la nave, y las vecinas se parecen
#   (misma zona, misma luz). Remuestreamos **bloques de 10 imágenes consecutivas**, no imágenes sueltas; así el intervalo
#   no es optimista.
# * **Pareado:** las dos configuraciones se evalúan sobre el mismo remuestreo; las semillas se promedian.
# * **Real:** el IC 95 % de la diferencia excluye 0 **y** |Δ| ≥ 0.01. **Equivalente:** todo el IC cae dentro de ±0.01.
#   Si no es ninguna de las dos, **inconcluso**. Con varias comparaciones a la vez, Holm.

# %% 1. funciones
def block_idx(n, B=NB, L=BLOCK, seed=0):
    """B remuestreos de n imágenes formados por bloques de L imágenes consecutivas."""
    L = min(L, n); rng = np.random.default_rng(seed); nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, (B, nb))
    return (starts[..., None] + np.arange(L)).reshape(B, -1)[:, :n]


def boot(confs, idx, metric=miou_np):
    """confs: lista por semilla de [N,4,4] -> métrica promedio de semillas para cada remuestreo [B]."""
    return np.mean([metric(c[idx].sum(1)) for c in confs], 0)


def point(confs, metric=miou_np):
    return float(np.mean([metric(c.sum(0)) for c in confs]))


def compare(a, b, metric=miou_np, tol=TOL):
    """Δ = a - b con IC 95 % por bloques; p bilateral del bootstrap; etiqueta real / equivalente / inconcluso."""
    idx = block_idx(a[0].shape[0]); d = boot(a, idx, metric) - boot(b, idx, metric)
    delta = point(a, metric) - point(b, metric); lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    label = "real" if (lo > 0 or hi < 0) and abs(delta) >= tol else "equivalente" if (lo > -tol and hi < tol) else "inconcluso"
    return dict(delta=delta, lo=lo, hi=hi, p=p, label=label)


def holm(rows, alpha=0.05):
    """Ajuste de Holm sobre las p de una familia de comparaciones; 'real' exige además p ajustada < alpha."""
    order = np.argsort([r["p"] for r in rows]); m = len(rows); prev = 0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (m - k) * rows[i]["p"])); rows[i]["p_holm"] = prev
        if rows[i]["label"] == "real" and prev >= alpha:
            rows[i]["label"] = "inconcluso"
    return rows


fmt = lambda r: f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {r['label']}"
r0 = lambda arrs: [a[:, 0] if a.ndim == 4 else a for a in arrs]       # radio 0 (sin excluir bordes)
VERDICTS = []

# %% [markdown]
# ## 2. P1 — ¿la brecha es de apariencia?

# %% 2a. tabla y comparaciones contra las imágenes originales
P1 = {t: group(f"P1_{t}", "conf")[0] for t in ("original", "hist_eq", "clahe", "standardize", "blur")}
rows, cmp1 = [], {}
for t, g in P1.items():
    if not g["M1"]:
        continue
    row = {"imagen": t, "semillas": len(g["M1"])}
    for k in ("M1", "M2", "M3"):
        pts = [miou_np(c[:, 0].sum(0)) for c in g[k]]; row[k] = f"{np.mean(pts):.3f} ± {np.std(pts):.3f}"
    if t != "original":
        for k in ("M1", "M2"):
            cmp1[(t, k)] = compare(r0(g[k]), r0(P1["original"][k])); row[f"Δ {k}"] = fmt(cmp1[(t, k)])
    rows.append(row)
T1 = pd.DataFrame(rows); T1.to_csv(OUT / "P1_table.csv", index=False); print(T1.to_string(index=False))

# %% 2b. IoU por clase con y sin desenfoque (¿cae más la arena?)
if P1["blur"]["M1"]:
    per_class = lambda g: np.mean([iou_np((c1[:, 0].sum(0) + c2[:, 0].sum(0))) for c1, c2 in zip(g["M1"], g["M2"])], 0)
    drop = per_class(P1["original"]) - per_class(P1["blur"])
    print("caída de IoU por clase con desenfoque (M1+M2):", dict(zip(CLASS_NAMES, np.round(drop, 3))))

# %% 2c. veredictos HP1 y HP1b (reglas de hypotheses_v2.json)
if cmp1:
    photo = [cmp1[(t, k)] for t in ("hist_eq", "clahe", "standardize") for k in ("M1", "M2")]
    holm(photo)
    gain = any(cmp1[(t, "M1")]["label"] == "real" and cmp1[(t, "M1")]["delta"] > 0 and cmp1[(t, "M2")]["label"] == "real"
               and cmp1[(t, "M2")]["delta"] > 0 for t in ("hist_eq", "clahe", "standardize"))
    ok = all(r["label"] == "equivalente" or (r["label"] == "real" and r["delta"] < 0) for r in photo)
    VERDICTS.append(("HP1", "refutada" if gain else "apoyada" if ok else "inconclusa",
                     "; ".join(f"{t} {k}: {fmt(cmp1[(t, k)])}" for t in ("hist_eq", "clahe", "standardize") for k in ("M1", "M2"))))
    b1, b2 = cmp1[("blur", "M1")], cmp1[("blur", "M2")]
    real_drop = all(r["label"] == "real" and r["delta"] < 0 for r in (b1, b2))
    sand = CLASS_NAMES[int(np.argmax(drop))] == "sand"
    VERDICTS.append(("HP1b", "apoyada" if real_drop and sand else "refutada" if b1["label"] == b2["label"] == "equivalente" else "inconclusa",
                     f"M1 {fmt(b1)}; M2 {fmt(b2)}; mayor caída: {CLASS_NAMES[int(np.argmax(drop))]}"))

# %% 2d. figura P1
if cmp1:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ts = ["hist_eq", "clahe", "standardize", "blur"]
    for j, k in enumerate(("M1", "M2")):
        y = np.arange(len(ts)) + (j - 0.5) * 0.3
        d = [cmp1[(t, k)] for t in ts]
        ax.errorbar([r["delta"] for r in d], y, xerr=[[r["delta"] - r["lo"] for r in d], [r["hi"] - r["delta"] for r in d]],
                    fmt="o", ms=5, lw=1.5, color=["#2a6fdb", "#e07b39"][j], label=k, capsize=0)
    ax.axvspan(-TOL, TOL, color="0.92", zorder=0, label="±0.01 (equivalence)"); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(ts)), ts); ax.set_xlabel("Δ mIoU vs. original images (95 % CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures/P1_apariencia.png", dpi=200)

# %% [markdown]
# ## 3. P2 — ¿dónde conviene adaptar?

# %% 3a. tabla: cada zona frente a la sonda y entre zonas
P2 = {z: group(f"P2_{z}", "conf") for z in ("probe", "early", "mid", "late")}
rows = []
for z, (g, meta) in P2.items():
    if not g["M1"]:
        continue
    row = {"zona": z, "semillas": len(g["M1"]), "LoRA (k)": round(meta[0].get("trainable_lora", 0) / 1e3), "min/corrida": round(np.mean([m["minutes"] for m in meta]), 1)}
    for k in ("M1", "M2", "M3"):
        pts = [miou_np(c[:, 0].sum(0)) for c in g[k]]; row[k] = f"{np.mean(pts):.3f} ± {np.std(pts):.3f}"
    if z != "probe" and P2["probe"][0]["M1"]:
        for k in ("M1", "M2"):
            row[f"Δ sonda {k}"] = fmt(compare(r0(g[k]), r0(P2["probe"][0][k])))
    rows.append(row)
T2 = pd.DataFrame(rows); T2.to_csv(OUT / "P2_table.csv", index=False); print(T2.to_string(index=False))

pairs = [("late", "early"), ("late", "mid"), ("mid", "early")]
cmp2 = {}
if all(P2[z][0]["M1"] for z in ("early", "mid", "late")):
    fam = []
    for a, b in pairs:
        for k in ("M1", "M2"):
            cmp2[(a, b, k)] = compare(r0(P2[a][0][k]), r0(P2[b][0][k])); fam.append(cmp2[(a, b, k)])
    holm(fam)
    for (a, b, k), r in cmp2.items():
        print(f"{a} - {b} en {k}: {fmt(r)} (p Holm {r['p_holm']:.3f})")
    le = [cmp2[("late", "early", k)] for k in ("M1", "M2")]
    v = ("apoyada" if all(r["label"] == "real" and r["delta"] > 0 for r in le) else
         "refutada" if all(r["label"] == "real" and r["delta"] < 0 for r in le) else "inconclusa")
    VERDICTS.append(("HP2", v, "; ".join(f"{a}-{b} {k}: {fmt(r)}" for (a, b, k), r in cmp2.items())))

# %% [markdown]
# ## 4. P3 — ¿dónde falla y sabe que falla?
# Usamos la sonda sobre las imágenes originales (cuaderno A). Tres miradas:
# 1. **Acuerdo entre expertos:** en las imágenes de M1, los píxeles que también etiquetaron 3 expertos (M3) son de
#    alto acuerdo; los que solo están en M1 son de bajo acuerdo.
# 2. **Bordes:** el mIoU al excluir una franja alrededor de los bordes de las etiquetas.
# 3. **Calibración:** si la confianza del modelo coincide con su tasa de acierto (ECE), en general y donde predice roca grande.

# %% 4a. acuerdo entre expertos
ag = [load(f"P1_original_s{s}") for s in SEEDS]; ag = [r[1]["M1__agree"] for r in ag if r]
if ag:
    hi_, lo_ = [a[:, 0] for a in ag], [a[:, 1] for a in ag]
    r = compare(hi_, lo_, tol=0.05)
    print(f"mIoU alto acuerdo {point(hi_):.3f} | bajo acuerdo {point(lo_):.3f} | diferencia {fmt(r)}")
    print("IoU por clase (alto / bajo):", {c: (round(a, 3), round(b, 3)) for c, a, b in
          zip(CLASS_NAMES, np.mean([iou_np(h.sum(0)) for h in hi_], 0), np.mean([iou_np(l.sum(0)) for l in lo_], 0))})
    VERDICTS.append(("HP3a", "apoyada" if r["delta"] >= 0.05 and r["lo"] > 0 else
                     "refutada" if r["lo"] > -0.05 and r["hi"] < 0.05 else "inconclusa", fmt(r)))

# %% 4b. bordes: mIoU al excluir franjas de 0, 4, 8, 16 px
g = P1["original"]
if g["M1"]:
    for k in ("M1", "M2"):
        print(k, {rad: round(np.mean([miou_np(c[:, j].sum(0)) for c in g[k]]), 3) for j, rad in enumerate((0, 4, 8, 16))})

# %% 4c. calibración
cal = group("P1_original", "calib")[0]
if cal["M1"]:
    pooled = [np.concatenate([a, b]) for a, b in zip(cal["M1"], cal["M2"])]           # M1 + M2, por semilla
    ece_all = np.mean([ece(p.sum(0))[0] for p in pooled])
    gap = lambda p: (p[:, 1, :, 1].sum() - p[:, 1, :, 2].sum()) / max(p[:, 1, :, 0].sum(), 1)   # confianza - acierto, predice roca grande
    idx = block_idx(pooled[0].shape[0])
    gaps = np.mean([[gap(p[i]) for i in idx] for p in pooled], 0); g_pt = float(np.mean([gap(p) for p in pooled]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    miss = np.mean([1 - p[:, 2, :, 2].sum() / max(p[:, 2, :, 0].sum(), 1) for p in pooled])
    print(f"ECE (M1+M2): {ece_all:.3f} | donde predice roca grande: confianza - acierto = {g_pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f" | píxeles de roca grande no detectados: {miss:.2f}")
    VERDICTS.append(("HP3b", "apoyada" if ece_all <= 0.05 and g_pt >= 0.10 and lo > 0 else "refutada" if hi < 0.10 else "inconclusa",
                     f"ECE {ece_all:.3f}; brecha roca grande {g_pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"))
    # diagrama de confiabilidad
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for gi, (lab, col) in enumerate((("all labelled pixels", "#2a6fdb"), ("predicted big_rock", "#e07b39"))):
        s = sum(p.sum(0)[gi] for p in pooled); m = s[:, 0] > 50
        ax.plot((s[m, 1] / s[m, 0]), (s[m, 2] / s[m, 0]), "o-", ms=4, lw=1.5, color=col, label=lab)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--"); ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "figures/P3_calibracion.png", dpi=200)

# %% [markdown]
# ## 5. Teardown (preliminar)

# %% 5. tabla del teardown
tp = next((f / "teardown.json" for f in V2 if (f / "teardown.json").exists()), None)
if tp:
    T = json.load(open(tp))
    td = pd.DataFrame({"parámetros (M)": {k: v / 1e6 for k, v in T["params"].items()}, "GFLOPs (1 imagen 512 px)": T["gflops"]}).round(2)
    print(td.to_string()); print("latencia (ms):", {k: round(v, 1) for k, v in T["latency_ms"].items()}, "| memoria (MB):", {k: round(v) for k, v in T["peak_mem_mb"].items()})
    print("features de Marte por bloque:", [(r["block"], round(r["mean_norm"], 1), round(r["cos_in_out"], 3)) for r in T["mars_features"]])
    td.to_csv(OUT / "teardown_table.csv")

# %% [markdown]
# ## 6. Relectura post-hoc de la corrida final
# **Post-hoc:** se aplica la estadística nueva (bloques + equivalencia) a comparaciones ya hechas. No reemplaza los veredictos
# registrados de H1–H6; muestra cuánto cambian los intervalos al tener en cuenta que las imágenes vecinas se parecen.

# %% 6. comparaciones clave de la corrida final
def v1(names, key):
    out = {k: [] for k in ("M1", "M2")}
    for n in names:
        p = V1 / f"{n}.npz"
        if p.exists():
            z = np.load(p)
            for k in out:
                out[k].append(z[f"{k}__{key}"][:, 0])
    return out

if V1.exists():
    A = lambda l: v1([f"A_layers_s{s}" for s in SEEDS], l)
    checks = {"H1: bloque 10 - última capa": (A("10"), A("last_norm")),
              "H1: bloque 12 - última capa": (A("12"), A("last_norm")),
              "H3: DINOv3 - DINOv2": (v1([f"D_dinov3_b16_lvd_s{s}" for s in SEEDS], "probe"), v1([f"D_dinov2_b14_s{s}" for s in SEEDS], "probe")),
              "H4: decoder - sonda": (v1([f"B_decoder512_s{s}" for s in SEEDS], "decoder"), A("last_norm")),
              "FT4 - sonda (N=4000)": (v1([f"C_FT4_n4000_s{s}" for s in SEEDS], "probe"), v1([f"C_LP_n4000_s{s}" for s in SEEDS], "probe"))}
    rows = []
    for name, (a, b) in checks.items():
        for k in ("M1", "M2"):
            if a[k] and b[k]:
                r = compare(a[k], b[k]); rows.append({"comparación": name, "set": k, "Δ [IC bloques]": fmt(r),
                                                       "ancho IC": round(r["hi"] - r["lo"], 3)})
    T6 = pd.DataFrame(rows); T6.to_csv(OUT / "posthoc_final_run.csv", index=False); print(T6.to_string(index=False))

# %% [markdown]
# ## 6b. Baseline supervisado entrenado por nosotros (DeepLabV3 ResNet-50, mismo split y test)

# %% 6b. tabla del baseline frente a la sonda lineal
BLN = {"BL_pre_full": "DeepLabV3 R50 (ImageNet), 15,564 img", "BL_scratch_full": "DeepLabV3 R50 (sin preentrenar), 15,564 img"}
PROBE = P1["original"]
rows = []
for k0, lab in (("probe", "sonda lineal DINOv3 congelado (4.6 k)"), *BLN.items()):
    g, meta = (PROBE, [{}]) if k0 == "probe" else group(k0, "conf")
    if not g["M1"]:
        continue
    row = {"modelo": lab, "semillas": len(g["M1"])}
    for k in ("M1", "M2", "M3"):
        pts = [miou_np(c[:, 0].sum(0)) for c in g[k]]; row[k] = f"{np.mean(pts):.3f}" + (f" ± {np.std(pts):.3f}" if len(pts) > 1 else "")
    if k0 != "probe":
        for k in ("M1", "M2"):
            row[f"Δ vs sonda {k}"] = fmt(compare(r0(g[k]), r0(PROBE[k])))
        row["épocas"] = meta[0].get("epochs_run", ""); row["pasos"] = meta[0].get("steps", "")
    rows.append(row)
if len(rows) > 1:
    T7 = pd.DataFrame(rows); T7.to_csv(OUT / "baseline_table.csv", index=False); print(T7.to_string(index=False))

# %% [markdown]
# ## Veredictos

# %% 7. veredictos
V = pd.DataFrame(VERDICTS, columns=["hipótesis", "veredicto", "evidencia"]); V.to_csv(OUT / "verdicts_v2.csv", index=False)
for _, r in V.iterrows():
    print(f"{r['hipótesis']:5s} {r['veredicto']:11s} {r['evidencia']}")
