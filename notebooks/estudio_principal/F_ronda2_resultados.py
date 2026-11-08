# %% [markdown]
# # AI4Mars × DINOv3 — Cuaderno F: ronda 2, veredictos
#
# Solo lee resultados (CPU, sin dataset ni GPU) y aplica las reglas de `hipotesis_ronda2.json`, escritas **antes** de correr.
# Las tablas y figuras de entrenamiento están en el cuaderno E (se generan en Kaggle); aquí están los **veredictos**.
#
# | Parte | Hipótesis | Corrida que la alimenta |
# |---|---|---|
# | 1 | Relectura **post hoc** de la ronda 1: margen 0.02, cotas, efecto mínimo detectable (sin entrenar nada nuevo) | ronda 1 (`raw/{a,b,d}`) |
# | 2 | **HR1** pérdida · **HR2** LoRA sobre el decoder · **HR3** frente a lo publicado | E → `raw/e` |
# | 3 | **HRA / HRAb** apariencia de la imagen | A con T → `raw/a_T` |
# | 4 | **HRW** dónde adaptar | B con T → `raw/b_T` |
# | 5 | **HRB** sonda frente a DeepLabV3 entrenado más tiempo | D con T → `raw/d_T` |
#
# **Cómo se lee un veredicto.** Cada hipótesis es una pregunta de tres respuestas, y las tres son resultados concluyentes: «A mejor», «sin diferencia» (equivalente, ±0.01)
# o «B mejor». Solo es «inconcluso» si el intervalo cruza un margen; por eso cada veredicto lleva al lado la **mitad del intervalo** (la resolución real de la prueba).
# Todo se decide en **T** (2,000 imágenes); M1 y M2 se muestran al lado.
# Si falta alguna corrida, esa parte se salta y lo dice.

# %% 0. Configuración
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(globals().get("__file__", "notebooks/estudio_principal/F_ronda2_resultados.py")).resolve().parent
sys.path.insert(0, str(HERE))
from a4m_lib import miou_np, iou_np, ece, CLASS_NAMES
from a4m_stats import block_idx, contrast, holm, single, metric_boot, metric_point, label_of, fmt, r0, br_iou

BASE = Path(os.environ.get("A4M_BASE", HERE.parents[1] / "results" / "estudio_principal"))
RAW = lambda *d: [BASE / "raw" / x for x in d]
OUT = Path(os.environ.get("A4M_OUT_F", BASE)); (OUT / "figures").mkdir(parents=True, exist_ok=True)
H = json.load(open(HERE / "hipotesis_ronda2.json")); C = H["common"]
TOL, TOL_BR = C["tol"], C["tol_big_rock_iou"]
DECISION = "T"
VERDICTS = []
print("hipótesis de la ronda 2 registradas:", H["registered_at"])

# %% [markdown]
# ## Checklist de leakage (se corre SIEMPRE antes de leer cualquier resultado)
# Cada corrida de Kaggle guarda su `auditoria_datos.json` (cuaderno E, A, B y D). Aquí se comprueba, para cada una: (1) ningún archivo en dos conjuntos, (2) val y T a ≥ 24 h de
# todo el entrenamiento, (3) cuántas imágenes se quitaron del entrenamiento por casi-duplicados de las de evaluación. **Si algo no se cumple, este cuaderno se detiene.**

# %% 0b. checklist
AUDITS = sorted(BASE.glob("raw/*/auditoria_datos.json"))
GAP_S = 86400
if not AUDITS:
    print("AVISO: no hay auditoria_datos.json en raw/*/ — copia ahí el de cada corrida de Kaggle antes de interpretar nada.")
for f in AUDITS:
    a = json.load(open(f)); bad = {k: v for k, v in a["stem_overlap"].items() if v}
    assert not bad, f"LEAKAGE en {f.parent.name}: archivos en dos conjuntos {bad}"
    for k in ("val", "T"):
        if k in a["clock"]:
            assert a["clock"][k]["min_s"] >= GAP_S, f"LEAKAGE en {f.parent.name}: {k} a solo {a['clock'][k]['min_s']} s del entrenamiento"
    print(f"OK {f.parent.name:8s} | n = {a['n']} | solapes de archivos: 0 | distancia mínima val/T al entrenamiento: "
          + ", ".join(f"{k} {a['clock'][k]['min_s'] / 3600:.0f} h" for k in ("val", "T") if k in a["clock"])
          + f" | expertos M1-M3: {min(a['clock'][k]['min_s'] for k in ('M1', 'M2', 'M3')) / 3600:.1f} h"
          + f" | casi-duplicados quitados del entrenamiento: {a.get('dedup_removed', 0)} (umbral {a['dedup']} bits)")


def load(name, folders):
    for f in folders:
        j, z = Path(f) / f"{name}.json", Path(f) / f"{name}.npz"
        if j.exists() and z.exists():
            return json.load(open(j)), dict(np.load(z))
    return None


def group(prefix, folders, seeds, key="conf"):
    """{conjunto: [matriz por semilla]} y los metadatos de cada semilla."""
    res = [load(f"{prefix}_s{s}", folders) for s in seeds]; res = [r for r in res if r]
    return {k: [r[1][f"{k}__{key}"] for r in res if f"{k}__{key}" in r[1]] for k in ("val", "T", "M1", "M2", "M3")}, [r[0] for r in res]


def three_way(r, a, b):
    """Traduce una comparación a una de las tres respuestas definidas de antemano."""
    if r["label"] == "equivalente":
        return "sin diferencia"
    if r["label"] == "real":
        return f"{a} mejor" if r["delta"] > 0 else f"{b} mejor"
    return "inconcluso"


def ci_row(r):
    return f"{fmt(r)} (mitad del IC {r['half_ci']:.3f}, cota {r['bound']:.3f})"

# %% [markdown]
# ## Parte 1 — relectura de la ronda 1 (post hoc)
# Para cada comparación registrada en la ronda 1 (HP1, HP1b, HP2): el veredicto con el margen registrado (0.01), el que saldría con 0.02, y **la cota**
# (el mayor efecto compatible con los datos). «Inconcluso con tol 0.01 y cota 0.015» se lee como «si hay efecto, es de a lo sumo 1.5 puntos de mIoU».
# No cambia ningún veredicto de la ronda 1.

# %% 1. sensibilidad de la ronda 1
R1 = RAW("a", "b", "d")
P1 = {t: group(f"P1_{t}", R1, (0, 1, 2))[0] for t in ("original", "hist_eq", "clahe", "standardize", "blur")}
P2 = {z: group(f"P2_{z}", R1, (0, 1))[0] for z in ("early", "mid", "late")}
specs = [("HP1", f"{t} − original", P1[t], P1["original"]) for t in ("hist_eq", "clahe", "standardize")] + [("HP1b", "blur − original", P1["blur"], P1["original"])] \
      + [("HP2", f"{a} − {b}", P2[a], P2[b]) for a, b in (("late", "early"), ("late", "mid"), ("mid", "early"))]
rows = []
for hp, name, a, b in specs:
    for k in ("M1", "M2"):
        if not a[k] or not b[k]:
            continue
        r1 = contrast([r0(a[k])], [r0(b[k])], tol=TOL); r2 = contrast([r0(a[k])], [r0(b[k])], tol=0.02)
        rows.append({"hip": hp, "comparación": name, "conjunto": k, "Δ": round(r1["delta"], 4), "IC95": f"[{r1['lo']:+.3f}, {r1['hi']:+.3f}]", "tol 0.01": r1["label"],
                     "tol 0.02 (post hoc)": r2["label"], "cota |efecto| ≤": round(r1["bound"], 3), "mitad del IC": round(r1["half_ci"], 3)})
if rows:
    T_S = pd.DataFrame(rows); T_S.to_csv(OUT / "sensibilidad_ronda1.csv", index=False); print(T_S.to_string(index=False))
    print(f"\nEfecto mínimo detectable con 322 imágenes (mediana de la mitad del IC): {T_S['mitad del IC'].median():.3f} mIoU")
    print("Comparaciones decididas con tol 0.01:", int((T_S['tol 0.01'] != 'inconcluso').sum()), "de", len(T_S), "| con tol 0.02:", int((T_S['tol 0.02 (post hoc)'] != 'inconcluso').sum()))
else:
    print("no se encontraron resultados de la ronda 1 en", [str(p) for p in R1])

# %% [markdown]
# ## Parte 2 — ronda 2 (cuaderno E): HR1, HR2, HR3

# %% 2a. carga
VAR = ("dec_ce", "dec_lov", "lora_ce", "lora_lov")
G2 = {v: group(f"R2_{v}", RAW("e"), (0, 1, 2)) for v in VAR}
have2 = all(G2[v][0].get(DECISION) for v in VAR)
if not have2:
    print("Ronda 2: faltan resultados con el conjunto", DECISION, "en", [str(p) for p in RAW("e")], "→ se salta esta parte.")
else:
    g = {v: G2[v][0] for v in VAR}
    print("épocas entrenadas:", {v: G2[v][1][0]["epochs"] for v in VAR}, "| mejor época por semilla:", {v: [m["best_epoch"] + 1 for m in G2[v][1]] for v in VAR},
          "| T =", G2["dec_ce"][1][0].get("n_T"), "imágenes, entrenamiento:", G2["dec_ce"][1][0]["n"])

# %% 2b. HR1: la pérdida; HR2: LoRA (decididas en T; M1 y M2 se reportan)
if have2:
    ev = {}
    for k in (DECISION, "M1", "M2"):
        ev[k] = dict(loss_br=contrast([r0(g["dec_lov"][k]), r0(g["lora_lov"][k])], [r0(g["dec_ce"][k]), r0(g["lora_ce"][k])], metric=br_iou, tol=TOL_BR),
                     loss_all=contrast([r0(g["dec_lov"][k]), r0(g["lora_lov"][k])], [r0(g["dec_ce"][k]), r0(g["lora_ce"][k])]),
                     lora=contrast([r0(g["lora_ce"][k]), r0(g["lora_lov"][k])], [r0(g["dec_ce"][k]), r0(g["dec_lov"][k])]))
    fp = {k: (np.mean([c[:, 0].sum(0)[:3, 3].sum() for v in ("dec_ce", "lora_ce") for c in g[v][k]]), np.mean([c[:, 0].sum(0)[:3, 3].sum() for v in ("dec_lov", "lora_lov") for c in g[v][k]])) for k in (DECISION, "M1", "M2", "M3")}
    d = ev[DECISION]
    drop = d["loss_all"]["label"] == "real" and d["loss_all"]["delta"] < 0
    ans = three_way(d["loss_br"], "Lovász", "CE")
    ans = {"CE mejor": "Lovász peor"}.get(ans, ans)
    if ans == "Lovász mejor" and drop:
        ans = "mixto (mejora big_rock, empeora el mIoU global)"
    VERDICTS.append(("HR1 (Lovász, IoU big_rock)", ans, f"T: {ci_row(d['loss_br'])}; mIoU global T {fmt(d['loss_all'])}; M1 {fmt(ev['M1']['loss_br'])}; M2 {fmt(ev['M2']['loss_br'])}; "
                     + "; ".join(f"FP big_rock {k}: {int(a)} → {int(b)} px" for k, (a, b) in fp.items())))
    VERDICTS.append(("HR2 (LoRA sobre decoder, mIoU)", three_way(ev[DECISION]["lora"], "LoRA", "decoder solo").replace("decoder solo mejor", "LoRA peor"),
                     f"T: {ci_row(ev[DECISION]['lora'])}; M1 {fmt(ev['M1']['lora'])}; M2 {fmt(ev['M2']['lora'])}"))

# %% 2c. HR3: frente a lo publicado (M1, M2, M3), mejor variante elegida por validación
if have2:
    best = max(VAR, key=lambda v: np.mean([m["val_miou"] for m in G2[v][1]])); print("mejor variante por validación:", best)
    rows, status, exceeds = [], {}, {}
    for k in ("M1", "M2", "M3"):
        pt, lo, hi = single(r0(G2[best][0][k]))
        for pub, vals in C["published"].items():
            if pub == "note":
                continue
            ref = vals[k]; st = "supera" if lo >= ref else "por debajo" if hi < ref else "empata"
            rows.append({"conjunto": k, "nuestra": f"{pt:.3f} [{lo:.3f}, {hi:.3f}]", "publicado": pub, "cifra": ref, "estado": st})
            if pub.startswith("SegFormer"):
                status[k] = st
            if pub.startswith("UPerNet"):
                exceeds[k] = st
    T3 = pd.DataFrame(rows); T3.to_csv(OUT / "R2_vs_publicado.csv", index=False); print(T3.to_string(index=False))
    ans = "refutada" if "por debajo" in status.values() else "apoyada"
    extra = " | supera a SegFormer-B3 en los tres conjuntos" if all(s == "supera" for s in status.values()) else ""
    VERDICTS.append(("HR3 (vs publicado)", ans, f"mejor variante {best}; vs SegFormer-B3: " + ", ".join(f"{k} {status[k]}" for k in status)
                     + " | vs UPerNet-Swin-S: " + ", ".join(f"{k} {exceeds[k]}" for k in exceeds) + extra + " (indicativo: otro split y convención)"))

# %% [markdown]
# ## Parte 3 — HRA, HRAb: ¿la brecha es de apariencia? (cuaderno A con T)

# %% 3. HRA / HRAb
PA = {t: group(f"P1_{t}", RAW("a_T"), (0, 1, 2))[0] for t in ("original", "hist_eq", "clahe", "standardize", "blur")}
if not PA["original"].get(DECISION):
    print("HRA: faltan resultados con T en", [str(p) for p in RAW("a_T")], "→ se salta.")
else:
    rows = [contrast([r0(PA[t][DECISION])], [r0(PA["original"][DECISION])]) | {"t": t} for t in ("hist_eq", "clahe", "standardize")]
    holm(rows); out = [three_way(r, "transformada", "original") for r in rows]
    gain = any(r["label"] == "real" and r["delta"] > 0 for r in rows); ok = all(o in ("sin diferencia", "original mejor") for o in out)
    VERDICTS.append(("HRA (apariencia)", "apoyada" if ok else "refutada" if gain else "inconclusa",
                     "; ".join(f"{r['t']} T {ci_row(r)}" for r in rows) + "; M1: " + ", ".join(f"{t} {fmt(contrast([r0(PA[t]['M1'])], [r0(PA['original']['M1'])]))}" for t in ("hist_eq", "clahe", "standardize"))))
    rb = contrast([r0(PA["blur"][DECISION])], [r0(PA["original"][DECISION])])
    pc = lambda gq: np.mean([iou_np(c.sum(0)) for c in r0(gq[DECISION])], 0)
    drops = pc(PA["original"]) - pc(PA["blur"]); soil_top = int(np.argmax(drops)) == 0
    ans = ("apoyada" if rb["label"] == "equivalente" and soil_top else "refutada" if rb["label"] == "real" and rb["delta"] < 0 else "inconclusa")
    VERDICTS.append(("HRAb (textura)", ans, f"blur − original T {ci_row(rb)}; caída de IoU por clase: " + ", ".join(f"{c} {x:+.3f}" for c, x in zip(CLASS_NAMES, drops))))

# %% [markdown]
# ## Parte 4 — HRW: ¿dónde adaptar? (cuaderno B con T, ≥ 6 épocas)

# %% 4. HRW
PB = {z: group(f"P2T_{z}", RAW("b_T"), (0, 1))[0] for z in ("early", "mid", "late")}
if not all(PB[z].get(DECISION) for z in PB):
    print("HRW: faltan resultados con T en", [str(p) for p in RAW("b_T")], "→ se salta.")
else:
    pairs = (("late", "early"), ("late", "mid"), ("mid", "early"))
    rows = [contrast([r0(PB[a][DECISION])], [r0(PB[b][DECISION])]) | {"a": a, "b": b} for a, b in pairs]; holm(rows)
    if rows[0]["label"] == "real" and rows[0]["delta"] > 0:
        ans = "late mejor"
    elif rows[0]["label"] == "real" and rows[0]["delta"] < 0:
        ans = "early mejor"
    elif all(r["label"] == "equivalente" for r in rows):
        ans = "la zona no importa"
    else:
        ans = "inconcluso"
    VERDICTS.append(("HRW (dónde adaptar)", ans, "; ".join(f"{r['a']}−{r['b']} T {ci_row(r)}" for r in rows) + "; M1: "
                     + ", ".join(f"{a}−{b} {fmt(contrast([r0(PB[a]['M1'])], [r0(PB[b]['M1'])]))}" for a, b in pairs)))

# %% [markdown]
# ## Parte 5 — HRB: la sonda lineal frente a DeepLabV3 entrenado más tiempo (cuaderno D con T)

# %% 5. HRB
BL = load("BL_pre_full_s0", RAW("d_T"))
PROBE = PA["original"] if PA["original"].get(DECISION) else None
if BL is None or PROBE is None or f"{DECISION}__conf" not in BL[1]:
    print("HRB: faltan resultados con T (D y/o A) → se salta.")
else:
    r = contrast([r0(PROBE[DECISION])], [[BL[1][f"{DECISION}__conf"][:, 0]]])
    VERDICTS.append(("HRB (sonda vs DeepLabV3)", three_way(r, "sonda", "DeepLabV3"), f"T: {ci_row(r)}; épocas del baseline: {BL[0].get('epochs', BL[0].get('best_epoch'))}"))

# %% 6. veredictos
if VERDICTS:
    V = pd.DataFrame(VERDICTS, columns=["hipótesis", "veredicto", "evidencia"]); V.to_csv(OUT / "veredictos_ronda2.csv", index=False)
    for _, r in V.iterrows():
        print(f"{r['hipótesis']:32s} {r['veredicto']:20s} {r['evidencia']}")
