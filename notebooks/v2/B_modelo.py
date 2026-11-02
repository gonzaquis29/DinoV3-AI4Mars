# %% [markdown]
# # AI4Mars × DINOv3 — v2 · Cuaderno B: el modelo
#
# **P2 — ¿dónde conviene adaptar DINOv3 para Marte?**
#
# El ViT-B/16 tiene 12 bloques. Adaptamos **solo dos bloques seguidos** con LoRA, siempre con el mismo presupuesto
# (rango 8 en todas las capas lineales de esos dos bloques), en tres zonas:
#
# | zona | bloques | si gana esta zona, sugiere que… |
# |---|---|---|
# | inicio | 1–2 | la brecha es de bajo nivel (apariencia, textura) |
# | medio | 6–7 | la brecha está en representaciones intermedias |
# | final | 11–12 | la brecha es semántica (cómo se agrupan los conceptos) |
#
# Control: la misma sonda lineal sin adaptar nada, entrenada con la misma receta.
# El resto del modelo queda congelado. Hipótesis HP2 en `hypotheses_v2.json` (escrita antes de correr).
#
# ## Cómo se corre (dos modos)
# | modo | cómo se activa | entrenamiento |
# |---|---|---|
# | **final** | `A4M_FULL=1` (y `A4M_ZONES=early,mid,late`) | **15,564** imágenes, **2 épocas** (~1,900 pasos) por corrida, **2 semillas**, LR fijo (el que ganó en validación en el piloto), caché de tokens en disco, una caché por zona que empieza justo antes de sus bloques |
# | piloto | (sin variable) | 2,000 imágenes, 10 épocas, 3 semillas, LR elegido en validación entre dos valores |
#
# En el modo final el control «sin adaptar» es la sonda de P1 (misma receta); no se vuelve a entrenar aquí.
#
# **Truco para que sea barato:** las imágenes pasan **una sola vez** por la entrada del modelo (patch embedding) y guardamos
# esos tokens. Cada entrenamiento parte de ahí y solo recalcula los 12 bloques. Sin aumento de datos (igual que la corrida final).

# %% 0. Configuración
import sys, os, json, time, shutil, threading, gc
from pathlib import Path

LIB = next((f for d in range(6) for f in Path("/kaggle/input").glob("*/" * d + "a4m_lib.py")), Path.cwd() / "a4m_lib.py")
sys.path.insert(0, str(LIB.parent))                                # a4m_lib.py: dataset adjunto en Kaggle, o junto al cuaderno
from a4m_lib import *
import a4m_lib

CFG = dict(size=512, n_pool=2000, n_val=500, seeds=(0, 1, 2), epochs=10, bs=16, head_lr=1e-3,
           rank=8, lora_lrs=(2e-4, 1e-3),                      # el LR de LoRA se elige en validación (semilla 0) para cada zona
           zones={"early": (0, 1), "mid": (5, 6), "late": (10, 11)})   # índices 0-based: bloques 1-2, 6-7, 11-12
FIXED_LR = {"early": 2e-4, "mid": 1e-3, "late": 1e-3}            # versión final: los LR que ganaron en validación en el piloto (N=2,000); no se vuelven a barrer
ZONES_TO_RUN = [z for z in os.environ.get("A4M_ZONES", "early,mid,late").split(",") if z]
if FULL:                                                        # versión final: todas las imágenes de entrenamiento
    CFG.update(epochs=2, patience=None, seeds=(0, 1))           # 2 épocas (~1,900 pasos) por corrida: es lo que cabe en el tiempo; mejor época por validación
if SMOKE:
    CFG.update(size=64, n_pool=24, n_val=6, seeds=(0, 1), epochs=2, bs=4)
if QUICK:
    CFG.update(n_pool=64, seeds=(0, 1), epochs=1, lora_lrs=(2e-4,))
setup_dirs()
DEVS = devices()
shutil.copy(Path(a4m_lib.__file__).with_name("hypotheses_v2.json"), OUT / "hypotheses_v2.json")
print("GPUs:", DEVS)

# %% [markdown]
# ## LoRA en 15 líneas
# Una capa lineal congelada `W` se reemplaza por `W x + (B A x) · escala`, donde `A` (r × entrada) y `B` (salida × r)
# son pequeñas y entrenables. `B` empieza en cero, así que al inicio el modelo es exactamente el original.

# %% 1. LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.base, self.scale = base, (alpha or r) / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def add_lora(block, r):
    """Reemplaza cada nn.Linear del bloque (atención q, k, v, o y las dos del MLP) por su versión LoRA."""
    for name, mod in list(block.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, r))

# %% [markdown]
# ## Datos y tokens de entrada (una sola vez)

# %% 2. imágenes y tokens
ROOT = make_fake_dataset(OUT / "fake_data") if SMOKE else find_root()
SPLITS = build_splits(ROOT, n_val=CFG["n_val"])
if FULL:
    TMP, CFG["n_pool"] = pick_tmp(len(SPLITS["train"]))
    print("n de entrenamiento:", CFG["n_pool"], "| zonas:", ZONES_TO_RUN, "| caché en", TMP)
pool = np.random.default_rng(0).permutation(len(SPLITS["train"]))[:CFG["n_pool"]]      # mismo subconjunto que el cuaderno A y la etapa D
DATA = {"train": load_images([SPLITS["train"][i] for i in pool], CFG["size"], native_labels=False)}
for k in EVAL_SETS:
    DATA[k] = load_images(SPLITS[k], CFG["size"], native_labels=True)
M3_FOR_M1 = load_m3_labels(SPLITS["M1_m3label"], DATA["M1"][1])

bb = load_backbone(DEVS[0]); info = bb_info(bb); grid = CFG["size"] // info["patch"]
t0 = time.time()
def build_caches(start, tag=None):
    """Tokens que entran al bloque `start` (0 = justo después del patch embedding). En la versión final el train va a disco."""
    return {k: extract_tokens(bb, ImageSet(*DATA[k], extra=M3_FOR_M1 if k == "M1" else None), start, DEVS[0],
                              disk_path=TMP / f"p2_{tag}.npy" if FULL and k == "train" else None)
            for k in ("train",) + EVAL_SETS}

if not FULL:
    CACHES = build_caches(0)
    print(f"tokens de entrada: {(time.time() - t0) / 60:.1f} min | train {tuple(CACHES['train'][0].shape)}")
    del DATA

# %% 3. comprobación: recalcular los 12 bloques desde los tokens guardados reproduce el modelo original
with torch.no_grad():
    pos0 = rope(bb, CFG["size"], DEVS[0])
    net = ReplayNet(info, grid, pos0, start=0).to(DEVS[0]).eval()
    x = torch.stack([ImageSet(*load_images(SPLITS["val"][:2], CFG["size"], True, workers=1))[i][0] for i in range(2)]).to(DEVS[0])
    out = bb(pixel_values=x, output_hidden_states=True)
    h = out.hidden_states[0]
    for blk in net.blocks:
        o = blk(h, position_embeddings=pos0); h = o[0] if isinstance(o, (tuple, list)) else o
    err = (net.norm(h) - out.last_hidden_state).abs().max().item()
assert err <= 1e-3 * max(out.last_hidden_state.abs().max().item(), 1.0), f"replay error {err:.2e}"
print(f"réplica OK (error máximo {err:.1e})"); del net

# %% [markdown]
# ## Un entrenamiento
# `zone=None` es el control (solo la sonda). Con una zona, se agrega LoRA a sus dos bloques y se entrena LoRA + sonda.

# %% 4. una corrida
def run(zone, seed, lora_lr, device, name):
    if have(name):
        return json.load(open(find_art(f"{name}.json")))
    seed_all(seed)
    first = CFG["zones"][zone][0] if zone else info["depth"]          # desde qué bloque hace falta calcular gradientes
    start = first if FULL else 0                                      # versión final: la caché ya empieza en el primer bloque de la zona
    caches = ZC[zone] if FULL else CACHES
    net = ReplayNet(info, grid, rope(bb, CFG["size"], device), start=start, grad_from=first)
    if zone:
        for b in CFG["zones"][zone]:
            add_lora(net.blocks[b - start], CFG["rank"])
    net.to(device); t0 = time.time()
    log = lambda m: print(f"[{name}]{m}")
    best, ep, hist = fit(net, caches, CFG["n_pool"], seed, CFG["epochs"], CFG["bs"],
                         {"head": CFG["head_lr"], "adapter": lora_lr}, device, log=log, patience=CFG.get("patience"))
    n_lora = sum(p.numel() for n, p in net.named_parameters() if "lora" in n)
    meta = dict(part="P2", zone=zone or "probe", seed=seed, lora_lr=lora_lr, rank=CFG["rank"], n=CFG["n_pool"], epochs=CFG["epochs"],
                val_miou=best, best_epoch=ep, val_hist=hist, trainable_lora=n_lora,
                trainable_head=sum(p.numel() for p in net.head.parameters()), minutes=round((time.time() - t0) / 60, 1))
    save_result(name, meta, test_all(net, caches, device))
    print(f"[{name}] listo: val {best:.4f}, {meta['minutes']} min, LoRA {n_lora / 1e3:.0f} k parámetros")
    del net; torch.cuda.empty_cache() if device.type == "cuda" else None
    return meta

# %% 5. una zona completa: elegir el LR con la semilla 0 en validación, luego semillas 1 y 2 con ese LR
def run_zone(zone, device):
    if zone is None:
        for s in CFG["seeds"]:
            run(None, s, 0.0, device, f"P2_probe_s{s}")
        return
    sweep = {lr: run(zone, CFG["seeds"][0], lr, device, f"P2_{zone}_lr{lr:g}_s{CFG['seeds'][0]}") for lr in CFG["lora_lrs"]}
    lr = max(sweep, key=lambda k: sweep[k]["val_miou"])
    print(f"[{zone}] LR elegido en validación: {lr:g}")
    json.dump({"zone": zone, "val_by_lr": {f"{k:g}": v["val_miou"] for k, v in sweep.items()}, "chosen": lr},
              open(RES_DIR / f"P2_{zone}_lr_choice.json", "w"), indent=1)
    src = f"P2_{zone}_lr{lr:g}_s{CFG['seeds'][0]}"
    for ext in (".json", ".npz"):
        shutil.copy(find_art(src + ext), RES_DIR / f"P2_{zone}_s{CFG['seeds'][0]}{ext}")
    for s in CFG["seeds"][1:]:
        run(zone, s, lr, device, f"P2_{zone}_s{s}")

# %% [markdown]
# ## Correr todo, repartido en las dos GPUs
# La zona «inicio» es la más cara (gradientes por los 12 bloques), así que va sola en una GPU; el resto en la otra.

# %% 6. lanzar
JOBS = [["early"], ["mid", "late", None]] if len(DEVS) > 1 else [["early", "mid", "late", None]]
ZC = {}

def run_zone_full(zone):
    """Versión final: una caché por zona (en disco), las dos GPUs reparten las corridas de la zona."""
    t0 = time.time(); ZC[zone] = build_caches(CFG["zones"][zone][0], zone)
    print(f"[{zone}] tokens de entrada del bloque {CFG['zones'][zone][0] + 1}: {(time.time() - t0) / 60:.1f} min")
    def par(jobs):
        out, th = {}, []
        def go(i, j):
            try:
                out[i] = run(*j[:3], DEVS[i % len(DEVS)], j[3])
            except Exception as e:
                import traceback; traceback.print_exc(); out[i] = None; errors.append((j[3], repr(e)))
        for i, j in enumerate(jobs):
            th.append(threading.Thread(target=go, args=(i, j))); th[-1].start()
        for t in th:
            t.join()
        return out
    lr = FIXED_LR[zone]
    json.dump({"zone": zone, "chosen": lr, "note": "LR fijado con la validación del piloto (N=2,000); no se barrió de nuevo"}, open(RES_DIR / f"P2_{zone}_lr_choice.json", "w"), indent=1)
    par([(zone, s, lr, f"P2_{zone}_s{s}") for s in CFG["seeds"]])
    ZC[zone]["train"][0].close(); ZC.pop(zone); gc.collect()
    print(f"[{zone}] zona terminada en {(time.time() - t0) / 60:.1f} min")
errors = []

def worker(zones, device):
    for z in zones:
        try:
            run_zone(z, device)
        except Exception as e:
            import traceback; traceback.print_exc(); errors.append((z, repr(e)))

t0 = time.time()
if FULL:
    for z in ZONES_TO_RUN:
        try:
            run_zone_full(z)
        except Exception as e:
            import traceback; traceback.print_exc(); errors.append((z, repr(e)))
else:
    threads = [threading.Thread(target=worker, args=(zs, DEVS[i])) for i, zs in enumerate(JOBS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
print(f"P2 terminado en {(time.time() - t0) / 60:.1f} min | errores: {errors or 'ninguno'}")

# %% 7. tabla rápida (intervalos de confianza y veredicto en el cuaderno C)
for z in ["probe", "early", "mid", "late"]:
    res = [load_result(f"P2_{z}_s{s}") for s in CFG["seeds"]]
    res = [r for r in res if r]
    if res:
        m = {k: np.mean([miou_np(r[1][f"{k}__conf"][:, 0].sum(0)) for r in res]) for k in ("val", "M1", "M2", "M3")}
        print(f"{z:6s} semillas {len(res)} | " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
              + f" | LoRA {res[0][0]['trainable_lora'] / 1e3:.0f} k | {np.mean([r[0]['minutes'] for r in res]):.0f} min/corrida")
