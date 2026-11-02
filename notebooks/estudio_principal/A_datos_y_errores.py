# %% [markdown]
# # AI4Mars × DINOv3 — Cuaderno A: datos y errores
#
# **Pregunta central:** ¿cómo se extienden las features de DINOv3 (preentrenado con imágenes web) al dominio de Marte?
#
# Este cuaderno cubre:
#
# | Sección | Qué responde |
# |---|---|
# | 1. Teardown | Cuánto pesa y cuánto cuesta cada bloque del ViT-B/16 (parámetros, FLOPs, latencia, memoria) y cómo cambian las features de Marte bloque a bloque |
# | 2. **P1 — datos** | ¿La brecha Web→Marte es de *apariencia*? Misma sonda, imágenes con la apariencia modificada |
# | 3. **P3 — errores** (datos) | Guarda, para cada imagen de test, la confianza por píxel y el acuerdo entre expertos. El análisis está en el cuaderno C |
#
# Las hipótesis se escribieron **antes** de correr (`hipotesis_registradas.json`; la versión final se volvió a registrar en `hipotesis_datos_completos.json`, con fecha).
#
# ## Cómo se corre (dos modos)
# | modo | cómo se activa | entrenamiento | para qué |
# |---|---|---|---|
# | **final** | `A4M_FULL=1` | las **15,564** imágenes de entrenamiento, sonda hasta 10 épocas con parada temprana (paciencia 2; cortó casi todas en la época 5, antes de que el LR bajara: limitación declarada), las 3 semillas comparten lotes, caché de tokens en disco | resultados del artículo (`results/estudio_principal/`) |
# | piloto | (sin variable) | **2,000** imágenes (el mismo subconjunto de la etapa D de la corrida final), 20 épocas | réplica independiente (`results/estudio_principal_piloto/`) |
#
# Se corre en Kaggle (T4); `A4M_SMOKE=1` hace una prueba mínima sin datos reales ni GPU.
# Todo lo que no ayuda a explicar el experimento (lectura de archivos, bucle de entrenamiento, métricas) está en `a4m_lib.py`.

# %% 0. Configuración
import sys, json, time, shutil
from pathlib import Path

LIB = next((f for d in range(6) for f in Path("/kaggle/input").glob("*/" * d + "a4m_lib.py")), Path.cwd() / "a4m_lib.py")
sys.path.insert(0, str(LIB.parent))                                # a4m_lib.py: dataset adjunto en Kaggle, o junto al cuaderno
from a4m_lib import *
import a4m_lib

CFG = dict(size=512, n_pool=2000, n_val=500, seeds=(0, 1, 2), epochs=20, bs=16, head_lr=1e-3, patience=None)
if FULL:                                                          # versión final: todas las imágenes de entrenamiento, parada temprana
    CFG.update(epochs=10, patience=2)
if SMOKE:
    CFG.update(size=64, n_pool=24, n_val=6, seeds=(0, 1), epochs=2, bs=4)
if QUICK:
    CFG.update(n_pool=64, seeds=(0,), epochs=1)
setup_dirs()
DEV = devices()[0]
hyp = Path(a4m_lib.__file__).with_name("hipotesis_registradas.json")
shutil.copy(hyp, OUT / "hipotesis_registradas.json")
print("device:", DEV, "| hipótesis registradas el", json.load(open(hyp))["registered_at"])

# %% [markdown]
# ## Datos
# Mismo split que la corrida final: 500 imágenes de validación (las últimas por reloj de la nave) y el resto para entrenar.
# En el modo final se usan todas las de entrenamiento; en el piloto, 2,000. El caso «imágenes originales» debe reproducir el
# resultado ya conocido (≈ 0.677 mIoU en M1): es un control de que nada cambió.

# %% 1. cargar imágenes una sola vez (en RAM, a 512 px, un canal)
ROOT = make_fake_dataset(OUT / "fake_data") if SMOKE else find_root()
SPLITS = build_splits(ROOT, n_val=CFG["n_val"])
if FULL:
    TMP, CFG["n_pool"] = pick_tmp(len(SPLITS["train"]))
    print("n de entrenamiento:", CFG["n_pool"], "| caché en", TMP)
pool = np.random.default_rng(0).permutation(len(SPLITS["train"]))[:CFG["n_pool"]]
DATA = {"train": load_images([SPLITS["train"][i] for i in pool], CFG["size"], native_labels=False)}
for k in EVAL_SETS:
    DATA[k] = load_images(SPLITS[k], CFG["size"], native_labels=True)
M3_FOR_M1 = load_m3_labels(SPLITS["M1_m3label"], DATA["M1"][1])
print({k: len(v[0]) for k, v in DATA.items()})

# %% [markdown]
# ## 1. Teardown
# No es una pregunta de investigación: describe el modelo para entender sus costos y justificar dónde intervenir (cuaderno B).

# %% 1a. parámetros por bloque
bb = load_backbone(DEV); info = bb_info(bb)
count = lambda m: sum(p.numel() for p in m.parameters())
TEAR = {"params": {"patch_embed": count(bb.embeddings),
                   **{f"block_{i + 1}": count(b) for i, b in enumerate(info["blocks"])},
                   "final_norm": count(info["norm"]), "total": count(bb)}}
print({k: f"{v / 1e6:.2f} M" for k, v in TEAR["params"].items()})

# %% 1b. FLOPs por bloque (una imagen de 512 px), contados por PyTorch
from torch.utils.flop_counter import FlopCounterMode

def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()

x1 = to_input(DATA["val"][0][0])[None].to(DEV)
pos = rope(bb, CFG["size"], DEV)
with torch.no_grad():
    hs = bb(pixel_values=x1, output_hidden_states=True).hidden_states
    TEAR["gflops"] = {"patch_embed": flops(lambda: bb.embeddings(x1)) / 1e9}
    for i, blk in enumerate(info["blocks"]):
        TEAR["gflops"][f"block_{i + 1}"] = flops(lambda: blk(hs[i], position_embeddings=pos)) / 1e9
    TEAR["gflops"]["total"] = flops(lambda: bb(pixel_values=x1)) / 1e9
print({k: round(v, 2) for k, v in TEAR["gflops"].items()})

# %% 1c. latencia y memoria en esta GPU (fp16, sin gradientes)
def latency_ms(fn, n=20):
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize(DEV)
    t = time.time()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize(DEV)
    return (time.time() - t) / n * 1000

TEAR["latency_ms"], TEAR["peak_mem_mb"] = {}, {}
for bs in (1, 16):
    xb = x1.expand(bs, -1, -1, -1).contiguous()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEV)
    with torch.no_grad(), autocast(DEV):
        TEAR["latency_ms"][f"bs{bs}"] = latency_ms(lambda: bb(pixel_values=xb))
    if DEV.type == "cuda":
        TEAR["peak_mem_mb"][f"bs{bs}"] = torch.cuda.max_memory_allocated(DEV) / 2**20
with torch.no_grad(), autocast(DEV):
    h0 = hs[0].expand(16, -1, -1).contiguous()
    TEAR["latency_ms"]["one_block_bs16"] = latency_ms(lambda: info["blocks"][5](h0, position_embeddings=pos))
print(TEAR["latency_ms"], TEAR["peak_mem_mb"])

# %% 1d. ¿cómo cambian las features de Marte bloque a bloque? (32 imágenes de validación)
xs = torch.stack([to_input(g) for g in DATA["val"][0][:32]]).to(DEV)
with torch.no_grad(), autocast(DEV):
    hs = [h.float() for h in bb(pixel_values=xs, output_hidden_states=True).hidden_states]
TEAR["mars_features"] = [{"block": i, "mean_norm": hs[i].norm(dim=-1).mean().item(),
                          "cos_in_out": F.cosine_similarity(hs[i - 1], hs[i], dim=-1).mean().item()} for i in range(1, len(hs))]
for r in TEAR["mars_features"]:
    print(f"bloque {r['block']:2d}: norma media {r['mean_norm']:8.1f} | cos(entrada, salida) {r['cos_in_out']:.3f}")
json.dump(TEAR, open(RES_DIR / "teardown.json", "w"), indent=1)
del hs, xs

# %% [markdown]
# ## 2. P1 — ¿la brecha es de apariencia?
# Las fotos del rover son en escala de grises, con poco contraste y una iluminación distinta a la de las fotos web.
# Si eso fuera lo que limita a DINOv3, **arreglar la apariencia** debería mejorar la sonda. Probamos tres arreglos
# (no cambian el contenido, solo cómo se ve la imagen) y un cuarto cambio que **quita textura fina** (desenfoque):
# si la sonda depende de la textura, caerá, sobre todo en arena frente a suelo.
#
# El modelo, la sonda, las imágenes y las semillas son los mismos en todos los casos; solo cambia la imagen de entrada
# (en entrenamiento y en test).

# %% 2a. las cinco versiones de las imágenes
import cv2, gc

def hist_eq(g):                       # ecualización global del histograma
    return cv2.equalizeHist(g)

def clahe(g):                         # ecualización local con límite de contraste
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)

def standardize(g):                   # misma media y desviación en todas las imágenes (las de ImageNet en gris)
    z = (g.astype(np.float32) - g.mean()) / (g.std() + 1e-6)
    return np.clip(z * 0.226 * 255 + 0.449 * 255, 0, 255).astype(np.uint8)

def blur(g):                          # quita la textura fina (sigma = 2 px a 512 px)
    return cv2.GaussianBlur(g, (0, 0), 2)

TRANSFORMS = {"original": None, "hist_eq": hist_eq, "clahe": clahe, "standardize": standardize, "blur": blur}

# %% 2b. la sonda lineal: 4,612 parámetros sobre las features de la última capa
class Probe(nn.Module):
    def __init__(self, info, grid):
        super().__init__()
        self.n_prefix, self.grid = info["n_prefix"], grid
        self.head = lin_head(info["C"])          # BatchNorm + capa lineal 1x1 (768 -> 4 clases)

    def forward(self, tok):
        return self.head(to_map(tok.float(), self.n_prefix, self.grid))

# %% 2b'. entrenar varias sondas a la vez (versión final): mismos lotes, inicializaciones distintas por semilla, parada temprana por sonda
def fit_multi(nets, caches, seeds, epochs, bs, lr, device, patience, log=print):
    toks, labs = caches["train"][0], caches["train"][1]; n = len(toks); steps = n // bs
    opts, scheds, scalers = [], [], []
    for net in nets:
        net.to(device)
        o = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        opts.append(o); scheds.append(torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr, total_steps=epochs * steps, pct_start=0.1))
        scalers.append(torch.amp.GradScaler("cuda", enabled=device.type == "cuda"))
    k = len(nets); best, best_ep, state, hist, active = [-1.0] * k, [-1] * k, [None] * k, [[] for _ in range(k)], [True] * k
    t0 = time.time()
    for ep in range(epochs):
        for net in nets:
            net.train()
        order = np.random.default_rng(ep).permutation(n)
        for b in range(steps):
            idx = order[b * bs:(b + 1) * bs]
            t = toks[idx].to(device, non_blocking=True); y = labs[torch.from_numpy(idx)].to(device).long()
            for j, net in enumerate(nets):
                if not active[j]:
                    continue
                with autocast(device):
                    lg = net(t)
                loss = F.cross_entropy(F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False), y, ignore_index=IGNORE)
                opts[j].zero_grad(set_to_none=True); scalers[j].scale(loss).backward(); scalers[j].step(opts[j]); scalers[j].update(); scheds[j].step()
        for j, net in enumerate(nets):
            if not active[j]:
                continue
            v = float(miou_np(evaluate(net, caches["val"], device, radii=(0,))["conf"][:, 0].sum(0))); hist[j].append(v)
            if v > best[j]:
                best[j], best_ep[j] = v, ep; state[j] = {kk: x.detach().clone() for kk, x in net.state_dict().items()}
            if patience is not None and ep - best_ep[j] >= patience:
                active[j] = False
        log(f"    época {ep}: val " + " | ".join(f"s{seeds[j]} {hist[j][-1]:.4f}" + ("" if active[j] else " (parada)") for j in range(k)) + f" | {(time.time() - t0) / 60:.1f} min")
        if not any(active):
            break
    for j, net in enumerate(nets):
        net.load_state_dict(state[j])
    return best, best_ep, hist

# %% 2c. correr P1: para cada versión, extraer features una vez y entrenar la sonda con 3 semillas
grid = CFG["size"] // info["patch"]
for tname, fn in TRANSFORMS.items():
    names = [f"P1_{tname}_s{s}" for s in CFG["seeds"]]
    if all(have(n) for n in names):
        print(f"[{tname}] ya estaba"); continue
    t0 = time.time()
    caches = {k: extract_tokens(bb, ImageSet(*DATA[k], transform=fn, extra=M3_FOR_M1 if k == "M1" else None), "last", DEV,
                                disk_path=TMP / f"p1_{tname}.npy" if FULL and k == "train" else None)
              for k in ("train",) + EVAL_SETS}
    print(f"[{tname}] features en {(time.time() - t0) / 60:.1f} min")
    if FULL:                                          # las 3 semillas se entrenan a la vez sobre los mismos lotes (se lee el disco una vez)
        nets = []
        for seed in CFG["seeds"]:
            seed_all(seed); nets.append(Probe(info, grid))
        t0 = time.time()
        res = fit_multi(nets, caches, CFG["seeds"], CFG["epochs"], CFG["bs"], CFG["head_lr"], DEV, CFG["patience"], log=print)
        for net, seed, name, best, ep, hist in zip(nets, CFG["seeds"], names, *res):
            meta = dict(part="P1", transform=tname, seed=seed, n=CFG["n_pool"], epochs_max=CFG["epochs"], epochs_run=len(hist), patience=CFG["patience"],
                        val_miou=best, best_epoch=ep, val_hist=hist, trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                        minutes=round((time.time() - t0) / 60, 1))
            save_result(name, meta, test_all(net, caches, DEV))
            print(f"  {name}: val {best:.4f} (época {ep}, {len(hist)} épocas) | M1 {miou_np(np.load(RES_DIR / f'{name}.npz')['M1__conf'][:, 0].sum(0)):.4f}")
        net = nets[0]
        caches["train"][0].close()
        del caches, nets, net; gc.collect(); torch.cuda.empty_cache() if DEV.type == "cuda" else None
        continue
    for seed, name in zip(CFG["seeds"], names):
        if have(name):
            continue
        net = Probe(info, grid); t0 = time.time()
        best, ep, hist = fit(net, caches, CFG["n_pool"], seed, CFG["epochs"], CFG["bs"], {"head": CFG["head_lr"]}, DEV, log=lambda *a: None)
        meta = dict(part="P1", transform=tname, seed=seed, n=CFG["n_pool"], epochs=CFG["epochs"], val_miou=best, best_epoch=ep,
                    val_hist=hist, trainable=sum(p.numel() for p in net.parameters() if p.requires_grad), minutes=round((time.time() - t0) / 60, 1))
        save_result(name, meta, test_all(net, caches, DEV))
        print(f"  {name}: val {best:.4f} (época {ep}) | M1 {miou_np(np.load(RES_DIR / f'{name}.npz')['M1__conf'][:, 0].sum(0)):.4f}")
    del caches, net; gc.collect(); torch.cuda.empty_cache() if DEV.type == "cuda" else None
    print(f"  [{tname}] RAM libre {int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) / 2**20:.1f} GB")

# %% [markdown]
# ## Resumen rápido (los intervalos de confianza y los veredictos están en el cuaderno C)

# %% 3. tabla rápida: mIoU medio de 3 semillas por versión de la imagen
rows = []
for tname in TRANSFORMS:
    res = [load_result(f"P1_{tname}_s{s}") for s in CFG["seeds"]]
    res = [r for r in res if r]
    if res:
        rows.append({"imagen": tname, **{k: round(float(np.mean([miou_np(r[1][f"{k}__conf"][:, 0].sum(0)) for r in res])), 4)
                                          for k in ("val", "M1", "M2", "M3")}})
for r in rows:
    print(r)
calib = load_result(f"P1_original_s{CFG['seeds'][0]}")[1]
agree = calib["M1__agree"].sum(0)
print(f"P3 (vista previa, semilla 0): mIoU en píxeles de alto acuerdo {miou_np(agree[0]):.3f} | bajo acuerdo {miou_np(agree[1]):.3f}")
print(f"P3 (vista previa, semilla 0): ECE en M1 {ece(calib['M1__calib'].sum(0))[0]:.3f}")
print("listo:", sorted(p.name for p in RES_DIR.glob("*.json")))
