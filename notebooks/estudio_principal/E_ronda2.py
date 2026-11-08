# %% [markdown]
# # AI4Mars × DINOv3 — Cuaderno E: ronda 2 (receta completa, 2×2, entrenamiento largo)
#
# **Un solo cuaderno, de punta a punta, pensado para correr asíncrono en Kaggle.** Cada fase es una sección; al terminar, el cuaderno ejecutado queda con
# sus tablas y figuras como evidencia.
#
# | Fase | Qué hace |
# |---|---|
# | 0 | Configuración (épocas, semillas, LR) |
# | 1 | **Auditoría de datos / leakage**: solapes de archivos (asegura 0), distancia temporal y casi-duplicados entre test y entrenamiento |
# | 2 | Datos en RAM |
# | 3 | Modelo: LoRA, decoder multicapa, pérdida Lovász |
# | 4 | **Entrenamiento**: 4 variantes a la vez sobre un tronco congelado compartido |
# | 5 | **Evaluación** final de la mejor época de cada variante (val, T, M1, M2, M3) y guardado de resultados y pesos |
# | 6 | **Resultados y gráficos**: tabla, curvas de entrenamiento, IoU por clase, falsos positivos de `big_rock`, ejemplos cualitativos, calibración |
#
# Las hipótesis (HR1, HR2, HR3, …) y sus reglas de decisión están en `hipotesis_ronda2.json`, escritas **antes** de correr. Los veredictos los calcula el cuaderno F
# (CPU) a partir de los resultados que este deja.
#
# ## Diseño
# | variante | cuello | pérdida |
# |---|---|---|
# | `dec_ce` | decoder multicapa (bloques 3, 6, 9, 12), backbone congelado | entropía cruzada |
# | `dec_lov` | igual | entropía cruzada + Lovász-softmax |
# | `lora_ce` | igual, con LoRA (rango 8) en los bloques 11–12 | entropía cruzada |
# | `lora_lov` | igual, con LoRA en 11–12 | entropía cruzada + Lovász-softmax |
#
# Es un 2×2 (LoRA sí/no × pérdida) para atribuir la mejora a una pieza. Las 4 variantes comparten el tronco congelado (bloques 1–10) y **los mismos lotes y aumento** en cada paso.
#
# ## Cómo se evita el leakage y se logra potencia
# * **Partición por bloques en el tiempo**: el conjunto se ordena por el reloj real de la nave (no por nombre de archivo, que empieza por la cámara), val y T son bloques de 100 imágenes
#   repartidos por toda la misión (como los conjuntos de expertos) y se **quita del entrenamiento toda imagen a menos de 24 h** de val o T. La auditoría lo comprueba con un assert.
# * **T** (2,000 imágenes) es donde se deciden las hipótesis: con ~6× más imágenes que M1/M2, los intervalos son ~2.5× más estrechos. **M1, M2, M3** (expertos) solo sirven para compararse con lo publicado.
# * La época y todo lo demás se eligen **solo en val**. Ni T ni M1–M3 se usan para elegir nada.
# * La fase 1 **falla** (assert) si algún archivo está en dos conjuntos, y registra casi-duplicados y distancias de reloj en `auditoria_datos.json`.
#
# ## Variables de entorno
# | variable | efecto |
# |---|---|
# | `A4M_EPOCHS` | épocas por semilla (20 por defecto) |
# | `A4M_SEEDS` | semillas, p. ej. `0,1,2` (por defecto) |
# | `A4M_NBIG` | tamaño de T (2000 por defecto; 0 = sin T, reproduce la partición de la ronda 1) |
# | `A4M_DEDUP` | quita del entrenamiento las imágenes a ≤ esa distancia de Hamming (en bits de 256; 8 por defecto, 0 = no quitar) de cualquier imagen de evaluación |
# | `A4M_QUICK=1` / `A4M_SMOKE=1` | prueba corta en GPU (400 imágenes, 1 época; mide el costo por paso) / prueba en CPU con datos falsos |

# %% 0. Configuración
import sys, os, json, time, shutil, threading, gc, copy, random
from pathlib import Path

os.environ.setdefault("A4M_NBIG", "2000")                           # debe fijarse antes de importar a4m_lib
LIB = next((f for d in range(6) for f in Path("/kaggle/input").glob("*/" * d + "a4m_lib.py")), Path.cwd() / "a4m_lib.py")
sys.path.insert(0, str(LIB.parent))                                # a4m_lib.py y a4m_stats.py: dataset adjunto en Kaggle, o junto al cuaderno
from a4m_lib import *
import a4m_lib

CFG = dict(size=512, n_val=500, seeds=tuple(int(s) for s in os.environ.get("A4M_SEEDS", "0,1,2").split(",")),
           epochs=int(os.environ.get("A4M_EPOCHS", 20)), bs=16, head_lr=5e-4, lora_lr=3e-4, rank=8, dim=256,
           taps=(3, 6, 9, 12), lora_blocks=(10, 11), lovasz_weight=1.0, clip=1.0)     # hidden_states 3, 6, 9, 12 como el decoder de la corrida final; LoRA en los bloques 11-12 (0-based 10, 11)
VARIANTS = {"dec_ce": dict(lora=False, lov=False), "dec_lov": dict(lora=False, lov=True),
            "lora_ce": dict(lora=True, lov=False), "lora_lov": dict(lora=True, lov=True)}
TAG = "R2"
if SMOKE:
    CFG.update(size=64, n_val=6, seeds=(0, 1), epochs=2, bs=4)
if QUICK:
    CFG.update(seeds=(0,), epochs=1)
N_POOL_CAP = 24 if SMOKE else 400 if QUICK else None
setup_dirs(); (OUT / "figures").mkdir(parents=True, exist_ok=True)
DEVS = devices()
shutil.copy(Path(a4m_lib.__file__).with_name("hipotesis_ronda2.json"), OUT / "hipotesis_ronda2.json")
print("GPUs:", DEVS, "| épocas", CFG["epochs"], "| semillas", CFG["seeds"], "| T =", N_BIG, "imágenes, margen de", GAP_S // 3600, "h")

# %% [markdown]
# ## Fase 1 — Auditoría de datos (leakage)
# Antes de entrenar nada: (1) **ningún archivo** puede estar en dos de {train, val, T, M1–M3} (se asegura con `assert`; M1, M2 y M3 son los mismos 322 cuadros con
# distinto acuerdo de expertos, por diseño); (2) cuán cerca en el tiempo está cada imagen de evaluación de su vecina de entrenamiento más próxima;
# (3) cuántas imágenes de evaluación tienen un casi-duplicado en el entrenamiento (hash perceptual de 256 bits, distancia de Hamming ≤ 8).
# Los casi-duplicados se **quitan del entrenamiento** (`A4M_DEDUP=8`), sin tocar jamás las imágenes de evaluación; el número quitado queda en el resultado.

# %% 1. auditoría
ROOT = make_fake_dataset(OUT / "fake_data") if SMOKE else find_root()
SPLITS = build_splits(ROOT, n_val=CFG["n_val"])
SPLITS, AUDIT = audit_splits(SPLITS, dedup=int(os.environ.get("A4M_DEDUP", 8)))
json.dump(AUDIT, open(OUT / "auditoria_datos.json", "w"), indent=1)

# %% [markdown]
# ## Fase 2 — Datos
# Todas las imágenes de entrenamiento en RAM como uint8 a 512 px (~7 GB con etiquetas); el aumento (recorte 50–100 % con probabilidad 0.5 y volteo
# horizontal con probabilidad 0.5) se hace en la GPU, como en la corrida final. Val, T y M1–M3 con etiquetas a tamaño nativo.

# %% 2. imágenes
train_pairs = SPLITS["train"][:N_POOL_CAP] if N_POOL_CAP else SPLITS["train"]
W = 1 if SMOKE else 8
DATA = {"train": load_images(train_pairs, CFG["size"], native_labels=False, workers=W)}
for k in EVAL_SETS:
    DATA[k] = load_images(SPLITS[k], CFG["size"], native_labels=True, workers=W)
M3_FOR_M1 = load_m3_labels(SPLITS["M1_m3label"], DATA["M1"][1])
N_TRAIN = len(DATA["train"][0]); print("imágenes de entrenamiento:", N_TRAIN, "|", {k: len(DATA[k][0]) for k in EVAL_SETS})
QI = np.linspace(0, len(DATA["M1"][0]) - 1, 6).astype(int)           # 6 imágenes de M1 para los ejemplos cualitativos: espaciadas por reloj, elegidas SIN mirar resultados
_MEAN_T, _STD_T = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)


def make_batch(idx, rng, device):
    imgs, labs = DATA["train"]
    g = torch.from_numpy(np.stack([imgs[i] for i in idx])).to(device).float().div(255).unsqueeze(1)
    l = torch.from_numpy(np.stack([labs[i] for i in idx])).to(device).float().unsqueeze(1)
    S = g.shape[-1]; xs, ys = [], []
    for i in range(len(idx)):
        xi, yi = g[i:i + 1], l[i:i + 1]
        if rng.random() < 0.5:
            c = int(S * rng.uniform(0.5, 1.0)); x0, y0 = rng.randint(0, S - c), rng.randint(0, S - c)
            xi = F.interpolate(xi[..., y0:y0 + c, x0:x0 + c], size=(S, S), mode="bilinear", align_corners=False)
            yi = F.interpolate(yi[..., y0:y0 + c, x0:x0 + c], size=(S, S), mode="nearest")
        if rng.random() < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        xs.append(xi); ys.append(yi)
    x = torch.cat(xs).expand(-1, 3, -1, -1)
    return (x - _MEAN_T.to(device)) / _STD_T.to(device), torch.cat(ys).squeeze(1).long()

# %% [markdown]
# ## Fase 3 — Modelo
# **LoRA** (igual que el cuaderno B): `W x + (B A x)·escala`, con `B` en cero al inicio.
# **Decoder**: el de la corrida final (proyección 1×1 por nivel → fusión 3×3 → dos etapas de ×2 → clasificador), ~3.5 M parámetros.
# **Lovász-softmax** (Berman et al., 2018): sustituto convexo por tramos del IoU; castiga los falsos positivos de una clase rara
# (`big_rock`) mucho más que la entropía cruzada, que los promedia entre millones de píxeles fáciles.

# %% 3. LoRA, decoder y pérdida
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__()
        self.base, self.scale = base, (alpha or r) / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def add_lora(block, r):
    for name, mod in list(block.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, cname, LoRALinear(child, r))


class Decoder(nn.Module):
    def __init__(self, C, n_in, dim=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(C), nn.Conv2d(C, dim, 1)) for _ in range(n_in)])
        self.fuse = nn.Sequential(nn.Conv2d(dim * n_in, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.up1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.BatchNorm2d(dim // 2), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Conv2d(dim // 2, dim // 4, 3, padding=1), nn.BatchNorm2d(dim // 4), nn.ReLU(True))
        self.cls = nn.Conv2d(dim // 4, NUM_CLASSES, 1)

    def forward(self, maps):                                           # logits a 4x la rejilla de tokens; quien llama los lleva al tamaño de la etiqueta
        z = self.fuse(torch.cat([p(f.float()) for p, f in zip(self.proj, maps)], 1))
        z = self.up1(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        z = self.up2(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        return self.cls(z)


class Variant(nn.Module):
    """Una cabeza (decoder) y, opcionalmente, copias de los bloques 11-12 con LoRA."""

    def __init__(self, info, lora):
        super().__init__()
        self.head = Decoder(info["C"], len(CFG["taps"]), CFG["dim"])
        self.blocks = None
        if lora:
            self.blocks = nn.ModuleList([copy.deepcopy(info["blocks"][i]) for i in CFG["lora_blocks"]])
            for p in self.blocks.parameters():
                p.requires_grad = False
            for b in self.blocks:
                add_lora(b, CFG["rank"])

    def tail(self, h, pos, norm):
        for blk in self.blocks:
            o = blk(h, position_embeddings=pos); h = o[0] if isinstance(o, (tuple, list)) else o
        return norm(h)

    def train(self, mode=True):
        super().train(mode)
        if self.blocks is not None:
            self.blocks.eval()                                          # sin dropout en el backbone
        return self


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum(); inter = gts - gt_sorted.cumsum(0); union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    if len(gt_sorted) > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_softmax(prob, y):
    """Lovász-softmax sobre todo el lote (clases presentes), ignorando IGNORE. prob [B,C,H,W] ya en softmax, y [B,H,W]."""
    C = prob.shape[1]
    p = prob.permute(0, 2, 3, 1).reshape(-1, C); yy = y.reshape(-1); keep = yy != IGNORE
    p, yy = p[keep], yy[keep]; losses = []
    for c in range(C):
        fg = (yy == c).float()
        if fg.sum() == 0:
            continue
        err_sorted, perm = torch.sort((fg - p[:, c]).abs(), descending=True)
        losses.append(torch.dot(err_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if losses else p.sum() * 0.0


class Ctx:
    """Backbone congelado en un dispositivo, con su RoPE y sus dimensiones. `shared()` lo corre UNA vez por lote para las cuatro variantes."""

    def __init__(self, device):
        self.device = device; self.bb = load_backbone(device); self.info = bb_info(self.bb)
        self.grid = CFG["size"] // self.info["patch"]; self.pos = rope(self.bb, CFG["size"], device)
        self.norm, self.depth, self.npre = self.info["norm"], self.info["depth"], self.info["n_prefix"]
        assert CFG["taps"][-1] == self.depth and CFG["lora_blocks"] == (self.depth - 2, self.depth - 1)

    @torch.no_grad()
    def shared(self, x):
        """Mapas de los bloques 3, 6, 9; salida normalizada del bloque 12 (variantes sin LoRA); entrada al bloque 11 (variantes con LoRA)."""
        with autocast(self.device):
            hs = self.bb(pixel_values=x, output_hidden_states=True).hidden_states
            early = [to_map(hs[t], self.npre, self.grid) for t in CFG["taps"][:-1]]
            frozen_last = to_map(self.norm(hs[self.depth]), self.npre, self.grid)
        return early, frozen_last, hs[CFG["lora_blocks"][0]]

    def logits(self, v, sh):
        early, frozen_last, h_in = sh
        with autocast(self.device):
            last = frozen_last if v.blocks is None else to_map(v.tail(h_in, self.pos, self.norm), self.npre, self.grid)
            return v.head(early + [last])


def keep_state(v):
    return {k: x.detach().clone() for k, x in v.state_dict().items() if k.startswith("head.") or "lora" in k}

# %% [markdown]
# ## Fase 5 (herramientas) — Evaluación
# Una pasada por un conjunto evalúa **todas** las variantes con el mismo tronco. `full=True` produce lo mismo que `a4m_lib.test_all`: matrices de confusión por imagen con
# radios de borde, calibración y acuerdo entre expertos, de modo que los cuadernos de análisis las leen igual que las de la ronda 1.

# %% 5a. evaluación
@torch.no_grad()
def eval_all(ctx, V, key, full):
    ds = ImageSet(*DATA[key], extra=M3_FOR_M1 if key == "M1" else None)
    out = {n: dict(conf=[], calib=[], agree=[]) for n in V}
    for v in V.values():
        v.eval()
    for batch in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
        x = batch[0].to(ctx.device); y = batch[1].to(ctx.device).long(); sh = ctx.shared(x)
        for n, v in V.items():
            lg = F.interpolate(ctx.logits(v, sh).float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            pred = lg.argmax(1)
            out[n]["conf"].append(per_image_conf(pred, y, RADII if full else (0,)))
            if full and key != "val":
                out[n]["calib"].append(calib_bins(lg.softmax(1), y))
                if len(batch) > 2:
                    y3 = batch[2].to(ctx.device).long()
                    hi = torch.where(y3 != IGNORE, y, torch.full_like(y, IGNORE)); lo = torch.where(y3 == IGNORE, y, torch.full_like(y, IGNORE))
                    out[n]["agree"].append(np.stack([per_image_conf(pred, hi, (0,))[:, 0], per_image_conf(pred, lo, (0,))[:, 0]], 1))
    return {n: {k: np.concatenate(v) for k, v in d.items() if v} for n, d in out.items()}


@torch.no_grad()
def predict_some(ctx, V, key, idxs):
    """Mapas de clases (uint8, tamaño de la etiqueta) de unas pocas imágenes, para los ejemplos cualitativos."""
    ds = ImageSet(*DATA[key]); out = {n: [] for n in V}
    for v in V.values():
        v.eval()
    for i in idxs:
        x, y = ds[i][0][None].to(ctx.device), ds[i][1]; sh = ctx.shared(x)
        for n, v in V.items():
            lg = F.interpolate(ctx.logits(v, sh).float(), size=tuple(y.shape[-2:]), mode="bilinear", align_corners=False)
            out[n].append(lg.argmax(1)[0].cpu().numpy().astype(np.uint8))
    return {n: np.stack(p) for n, p in out.items()}

# %% [markdown]
# ## Fase 4 — Entrenamiento (una semilla = las cuatro variantes juntas)
# Optimizador (AdamW), planificador (OneCycle, 10 % de calentamiento) y escalador de precisión mixta **independientes por variante**; recorte de gradiente 1.0.
# Cada variante conserva su mejor época según el mIoU de **validación**. Se escribe un checkpoint por época (si Kaggle corta, se reanuda ahí).
# Al terminar la semilla (**fase 5**) cada variante, en su mejor época, se evalúa en val, T, M1, M2 y M3 y se guardan resultados, pesos y ejemplos cualitativos.

# %% 4. una semilla
INIT_LOCK = threading.Lock()


def run_seed(seed, device):
    names = {n: f"{TAG}_{n}_s{seed}" for n in VARIANTS}
    if all(have(nm) for nm in names.values()):
        print(f"[seed {seed}] ya está hecha"); return
    ctx = Ctx(device)
    with INIT_LOCK:                                                     # semillas globales de torch/random: construir de una en una
        seed_all(seed); V = {n: Variant(ctx.info, s["lora"]).to(device) for n, s in VARIANTS.items()}
    steps = N_TRAIN // CFG["bs"]; E = CFG["epochs"]
    opts, scheds, scalers, params = {}, {}, {}, {}
    for n, v in V.items():
        head = [p for p in v.head.parameters()]; lora = [p for nn_, p in v.named_parameters() if p.requires_grad and nn_.startswith("blocks.")]
        groups = [{"params": head, "lr": CFG["head_lr"]}] + ([{"params": lora, "lr": CFG["lora_lr"]}] if lora else [])
        opts[n] = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheds[n] = torch.optim.lr_scheduler.OneCycleLR(opts[n], max_lr=[g["lr"] for g in groups], total_steps=E * steps, pct_start=0.1)
        scalers[n] = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); params[n] = head + lora
    best = {n: -1.0 for n in V}; best_ep = {n: -1 for n in V}; best_state = {n: None for n in V}
    hist = {n: [] for n in V}; loss_hist = {n: [] for n in V}; start = 0; minutes_before = 0.0
    resume_p = RES_DIR / f"{TAG}_s{seed}_resume.pt"; found = find_art(resume_p.name)
    if found:
        s = torch.load(found, map_location=device, weights_only=False)
        for n, v in V.items():
            v.load_state_dict(s["models"][n], strict=False); opts[n].load_state_dict(s["opts"][n])
            scheds[n].load_state_dict(s["scheds"][n]); scalers[n].load_state_dict(s["scalers"][n])
        best, best_ep, best_state, hist, loss_hist, minutes_before = s["best"], s["best_ep"], s["best_state"], s["hist"], s["loss_hist"], s["minutes"]
        start = s["epoch"] + 1; print(f"[seed {seed}] reanudado en la época {start}")
    t0 = time.time()
    for ep in range(start, E):
        for v in V.values():
            v.train()
        order = np.random.default_rng(seed * 100003 + ep).permutation(N_TRAIN); arng = random.Random(seed * 7919 + ep)
        run = {n: 0.0 for n in V}; te = time.time(); n_logged = 0; t_probe = min(24, steps - 1)
        for b in range(steps):
            x, y = make_batch(np.sort(order[b * CFG["bs"]:(b + 1) * CFG["bs"]]), arng, device); sh = ctx.shared(x)
            for n, v in V.items():
                lo = ctx.logits(v, sh).float()
                lg = F.interpolate(lo, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(lg, y, ignore_index=IGNORE)
                if VARIANTS[n]["lov"]:                                    # Lovász en la rejilla del decoder (4x la de tokens): ordenar 4 M de píxeles por paso era el cuello de botella
                    loss = loss + CFG["lovasz_weight"] * lovasz_softmax(lo.softmax(1), y[:, 2::4, 2::4])
                opts[n].zero_grad(set_to_none=True); scalers[n].scale(loss).backward(); scalers[n].unscale_(opts[n])
                nn.utils.clip_grad_norm_(params[n], CFG["clip"]); scalers[n].step(opts[n]); scalers[n].update(); scheds[n].step()
                if b % 20 == 0:
                    run[n] += loss.item()
            n_logged += b % 20 == 0
            if ep == start and device.type == "cuda" and b == 4:
                torch.cuda.synchronize(); t_warm = time.time()               # los primeros pasos incluyen calentamiento (cudnn, memoria): no cuentan
            if ep == start and b == t_probe and b > 4 and device.type == "cuda":
                torch.cuda.synchronize(); full_steps = len(SPLITS["train"]) // 16; sps = (time.time() - t_warm) / (b - 4)
                print(f"[seed {seed}] COSTO MEDIDO: {sps:.2f} s/paso (4 variantes, bs {CFG['bs']}) → ~{sps * full_steps / 60:.1f} min/época con {full_steps} pasos, "
                      f"~{sps * full_steps * 20 / 3600:.1f} h para 20 épocas por semilla")
        res = eval_all(ctx, V, "val", full=False); line = []
        for n in V:
            vm = float(miou_np(res[n]["conf"][:, 0].sum(0))); hist[n].append(vm); loss_hist[n].append(run[n] / max(1, n_logged))
            if vm > best[n]:
                best[n], best_ep[n], best_state[n] = vm, ep, keep_state(V[n])
            line.append(f"{n} {vm:.4f}")
        print(f"[seed {seed}] ép {ep + 1}/{E} ({(time.time() - te) / 60:.1f} min) val: " + " | ".join(line))
        torch.save({"epoch": ep, "models": {n: keep_state(v) for n, v in V.items()}, "opts": {n: o.state_dict() for n, o in opts.items()},
                    "scheds": {n: o.state_dict() for n, o in scheds.items()}, "scalers": {n: o.state_dict() for n, o in scalers.items()},
                    "best": best, "best_ep": best_ep, "best_state": best_state, "hist": hist, "loss_hist": loss_hist,
                    "minutes": minutes_before + (time.time() - t0) / 60}, resume_p)
    # ---- fase 5: evaluación final de la mejor época de cada variante
    minutes = round(minutes_before + (time.time() - t0) / 60, 1)
    for n, v in V.items():
        v.load_state_dict(best_state[n], strict=False)
    arrays = {n: {} for n in V}
    for k in EVAL_SETS:
        r = eval_all(ctx, V, k, full=True)
        for n in V:
            for kk, a in r[n].items():
                arrays[n][f"{k}__{kk}"] = a
    qual = predict_some(ctx, V, "M1", QI)
    for n in V:
        arrays[n]["qual__pred"] = qual[n]
    torch.save({n: best_state[n] for n in V}, RES_DIR / f"{TAG}_s{seed}_weights.pt")
    for n, v in V.items():
        meta = dict(part="R2", variant=n, lora=VARIANTS[n]["lora"], lovasz=VARIANTS[n]["lov"], seed=seed, epochs=E, n=N_TRAIN, n_T=N_BIG, gap_s=GAP_S,
                    head_lr=CFG["head_lr"], lora_lr=CFG["lora_lr"], rank=CFG["rank"], taps=list(CFG["taps"]), val_miou=best[n], best_epoch=best_ep[n],
                    val_hist=hist[n], loss_hist=loss_hist[n], trainable_head=sum(p.numel() for p in v.head.parameters()),
                    trainable_lora=sum(p.numel() for k_, p in v.named_parameters() if "lora" in k_), minutes_all_variants=minutes,
                    audit_near_dup_le8={k: d["le8"] for k, d in AUDIT["near_duplicates"].items()}, dedup=AUDIT["dedup"], dedup_removed=AUDIT.get("dedup_removed", 0))
        save_result(names[n], meta, arrays[n])
        print(f"[{names[n]}] mejor época {best_ep[n] + 1}/{E}, val {best[n]:.4f}")
    if resume_p.exists():
        resume_p.unlink()
    del V, ctx; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# %% [markdown]
# ## Correr: una semilla por GPU
# Con dos GPU, cada una toma las semillas por turnos (una GPU: todas en secuencia). Cada semilla termina sola y deja sus resultados; si Kaggle corta,
# se vuelve a lanzar adjuntando la salida anterior como dataset y solo se rehace lo que falta (o se reanuda la semilla a medias desde su checkpoint de época).

# %% 4b. lanzar
errors = []


def worker(seeds, device):
    for s in seeds:
        try:
            run_seed(s, device)
        except Exception as e:
            import traceback; traceback.print_exc(); errors.append((s, repr(e)))


t0 = time.time()
shards = [CFG["seeds"][i::len(DEVS)] for i in range(len(DEVS))]
threads = [threading.Thread(target=worker, args=(sh, DEVS[i])) for i, sh in enumerate(shards) if sh]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Ronda 2 terminada en {(time.time() - t0) / 60:.1f} min | errores: {errors or 'ninguno'}")

# %% [markdown]
# ## Fase 6 — Resultados y gráficos
# Todo se lee de los resultados guardados. Los intervalos son bootstrap por bloques de 10 imágenes consecutivas (`a4m_stats`). Los **veredictos** de las hipótesis
# (HR1, HR2, HR3 …) los calcula el cuaderno F con las reglas registradas; aquí se ve el comportamiento.

# %% 6a. tabla principal
import matplotlib.pyplot as plt
from a4m_stats import single, contrast, r0, br_iou, fmt

SETS = tuple(k for k in ("T", "M1", "M2", "M3") if k in EVAL_SETS)
RES = {n: [r for r in (load_result(f"{TAG}_{n}_s{s}") for s in CFG["seeds"]) if r] for n in VARIANTS}
COL = {"dec_ce": "#1f77b4", "dec_lov": "#ff7f0e", "lora_ce": "#2ca02c", "lora_lov": "#d62728"}
CLASS_COL = ["#a1887f", "#546e7a", "#ffd54f", "#e53935"]


def done(fig, name):
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"{name}.png", dpi=150)
    plt.show() if "ipykernel" in sys.modules else plt.close(fig)


def seeds_conf(n, k):
    return [r[1][f"{k}__conf"][:, 0] for r in RES[n]]


import pandas as pd
rows = []
for n in VARIANTS:
    if not RES[n]:
        continue
    row = {"variante": n, "semillas": len(RES[n]), "mejor época": [r[0]["best_epoch"] + 1 for r in RES[n]]}
    for k in SETS:
        pt, lo, hi = single(seeds_conf(n, k)); row[k] = f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"
    rows.append(row)
TABLE = pd.DataFrame(rows); TABLE.to_csv(OUT / "R2_table.csv", index=False); print(TABLE.to_string(index=False))
print("\nIC 95 % por bootstrap de bloques. Publicado (indicativo): SegFormer-B3 M1 0.69 / M2 0.74 / M3 0.93; UPerNet Swin-S 0.70 / 0.74 / 0.94.")

# %% 6b. curvas de entrenamiento
if any(RES.values()):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in VARIANTS:
        if RES[n]:
            h = np.mean([r[0]["val_hist"] for r in RES[n]], 0); l = np.mean([r[0]["loss_hist"] for r in RES[n]], 0); x = np.arange(1, len(h) + 1)
            ax[0].plot(x, h, color=COL[n], marker="o", ms=3, label=n); ax[1].plot(x, l, color=COL[n], marker="o", ms=3, label=n)
    ax[0].set(xlabel="época", ylabel="mIoU de validación", title="Validación (media de semillas)"); ax[1].set(xlabel="época", ylabel="pérdida de entrenamiento", title="Entrenamiento")
    ax[0].legend(); done(fig, "6b_curvas")

# %% 6c. IoU por clase y falsos positivos de big_rock
if any(RES.values()):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, k in zip(ax[:2], [s for s in ("T", "M1") if s in SETS]):
        w = 0.2
        for j, n in enumerate(VARIANTS):
            if RES[n]:
                iou = np.mean([iou_np(c.sum(0)) for c in seeds_conf(n, k)], 0); a.bar(np.arange(4) + (j - 1.5) * w, iou, w, color=COL[n], label=n)
        a.set_xticks(range(4)); a.set_xticklabels(CLASS_NAMES); a.set(ylabel=f"IoU en {k}", title=f"IoU por clase en {k}")
    ax[0].legend(fontsize=8)
    fpk = [(k, n, np.mean([c.sum(0)[:3, 3].sum() for c in seeds_conf(n, k)]) / 1e3) for k in ("M3",) if k in SETS for n in VARIANTS if RES[n]]
    ax[2].bar([f"{n}" for _, n, _ in fpk], [v for *_, v in fpk], color=[COL[n] for _, n, _ in fpk]); ax[2].set(ylabel="miles de píxeles", title="Falsos positivos de big_rock en M3")
    ax[2].tick_params(axis="x", rotation=20); done(fig, "6c_iou_clase")

# %% 6d. ejemplos cualitativos (6 imágenes de M1 fijadas de antemano, semilla 0)
if all(RES[n] for n in VARIANTS):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(CLASS_COL + ["#000000"]); show = lambda m: np.where(m == IGNORE, 4, m)
    fig, ax = plt.subplots(len(QI), 2 + len(VARIANTS), figsize=(2.6 * (2 + len(VARIANTS)), 2.5 * len(QI)))
    for r, i in enumerate(QI):
        lab = DATA["M1"][1][i]; img = np.asarray(Image.fromarray(DATA["M1"][0][i]).resize(lab.shape[::-1]))
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 1].imshow(show(lab), cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        for j, n in enumerate(VARIANTS):
            ax[r, 2 + j].imshow(RES[n][0][1]["qual__pred"][r], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        for a in ax[r]:
            a.axis("off")
    for j, t in enumerate(["imagen", "etiqueta M1 (negro = sin etiquetar)"] + list(VARIANTS)):
        ax[0, j].set_title(t, fontsize=9)
    ax[-1, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COL], labels=CLASS_NAMES, loc="lower left", fontsize=7)
    done(fig, "6d_cualitativo")

# %% 6e. calibración (¿sabe el modelo cuándo falla?)
if all(RES[n] for n in VARIANTS):
    k = "T" if "T" in SETS else "M1"; fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for n in VARIANTS:
        cal = np.mean([r[1][f"{k}__calib"].sum(0) for r in RES[n]], 0)                   # [3 grupos, N_BINS, 3]
        for a, g, t in ((ax[0], 0, "todos los píxeles etiquetados"), (ax[1], 1, "píxeles predichos big_rock")):
            cnt, sc, cc = cal[g, :, 0], cal[g, :, 1], cal[g, :, 2]; ok = cnt > 0
            a.plot(sc[ok] / cnt[ok], cc[ok] / cnt[ok], marker="o", ms=3, color=COL[n], label=f"{n} (ECE {ece(cal[g]):.3f})"); a.set_title(f"Calibración en {k}: {t}", fontsize=9)
    for a in ax:
        a.plot([0, 1], [0, 1], "k--", lw=0.8); a.set(xlabel="confianza", ylabel="exactitud"); a.legend(fontsize=7)
    done(fig, "6e_calibracion")

# %% 6f. cierre
print("Resultados en", RES_DIR, "| figuras en", OUT / "figures", "| auditoría en", OUT / "auditoria_datos.json")
print("Siguiente paso: bajar la salida de Kaggle a results/estudio_principal/raw/e y correr F_ronda2_resultados.py (veredictos de HR1, HR2, HR3).")
