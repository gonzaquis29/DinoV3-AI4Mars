# %% [markdown]
# # AI4Mars × DINOv3 — Cuaderno D: baseline supervisado entrenado por nosotros
#
# Para poder decir «frente a un modelo clásico entrenado igual», entrenamos **DeepLabV3 con ResNet-50** de punta a punta,
# con el mismo split, las mismas imágenes de test (M1, M2, M3), el mismo tamaño de entrada (512 px) y las mismas métricas
# que los demás cuadernos. Variantes:
#
# | nombre | backbone | entrenamiento |
# |---|---|---|
# | `BL_pre_full` | ResNet-50 preentrenado en ImageNet | las 15,564 imágenes de entrenamiento |
# | `BL_scratch_full` | ResNet-50 sin preentrenar | las 15,564 imágenes |
#
# El presupuesto de entrenamiento se fija en **minutos** (`BUDGET_MIN`): se mide la velocidad en unos pasos de prueba y se calcula
# cuántos pasos caben. Los pasos y épocas realmente usados quedan en cada JSON. Parada temprana (paciencia de 3 épocas en validación); mejor época elegida en validación, nunca en test.
#
# **Límite conocido:** con 70 min por entrenamiento caben solo ~3 épocas (~2,900 pasos). El modelo preentrenado ya se había estabilizado y el que parte de cero seguía mejorando:
# el baseline está **subentrenado** y es de **1 semilla**. La comparación con la sonda lineal debe leerse con esa reserva.

# %% 0. Configuración
import sys, json, time, threading, os
from pathlib import Path

LIB = next((f for d in range(6) for f in Path("/kaggle/input").glob("*/" * d + "a4m_lib.py")), Path.cwd() / "a4m_lib.py")
sys.path.insert(0, str(LIB.parent))
from a4m_lib import *
import a4m_lib
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

CFG = dict(size=512, n_val=500, bs=16, lr=1e-3, wd=1e-4, max_epochs=20, patience=3, budget_min=float(os.environ.get("BUDGET_MIN", 75)), seed=0)
if SMOKE:
    CFG.update(size=64, n_val=6, bs=4, max_epochs=2, budget_min=1)
if QUICK:
    CFG.update(max_epochs=1, budget_min=3)
setup_dirs(); DEVS = devices(); print("GPUs:", DEVS, "| presupuesto por entrenamiento:", CFG["budget_min"], "min")

# %% 1. datos (todas las imágenes en RAM como uint8)
ROOT = make_fake_dataset(OUT / "fake_data") if SMOKE else find_root()
SPLITS = build_splits(ROOT, n_val=CFG["n_val"])
if QUICK:
    SPLITS["train"] = SPLITS["train"][:64]
pool2000 = np.random.default_rng(0).permutation(len(SPLITS["train"]))[:2000]          # mismo subconjunto que los cuadernos A y B

def stack_set(pairs, native):
    imgs, labs = load_images(pairs, CFG["size"], native_labels=native)
    return np.stack(imgs), (np.stack(labs) if native is False else labs)

TRAIN_X, TRAIN_Y = stack_set(SPLITS["train"], False)
EVAL = {}
for k in EVAL_SETS:
    imgs, labs = load_images(SPLITS[k], CFG["size"], native_labels=True)
    EVAL[k] = (np.stack(imgs), labs)
M3_FOR_M1 = load_m3_labels(SPLITS["M1_m3label"], EVAL["M1"][1])
print("train", TRAIN_X.shape, "| test", {k: len(v[0]) for k, v in EVAL.items()})

MEAN_G, STD_G = float(np.mean(MEAN)), float(np.mean(STD))
def to_gpu(x_u8, device):
    x = torch.from_numpy(x_u8).to(device).float().div(255)
    return ((x - MEAN_G) / STD_G).unsqueeze(1).expand(-1, 3, -1, -1)

# %% 2. el modelo y la evaluación
def make_model(pretrained):
    kw = dict(weights=None, num_classes=NUM_CLASSES, aux_loss=False)
    if pretrained and not SMOKE:
        kw["weights_backbone"] = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    else:
        kw["weights_backbone"] = None
    return deeplabv3_resnet50(**kw)

@torch.no_grad()
def evaluate_img(net, name, device, radii=RADII, extra=False, bs=8):
    net.eval(); X, labs = EVAL[name]
    conf, cal, agree = [], [], []
    for i in range(0, len(X), bs):
        x = to_gpu(X[i:i + bs], device)
        y = torch.from_numpy(np.stack(labs[i:i + bs])).to(device).long()
        with autocast(device):
            lg = net(x)["out"]
        lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = lg.argmax(1)
        conf.append(per_image_conf(pred, y, radii))
        if extra:
            cal.append(calib_bins(lg.softmax(1), y))
            if name == "M1":
                y3 = torch.from_numpy(np.stack(M3_FOR_M1[i:i + bs])).to(device).long()
                hi = torch.where(y3 != IGNORE, y, torch.full_like(y, IGNORE)); lo = torch.where(y3 == IGNORE, y, torch.full_like(y, IGNORE))
                agree.append(np.stack([per_image_conf(pred, hi, (0,))[:, 0], per_image_conf(pred, lo, (0,))[:, 0]], 1))
    out = {f"{name}__conf": np.concatenate(conf)}
    if cal:
        out[f"{name}__calib"] = np.concatenate(cal)
    if agree:
        out[f"{name}__agree"] = np.concatenate(agree)
    return out

def train_step(net, opt, scaler, idx, device, flip=True):
    x = TRAIN_X[idx]; y = TRAIN_Y[idx]
    if flip and np.random.rand() < 0.5:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    x = to_gpu(np.ascontiguousarray(x), device); y = torch.from_numpy(np.ascontiguousarray(y)).to(device).long()
    with autocast(device):
        lg = net(x)["out"]
    loss = F.cross_entropy(lg.float(), y, ignore_index=IGNORE)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return loss

def make_opt(net, steps_total):
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps_total, pct_start=0.05)
    return opt, sched

# %% 3. un entrenamiento con presupuesto de tiempo
def run(name, pretrained, rows, seed, device):
    if have(name):
        return
    seed_all(seed); n = len(rows); bs = CFG["bs"]; spe = max(1, n // bs)
    # 3a. medir la velocidad con 10 pasos en un modelo desechable
    probe_net = make_model(pretrained).to(device).train(); o, _ = make_opt(probe_net, 100); sc = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed)
    for _ in range(2):
        train_step(probe_net, o, sc, np.sort(rng.choice(rows, bs, replace=False)), device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(10):
        train_step(probe_net, o, sc, np.sort(rng.choice(rows, bs, replace=False)), device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    sps = (time.time() - t0) / 10; del probe_net, o; torch.cuda.empty_cache() if device.type == "cuda" else None
    epochs = int(max(1, min(CFG["max_epochs"], CFG["budget_min"] * 60 / (sps * spe * 1.08))))        # 1.08: tiempo de la validación
    print(f"[{name}] {sps:.2f} s/paso -> {epochs} épocas de {spe} pasos ({epochs * spe} pasos, ~{epochs * spe * sps / 60:.0f} min)")
    # 3b. entrenar de verdad
    seed_all(seed); net = make_model(pretrained).to(device); opt, sched = make_opt(net, epochs * spe)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"); t0 = time.time()
    best, best_ep, best_state, hist = -1.0, -1, None, []
    for ep in range(epochs):
        net.train(); order = np.random.default_rng(seed * 7919 + ep).permutation(rows)
        for b in range(spe):
            train_step(net, opt, scaler, np.sort(order[b * bs:(b + 1) * bs]), device); sched.step()
        v = float(miou_np(evaluate_img(net, "val", device, radii=(0,))["val__conf"][:, 0].sum(0))); hist.append(v)
        if v > best:
            best, best_ep = v, ep; best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        print(f"[{name}] época {ep}: val mIoU {v:.4f} ({(time.time() - t0) / 60:.1f} min)")
        if ep - best_ep >= CFG["patience"]:
            print(f"[{name}] parada temprana: sin mejora en {CFG['patience']} épocas"); break
    net.load_state_dict(best_state)
    arrays = {}
    for k in EVAL_SETS:
        arrays.update(evaluate_img(net, k, device, extra=k != "val"))
    meta = dict(part="BL", pretrained=pretrained, n=len(rows), seed=seed, epochs_planned=epochs, epochs_run=len(hist), steps=len(hist) * spe, patience=CFG['patience'], batch=bs, lr=CFG["lr"],
                val_miou=best, best_epoch=best_ep, val_hist=hist, params_M=round(sum(p.numel() for p in net.parameters()) / 1e6, 1),
                minutes=round((time.time() - t0) / 60, 1))
    save_result(name, meta, arrays)
    print(f"[{name}] listo: val {best:.4f}, M1 {miou_np(arrays['M1__conf'][:, 0].sum(0)):.4f}, {meta['minutes']} min")
    del net; torch.cuda.empty_cache() if device.type == "cuda" else None

ALL = np.arange(len(TRAIN_X))
JOBS = ([[("BL_pre_full_s0", True, ALL)], [("BL_scratch_full_s0", False, ALL)]]
        if len(DEVS) > 1 else [[("BL_pre_full_s0", True, ALL), ("BL_scratch_full_s0", False, ALL)]])
if SMOKE or QUICK:
    pool2000 = ALL[:min(len(ALL), 40)]
    JOBS = [[(n, p, (pool2000 if "n2000" in n else ALL))] for js in JOBS for (n, p, _) in js]

# %% 4. lanzar (una GPU por lista de trabajos)
errors = []
def worker(jobs, device):
    for name, pre, rows in jobs:
        try:
            run(name, pre, rows, CFG["seed"], device)
        except Exception as e:
            import traceback; traceback.print_exc(); errors.append((name, repr(e)))
threads = [threading.Thread(target=worker, args=(js, DEVS[i % len(DEVS)])) for i, js in enumerate(JOBS)]
for t in threads: t.start()
for t in threads: t.join()
print("errores:", errors or "ninguno")
for n in sorted(p.stem for p in RES_DIR.glob("BL_*.json")):
    r = load_result(n)
    print(n, {k: round(float(miou_np(r[1][f"{k}__conf"][:, 0].sum(0))), 4) for k in ("val", "M1", "M2", "M3")}, "| pasos", r[0]["steps"], "| min", r[0]["minutes"])
