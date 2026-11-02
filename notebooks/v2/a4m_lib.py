"""a4m_lib — plumbing shared by the v2 notebooks (data, token caches, training loop, metrics, result store).

The notebooks keep the parts worth explaining (image transforms, LoRA, the experiment loops, the analysis);
everything here is bookkeeping. Conventions are the same as the final run (notebooks/ai4mars_dinov3_final.py):
M1/M2/M3 never merged, validation = last 500 training images by spacecraft clock, train images that appear in a
test set are dropped, images bilinear / labels nearest, label 255 never counted, mIoU over classes with non-empty union.
"""
import os, json, time, random, copy, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

SMOKE = bool(os.environ.get("A4M_SMOKE"))          # tiny offline self-test (fake data, tiny random ViT)
FULL = bool(os.environ.get("A4M_FULL"))            # train on all ~15.5k training images (token caches go to disk)
QUICK = bool(os.environ.get("A4M_QUICK"))          # real data and model, few images and 1 epoch: checks the pipeline on Kaggle
INPUT = Path(os.environ.get("A4M_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("A4M_OUT", "/kaggle/working"))
RES_DIR = OUT / "results"

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES, IGNORE = 4, 255
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RADII = (0, 4, 8, 16)                               # boundary band (px at native label size) excluded from scoring
N_BINS = 15                                         # confidence bins for calibration
EVAL_SETS = ("val", "M1", "M2", "M3")
_LUT = np.full(256, IGNORE, np.uint8); _LUT[:NUM_CLASSES] = np.arange(NUM_CLASSES)
IMG_EXT = (".jpg", ".jpeg", ".png")


def setup_dirs():
    RES_DIR.mkdir(parents=True, exist_ok=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def devices():
    n = torch.cuda.device_count()
    return [torch.device(f"cuda:{i}") for i in range(n)] or [torch.device("cpu")]


def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


# ---------------------------------------------------------------- data
def make_fake_dataset(root, n_train=40, n_test=6, size=96):
    """SMOKE only: a tiny AI4Mars-shaped folder with random gray images and blocky labels."""
    rng = np.random.default_rng(0)
    nc = Path(root) / "msl" / "ncam"
    dirs = [nc / "images" / "edr", nc / "labels" / "train"] + [nc / "labels" / "test" / f"masked-gold-min{k}-100agree" for k in (1, 2, 3)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    def lab():
        l = np.kron(rng.integers(0, 4, (6, 6)), np.ones((size // 6 + 1,) * 2, int))[:size, :size].astype(np.uint8)
        l[rng.random(l.shape) < 0.1] = 255
        return l
    for i in range(n_train + n_test):
        stem = f"NLA_{400000000 + i}EDR_F0000000NCAM00000M1"
        Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8)).convert("RGB").save(dirs[0] / f"{stem}.JPG")
        if i < n_train:
            Image.fromarray(lab()).save(dirs[1] / f"{stem}.png")
        else:
            l1 = lab()
            for k, d in zip((1, 2, 3), dirs[2:]):
                lk = l1.copy(); lk[rng.random(l1.shape) < 0.15 * (k - 1)] = 255
                Image.fromarray(lk).save(d / f"{stem}_merged.png")
    return Path(root)


def find_root(search_dir=None):
    search_dir = Path(search_dir or INPUT)
    for dp, dns, _ in os.walk(search_dir):
        if "msl" in dns and (Path(dp) / "msl" / "ncam").is_dir():
            return Path(dp)
        if len(Path(dp).relative_to(search_dir).parts) >= 6:
            dns[:] = []
    raise FileNotFoundError(f"no folder containing msl/ncam under {search_dir}")


def _pairs(img_dir, lbl_dir):
    idx = {Path(f).stem.lower(): str(Path(img_dir) / f) for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)}
    out = []
    for f in sorted(os.listdir(lbl_dir)):
        if f.lower().endswith(".png"):
            s = Path(f).stem.lower()
            ip = idx.get(s) or idx.get(s[:-len("_merged")] if s.endswith("_merged") else s)
            if ip:
                out.append((ip, str(Path(lbl_dir) / f)))
    return out


def build_splits(root, n_val=500):
    """Same split as the final run. Also returns, for each M1 image, the M3 label path (or None): pixels labelled in
    M1 but not in M3 are the low-agreement pixels used by P3."""
    nc = Path(root) / "msl" / "ncam"
    img_dir = nc / "images" / "edr"
    tests = {f"M{k}": _pairs(img_dir, nc / "labels" / "test" / f"masked-gold-min{k}-100agree") for k in (1, 2, 3)}
    allp = _pairs(img_dir, nc / "labels" / "train")
    test_stems = {Path(i).stem.lower() for ps in tests.values() for i, _ in ps}
    allp = [p for p in allp if Path(p[0]).stem.lower() not in test_stems]
    allp.sort(key=lambda p: Path(p[1]).stem)                     # file names start with the spacecraft clock
    splits = {"train": allp[:-n_val], "val": allp[-n_val:], **tests}
    m3 = {Path(i).stem.lower(): l for i, l in tests["M3"]}
    splits["M1_m3label"] = [m3.get(Path(i).stem.lower()) for i, _ in tests["M1"]]
    print(" | ".join(f"{k} {len(v)}" for k, v in splits.items() if k != "M1_m3label"),
          f"| M1 images with an M3 label: {sum(x is not None for x in splits['M1_m3label'])}")
    return splits


def read_label(path):
    m = np.array(Image.open(path))
    return _LUT[m[..., 0] if m.ndim == 3 else m]


def load_gray(path, size):
    """NavCam frames are grayscale (checked in the final run): keep one uint8 channel at size x size."""
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR))


def to_input(gray_u8):
    """uint8 [S,S] -> normalised float [3,S,S] (gray replicated to 3 channels, ImageNet statistics)."""
    x = torch.from_numpy(np.array(gray_u8)).float().div(255)[None].expand(3, -1, -1)
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class ImageSet(Dataset):
    """In-RAM uint8 images + labels. transform: function uint8[S,S] -> uint8[S,S] applied before normalising.
    Labels are kept at native size for evaluation sets, at S for training."""

    def __init__(self, imgs, labels, transform=None, extra=None):
        self.imgs, self.labels, self.transform, self.extra = imgs, labels, transform, extra

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        g = self.imgs[i] if self.transform is None else self.transform(self.imgs[i])
        y = torch.from_numpy(self.labels[i].astype(np.int64))
        if self.extra is None:
            return to_input(g), y
        return to_input(g), y, torch.from_numpy(self.extra[i].astype(np.int64))


def _read_one(a):
    ip, lp, size, native = a
    g = load_gray(ip, size)
    y = read_label(lp)
    if not native:
        y = np.asarray(Image.fromarray(y).resize((size, size), Image.NEAREST))
    return g, y


def load_images(pairs, size, native_labels, workers=8):
    """Decodes images once (multiprocessing); returns list of uint8 images and list of uint8 labels."""
    import multiprocessing as mp
    args = [(ip, lp, size, native_labels) for ip, lp in pairs]
    if workers <= 1:
        res = [_read_one(a) for a in tqdm(args, desc="decode", leave=False)]
    else:
        with mp.get_context("fork").Pool(workers) as p:
            res = list(tqdm(p.imap(_read_one, args, chunksize=16), total=len(args), desc="decode", leave=False))
    return [r[0] for r in res], [r[1] for r in res]


def load_m3_labels(paths, like):
    """M3 labels aligned with M1 (all-ignore where an image has no M3 label)."""
    return [read_label(p) if p else np.full_like(l, IGNORE) for p, l in zip(paths, like)]


# ---------------------------------------------------------------- backbone
def find_local_dinov3():
    if SMOKE or not INPUT.exists():
        return None
    for d in range(4):
        for cfgp in INPUT.glob("/".join(["*"] * d + ["config.json"])):
            try:
                c = json.load(open(cfgp))
            except Exception:
                continue
            if c.get("model_type") == "dinov3_vit" and c.get("hidden_size") == 768 and (cfgp.parent / "model.safetensors").exists():
                return str(cfgp.parent)
    return None


def load_backbone(device, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m"):
    from transformers import AutoModel
    if SMOKE:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        torch.manual_seed(0)
        m = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=12, num_attention_heads=2,
                                           patch_size=16, num_register_tokens=4, image_size=64))
    else:
        mid = find_local_dinov3() or model_id
        try:
            m = AutoModel.from_pretrained(mid, dtype=torch.float32)
        except TypeError:
            m = AutoModel.from_pretrained(mid, torch_dtype=torch.float32)
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device).eval()


def bb_info(bb):
    cfg = bb.config
    lists = [mod for _, mod in bb.named_modules() if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers]
    blocks = max(lists, key=lambda ml: sum(p.numel() for p in ml.parameters()))
    norm = getattr(bb, "norm", None) or getattr(bb, "layernorm", None)
    return dict(blocks=blocks, depth=int(cfg.num_hidden_layers), C=int(cfg.hidden_size), patch=int(cfg.patch_size),
                n_prefix=1 + int(getattr(cfg, "num_register_tokens", 0) or 0), norm=norm)


def rope(bb, size, device):
    """RoPE position embeddings for a size x size input, computed where the backbone lives, then moved to `device`."""
    home = next(bb.parameters()).device
    with torch.no_grad():
        return tuple(p.to(device) for p in bb.rope_embeddings(torch.zeros(1, 3, size, size, device=home)))


def to_map(tok, n_prefix, grid):
    t = tok[:, n_prefix:, :]
    return t.transpose(1, 2).reshape(t.shape[0], -1, grid, grid)


def lin_head(C):
    return nn.Sequential(nn.BatchNorm2d(C), nn.Conv2d(C, NUM_CLASSES, 1))


def pick_tmp(n_train, tokens=1025, dim=768):
    """Scratch dir with the most free space (Kaggle's /kaggle/working only has 20 GB). If even that cannot hold the fp16 token
    cache of n_train images, returns how many images fit, so the run shrinks instead of dying (the n used is stored in every result)."""
    best = None
    for c in (Path("/kaggle/temp"), Path("/tmp"), Path("/var/tmp"), OUT / "tmp"):
        try:
            c.mkdir(parents=True, exist_ok=True); free = shutil.disk_usage(c).free / 2**30
        except Exception:
            continue
        print(f"  disco libre en {c}: {free:.0f} GB")
        if best is None or free > best[1]:
            best = (c, free)
    per_img = tokens * dim * 2 / 2**30
    fit = int((best[1] - 4) / per_img)
    if fit < n_train:
        print(f"  AVISO: la caché de {n_train} imágenes ({n_train * per_img:.0f} GB) no cabe; se usarán {fit}")
    return best[0], min(n_train, fit)


class DiskTokens:
    """fp16 token cache [N,T,C] stored in a .npy memmap (the full training set does not fit in RAM). Behaves like the tensor
    the training loop expects: len(), slicing and indexing with an index array return torch tensors."""

    def __init__(self, path, shape):
        self.path, self.shape = str(path), tuple(shape)
        self.mm = np.lib.format.open_memmap(self.path, mode="w+", dtype=np.float16, shape=self.shape)

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, sl, v):
        self.mm[sl] = v.numpy()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return torch.from_numpy(np.ascontiguousarray(self.mm[idx]))
        idx = np.asarray(idx); o = np.argsort(idx)
        out = np.empty((len(idx),) + self.shape[1:], np.float16); out[o] = self.mm[idx[o]]
        return torch.from_numpy(out)

    def close(self):
        try:
            self.mm.flush(); del self.mm
        finally:
            if os.path.exists(self.path):
                os.remove(self.path)


@torch.no_grad()
def extract_tokens(bb, ds, take, device, bs=16, disk_path=None):
    """take: hidden_states index (0 = patch embeddings, before block 1) or 'last' (final normed output).
    Returns fp16 tokens [N,T,C] and the label tensors of the dataset."""
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)   # 0: forked workers got OOM-killed on Kaggle (v2-a, 4th transform)
    toks, labs = None, []; i = 0
    for batch in tqdm(dl, leave=False, desc=f"tokens@{take}"):
        x = batch[0].to(device)
        with autocast(device):
            o = bb(pixel_values=x, output_hidden_states=take != "last")
        t = (o.last_hidden_state if take == "last" else o.hidden_states[take]).half()
        assert torch.isfinite(t).all(), "non-finite tokens"
        if toks is None:
            shape = (len(ds),) + tuple(t.shape[1:])
            toks = DiskTokens(disk_path, shape) if disk_path else torch.empty(shape, dtype=torch.float16)
        toks[i:i + len(x)] = t.cpu(); i += len(x)
        labs.append([b.to(torch.uint8) for b in batch[1:]])
    labs = [torch.cat([l[j] for l in labs]) for j in range(len(labs[0]))]
    return (toks, *labs)


class ReplayNet(nn.Module):
    """Cached tokens taken before block `start` -> blocks start..depth-1 -> final norm -> linear head.
    Blocks are frozen copies of the backbone; blocks before `grad_from` run without autograd (nothing to train
    there, saves time and memory). Trainable parts are added by the caller (e.g. LoRA) plus the head."""

    def __init__(self, info, grid, pos, start, grad_from=None):
        super().__init__()
        self.start, self.grid, self.pos, self.n_prefix = start, grid, pos, info["n_prefix"]
        self.blocks = nn.ModuleList([copy.deepcopy(b) for b in info["blocks"][start:]])
        self.norm = copy.deepcopy(info["norm"])
        for p in list(self.blocks.parameters()) + list(self.norm.parameters()):
            p.requires_grad = False
        self.grad_from = start if grad_from is None else grad_from
        self.head = lin_head(info["C"])

    def forward(self, tok):
        h = tok.float()
        for i, blk in enumerate(self.blocks):
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.start + i >= self.grad_from):
                o = blk(h, position_embeddings=self.pos)
            h = o[0] if isinstance(o, (tuple, list)) else o
        return self.head(to_map(self.norm(h), self.n_prefix, self.grid))

    def train(self, mode=True):
        super().train(mode); self.blocks.eval(); self.norm.eval(); return self      # no dropout in the backbone copy


# ---------------------------------------------------------------- metrics
def iou_np(c):
    c = np.asarray(c, np.float64); tp = np.diagonal(c, axis1=-2, axis2=-1); u = c.sum(-1) + c.sum(-2) - tp
    return np.where(u > 0, tp / np.maximum(u, 1), np.nan)


def miou_np(c):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(iou_np(c), axis=-1)


def per_image_conf(pred, y, radii):
    """int32 [B,R,4,4]; radius r>0 drops pixels within r px of a label change or an unlabelled pixel."""
    B = y.shape[0]; yf = y.float().unsqueeze(1); out = []
    base = torch.arange(B, device=y.device).view(B, 1, 1) * 16
    for r in radii:
        keep = y != IGNORE
        if r > 0:
            k = 2 * r + 1
            keep = keep & (F.max_pool2d(yf, k, 1, r) == -F.max_pool2d(-yf, k, 1, r)).squeeze(1)
        out.append(torch.bincount((base + y.clamp(max=3) * 4 + pred)[keep], minlength=B * 16).view(B, 4, 4))
    return torch.stack(out, 1).cpu().numpy().astype(np.int32)


def calib_bins(prob, y):
    """Per-image reliability statistics, float32 [B,3,N_BINS,3]: groups (all labelled pixels, pixels predicted
    big_rock, pixels whose label is big_rock) x confidence bins x (count, sum of confidence, number correct)."""
    conf, pred = prob.max(1)
    keep = y != IGNORE
    groups = (keep, keep & (pred == 3), keep & (y == 3))
    b = (conf * N_BINS).long().clamp(max=N_BINS - 1)
    correct = (pred == y).float()
    out = torch.zeros(y.shape[0], 3, N_BINS, 3, device=y.device)
    for i in range(y.shape[0]):
        for g, m in enumerate(groups):
            mi = m[i]
            if mi.any():
                bi = b[i][mi]
                out[i, g, :, 0] = torch.bincount(bi, minlength=N_BINS).float()
                out[i, g, :, 1] = torch.bincount(bi, weights=conf[i][mi].float(), minlength=N_BINS)
                out[i, g, :, 2] = torch.bincount(bi, weights=correct[i][mi], minlength=N_BINS)
    return out.cpu().numpy().astype(np.float32)


def ece(bins):
    """bins [..., N_BINS, 3] summed over images -> expected calibration error (confidence - accuracy, weighted)."""
    n, s, c = bins[..., 0], bins[..., 1], bins[..., 2]
    tot = n.sum(-1)
    return np.abs(s - c).sum(-1) / np.maximum(tot, 1)


# ---------------------------------------------------------------- evaluation and training on cached tokens
@torch.no_grad()
def evaluate(net, cache, device, radii=RADII, extra=False, bs=8):
    """cache = (tokens, labels[, m3 labels]). Returns per-image confusions (and P3 statistics when extra=True)."""
    net.eval(); toks, labs = cache[0], cache[1]
    conf, cal, agree = [], [], []
    for i in range(0, len(toks), bs):
        t = toks[i:i + bs].to(device); y = labs[i:i + bs].to(device).long()
        with autocast(device):
            lg = net(t)
        lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = lg.argmax(1)
        conf.append(per_image_conf(pred, y, radii))
        if extra:
            cal.append(calib_bins(lg.softmax(1), y))
            if len(cache) > 2:
                y3 = cache[2][i:i + bs].to(device).long()
                hi = torch.where(y3 != IGNORE, y, torch.full_like(y, IGNORE))      # labelled in M1 and in M3
                lo = torch.where(y3 == IGNORE, y, torch.full_like(y, IGNORE))      # labelled in M1 only
                agree.append(np.stack([per_image_conf(pred, hi, (0,))[:, 0], per_image_conf(pred, lo, (0,))[:, 0]], 1))
    out = {"conf": np.concatenate(conf)}
    if cal:
        out["calib"] = np.concatenate(cal)
    if agree:
        out["agree"] = np.concatenate(agree)
    return out


def fit(net, caches, pool_n, seed, epochs, bs, lrs, device, log=print, patience=None):
    """Trains the parameters in `lrs` ({"head": lr, "adapter": lr}) on the first pool_n cached training images
    (seed-specific order), keeps the epoch with the best validation mIoU. Returns (best val mIoU, epoch, history)."""
    toks, labs = caches["train"][0], caches["train"][1]
    net.to(device); seed_all(seed)
    groups = [{"params": [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("head")], "lr": lrs["head"]}]
    other = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("head")]
    if other:
        groups.append({"params": other, "lr": lrs["adapter"]})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = max(1, pool_n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_state, best_ep, hist = -1.0, None, -1, []
    order0 = np.random.default_rng(1000 + seed).permutation(pool_n)
    for ep in range(epochs):
        net.train(); order = np.random.default_rng(seed * 7919 + ep).permutation(order0)
        for b in range(steps):
            idx = torch.from_numpy(order[b * bs:(b + 1) * bs])
            t = toks[idx].to(device, non_blocking=True); y = labs[idx].to(device).long()
            with autocast(device):
                lg = net(t)
            loss = F.cross_entropy(F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False), y, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        v = float(miou_np(evaluate(net, caches["val"], device, radii=(0,))["conf"][:, 0].sum(0))); hist.append(v)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items() if not k.startswith(("blocks.", "norm."))
                          or "lora" in k}
        log(f"    ep {ep}: val mIoU {v:.4f}")
        if patience is not None and ep - best_ep >= patience:
            log(f"    parada temprana en la época {ep} (sin mejora en {patience} épocas)"); break
    net.load_state_dict(best_state, strict=False)
    return best, best_ep, hist


def test_all(net, caches, device):
    """Final evaluation of a trained net on val/M1/M2/M3; P3 statistics on the test sets."""
    arrays = {}
    for k in EVAL_SETS:
        r = evaluate(net, caches[k], device, extra=k != "val")
        arrays[f"{k}__conf"] = r["conf"]
        if "calib" in r:
            arrays[f"{k}__calib"] = r["calib"]
        if "agree" in r:
            arrays[f"{k}__agree"] = r["agree"]
    return arrays


# ---------------------------------------------------------------- result store
def find_art(fname):
    p = RES_DIR / fname
    if p.exists():
        return p
    if INPUT.exists():
        for d in range(5):
            hit = next(INPUT.glob("/".join(["*"] * d + [fname])), None)
            if hit:
                return hit
    return None


def have(name):
    return find_art(f"{name}.json") is not None and find_art(f"{name}.npz") is not None


def save_result(name, meta, arrays):
    np.savez_compressed(RES_DIR / f"{name}.npz", **arrays)
    json.dump({**meta, "name": name, "radii": list(RADII), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(RES_DIR / f"{name}.json", "w"), indent=1)


def load_result(name, folder=None):
    if folder is not None:
        j, z = Path(folder) / f"{name}.json", Path(folder) / f"{name}.npz"
        if not (j.exists() and z.exists()):
            return None
    else:
        j, z = find_art(f"{name}.json"), find_art(f"{name}.npz")
        if not (j and z):
            return None
    return json.load(open(j)), dict(np.load(z))
