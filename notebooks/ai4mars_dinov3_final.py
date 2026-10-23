# %% [markdown]
# # AI4Mars x DINOv3 — final experiment suite
#
# One resumable notebook that produces the evidence for the paper. Every stage writes its
# results to `results/` (per-image confusion matrices + a JSON), skips itself if the result
# already exists, and can be run in different sessions / accounts. Stage `E` only reads
# results, so it runs anywhere once the result files are attached as an input or pulled
# from the Hugging Face repo.
#
# | stage | question | what runs |
# |---|---|---|
# | `A` | which layer? (H1) | frozen DINOv3, one linear probe per layer, **full train set**, one forward pass for all layers, N seeds |
# | `B` | does a bigger head help? (H4) | decoder head (3.5M params), N seeds |
# | `C` | label efficiency and partial fine-tuning (H2) | cached block-8 tokens; LP vs fine-tune last 2 / last 4 blocks at N = 250 ... 4000 labelled images, N seeds |
# | `D` | what does pretraining buy? (H3) | DINOv3 vs DINOv2 vs random-init, same probe |
# | `E` | analysis | bootstrap CIs, paired tests, label-noise ceiling, big_rock by size, verdicts, figures |
#
# Conventions kept from the earlier runs: M1/M2/M3 are never merged; val is a temporal hold-out
# of the train split; train stems that appear in a test set are dropped; images bilinear, labels
# nearest, 255 never counted; mIoU averages only classes with non-empty union.
#
# **Statistics.** Every metric is stored as a per-image confusion matrix, so any number can be
# bootstrapped over images. Differences between configurations are paired bootstraps on the same
# images. A difference is called real only if its 95% CI excludes 0 **and** |delta| >= `tol`.
#
# **Hypotheses are written before any run** (cell 1) and saved with a timestamp.
# Where a hypothesis was informed by the earlier pilot, the cell says so.

# %% 0. config + imports
import os, json, time, random, math, copy, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

SMOKE = bool(os.environ.get("A4M_SMOKE"))               # tiny offline self-test, never set on Kaggle
INPUT = Path(os.environ.get("A4M_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("A4M_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)
RES_DIR = OUT / "results"; RES_DIR.mkdir(exist_ok=True)

CFG = dict(
    # ---- what to run in THIS session (edit per session; results are cached) ----
    stages=["A", "B", "C", "D", "E"],
    seeds=dict(A=(0, 1, 2), B=(0, 1, 2), C=(0, 1, 2), D=(0, 1, 2)),
    # ---- shared ----
    model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    size=512, bs=16, n_val=500, num_workers=4, gpu=0,
    tol=0.01,                                           # smallest mIoU difference worth interpreting
    n_boot=2000,
    radii=(0, 4, 8, 16),                                # boundary band (px at native 1024) excluded from scoring
    # ---- A: layer-wise probes ----
    layers=(2, 4, 6, 8, 10, 12, "last_norm"), A_epochs=4, A_lr=1e-3,
    # ---- B: decoder ----
    decoder_sizes=(512,), decoder_epochs=12, decoder_lr=5e-4, decoder_eval_every=2, reuse_legacy_decoder=False,
    # ---- C: label efficiency + partial fine-tuning (tokens cached at block depth-4) ----
    pool=4000, fracs=(250, 500, 1000, 2000, 4000), C_epochs=10, C_bs=16,
    head_lr=1e-3, ft_lrs=(1e-5, 5e-5), ft_sweep_n=1000,
    # ---- D: backbone comparison, probe on cached last-layer tokens ----
    D_n=2000, D_epochs=20,
    backbones=[
        ("dinov3_b16_lvd", "facebook/dinov3-vitb16-pretrain-lvd1689m", 512, False),
        ("dinov2_b14",     "facebook/dinov2-base",                    504, False),   # patch 14: grid 36 vs 32 (confound, state it)
        ("dinov3_b16_rand", "facebook/dinov3-vitb16-pretrain-lvd1689m", 512, True),
        ("dinov2_b14_448", "facebook/dinov2-base",                    448, False),   # extra run (H3b): patch 14 at 448 px -> grid 32, as DINOv3 at 512
    ],
    # ---- two-GPU split: with >=2 GPUs the notebook spawns one worker per job, then runs E itself ----
    parallel=True,
    budget_h=float(os.environ.get("A4M_BUDGET_H", 7.5)),                                       # per process: no new unit of work starts if it would end past this
)
# Jobs balance the wall time (~4-5 h each by the benchmark): gpu0 = A + D + C seeds 0,1; gpu1 = decoder seeds 0,1,2 + C seed 2.
# Each worker is a copy of this script selected by the A4M_JOB env var.
JOBS = {
    "gpu0": dict(gpu=0, stages=["A", "C", "D"], seeds=dict(A=(0, 1, 2), C=(0, 1), D=(0, 1, 2))),
    "gpu1": dict(gpu=1, stages=["B", "C"], seeds=dict(B=(0, 1, 2), C=(2,))),
}
JOB = os.environ.get("A4M_JOB")
if SMOKE:                                              # shrink everything for the offline self-test
    CFG.update(seeds=dict(A=(0, 1), B=(0, 1), C=(0, 1), D=(0, 1)), size=64, bs=4, n_val=6, num_workers=0,
               A_epochs=2, decoder_epochs=2, pool=16, fracs=(4, 8, 16), C_epochs=2, C_bs=4, ft_sweep_n=8,
               D_n=12, D_epochs=2, n_boot=50, radii=(0, 2),
               backbones=[("dinov3_b16_lvd", "facebook/dinov3-vitb16-pretrain-lvd1689m", 64, False),
                          ("dinov2_b14", "facebook/dinov2-base", 56, False),
                          ("dinov3_b16_rand", "facebook/dinov3-vitb16-pretrain-lvd1689m", 64, True)])
if os.environ.get("A4M_ONLY_E"):                        # analysis only: run on downloaded results, no dataset needed
    CFG["stages"], CFG["parallel"] = ["E"], False
if os.environ.get("A4M_STAGES"):                        # e.g. A4M_STAGES=D A4M_BACKBONES=dinov2_b14_448: run just that, single process
    CFG["stages"], CFG["parallel"] = os.environ["A4M_STAGES"].split(","), False
if os.environ.get("A4M_BACKBONES"):
    _bk = set(os.environ["A4M_BACKBONES"].split(","))
    CFG["backbones"] = [b for b in CFG["backbones"] if b[0] in _bk]
if JOB:                                                 # worker process: take this job's stages/seeds/GPU
    CFG.update(JOBS[JOB]); CFG["parallel"] = False

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES, IGNORE_INDEX, K = 4, 255, 4
IMG_EXT = (".jpg", ".jpeg", ".png")
_LUT = np.full(256, IGNORE_INDEX, np.uint8); _LUT[:NUM_CLASSES] = np.arange(NUM_CLASSES)

DEVICE = torch.device(f"cuda:{CFG['gpu']}" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
TOL = CFG["tol"]
print(f"[{JOB or 'main'}]", DEVICE, f"| {torch.cuda.device_count()} GPU(s) visible | torch", torch.__version__)


def _secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)


HF_TOKEN, HF_REPO = _secret("HF_TOKEN"), _secret("HF_REPO")   # HF_REPO optional: where results are mirrored
if HF_TOKEN and not SMOKE:
    from huggingface_hub import login
    login(HF_TOKEN)


T0 = time.time()


def over_budget(est_h, what=""):
    used = (time.time() - T0) / 3600
    if used + est_h > CFG["budget_h"]:
        print(f"  [budget] {used:.1f} h used + ~{est_h} h needed > {CFG['budget_h']} h: skipping {what}")
        return True
    return False


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def autocast():
    return torch.autocast(DEVICE.type, dtype=torch.float16, enabled=AMP)


# %% [markdown]
# ## 1. Hypotheses, registered before any run
#
# Tolerance `tol` = 0.01 mIoU. "Real" = paired-bootstrap 95% CI of the difference excludes 0 and |delta| >= tol.
# H1 and the shape of H4 were informed by an earlier 2,000-image pilot (last layer best, decoder ~ linear probe);
# H2, H3, H5, H6 were not tested before.

# %% 1. register
HYPOTHESES = {
    "H1": {"claim": "Intermediate layers beat the last layer (E1 proposal).",
           "expect": "NOT supported: last_norm is best or within tol of the best intermediate layer (pilot: block 8 was -0.032).",
           "rule": "supported if best-by-val intermediate beats last_norm (real); refuted if last_norm beats it (real); else inconclusive. Judged on M1 and M2."},
    "H2": {"claim": "The gap fine-tuning minus linear probing grows with the number of labelled images.",
           "expect": "With <=500 images FT-2 is within tol of LP; with 4000 images FT-2/FT-4 beat LP by >= tol.",
           "rule": "supported if delta(N_min) < tol, delta(N_max) real and >= tol, and delta(N_max)-delta(N_min) >= tol (FT-2 vs LP, M1 and M2)."},
    "H3": {"claim": "DINOv3 beats DINOv2 as a frozen backbone, but by little because Mars is far from both pretraining sets; both beat random init by a lot.",
           "expect": "DINOv3 - DINOv2 < 0.03 mIoU; pretrained - random > 0.2.",
           "rule": "supported if DINOv3-DINOv2 is real (>= tol) and both beat random by > 0.1."},
    "H4": {"claim": "A 3.5M-param decoder head does not beat a 4.6k-param linear probe at full data on M1/M2 (capacity saturation).",
           "expect": "|delta| < tol on M1 and M2; any gain appears only in M3.",
           "rule": "supported if the decoder-minus-LP difference on M1 and M2 is not real in favour of the decoder."},
    "H5": {"claim": "Part of the M1/M2 ceiling is label noise at region borders: excluding a band around label edges raises every model and shrinks the spread between models.",
           "expect": "mIoU(r=8) - mIoU(r=0) >= 0.03 on M1 for every config, spread across configs smaller at r=8.",
           "rule": "supported if both hold."},
    "H6": {"claim": "big_rock failure is size-driven: small or fragmented boulders are missed, large contiguous ones are found.",
           "expect": "Recall of the largest third of big_rock components exceeds the smallest third by >= 0.2.",
           "rule": "supported if that gap is >= 0.2 (LP last_norm, pooled M1+M2)."},
}
hp = OUT / "hypotheses_preregistered.json"
if not hp.exists():
    json.dump({"registered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tol": TOL, "hypotheses": HYPOTHESES},
              open(hp, "w"), indent=2)
print("hypotheses registered:", hp)

# %% 2. data  (conventions as in the earlier runs)
def find_root(search_dir=INPUT):
    for dp, dns, _ in os.walk(search_dir):
        if "msl" in dns and (Path(dp) / "msl" / "ncam").is_dir():
            return Path(dp)
        if len(Path(dp).relative_to(search_dir).parts) >= 6:
            dns[:] = []
    raise FileNotFoundError(f"no folder containing msl/ncam under {search_dir}")


def locate(root):
    ncam = Path(root) / "msl" / "ncam"
    img_dir, train_dir = ncam / "images" / "edr", ncam / "labels" / "train"
    assert img_dir.is_dir() and train_dir.is_dir(), (img_dir, train_dir)
    tests = {}
    for k in (1, 2, 3):
        d = ncam / "labels" / "test" / f"masked-gold-min{k}-100agree"
        if d.is_dir() and any(d.glob("*.png")):
            tests[f"M{k}"] = d
    missing = [f"M{k}" for k in (1, 2, 3) if f"M{k}" not in tests]
    assert not missing, f"missing test labels: {missing}"
    return img_dir, train_dir, tests


def make_pairs(img_dir, lbl_dir, verbose=True):
    idx = {Path(f).stem.lower(): str(Path(img_dir) / f)
           for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)}
    pairs, unmatched = [], []
    for f in sorted(os.listdir(lbl_dir)):
        if not f.lower().endswith(".png"):
            continue
        stem = Path(f).stem.lower()
        ip = idx.get(stem) or idx.get(stem[:-len("_merged")] if stem.endswith("_merged") else stem)
        (pairs if ip else unmatched).append((ip, str(Path(lbl_dir) / f)) if ip else f)
    if verbose:
        print(f"{Path(lbl_dir).name}: {len(pairs)} paired, {len(unmatched)} unmatched")
    return pairs


def build_splits(root, n_val=500, verbose=True):
    img_dir, train_dir, test_dirs = locate(root)
    tests = {k: make_pairs(img_dir, d, verbose) for k, d in test_dirs.items()}
    allp = make_pairs(img_dir, train_dir, verbose)
    test_stems = {Path(i).stem.lower() for ps in tests.values() for i, _ in ps}
    before = len(allp)
    allp = [p for p in allp if Path(p[0]).stem.lower() not in test_stems]
    allp.sort(key=lambda p: Path(p[1]).stem)               # filenames start with the spacecraft clock
    val, train = allp[-n_val:], allp[:-n_val]
    if verbose:
        print(f"leakage removed: {before - len(allp)} | train {len(train)} | val {len(val)} | "
              + " | ".join(f"{k} {len(v)}" for k, v in tests.items()))
    return {"train": train, "val": val, **tests}


def read_label(path):
    m = np.array(Image.open(path))
    return _LUT[m[..., 0] if m.ndim == 3 else m]


class AI4MarsSeg(Dataset):
    """native_labels=True (eval): labels stay at native size; the caller upsamples logits to them."""

    def __init__(self, pairs, size, mean, std, train=False, native_labels=False, scale_crop=(0.5, 1.0)):
        self.pairs, self.size, self.train = pairs, size, train
        self.native_labels, self.scale_crop = native_labels, scale_crop
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img, lbl = Image.open(ip).convert("RGB"), Image.fromarray(read_label(lp))
        assert img.size == lbl.size, (ip, img.size, lbl.size)
        if self.train:
            if self.scale_crop and random.random() < 0.5:
                W, H = img.size
                s = random.uniform(*self.scale_crop)
                cw, ch = int(W * s), int(H * s)
                x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
                img, lbl = img.crop((x0, y0, x0 + cw, y0 + ch)), lbl.crop((x0, y0, x0 + cw, y0 + ch))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT); lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if not self.native_labels:
            lbl = lbl.resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, torch.from_numpy(np.asarray(lbl).astype(np.int64))


if os.environ.get("A4M_ONLY_E"):
    ROOT, SPLITS = None, None
else:
    ROOT = find_root()
    SPLITS = build_splits(ROOT, n_val=CFG["n_val"])
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)       # DINOv3 and DINOv2 both use ImageNet stats
NW = CFG["num_workers"]
EVAL_SETS = ["val", "M1", "M2", "M3"]


def train_loader(pairs, size, bs, seed):
    g = torch.Generator().manual_seed(seed)                     # seeds order, augmentation and worker RNG
    return DataLoader(AI4MarsSeg(pairs, size, MEAN, STD, train=True), batch_size=bs, shuffle=True,
                      num_workers=NW, drop_last=True, pin_memory=AMP, generator=g)


def eval_loaders(size, bs=4, sets=EVAL_SETS):
    return {k: DataLoader(AI4MarsSeg(SPLITS[k], size, MEAN, STD, native_labels=True), batch_size=bs,
                          shuffle=False, num_workers=NW, pin_memory=AMP) for k in sets}



# %% 2b. one-time image cache: decode JPEG/PNG once, keep uint8 at the training size (the data loader was the bottleneck)
CACHE_DIR = Path(os.environ.get("A4M_CACHE", "/kaggle/temp"))
if not CACHE_DIR.parent.exists() or not os.access(CACHE_DIR.parent, os.W_OK):
    CACHE_DIR = OUT / "cache"
_MEAN_T, _STD_T = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)


def _cache_paths(S, tag=""):
    return CACHE_DIR / f"train_img_{S}{tag}.npy", CACHE_DIR / f"train_lbl_{S}{tag}.npy", CACHE_DIR / f"cache_{S}{tag}.ok"


def _load_pair(a):
    ip, lp, S = a
    img = np.asarray(Image.open(ip).convert("RGB").resize((S, S), Image.BILINEAR))
    lbl = np.asarray(Image.fromarray(read_label(lp)).resize((S, S), Image.NEAREST))
    return img, lbl


def ensure_cache(S=None, pairs=None, tag=""):
    """Idempotent. images are stored 1-channel when the NavCam frames are gray (checked), labels at the same size."""
    import multiprocessing as mp
    S = S or CFG["size"]; pairs = pairs if pairs is not None else SPLITS["train"]
    img_p, lbl_p, ok = _cache_paths(S, tag)
    if ok.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    probe = [_load_pair((ip, lp, S))[0] for ip, lp in pairs[:32]]
    gray = all(np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]) for a in probe)
    shape = (len(pairs), S, S) if gray else (len(pairs), S, S, 3)
    imgs = np.lib.format.open_memmap(img_p, mode="w+", dtype=np.uint8, shape=shape)
    lbls = np.lib.format.open_memmap(lbl_p, mode="w+", dtype=np.uint8, shape=(len(pairs), S, S))
    t0 = time.time()
    with mp.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
        for i, (a, l) in enumerate(tqdm(pool.imap(_load_pair, [(ip, lp, S) for ip, lp in pairs], chunksize=16),
                                        total=len(pairs), desc="building image cache")):
            imgs[i] = a[..., 0] if gray else a; lbls[i] = l
    imgs.flush(); lbls.flush(); del imgs, lbls
    ok.write_text(json.dumps({"n": len(pairs), "size": S, "gray": gray}))
    print(f"image cache: {len(pairs)} images at {S}px ({'gray' if gray else 'rgb'}) in {(time.time() - t0) / 60:.1f} min -> {CACHE_DIR}")


def gpu_aug(g, l, rng):
    """uint8 [B,S,S(,3)] images + [B,S,S] labels -> normalised float [B,3,S,S], long [B,S,S] on DEVICE.
    Same augmentation as before (50%: random crop of 50-100% of the side, resized back; 50%: h-flip)."""
    g = g.to(DEVICE, non_blocking=True).float() / 255; l = l.to(DEVICE, non_blocking=True).long()
    g = g.unsqueeze(1) if g.dim() == 3 else g.permute(0, 3, 1, 2)
    S = g.shape[-1]; xs, ys = [], []
    for i in range(g.shape[0]):
        xi, yi = g[i:i + 1], l[i:i + 1].unsqueeze(1).float()
        if rng.random() < 0.5:
            c = int(S * rng.uniform(0.5, 1.0)); x0, y0 = rng.randint(0, S - c), rng.randint(0, S - c)
            xi = F.interpolate(xi[..., y0:y0 + c, x0:x0 + c], size=(S, S), mode="bilinear", align_corners=False)
            yi = F.interpolate(yi[..., y0:y0 + c, x0:x0 + c], size=(S, S), mode="nearest")
        if rng.random() < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        xs.append(xi); ys.append(yi)
    x = torch.cat(xs); x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
    return (x - _MEAN_T.to(DEVICE)) / _STD_T.to(DEVICE), torch.cat(ys).squeeze(1).long()


class CacheTrain:
    """Training batches from the memory-mapped cache; augmentation runs on the GPU."""

    def __init__(self, S, bs, seed, tag=""):
        img_p, lbl_p, _ = _cache_paths(S, tag)
        self.imgs, self.lbl = np.load(img_p, mmap_mode="r"), np.load(lbl_p, mmap_mode="r")
        self.bs, self.seed = bs, seed

    def __len__(self):
        return len(self.imgs) // self.bs

    def epoch(self, ep):
        perm = np.random.default_rng(self.seed * 100003 + ep).permutation(len(self.imgs)); arng = random.Random(self.seed * 7919 + ep)
        for b in range(len(self)):
            idx = np.sort(perm[b * self.bs:(b + 1) * self.bs])
            yield gpu_aug(torch.from_numpy(np.ascontiguousarray(self.imgs[idx])), torch.from_numpy(np.ascontiguousarray(self.lbl[idx])), arng)


class LoaderTrain:
    """Fallback for sizes other than the cache size (e.g. a 1024 decoder): the original DataLoader path."""

    def __init__(self, pairs, size, bs, seed):
        self.pairs, self.size, self.bs, self.seed = pairs, size, bs, seed

    def __len__(self):
        return len(self.pairs) // self.bs

    def epoch(self, ep):
        for x, y in train_loader(self.pairs, self.size, self.bs, self.seed * 1000 + ep):
            yield x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)


def make_train(size, bs, seed):
    if size == CFG["size"]:
        ensure_cache(size)
        return CacheTrain(size, bs, seed)
    return LoaderTrain(SPLITS["train"], size, bs, seed)


class CachedDS(Dataset):
    """Cached train images by index (no augmentation), resized to `size` if it differs from the cache size."""

    def __init__(self, idx, size):
        ensure_cache(CFG["size"])
        img_p, lbl_p, _ = _cache_paths(CFG["size"])
        self.imgs, self.lbl, self.idx, self.size = np.load(img_p, mmap_mode="r"), np.load(lbl_p, mmap_mode="r"), np.asarray(idx), size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = torch.from_numpy(np.array(self.imgs[self.idx[i]])).float() / 255
        g = g[None].expand(3, -1, -1) if g.dim() == 2 else g.permute(2, 0, 1)
        l = torch.from_numpy(np.array(self.lbl[self.idx[i]])).long()
        if self.size != g.shape[-1]:
            g = F.interpolate(g[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0]
            l = F.interpolate(l[None, None].float(), size=(self.size, self.size), mode="nearest")[0, 0].long()
        return (g - _MEAN_T[0]) / _STD_T[0], l


# %% 1b. multi-GPU launcher: one worker process per GPU, then this process runs stage E
import sys, subprocess
_SCRIPT = globals().get("__file__")
_N_GPU = int(os.environ.get("A4M_FAKE_GPUS", 0)) or torch.cuda.device_count()
if CFG["parallel"] and not JOB and _N_GPU >= 2 and _SCRIPT and not os.environ.get("A4M_BENCH"):
    print(f"launching {len(JOBS)} workers on {_N_GPU} GPUs")
    ensure_cache(CFG["size"])                           # build the shared image cache once, before the workers start
    procs = {j: subprocess.Popen([sys.executable, _SCRIPT], env={**os.environ, "A4M_JOB": j}) for j in JOBS}
    for j, p in procs.items():
        rc = p.wait(); print(f"worker {j} finished with exit code {rc}" + ("" if rc == 0 else "  <-- check its log above"))
    CFG["stages"] = ["E"]                                # the workers did the training; analysis runs here
elif CFG["parallel"] and not JOB:
    print("single-process mode (need >=2 GPUs and a script kernel for the two-GPU split)")

# %% 3. backbones
def find_local_dinov3():
    """A DINOv3 ViT-B/16 folder attached as a Kaggle dataset (config.json with model_type dinov3_vit), if any."""
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


DINOV3_LOCAL = find_local_dinov3()
if DINOV3_LOCAL:
    print("using attached DINOv3 weights:", DINOV3_LOCAL)


def load_backbone(model_id, random_init=False):
    from transformers import AutoModel, AutoConfig
    if DINOV3_LOCAL and "dinov3-vitb16" in model_id:
        model_id = DINOV3_LOCAL
    if SMOKE:                                                   # tiny random backbones, no download
        from transformers import DINOv3ViTConfig, DINOv3ViTModel, Dinov2Config, Dinov2Model
        if "dinov2" in model_id:
            return Dinov2Model(Dinov2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                            num_attention_heads=2, patch_size=14, image_size=56)).to(DEVICE).eval()
        return DINOv3ViTModel(DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=12,
                                              num_attention_heads=2, patch_size=16, num_register_tokens=4,
                                              image_size=64)).to(DEVICE).eval()
    if random_init:
        m = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
    else:
        try:
            m = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        except TypeError:
            m = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    return m.to(DEVICE).eval()


def freeze(m):
    for p in m.parameters():
        p.requires_grad = False
    return m.eval()


def bb_info(bb):
    cfg = bb.config
    blocks = [mod for _, mod in bb.named_modules()
              if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers]
    assert blocks, "transformer block list not found"
    blocks = max(blocks, key=lambda ml: sum(p.numel() for p in ml.parameters()))
    norm = getattr(bb, "norm", None) or getattr(bb, "layernorm", None)
    return dict(blocks=blocks, depth=int(cfg.num_hidden_layers), C=int(cfg.hidden_size),
                patch=int(cfg.patch_size), n_prefix=1 + int(getattr(cfg, "num_register_tokens", 0) or 0), norm=norm)


def to_map(tok, n_prefix, grid):
    t = tok[:, n_prefix:, :]
    assert t.shape[1] == grid * grid, (t.shape, grid)
    return t.transpose(1, 2).reshape(t.shape[0], -1, grid, grid)


def lin_head(C):
    return nn.Sequential(nn.BatchNorm2d(C), nn.Conv2d(C, NUM_CLASSES, 1))


# %% 4. metrics, per-image confusion, bootstrap
def iou_np(c):
    c = np.asarray(c, np.float64)
    tp = np.diagonal(c, axis1=-2, axis2=-1); union = c.sum(-1) + c.sum(-2) - tp
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def miou_np(c):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(iou_np(c), axis=-1)


def per_image_conf(pred, y, radii):
    """int32 [B, R, 4, 4]. radius r>0 drops pixels within r px of a label change or of an unlabeled pixel."""
    B = y.shape[0]; yf = y.float().unsqueeze(1); out = []
    base = torch.arange(B, device=y.device).view(B, 1, 1) * 16
    for r in radii:
        keep = y != IGNORE_INDEX
        if r > 0:
            k = 2 * r + 1
            mx, mn = F.max_pool2d(yf, k, 1, r), -F.max_pool2d(-yf, k, 1, r)
            keep = keep & (mx == mn).squeeze(1)
        idx = (base + y * 4 + pred)[keep]
        out.append(torch.bincount(idx, minlength=B * 16).view(B, 4, 4))
    return torch.stack(out, 1).cpu().numpy().astype(np.int32)


def big_rock_components(pred, y, min_px=25):
    from scipy import ndimage
    rows = []
    for b in range(y.shape[0]):
        gt = (y[b] == 3).cpu().numpy()
        if not gt.any():
            continue
        lab, n = ndimage.label(gt)
        pb = (pred[b] == 3).cpu().numpy()
        for i in range(1, n + 1):
            m = lab == i; a = int(m.sum())
            if a >= min_px:
                rows.append((a, float((pb & m).sum() / a)))
    return rows


def score_batch(logits, y, radii, comps):
    logits = logits.float()
    if logits.shape[-2:] != y.shape[-2:]:
        logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    pred = logits.argmax(1)
    return per_image_conf(pred, y, radii), (big_rock_components(pred, y) if comps else [])


@torch.no_grad()
def eval_images(predict, dl, radii=(0,), comps=False):
    """predict(x) -> {name: logits}. Returns {name: {"conf": int32[N,R,4,4], "comps": [(area, recall)]}}."""
    acc = {}
    for x, y in tqdm(dl, leave=False, desc="eval"):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            lg = predict(x)
        for name, logits in lg.items():
            c, cp = score_batch(logits, y, radii, comps)
            a = acc.setdefault(name, {"conf": [], "comps": []})
            a["conf"].append(c); a["comps"] += cp
    return {n: {"conf": np.concatenate(a["conf"]), "comps": a["comps"]} for n, a in acc.items()}


def boot_idx(n, B, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def boot_metric(confs, idx, metric=miou_np):
    """confs: list over seeds of [N,4,4]. Same image resample for every seed; returns [B] seed-mean metric."""
    vals = []
    for c in confs:
        s = np.concatenate([c[idx[i:i + 200]].sum(1) for i in range(0, len(idx), 200)])
        vals.append(metric(s))
    return np.mean(vals, 0)


def point_metric(confs, metric=miou_np):
    return float(np.mean([metric(c.sum(0)) for c in confs]))


def ci95(x):
    lo, hi = np.percentile(x, [2.5, 97.5]); return float(lo), float(hi)


def paired(confs_a, confs_b, metric=miou_np, B=None, seed=0):
    B = B or CFG["n_boot"]; n = confs_a[0].shape[0]; idx = boot_idx(n, B, seed)
    d = boot_metric(confs_a, idx, metric) - boot_metric(confs_b, idx, metric)
    delta = point_metric(confs_a, metric) - point_metric(confs_b, metric)
    lo, hi = ci95(d)
    return dict(delta=delta, lo=lo, hi=hi, real=bool((lo > 0 or hi < 0) and abs(delta) >= TOL))


# %% 5. result store (working dir -> attached inputs -> HF Hub)
def find_art(fname):
    p = RES_DIR / fname
    if p.exists():
        return p
    for d in range(4):
        for sub in ("", "results"):
            pat = "/".join(x for x in ("/".join(["*"] * d), sub, fname) if x)
            hit = next(INPUT.glob(pat), None) if INPUT.exists() else None
            if hit:
                return hit
    if HF_REPO and not SMOKE:
        try:
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(HF_REPO, fname))
        except Exception:
            pass
    return None


def push(*paths):
    if not HF_REPO or SMOKE:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(HF_REPO, private=True, exist_ok=True)
        for p in paths:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=Path(p).name, repo_id=HF_REPO)
    except Exception as e:
        print("  HF upload failed (files are still in results/):", type(e).__name__, e)


def have(name):
    return find_art(f"{name}.json") is not None and find_art(f"{name}.npz") is not None


def save_result(name, meta, confs, comps=None):
    """confs: {"M1__last_norm": int32[N,R,4,4], ...}; comps: {key: [(area, recall)]}."""
    np.savez_compressed(RES_DIR / f"{name}.npz", **confs)
    json.dump({**meta, "name": name, "radii": list(CFG["radii"]), "comps": comps or {}},
              open(RES_DIR / f"{name}.json", "w"), indent=1)
    push(RES_DIR / f"{name}.json", RES_DIR / f"{name}.npz")
    print(f"  saved {name}")


def load_result(name):
    j, z = find_art(f"{name}.json"), find_art(f"{name}.npz")
    if not (j and z):
        return None
    return json.load(open(j)), dict(np.load(z))


# %% 6. Stage A — layer-wise linear probes on the FULL train set (one forward pass for all layers)
def stage_A():
    size, bs = CFG["size"], CFG["bs"]
    layers = [str(l) for l in CFG["layers"]]
    bb = freeze(load_backbone(CFG["model_id"])); info = bb_info(bb); grid = size // info["patch"]
    elo = eval_loaders(size)
    for seed in CFG["seeds"]["A"]:
        name = f"A_layers_s{seed}"
        if have(name):
            print(f"[{name}] cached"); continue
        if over_budget(0.6, name):
            continue
        print(f"\n=== {name} ==="); seed_all(seed); t0 = time.time()
        heads = nn.ModuleDict({l: lin_head(info["C"]) for l in layers}).to(DEVICE)
        tdl = make_train(size, bs, seed)
        opt = torch.optim.AdamW(heads.parameters(), lr=CFG["A_lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["A_lr"], total_steps=CFG["A_epochs"] * len(tdl), pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)

        def feats(x):
            o = bb(pixel_values=x, output_hidden_states=True)
            return {l: to_map(o.last_hidden_state if l == "last_norm" else o.hidden_states[int(l)],
                              info["n_prefix"], grid) for l in layers}

        def predict(x):
            with torch.no_grad():
                f = feats(x)
            return {l: heads[l](f[l].float()) for l in layers}

        best = {l: (-1.0, None, -1) for l in layers}; val_hist = []
        for ep in range(CFG["A_epochs"]):
            heads.train()
            for x, y in tqdm(tdl.epoch(ep), total=len(tdl), desc=f"ep {ep}", leave=False):
                with autocast():
                    lg = predict(x)
                    loss = sum(F.cross_entropy(F.interpolate(v.float(), size=y.shape[-2:], mode="bilinear",
                                                             align_corners=False), y, ignore_index=IGNORE_INDEX)
                               for v in lg.values())
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
            heads.eval()
            r = eval_images(predict, elo["val"])
            row = {l: float(miou_np(r[l]["conf"][:, 0].sum(0))) for l in layers}
            val_hist.append(row)
            for l in layers:
                if row[l] > best[l][0]:
                    best[l] = (row[l], copy.deepcopy(heads[l].state_dict()), ep)
            print(f"ep {ep} val mIoU: " + " ".join(f"{l}:{row[l]:.3f}" for l in layers))
        for l in layers:
            heads[l].load_state_dict(best[l][1])
        heads.eval()
        confs, comps = {}, {}
        for k in ("val", "M1", "M2", "M3"):
            r = eval_images(predict, elo[k], CFG["radii"], comps=k != "val")
            for l in layers:
                confs[f"{k}__{l}"] = r[l]["conf"]
                if k != "val":
                    comps[f"{k}__{l}"] = r[l]["comps"]
        meta = dict(stage="A", seed=seed, size=size, layers=layers, epochs=CFG["A_epochs"],
                    val_miou={l: best[l][0] for l in layers}, best_epoch={l: best[l][2] for l in layers},
                    val_hist=val_hist, trainable_per_head=sum(p.numel() for p in heads[layers[0]].parameters()),
                    n_train=len(SPLITS["train"]), minutes=round((time.time() - t0) / 60, 1))
        save_result(name, meta, confs, comps)
    del bb; torch.cuda.empty_cache()


# %% 7. Stage B — decoder head (3.5M params), reuses the earlier run's checkpoint for seed 0 if attached
class DinoSeg(nn.Module):
    """Frozen backbone + 4-layer decoder head. layers: hidden_states indices, negative = from the end."""

    def __init__(self, model_id, layers=(-10, -7, -4, -1), dim=256):
        super().__init__()
        self.backbone = freeze(load_backbone(model_id)); info = bb_info(self.backbone)
        self.patch, self.n_prefix, C, depth = info["patch"], info["n_prefix"], info["C"], info["depth"]
        self.depth = depth; self.norm = info["norm"]
        self.layers = tuple(l if l > 0 else depth + 1 + l for l in layers)
        self.proj = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(C), nn.Conv2d(C, dim, 1)) for _ in self.layers])
        self.fuse = nn.Sequential(nn.Conv2d(dim * len(self.layers), dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(True))
        self.up1 = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.BatchNorm2d(dim // 2), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.Conv2d(dim // 2, dim // 4, 3, padding=1), nn.BatchNorm2d(dim // 4), nn.ReLU(True))
        self.cls = nn.Conv2d(dim // 4, NUM_CLASSES, 1)

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self, x):
        B, _, H, W = x.shape; h, w = H // self.patch, W // self.patch
        with torch.no_grad():
            hs = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states
            maps = []
            for l in self.layers:
                t = hs[l]
                if l == self.depth and self.norm is not None:
                    t = self.norm(t)
                maps.append(to_map(t, self.n_prefix, h))
        z = self.fuse(torch.cat([p(f.float()) for p, f in zip(self.proj, maps)], 1))
        z = self.up1(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        z = self.up2(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        return F.interpolate(self.cls(z), size=(H, W), mode="bilinear", align_corners=False)

    def head_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("backbone.")}


def fit_decoder(model, tdl, val_dl, epochs, lr, ckpt, resume):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tdl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    start, best, best_ep, hist = 0, -1.0, -1, []
    if resume.exists():
        s = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(s["model"], strict=False); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, best, best_ep, hist = s["epoch"] + 1, s["best"], s["best_ep"], s["hist"]
        print(f"  resumed at epoch {start}")
    for ep in range(start, epochs):
        model.train(); run = 0.0
        for x, y in tqdm(tdl.epoch(ep), total=len(tdl), desc=f"ep {ep}", leave=False):
            with autocast():
                lg = model(x)
            loss = F.cross_entropy(lg.float(), y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); run += loss.item()
        model.eval()
        if (ep + 1) % CFG["decoder_eval_every"] == 0 or ep == epochs - 1:      # val is ~1 min at native res
            r = eval_images(lambda x: {"m": model(x)}, val_dl)["m"]["conf"][:, 0].sum(0)
            v = float(miou_np(r)); hist.append({"epoch": ep, "loss": run / len(tdl), "val_miou": v})
            print(f"  ep {ep}: loss {run / len(tdl):.4f} | val mIoU {v:.4f}")
            if v > best:
                best, best_ep = v, ep; torch.save(model.head_state(), ckpt)
        else:
            hist.append({"epoch": ep, "loss": run / len(tdl)}); print(f"  ep {ep}: loss {run / len(tdl):.4f} (no eval)")
        torch.save({"epoch": ep, "model": model.head_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "best_ep": best_ep, "hist": hist}, resume)
    model.load_state_dict(torch.load(ckpt, weights_only=False), strict=False)
    return hist, best_ep, best


def stage_B():
    for size in CFG["decoder_sizes"]:
        bs = 16 if size <= 512 else 8
        elo = eval_loaders(size, bs=max(1, bs // 2))
        for seed in CFG["seeds"]["B"]:
            name = f"B_decoder{size}_s{seed}"
            if have(name):
                print(f"[{name}] cached"); continue
            if over_budget(1.3, name):
                continue
            print(f"\n=== {name} ==="); seed_all(seed); t0 = time.time()
            model = DinoSeg(CFG["model_id"]).to(DEVICE)
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            legacy = find_art(f"dinov3b_decoder_{size}.pt") if (seed == 0 and CFG["reuse_legacy_decoder"]) else None   # earlier run: other pipeline/epochs
            own = find_art(f"{name}.pt")
            hist, best_ep, best_val, src = [], -1, float("nan"), "trained"
            if legacy or own:
                sd = torch.load(legacy or own, map_location=DEVICE, weights_only=False)
                miss, unexp = model.load_state_dict(sd, strict=False)
                assert not unexp and not [k for k in miss if not k.startswith("backbone.")], (unexp, miss[:3])
                src = f"loaded {Path(legacy or own).name}"
            else:
                tdl = make_train(size, bs, seed)
                hist, best_ep, best_val = fit_decoder(model, tdl, elo["val"], CFG["decoder_epochs"],
                                                      CFG["decoder_lr"], RES_DIR / f"{name}.pt", RES_DIR / f"{name}_resume.pt")
                push(RES_DIR / f"{name}.pt")
            model.eval(); confs, comps = {}, {}
            for k in EVAL_SETS:
                r = eval_images(lambda x: {"decoder": model(x)}, elo[k], CFG["radii"], comps=k != "val")["decoder"]
                confs[f"{k}__decoder"] = r["conf"]
                if k != "val":
                    comps[f"{k}__decoder"] = r["comps"]
            meta = dict(stage="B", seed=seed, size=size, source=src, epochs=CFG["decoder_epochs"], hist=hist,
                        best_epoch=best_ep, val_miou=best_val, trainable=n_tr, minutes=round((time.time() - t0) / 60, 1))
            save_result(name, meta, confs, comps)
            del model; torch.cuda.empty_cache()


# %% 8. token caches + trainer (Stages C and D)
@torch.no_grad()
def extract_tokens(bb, ds, take, bs=16):
    """take: int (hidden_states index) or 'last' (final normed output). Returns fp16 tokens [N,T,C], uint8 labels."""
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=NW)
    toks = labs = None; i = 0
    for x, y in tqdm(dl, leave=False, desc="extract"):
        with autocast():
            o = bb(pixel_values=x.to(DEVICE), output_hidden_states=take != "last")
        t = (o.last_hidden_state if take == "last" else o.hidden_states[take]).half()
        assert torch.isfinite(t).all(), "non-finite tokens (fp16 overflow)"
        if toks is None:
            toks = torch.empty((len(ds),) + tuple(t.shape[1:]), dtype=torch.float16)
            labs = torch.empty((len(ds),) + tuple(y.shape[1:]), dtype=torch.uint8)
        toks[i:i + len(x)] = t.cpu(); labs[i:i + len(x)] = y.to(torch.uint8); i += len(x)
    return toks, labs


def build_caches(bb, size, take, pool_idx):
    caches = {"train": extract_tokens(bb, CachedDS(pool_idx, size), take)}
    for k in EVAL_SETS:
        caches[k] = extract_tokens(bb, AI4MarsSeg(SPLITS[k], size, MEAN, STD, native_labels=True), take)
    return caches


class TokenNet(nn.Module):
    """Cached tokens -> (n_replay frozen/trainable blocks) -> final norm -> 1x1 linear head."""

    def __init__(self, info, grid, pos, n_replay, n_train):
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(b) for b in info["blocks"][info["depth"] - n_replay:]]) if n_replay else nn.ModuleList()
        self.norm = copy.deepcopy(info["norm"]) if n_replay else None
        self.pos, self.grid, self.n_prefix = pos, grid, info["n_prefix"]
        for p in list(self.blocks.parameters()) + (list(self.norm.parameters()) if self.norm else []):
            p.requires_grad = False
        for b in self.blocks[len(self.blocks) - n_train:]:
            for p in b.parameters():
                p.requires_grad = True
        if n_train and self.norm:
            for p in self.norm.parameters():
                p.requires_grad = True
        self.head = lin_head(info["C"])

    def forward(self, tok):
        h = tok.float()
        for blk in self.blocks:
            o = blk(h, position_embeddings=self.pos)
            h = o[0] if isinstance(o, (tuple, list)) else o
        if self.norm is not None:
            h = self.norm(h)
        return self.head(to_map(h, self.n_prefix, self.grid))

    def block_params(self):
        return [p for p in list(self.blocks.parameters()) + (list(self.norm.parameters()) if self.norm else []) if p.requires_grad]


def eval_tokens(net, cache, radii=(0,), comps=False, bs=16):
    toks, labs = cache; net.eval(); acc = {"conf": [], "comps": []}
    with torch.no_grad():
        for i in range(0, len(toks), bs):
            t = toks[i:i + bs].to(DEVICE); y = labs[i:i + bs].to(DEVICE).long()
            with autocast():
                lg = net(t)
            c, cp = score_batch(lg, y, radii, comps)
            acc["conf"].append(c); acc["comps"] += cp
    return {"conf": np.concatenate(acc["conf"]), "comps": acc["comps"]}


def train_tokens(net, caches, n, seed, epochs, bs, head_lr, block_lr, sweep=False):
    """Trains on the first n images of a seed-specific permutation of the cached pool; selects by val mIoU."""
    toks, labs = caches["train"]
    perm = np.random.default_rng(1000 + seed).permutation(len(toks))[:n]
    net.to(DEVICE); seed_all(seed)
    groups = [{"params": list(net.head.parameters()), "lr": head_lr}]
    bp = net.block_params()
    if bp:
        groups.append({"params": bp, "lr": block_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, best_ep, hist = -1.0, None, -1, []
    for ep in range(epochs):
        net.train()
        for blk in net.blocks:
            blk.eval()                                           # no dropout/stochastic depth in the replayed blocks
        order = np.random.default_rng(seed * 7919 + ep).permutation(perm)
        for b in range(steps):
            idx = torch.from_numpy(order[b * bs:(b + 1) * bs])
            t = toks[idx].to(DEVICE, non_blocking=True); y = labs[idx].to(DEVICE).long()
            with autocast():
                lg = net(t)
            loss = F.cross_entropy(F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False),
                                   y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        v = float(miou_np(eval_tokens(net, caches["val"])["conf"][:, 0].sum(0))); hist.append(v)
        if v > best:
            best, best_ep = v, ep
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return best, best_ep, hist


def finish_run(name, meta, net, caches):
    confs, comps = {}, {}
    for k in EVAL_SETS:
        r = eval_tokens(net, caches[k], CFG["radii"], comps=k != "val")
        confs[f"{k}__probe"] = r["conf"]
        if k != "val":
            comps[f"{k}__probe"] = r["comps"]
    save_result(name, meta, confs, comps)


# %% 9. Stage C — label efficiency and partial fine-tuning on cached block-(depth-4) tokens
def stage_C():
    size, bs = CFG["size"], CFG["C_bs"]
    names = [f"C_{m}_n{n}_s{s}" for s in CFG["seeds"]["C"] for n in CFG["fracs"] for m in ("LP", "FT2", "FT4")]
    if all(have(n) for n in names) and have("C_sweep"):
        print("[C] all cached"); return
    bb = freeze(load_backbone(CFG["model_id"])); info = bb_info(bb); grid = size // info["patch"]; depth = info["depth"]
    take = depth - 4
    pool = np.random.default_rng(0).permutation(len(SPLITS["train"]))[:CFG["pool"]]
    print(f"caching tokens after block {take}: pool {len(pool)}")
    caches = build_caches(bb, size, take, pool)
    with torch.no_grad():
        pos = bb.rope_embeddings(torch.zeros(1, 3, size, size, device=DEVICE))
        pos = tuple(p.to(DEVICE) for p in pos)
        probe_net = TokenNet(info, grid, pos, 4, 0).to(DEVICE).eval()
        x = torch.stack([CachedDS(pool[:2], size)[i][0] for i in range(2)]).to(DEVICE)
        o = bb(pixel_values=x, output_hidden_states=True)
        h = o.hidden_states[take].float()
        for blk in probe_net.blocks:
            r = blk(h, position_embeddings=pos); h = r[0] if isinstance(r, (tuple, list)) else r
        err = (probe_net.norm(h) - o.last_hidden_state.float()).abs().max().item()
        scale = o.last_hidden_state.float().abs().max().item()
    assert err <= 1e-3 * max(scale, 1.0), f"block replay does not reproduce the backbone (max err {err:.2e}); fix before using Stage C"
    print(f"block replay check OK (max abs err {err:.2e})")
    del probe_net

    def run(method, n, seed, block_lr):
        n_train = {"LP": 0, "FT2": 2, "FT4": 4}[method]
        net = TokenNet(info, grid, pos, 4, n_train)
        t0 = time.time()
        best, ep, hist = train_tokens(net, caches, n, seed, CFG["C_epochs"], bs, CFG["head_lr"], block_lr)
        return net, dict(stage="C", method=method, n=n, seed=seed, block_lr=block_lr, val_miou=best, best_epoch=ep,
                         val_hist=hist, trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),
                         minutes=round((time.time() - t0) / 60, 1))

    # hyper-parameter sensitivity: block LR sweep at one size, one seed; the winner is reused everywhere
    sw = load_result("C_sweep")
    if sw is None:
        n_sw = min(CFG["ft_sweep_n"], CFG["pool"]); sweep = {}
        for m in ("FT2", "FT4"):
            for lr in CFG["ft_lrs"]:
                _, meta = run(m, n_sw, 0, lr); sweep[f"{m}@{lr:g}"] = meta["val_miou"]
                print(f"  sweep {m} lr {lr:g}: val mIoU {meta['val_miou']:.4f}")
        best_lr = {m: max(CFG["ft_lrs"], key=lambda lr: sweep[f"{m}@{lr:g}"]) for m in ("FT2", "FT4")}
        json.dump({"stage": "C_sweep", "n": n_sw, "val_miou": sweep, "best_lr": best_lr, "comps": {}, "radii": []},
                  open(RES_DIR / "C_sweep.json", "w"), indent=1)
        np.savez_compressed(RES_DIR / "C_sweep.npz", x=np.zeros(1)); push(RES_DIR / "C_sweep.json", RES_DIR / "C_sweep.npz")
    else:
        best_lr = sw[0]["best_lr"]
    print("block LR chosen on val:", best_lr)
    for seed in CFG["seeds"]["C"]:
        for n in CFG["fracs"]:
            for m in ("LP", "FT2", "FT4"):
                name = f"C_{m}_n{n}_s{seed}"
                if have(name) or over_budget(0.1, name):
                    continue
                print(f"\n=== {name} ===")
                net, meta = run(m, min(n, CFG["pool"]), seed, best_lr.get(m, 0.0))
                finish_run(name, meta, net, caches); del net; torch.cuda.empty_cache()
    del caches, bb


# %% 10. Stage D — backbone comparison (probe on cached last-layer tokens, same recipe for all)
def stage_D():
    for key, model_id, size, rand in CFG["backbones"]:
        names = [f"D_{key}_s{s}" for s in CFG["seeds"]["D"]]
        if all(have(n) for n in names):
            print(f"[D {key}] cached"); continue
        if over_budget(0.2, f"D {key}"):
            continue
        seed_all(0)
        bb = freeze(load_backbone(model_id, random_init=rand)); info = bb_info(bb); grid = size // info["patch"]
        pool = np.random.default_rng(0).permutation(len(SPLITS["train"]))[:CFG["D_n"]]
        print(f"\n--- {key}: {size}px, grid {grid}x{grid}, patch {info['patch']} ---")
        caches = build_caches(bb, size, "last", pool)
        for seed, name in zip(CFG["seeds"]["D"], names):
            if have(name):
                continue
            net = TokenNet(info, grid, None, 0, 0); t0 = time.time()
            best, ep, hist = train_tokens(net, caches, len(pool), seed, CFG["D_epochs"], CFG["C_bs"], CFG["head_lr"], 0.0)
            meta = dict(stage="D", key=key, model_id=model_id, random_init=rand, size=size, patch=info["patch"], seed=seed,
                        n=len(pool), val_miou=best, best_epoch=ep, val_hist=hist, minutes=round((time.time() - t0) / 60, 1))
            finish_run(name, meta, net, caches); del net
        del caches, bb; torch.cuda.empty_cache()


# %% 10b. benchmark mode (A4M_BENCH=1): measures seconds/step on this machine and projects each stage, then exits
def benchmark(n=12):
    def timed(fn):
        fn(); (torch.cuda.synchronize() if AMP else None); t = time.time()
        for _ in range(n):
            fn()
        (torch.cuda.synchronize() if AMP else None); return (time.time() - t) / n
    size, bs = CFG["size"], CFG["bs"]; ntr = len(SPLITS["train"]); spe = ntr // bs
    sub = SPLITS["train"][:256]; t0 = time.time(); ensure_cache(size, sub, tag="_bench"); build_s = time.time() - t0
    ct = CacheTrain(size, bs, 0, tag="_bench"); t0 = time.time(); k = 0
    for x, y in ct.epoch(0):
        k += 1
    data_s = (time.time() - t0) / max(k, 1)
    bb = freeze(load_backbone(CFG["model_id"])); info = bb_info(bb); grid = size // info["patch"]
    heads = nn.ModuleList([lin_head(info["C"]) for _ in range(7)]).to(DEVICE); opt = torch.optim.AdamW(heads.parameters(), 1e-3)

    def stepA():
        with autocast():
            with torch.no_grad():
                o = bb(pixel_values=x, output_hidden_states=True)
            f = to_map(o.last_hidden_state, info["n_prefix"], grid).float()
            loss = sum(F.cross_entropy(F.interpolate(h(f).float(), size=y.shape[-2:], mode="bilinear"), y, ignore_index=IGNORE_INDEX) for h in heads)
        opt.zero_grad(); loss.backward(); opt.step()
    a_s = timed(stepA)
    dec = DinoSeg(CFG["model_id"]).to(DEVICE); dopt = torch.optim.AdamW([p for p in dec.parameters() if p.requires_grad], 1e-3)

    def stepB():
        with autocast():
            loss = F.cross_entropy(dec(x).float(), y, ignore_index=IGNORE_INDEX)
        dopt.zero_grad(); loss.backward(); dopt.step()
    b_s = timed(stepB); del dec
    with torch.no_grad():
        pos = tuple(p.to(DEVICE) for p in bb.rope_embeddings(torch.zeros(1, 3, size, size, device=DEVICE)))
    net = TokenNet(info, grid, pos, 4, 4).to(DEVICE); nopt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-5)
    tok = torch.randn(CFG["C_bs"], info["n_prefix"] + grid * grid, info["C"], device=DEVICE).half()

    def stepC():
        with autocast():
            lg = net(tok)
            loss = F.cross_entropy(F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear"), y[:CFG["C_bs"]], ignore_index=IGNORE_INDEX)
        nopt.zero_grad(); loss.backward(); nopt.step()
    c_s = timed(stepC)
    c_steps = sum(max(1, min(f, CFG["pool"]) // CFG["C_bs"]) * CFG["C_epochs"] for f in CFG["fracs"]) * 3     # LP, FT2, FT4 (all costed as FT4)
    print(f"\n== benchmark ({DEVICE}, bs {bs}, {size}px, {os.cpu_count()} CPUs) ==")
    print(f"image cache build: {256 / build_s:.0f} img/s -> ~{ntr / (256 / build_s) / 60:.1f} min once for {ntr} images (cold dataset reads)")
    print(f"cached batch + GPU aug: {data_s:.3f} s/batch ({bs / data_s:.0f} img/s)")
    print(f"stage A step {a_s:.3f} s -> {spe * CFG['A_epochs'] * a_s / 3600:.2f} h per seed  (x{len(CFG['seeds']['A'])} seeds)")
    print(f"stage B step {b_s:.3f} s -> {spe * CFG['decoder_epochs'] * b_s / 3600:.2f} h per seed (+val), {CFG['decoder_epochs']} epochs")
    print(f"stage C step {c_s:.3f} s -> <= {c_steps * c_s / 3600:.2f} h per seed (upper bound, + val + token caching)")
    print("stage D: minutes (cached tokens)")


if os.environ.get("A4M_BENCH"):
    benchmark(); sys.exit(0)

# %% 11. run the training stages selected for this session
for st, fn in (("A", stage_A), ("B", stage_B), ("C", stage_C), ("D", stage_D)):
    if st in CFG["stages"]:
        t0 = time.time(); print(f"\n######## stage {st} ########"); fn()
        print(f"######## stage {st} done in {(time.time() - t0) / 60:.1f} min ########")

# %% [markdown]
# ## 12. Stage E — analysis (reads results only; runs on CPU)

# %% 12. load results, summary table
import pandas as pd
import matplotlib
if "get_ipython" not in globals():                      # plain script: never open a GUI window (plt.show() would block the run)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
SEEDS = CFG["seeds"]
SETS3 = ["M1", "M2", "M3"]
PUBLISHED = {   # Mohammad, Li & Gao, iSpaRo 2025, Table I, mIoU AS PUBLISHED (resized labels, own convention)
    "DeepLabV3+ R101 @512": (.67, .72, .82), "SegFormer-B3 @512": (.67, .72, .83),
    "SegFormer-B3 @1024": (.69, .74, .93), "UPerNet Swin-S @1024": (.70, .74, .94)}


def group_confs(prefix_fn, key, sets=SETS3, r=0):
    """{set: [conf[N,4,4] per seed]} for a configuration; missing seeds are skipped (and reported)."""
    out = {k: [] for k in sets}; meta = []
    for name in prefix_fn():
        res = load_result(name)
        if res is None:
            print(f"  [missing] {name}"); continue
        meta.append(res[0])
        for k in sets:
            out[k].append(res[1][f"{k}__{key}"][:, r])
    return out, meta


def A_names(): return [f"A_layers_s{s}" for s in SEEDS["A"]]
def B_names(size): return lambda: [f"B_decoder{size}_s{s}" for s in SEEDS["B"]]
def C_names(m, n): return lambda: [f"C_{m}_n{n}_s{s}" for s in SEEDS["C"]]
def D_names(key): return lambda: [f"D_{key}_s{s}" for s in SEEDS["D"]]


def summarize(label, confs):
    row = {"config": label, "seeds": len(confs["M1"])}
    if not confs["M1"]:
        return row
    n = confs["M1"][0].shape[0]; idx = boot_idx(n, CFG["n_boot"], 0)
    for k in SETS3:
        pts = [float(miou_np(c.sum(0))) for c in confs[k]]
        lo, hi = ci95(boot_metric(confs[k], idx))
        row[f"{k} mIoU"] = f"{np.mean(pts):.3f} ± {np.std(pts):.3f} [{lo:.3f}, {hi:.3f}]"
        row[f"{k} big_rock"] = f"{np.mean([iou_np(c.sum(0))[3] for c in confs[k]]):.3f}"
    return row


GROUPS = {}
for l in CFG["layers"]:
    GROUPS[f"A: probe {l if l == 'last_norm' else 'block ' + str(l)}"] = (A_names, str(l))
for size in CFG["decoder_sizes"]:
    GROUPS[f"B: decoder @{size}"] = (B_names(size), "decoder")
for m in ("LP", "FT2", "FT4"):
    GROUPS[f"C: {m} N={CFG['fracs'][-1]}"] = (C_names(m, CFG["fracs"][-1]), "probe")
for key, *_ in CFG["backbones"]:
    GROUPS[f"D: {key} N={CFG['D_n']}"] = (D_names(key), "probe")

if "E" in CFG["stages"]:
    CONFS = {}
    for lab, (fn, key) in GROUPS.items():
        CONFS[lab], _ = group_confs(fn, key)
    TABLE = pd.DataFrame([summarize(lab, c) for lab, c in CONFS.items()]).set_index("config")
    pub = pd.DataFrame({k: {n: v[i] for n, v in PUBLISHED.items()} for i, k in enumerate(SETS3)})
    pd.set_option("display.width", 250, "display.max_colwidth", None)
    print("mIoU: mean ± sd over seeds [95% bootstrap CI over images]"); print(TABLE.to_string())
    print("\nPublished baselines (point estimates, protocol differs; see notes):"); print(pub.to_string())
    TABLE.to_csv(OUT / "table_main.csv"); pub.to_csv(OUT / "table_published.csv")

# %% 13. hypothesis tests
def verdict_str(ok, refuted=False):
    return "supported" if ok else ("refuted" if refuted else "inconclusive")


VERD = []
if "E" in CFG["stages"]:
    # --- H1: layers (best intermediate picked on VAL, never on test)
    res = [load_result(n) for n in A_names()]; res = [r for r in res if r]
    if res:
        inter = [str(l) for l in CFG["layers"] if l != "last_norm"]
        best_l = max(inter, key=lambda l: np.mean([r[0]["val_miou"][l] for r in res]))
        ev, outs = [], []
        for k in ("M1", "M2"):
            la = [r[1][f"{k}__last_norm"][:, 0] for r in res]; bi = [r[1][f"{k}__{best_l}"][:, 0] for r in res]
            d = paired(bi, la); outs.append(d)
            ev.append(f"{k}: block {best_l} - last_norm = {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        VERD.append(("H1", verdict_str(all(o["real"] and o["delta"] > 0 for o in outs),
                                       all(o["real"] and o["delta"] < 0 for o in outs)), "; ".join(ev)))
    # --- H2: FT - LP vs N
    ev, ok_all, d_by = [], [], {}
    for k in ("M1", "M2"):
        ds = {}
        for n in CFG["fracs"]:
            ft = group_confs(C_names("FT2", n), "probe", sets=[k])[0][k]; lp = group_confs(C_names("LP", n), "probe", sets=[k])[0][k]
            if ft and lp:
                ds[n] = paired(ft, lp)
        d_by[k] = ds
        if len(ds) >= 2:
            n0, n1 = min(ds), max(ds)
            ok_all.append(ds[n0]["delta"] < TOL and ds[n1]["real"] and ds[n1]["delta"] >= TOL and ds[n1]["delta"] - ds[n0]["delta"] >= TOL)
            ev.append(f"{k}: FT2-LP " + ", ".join(f"N={n}: {d['delta']:+.3f}" for n, d in sorted(ds.items())))
    if ev:
        VERD.append(("H2", verdict_str(all(ok_all)), "; ".join(ev)))
    # --- H3: backbones
    keys = [b[0] for b in CFG["backbones"]][:3]            # DINOv3, DINOv2, random init; later entries are extra runs
    dc = {k: {s: group_confs(D_names(k), "probe", sets=[s])[0][s] for s in ("M1", "M2")} for k in keys}
    if all(dc[k]["M1"] for k in keys):
        v3, v2, rd = keys[0], keys[1], keys[2]; ev, oks = [], []
        for s in ("M1", "M2"):
            d32, d3r, d2r = paired(dc[v3][s], dc[v2][s]), paired(dc[v3][s], dc[rd][s]), paired(dc[v2][s], dc[rd][s])
            oks.append(d32["real"] and d32["delta"] >= TOL and d2r["delta"] > 0.1 and d3r["delta"] > 0.1)
            ev.append(f"{s}: v3-v2 {d32['delta']:+.3f} [{d32['lo']:+.3f},{d32['hi']:+.3f}], v3-rand {d3r['delta']:+.3f}, v2-rand {d2r['delta']:+.3f}")
        VERD.append(("H3", verdict_str(all(oks)), "; ".join(ev) + " (patch 14 vs 16 is a confound)"))
        # --- H3b (extra run, registered in hypotheses_extra.json): same comparison with DINOv2 at 448 px (grid 32x32)
        k448 = "dinov2_b14_448"
        if k448 in [b[0] for b in CFG["backbones"]]:
            d448 = {s: group_confs(D_names(k448), "probe", sets=[s])[0][s] for s in ("M1", "M2")}
            if d448["M1"]:
                ev, oks, inside = [], [], []
                for s in ("M1", "M2"):
                    d = paired(dc[v3][s], d448[s]); oks.append(d["real"]); inside.append(d["lo"] > -TOL and d["hi"] < TOL)
                    ev.append(f"{s}: v3-v2@448 {d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
                VERD.append(("H3b", verdict_str(all(oks), all(inside)), "; ".join(ev) + " (grid matched at 32x32; input pixels 448 vs 512 still differ)"))
    # --- H4: decoder vs linear probe at full data
    dec = {s: group_confs(B_names(CFG["decoder_sizes"][0]), "decoder", sets=[s])[0][s] for s in SETS3}
    lpf = {s: [load_result(n)[1][f"{s}__last_norm"][:, 0] for n in A_names() if load_result(n)] for s in SETS3}
    if dec["M1"] and lpf["M1"]:
        ev, oks = [], []
        for s in SETS3:
            d = paired(dec[s], lpf[s]); ev.append(f"{s}: decoder-LP {d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")
            if s != "M3":
                oks.append(not (d["real"] and d["delta"] > 0))
        VERD.append(("H4", verdict_str(all(oks), not all(oks)), "; ".join(ev)))
    # --- H5: border band
    cfgs = {"A last_norm": ([load_result(n) for n in A_names()], "last_norm"),
            "B decoder": ([load_result(n) for n in B_names(CFG["decoder_sizes"][0])()], "decoder")}
    band = {}
    for lab, (rs, key) in cfgs.items():
        rs = [r for r in rs if r]
        if rs:
            band[lab] = [float(np.mean([miou_np(r[1][f"M1__{key}"][:, i].sum(0)) for r in rs])) for i in range(len(CFG["radii"]))]
    if band:
        i8 = list(CFG["radii"]).index(8) if 8 in CFG["radii"] else len(CFG["radii"]) - 1
        gains = {k: v[i8] - v[0] for k, v in band.items()}
        spread0, spread8 = np.ptp([v[0] for v in band.values()]), np.ptp([v[i8] for v in band.values()])
        VERD.append(("H5", verdict_str(all(g >= 0.03 for g in gains.values()) and spread8 < spread0),
                     f"M1 mIoU r=0 -> r={CFG['radii'][i8]}: " + ", ".join(f"{k} {v[0]:.3f}->{v[i8]:.3f}" for k, v in band.items()) +
                     f"; spread across configs {spread0:.3f} -> {spread8:.3f}"))
    # --- H6: big_rock recall by component size
    rows = []
    for r in [load_result(n) for n in A_names()]:
        if r:
            for k in ("M1", "M2"):
                rows += r[0]["comps"].get(f"{k}__last_norm", [])
    if len(rows) >= 6:
        a = np.array([x[0] for x in rows]); rc = np.array([x[1] for x in rows])
        lo_t, hi_t = np.percentile(a, [100 / 3, 200 / 3])
        small, large = rc[a <= lo_t].mean(), rc[a >= hi_t].mean()
        VERD.append(("H6", verdict_str(large - small >= 0.2),
                     f"recall smallest third {small:.2f} vs largest third {large:.2f} (area cut-offs {lo_t:.0f}/{hi_t:.0f} px, {len(rows)} components)"))
    VT = pd.DataFrame(VERD, columns=["hypothesis", "verdict", "evidence"]).set_index("hypothesis")
    pd.set_option("display.max_colwidth", None); print(VT.to_string()); VT.to_csv(OUT / "hypothesis_verdicts.csv")

# %% 14. figures
if "E" in CFG["stages"]:
    def style():
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
    style()
    # (1) layer-wise
    res = [load_result(n) for n in A_names()]; res = [r for r in res if r]
    if res:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True); L = [str(l) for l in CFG["layers"]]
        for a, k in zip(ax, SETS3):
            m = np.array([[miou_np(r[1][f"{k}__{l}"][:, 0].sum(0)) for l in L] for r in res])
            a.errorbar(range(len(L)), m.mean(0), m.std(0), marker="o", capsize=3); a.set_xticks(range(len(L))); a.set_xticklabels(L)
            a.set_title(f"{k}: mIoU per layer (mean ± sd over {len(res)} seeds)"); a.set_xlabel("layer")
        ax[0].set_ylabel("mIoU"); plt.tight_layout(); plt.savefig(FIG / "fig_layers.png"); plt.show()
    # (2) label efficiency
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, k in zip(ax, ("M1", "M2")):
        for m, c in (("LP", "C0"), ("FT2", "C1"), ("FT4", "C3")):
            xs, mu, sd = [], [], []
            for n in CFG["fracs"]:
                cf = group_confs(C_names(m, n), "probe", sets=[k])[0][k]
                if cf:
                    v = [float(miou_np(c_.sum(0))) for c_ in cf]; xs.append(n); mu.append(np.mean(v)); sd.append(np.std(v))
            if xs:
                a.errorbar(xs, mu, sd, marker="o", capsize=3, label=m, color=c)
        a.set_xscale("log"); a.set_xlabel("labelled training images"); a.set_ylabel("mIoU"); a.set_title(f"{k}: label efficiency"); a.legend()
    plt.tight_layout(); plt.savefig(FIG / "fig_label_efficiency.png"); plt.show()
    # (3) border band
    if band:
        fig, a = plt.subplots(figsize=(5, 3.6))
        for lab, v in band.items():
            a.plot(CFG["radii"], v, marker="o", label=lab)
        a.set_xlabel("band excluded around label edges (px @ native res)"); a.set_ylabel("M1 mIoU"); a.legend()
        plt.tight_layout(); plt.savefig(FIG / "fig_border_band.png"); plt.show()
    # (4) big_rock recall by size
    if len(rows) >= 6:
        areas = np.array([x[0] for x in rows]); rcl = np.array([x[1] for x in rows])
        edges = np.unique(np.percentile(areas, [0, 25, 50, 75, 100])); bins = np.digitize(areas, edges[1:-1])
        fig, ax4 = plt.subplots(figsize=(5, 3.6))
        ax4.bar(range(len(edges) - 1), [rcl[bins == i].mean() if (bins == i).any() else 0 for i in range(len(edges) - 1)])
        ax4.set_xticks(range(len(edges) - 1)); ax4.set_xticklabels([f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(edges) - 1)], rotation=20)
        ax4.set_xlabel("big_rock component area (px)"); ax4.set_ylabel("recall"); ax4.set_title("big_rock found vs boulder size")
        plt.tight_layout(); plt.savefig(FIG / "fig_big_rock_size.png"); plt.show()

# %% 15. package outputs
if not JOB:
  keep = OUT / "keep"; keep.mkdir(exist_ok=True)
  for p in list(RES_DIR.glob("*.json")) + list(RES_DIR.glob("*.npz")) + list(OUT.glob("*.csv")) + list(FIG.glob("*.png")) + [hp]:
      if p.exists():
          shutil.copy(p, keep)
  shutil.make_archive(str(OUT / "ai4mars_results"), "zip", keep)
  print("packaged", len(list(keep.iterdir())), "files ->", OUT / "ai4mars_results.zip")
