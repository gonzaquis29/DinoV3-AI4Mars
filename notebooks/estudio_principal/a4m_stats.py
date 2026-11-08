"""a4m_stats — the statistics of the main study, shared by the training notebook (E) and the analysis notebooks (C, F).

Paired block bootstrap: test images are sorted by spacecraft clock and neighbours look alike (same place, same light), so we resample
BLOCKS of 10 consecutive images, not single images; every configuration is scored on the same resample (paired) and seeds are averaged.
A difference is
  real        95% CI excludes 0 and |delta| >= tol
  equivalent  the whole 95% CI lies inside (-tol, +tol)
  inconclusive otherwise
`bound` = max(|lo|, |hi|): the largest effect still compatible with the data ("the effect is at most X").
Holm's correction is applied inside each family of comparisons.
"""
import numpy as np
from a4m_lib import miou_np, iou_np

TOL, NB, BLOCK = 0.01, 2000, 10
br_iou = lambda c: iou_np(c)[..., 3]                                   # IoU of big_rock (class 3)
r0 = lambda arrs: [a[:, 0] if a.ndim == 4 else a for a in arrs]        # radius 0 (no border band excluded)


def block_idx(n, B=NB, L=BLOCK, seed=0):
    """B resamples of n images made of blocks of L consecutive images."""
    L = min(L, n); rng = np.random.default_rng(seed); nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, (B, nb))
    return (starts[..., None] + np.arange(L)).reshape(B, -1)[:, :n]


def metric_boot(confs, idx, metric=miou_np):
    """confs: list over seeds of per-image confusions [N,4,4] -> metric averaged over seeds, for each resample [B]."""
    return np.mean([metric(c[idx].sum(1)) for c in confs], 0)


def metric_point(confs, metric=miou_np):
    return float(np.mean([metric(c.sum(0)) for c in confs]))


def label_of(delta, lo, hi, tol):
    return "real" if (lo > 0 or hi < 0) and abs(delta) >= tol else "equivalente" if (lo > -tol and hi < tol) else "inconcluso"


def contrast(pos, neg, metric=miou_np, tol=TOL, seed=0):
    """pos, neg: lists of configurations (each = list over seeds of [N,4,4]). delta = mean(pos) - mean(neg), paired block bootstrap."""
    n = pos[0][0].shape[0]; idx = block_idx(n, seed=seed)
    d = np.mean([metric_boot(g, idx, metric) for g in pos], 0) - np.mean([metric_boot(g, idx, metric) for g in neg], 0)
    delta = np.mean([metric_point(g, metric) for g in pos]) - np.mean([metric_point(g, metric) for g in neg])
    lo, hi = np.percentile(d, [2.5, 97.5]); p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return dict(delta=delta, lo=lo, hi=hi, p=p, tol=tol, label=label_of(delta, lo, hi, tol), bound=max(abs(lo), abs(hi)), half_ci=(hi - lo) / 2)


def single(confs, metric=miou_np):
    """Point estimate and 95% block-bootstrap CI of one configuration."""
    idx = block_idx(confs[0].shape[0]); d = metric_boot(confs, idx, metric); lo, hi = np.percentile(d, [2.5, 97.5])
    return metric_point(confs, metric), float(lo), float(hi)


def holm(rows, alpha=0.05):
    order = np.argsort([r["p"] for r in rows]); m = len(rows); prev = 0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (m - k) * rows[i]["p"])); rows[i]["p_holm"] = prev
        if rows[i]["label"] == "real" and prev >= alpha:
            rows[i]["label"] = "inconcluso"
    return rows


fmt = lambda r: f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {r['label']}"
