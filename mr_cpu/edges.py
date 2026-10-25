"""Build the pose-graph edge table (seams + loops) from a frontend cache.

Seam k joins windows k and k+1 using their shared frames (VGGT-Long: last `overlap` frames of k,
first `overlap` frames of k+1). Its measurement Z_{k,k+1} maps window k+1 into window k.

A loop window holds frames range_a (inside window a, the later visit) followed by range_b
(inside window b, the earlier visit). As in VGGT-Long we align the loop window to a and to b:
  T_aL maps loop -> a,  T_bL maps loop -> b,  so Z_ba = T_bL @ inv(T_aL) maps a -> b,
and add the edge (i=b, j=a). Its features combine both alignments conservatively.
"""
from __future__ import annotations

import numpy as np

from . import lie
from .alignment import align_windows
from .cache import Cache

FEATURES = ("n_points", "valid_fraction", "conf_mean", "conf_median", "conf_logmean", "conf_p10",
            "inlier_fraction", "residual_rms", "residual_median", "char_length", "residual_rel",
            "spread_condition", "rotation_deg", "abs_log_scale", "translation_rel")


def _subset(win, frames):
    pos = {f: n for n, f in enumerate(win["frame_ids"].tolist())}
    idx = [pos[f] for f in frames]
    return {k: win[k][idx] for k in ("points", "points_conf", "c2w")}


def _combine(fa, fb):
    """Conservative merge of the two loop alignments' features."""
    out = {}
    for k in FEATURES:
        if k in ("conf_mean", "conf_median", "conf_logmean", "conf_p10", "valid_fraction", "inlier_fraction",
                 "n_points", "spread_condition"):
            out[k] = min(fa[k], fb[k])
        elif k in ("residual_rms", "residual_median", "residual_rel", "rotation_deg", "abs_log_scale", "translation_rel"):
            out[k] = max(fa[k], fb[k])
        else:
            out[k] = fa[k]
    return out


def build(cache_dir, delta=0.1, max_iters=5, rel_threshold=0.1, max_points=None, log=print):
    cache = Cache(cache_dir)
    ranges = cache.ranges
    rows, prev = [], cache.window(0)
    for k in range(len(ranges) - 1):
        nxt = cache.window(k + 1)
        shared = sorted(set(prev["frame_ids"].tolist()) & set(nxt["frame_ids"].tolist()))
        a, b = _subset(prev, shared), _subset(nxt, shared)
        z, f = align_windows(a["points"], a["points_conf"], b["points"], b["points_conf"], b["c2w"],
                             delta, max_iters, rel_threshold, max_points, seed=k)
        f.update(kind=0, i=k, j=k + 1, overlap=len(shared))
        rows.append((z, f))
        prev = nxt
        if log and (k % 50 == 0 or k == len(ranges) - 2):
            log(f"    seam {k}/{len(ranges) - 2}: scale {np.cbrt(np.linalg.det(z[:3, :3])):.3f}, "
                f"conf {f['conf_mean']:.2f}, inliers {f['inlier_fraction']:.2f}")
    for n, meta in enumerate(cache.loop_meta):
        lw = cache.loop(n)
        ca, cb = meta["chunk_a"], meta["chunk_b"]
        ra, rb = range(*meta["range_a"]), range(*meta["range_b"])
        # positional split (the two ranges may share frame ids when the visits are close in time)
        na = len(ra)
        if lw["frame_ids"][:na].tolist() != list(ra) or lw["frame_ids"][na:].tolist() != list(rb):
            raise ValueError(f"Loop window {n} frame ids do not match its manifest ranges")
        wa, wb = _subset(cache.window(ca), list(ra)), _subset(cache.window(cb), list(rb))
        t_al, fa = align_windows(wa["points"], wa["points_conf"], lw["points"][:na], lw["points_conf"][:na],
                                 lw["c2w"][:na], delta, max_iters, rel_threshold, max_points, seed=10_000 + n)
        t_bl, fb = align_windows(wb["points"], wb["points_conf"], lw["points"][na:], lw["points_conf"][na:],
                                 lw["c2w"][na:], delta, max_iters, rel_threshold, max_points, seed=20_000 + n)
        z = t_bl @ lie.inv(t_al[None])[0]
        f = _combine(fa, fb)
        f["conf_hist"], f["conf_hist2"] = fa["conf_hist"], fb["conf_hist"]
        # characteristic length of the edge residual must be in window-a (= node j) units
        f["char_length"] = float(np.median(np.abs(_depths(wa))))
        f.update(kind=1, i=cb, j=ca, overlap=len(ra) + len(rb), similarity=meta.get("similarity", np.nan))
        rows.append((z, f))
    table = {"z": np.stack([r[0] for r in rows]),
             "i": np.array([r[1]["i"] for r in rows]), "j": np.array([r[1]["j"] for r in rows]),
             "kind": np.array([r[1]["kind"] for r in rows]), "overlap": np.array([r[1]["overlap"] for r in rows]),
             "similarity": np.array([r[1].get("similarity", np.nan) for r in rows])}
    for key in FEATURES:
        table["f_" + key] = np.array([r[1][key] for r in rows], float)
    table["conf_hist"] = np.stack([r[1]["conf_hist"] for r in rows])
    table["conf_hist2"] = np.stack([r[1].get("conf_hist2", np.zeros_like(r[1]["conf_hist"])) for r in rows])
    frames, poses = [], []
    for k in range(len(ranges)):
        fid, c2w = cache.window_cameras(k)
        frames.append(fid)
        poses.append(c2w)
    return table, frames, poses


def _depths(w):
    rel = w["points"] - w["c2w"][:, None, None, :3, 3]
    d = np.einsum("nhwk,nk->nhw", rel, w["c2w"][:, :3, 2])
    c = w["points_conf"]
    keep = c > np.median(c) * .1
    return d[keep & (d > 0)]


def save(path, table, frames, poses):
    arrays = dict(table)
    arrays["window_lengths"] = np.array([len(f) for f in frames])
    arrays["window_frames"] = np.concatenate(frames)
    arrays["window_c2w"] = np.concatenate(poses)
    np.savez(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    cuts = np.cumsum(data.pop("window_lengths"))[:-1]
    frames = np.split(data.pop("window_frames"), cuts)
    poses = np.split(data.pop("window_c2w"), cuts)
    return data, frames, poses
