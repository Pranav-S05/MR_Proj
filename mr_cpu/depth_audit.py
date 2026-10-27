"""Pixel-level confidence audit against LiDAR depth (both VGGT heads).

For each window and each of its frames that has a LiDAR scan, LiDAR depth (camera 2 optical axis,
on the cached model grid) is compared with
  * the depth head: depth, depth_conf;
  * the point head: optical-axis depth of `points` in the window's predicted camera, points_conf.
A monocular window has no metric scale, so each head gets one scale per window,
s_k = median(lidar / prediction) over the window's LiDAR pixels; the relative error is
|s_k * prediction - lidar| / lidar. This is a scale-aligned shape audit (the scale uses the same
pixels it scores), not a metric-depth evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .alignment import camera_depths
from .cache import Cache


def collect(cache_dir, lidar_fn, max_per_frame=None, seed=0, log=print):
    """lidar_fn(frame_id, stride) -> (rows, cols, depth) or None if no scan for that frame.
    max_per_frame=None keeps every LiDAR pixel; an integer randomly subsamples each frame."""
    cache = Cache(cache_dir)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("window", "frame", "lidar", "d_conf", "d_err", "p_conf", "p_err")}
    for k in range(len(cache.ranges)):
        w = cache.window(k)
        stride = int(w["stride"]) if "stride" in w else 1
        samples = []
        for n, f in enumerate(w["frame_ids"].tolist()):
            hit = lidar_fn(f, stride)
            if hit is None or len(hit[2]) == 0:
                continue
            r, c, d = hit
            pdepth = camera_depths(w["points"][n:n + 1].astype(np.float64), w["c2w"][n:n + 1])[0][r, c]
            samples.append((f, d, w["depth"][n][r, c].astype(float), w["depth_conf"][n][r, c].astype(float),
                            pdepth, w["points_conf"][n][r, c].astype(float)))
        if not samples:
            continue
        lid = np.concatenate([s[1] for s in samples])
        dpred = np.concatenate([s[2] for s in samples])
        ppred = np.concatenate([s[4] for s in samples])
        okd, okp = dpred > 0, ppred > 0
        sd = np.median(lid[okd] / dpred[okd]) if okd.any() else np.nan
        sp = np.median(lid[okp] / ppred[okp]) if okp.any() else np.nan
        for f, d, dp, dc, pp, pc in samples:
            idx = np.arange(len(d))
            if max_per_frame and len(idx) > max_per_frame:
                idx = rng.choice(len(idx), max_per_frame, replace=False)
            out["window"].append(np.full(len(idx), k))
            out["frame"].append(np.full(len(idx), f))
            out["lidar"].append(d[idx])
            out["d_conf"].append(dc[idx])
            out["d_err"].append(np.abs(sd * dp[idx] - d[idx]) / d[idx])
            out["p_conf"].append(pc[idx])
            out["p_err"].append(np.where(pp[idx] > 0, np.abs(sp * pp[idx] - d[idx]) / d[idx], np.nan))
        if log and k % 50 == 0:
            log(f"    depth audit window {k}/{len(cache.ranges) - 1}")
    # lossless compact storage (hundreds of millions of pixels uncapped): confidences are float16 in the cache
    small = {"window": np.int32, "frame": np.int32, "d_conf": np.float16, "p_conf": np.float16}
    return {k: np.concatenate(v).astype(small.get(k, np.float32)) if v else np.array([]) for k, v in out.items()}


def kitti_lidar_fn(kitti_root, seq, model_hw, orig_hw):
    """LiDAR provider for a KITTI sequence (only frames whose .bin was downloaded).
    orig_hw comes from the cache manifest, so the images themselves are not needed here."""
    from .kitti import lidar_depth_on_grid, read_calib
    calib = read_calib(Path(kitti_root) / "sequences" / seq / "calib.txt")
    orig = tuple(orig_hw)
    velo = Path(kitti_root) / "sequences" / seq / "velodyne"

    def fn(frame, stride):
        path = velo / f"{frame:06d}.bin"
        if not path.exists():
            return None
        return lidar_depth_on_grid(path, calib, tuple(model_hw), orig, stride)
    return fn


def cache_lidar_fn(cache_dir):
    """Synthetic caches store reference depth samples in gt_depth/fNNNNNN.npz (rows, cols, depth)."""
    root = Path(cache_dir) / "gt_depth"

    def fn(frame, stride):
        path = root / f"f{frame:06d}.npz"
        if not path.exists():
            return None
        with np.load(path) as d:
            return d["rows"], d["cols"], d["depth"]
    return fn
