"""Synthetic stand-in for the VGGT frontend, for testing the CPU pipeline without a GPU.

Writes the same cache format as mr_gpu/run_vggt.py, from a REAL KITTI ground-truth trajectory:
  * scene: ground plane 1.65 m below the camera, side walls, far plane (sky = very far, low conf);
  * every window has its own random Sim(3) gauge (arbitrary rotation, translation, scale);
  * window "difficulty" d_k ~ logNormal; inside the window, frame f's geometry is distorted by
    a smooth Sim(3) about the window centre, Exp(xi_k * tau_f) with tau_f in [-1/2, 1/2] and
    |xi_k| proportional to d_k. Shared frames are distorted differently by consecutive windows,
    which is what creates seam errors;
  * pixel noise sigma_rel = 0.01 * sqrt(d_k) * u_p, and confidence C = 1 + 3 (0.01/sigma_rel)^1.6 * v_p,
    so E[error | C] ~ C^-0.63: confidence ranks errors but is NOT the loss-optimal alpha/C;
  * global descriptors are random Fourier features of the true position (revisits look alike);
  * reference depth samples ("LiDAR") on every 5th frame for the depth audit.
Everything here is labelled synthetic; it validates code paths and recoverable structure only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import kitti, lie
from .cache import find_loop_pairs, loop_windows, save_npz, window_ranges

CAM_HEIGHT, SKY_DEPTH = 1.65, 120.0


def _intrinsics(calib, orig_hw, model_hw):
    (oh, ow), (mh, mw) = orig_hw, model_hw
    k = calib["P2"][:, :3].copy()
    sx, sy = mw / ow, mh / oh
    return np.array([[k[0, 0] * sx, 0, (k[0, 2] + .5) * sx - .5],
                     [0, k[1, 1] * sy, (k[1, 2] + .5) * sy - .5], [0, 0, 1]])


def _scene_depth(rays, wall_l, wall_r, wall_top=6.0, far=60.0):
    """rays (h,w,3) with z=1 (camera y points down). Optical-axis depth of ground plane, two side
    walls of finite height, an end wall at `far` below the same height, else sky."""
    sky = np.full(rays.shape[:2], SKY_DEPTH)
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = np.where(rays[..., 1] > 1e-6, CAM_HEIGHT / rays[..., 1], np.inf)
        left = np.where(rays[..., 0] < -1e-6, -wall_l / rays[..., 0], np.inf)
        right = np.where(rays[..., 0] > 1e-6, wall_r / rays[..., 0], np.inf)
    walls = np.minimum(left, right)
    walls = np.where(rays[..., 1] * walls > -wall_top, walls, np.inf)   # above the roofline -> miss
    end = np.where(rays[..., 1] * far > -wall_top, far, np.inf)
    return np.minimum.reduce([sky, ground, walls, end])


def _sim3_about(centre, xi):
    """Sim(3) Exp(xi) applied about a world point (rotation/scale centred there)."""
    t_c = np.eye(4)
    t_c[:3, 3] = centre
    t_ci = np.eye(4)
    t_ci[:3, 3] = -centre
    return t_c @ lie.exp(xi[None])[0] @ t_ci


def _predict_frames(rng, gt, frames, centres, xis, gauge, k_model, grid, difficulty):
    """Points/poses of the given frames as one window would predict them."""
    rr, cc = grid
    pix = np.stack([cc, rr, np.ones_like(cc)], -1).astype(float)
    rays = pix @ np.linalg.inv(k_model).T
    g_inv = lie.inv(gauge[None])[0]
    out = {k: [] for k in ("c2w", "points", "points_conf", "depth", "depth_conf")}
    for n, f in enumerate(frames):
        walls = 6 + 3 * np.sin(np.array([f / 37., f / 53.])) ** 2
        z = _scene_depth(rays, *walls)
        cam_pts = rays * z[..., None]
        sky = z >= SKY_DEPTH
        u = np.exp(rng.normal(0, .5, z.shape))
        rel = np.where(sky, .3, .01 * np.sqrt(difficulty) * u)
        conf = np.where(sky, 1.02, 1 + 3 * (.01 / rel) ** 1.6 * np.exp(rng.normal(0, .3, z.shape)))
        noisy = cam_pts * (1 + rel * rng.normal(size=z.shape))[..., None]
        d = _sim3_about(centres[n], xis[n])
        world = (gt[f][:3, :3] @ noisy.reshape(-1, 3).T).T + gt[f][:3, 3]
        world = (d[:3, :3] @ world.T).T + d[:3, 3]
        win = (g_inv[:3, :3] @ world.T).T + g_inv[:3, 3]
        pose_world = d @ gt[f]
        s, r, t = lie.split(pose_world[None])
        c2w_world = lie.from_srt(1., r[0], t[0])[0]
        pose_win = g_inv @ c2w_world
        s2, r2, t2 = lie.split(pose_win[None])
        c2w = lie.from_srt(1., r2[0], t2[0])[0]
        pts = win.reshape(z.shape + (3,))
        depth = np.einsum("hwk,k->hw", pts - c2w[:3, 3], c2w[:3, 2]) * np.exp(rng.normal(0, .005, z.shape))
        out["c2w"].append(c2w)
        out["points"].append(pts.astype(np.float32))
        out["points_conf"].append(conf.astype(np.float16))
        out["depth"].append(depth.astype(np.float16))
        out["depth_conf"].append((conf * np.exp(rng.normal(0, .1, z.shape))).astype(np.float16))
    return {k: np.stack(v) for k, v in out.items()}


def _vggt_gauge(rng, first_pose):
    """Window -> world Sim(3) as VGGT defines it: the window frame is the first camera's frame
    (VGGT predicts frame 0 at the identity pose), and the scene is normalised so typical point
    distances are O(1). ~15 m per unit reproduces that; it matters because VGGT-Long's Huber
    threshold (delta = 0.1) is in window units, and because the origin sets the lever arm through
    which rotation errors appear as translation errors."""
    s, r, t = lie.split(first_pose[None])
    return lie.from_srt(15. * np.exp(rng.normal(0, .3)), r[0], t[0])[0]


def make_cache(kitti_root, seq, out, window=32, overlap=16, stride=4, seed=0, n_frames=None,
               loop_threshold=0.85, log=print):
    rng = np.random.default_rng(seed)
    gt = kitti.gt_cam2_poses(kitti_root, seq)
    if n_frames:
        gt = gt[:n_frames]
    calib = kitti.read_calib(Path(kitti_root) / "sequences" / seq / "calib.txt")
    orig_hw = (370, 1226) if int(seq) >= 4 else ((375, 1242) if int(seq) == 3 else (376, 1241))
    model_hw = kitti.model_input_size(*orig_hw)
    k_model = _intrinsics(calib, orig_hw, model_hw)
    rr, cc = np.meshgrid(np.arange(0, model_hw[0], stride), np.arange(0, model_hw[1], stride), indexing="ij")
    out = Path(out)
    ranges = window_ranges(len(gt), window, overlap)
    diffs = np.exp(rng.normal(0, .6, len(ranges)))
    for k, (a, b) in enumerate(ranges):
        frames = np.arange(a, b)
        tau = (frames - (a + b - 1) / 2) / max(window, 1)
        xi_k = rng.normal(0, 1, 7) * np.r_[[.004] * 3, [.02] * 3, .015] * diffs[k]
        centre = gt[frames, :3, 3].mean(0)
        xis = tau[:, None] * xi_k + rng.normal(0, 1, (len(frames), 7)) * np.r_[[2e-4] * 3, [.01] * 3, 1e-3]
        gauge = _vggt_gauge(rng, _sim3_about(centre, xis[0]) @ gt[frames[0]])
        pred = _predict_frames(rng, gt, frames, [centre] * len(frames), xis, gauge, k_model, (rr, cc), diffs[k])
        save_npz(out / "windows" / f"w{k:04d}.npz", frame_ids=frames, intrinsics=np.tile(k_model, (len(frames), 1, 1)),
                 stride=np.array(stride), **pred)
        if log and k % 50 == 0:
            log(f"    synthetic window {k}/{len(ranges) - 1}")
    # descriptors: random Fourier features of true position (+ noise)
    omega = rng.normal(0, 1 / 4., (128, 3))
    phase = rng.uniform(0, 2 * np.pi, 128)
    desc = np.cos(gt[:, :3, 3] @ omega.T + phase) + rng.normal(0, .15, (len(gt), 128))
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    np.save(out / "descriptors.npy", desc.astype(np.float16))
    pairs = find_loop_pairs(desc, threshold=loop_threshold)
    loops = loop_windows(ranges, pairs, half_window=10)
    for n, lw in enumerate(loops):
        fa, fb = np.arange(*lw["range_a"]), np.arange(*lw["range_b"])
        d = np.exp(rng.normal(.3, .6))
        parts = []
        xi_a = rng.normal(0, 1, 7) * np.r_[[.004] * 3, [.02] * 3, .015] * d
        gauge = _vggt_gauge(rng, _sim3_about(gt[fa, :3, 3].mean(0), xi_a) @ gt[fa[0]])
        for frames in (fa, fb):
            xi = xi_a if frames is fa else rng.normal(0, 1, 7) * np.r_[[.004] * 3, [.02] * 3, .015] * d
            parts.append(_predict_frames(rng, gt, frames, [gt[frames, :3, 3].mean(0)] * len(frames),
                                         [xi] * len(frames), gauge, k_model, (rr, cc), d))
        frames = np.r_[fa, fb]
        save_npz(out / "loops" / f"l{n:04d}.npz", frame_ids=frames, intrinsics=np.tile(k_model, (len(frames), 1, 1)),
                 stride=np.array(stride), **{k: np.concatenate([parts[0][k], parts[1][k]]) for k in parts[0]})
    # reference depth samples on every 5th frame
    gt_dir = out / "gt_depth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pix = np.stack([cc, rr, np.ones_like(cc)], -1).astype(float)
    rays = pix @ np.linalg.inv(k_model).T
    for f in range(0, len(gt), 5):
        z = _scene_depth(rays, *(6 + 3 * np.sin(np.array([f / 37., f / 53.])) ** 2))
        r_i, c_i = np.nonzero(z < 80)
        pick = rng.choice(len(r_i), min(1500, len(r_i)), replace=False)
        np.savez(gt_dir / f"f{f:06d}.npz", rows=r_i[pick], cols=c_i[pick], depth=z[r_i[pick], c_i[pick]])
    manifest = {"engine": "synthetic", "sequence": seq, "n_frames": len(gt), "window": window, "overlap": overlap,
                "stride": stride, "model_hw": list(model_hw), "orig_hw": list(orig_hw), "seed": seed,
                "windows": [list(r) for r in ranges], "loop_windows": loops, "loop_pairs": [list(p) for p in pairs],
                "window_difficulty": diffs.tolist()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kitti", default="data/kitti")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = make_cache(a.kitti, a.seq, a.out, a.window, a.overlap, a.stride, a.seed, a.frames)
    print(f"wrote {len(m['windows'])} windows and {len(m['loop_windows'])} loop windows to {a.out}")


if __name__ == "__main__":
    main()
