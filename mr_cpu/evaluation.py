"""Trajectory assembly and metrics.

assemble(): per-frame camera-to-world poses from window poses S_k and in-window cameras. Like
VGGT-Long's save_camera_poses, a frame seen by several windows takes the LAST window's estimate.
ate(): whole-sequence position ATE after ONE global Sim(3) (Umeyama) alignment, the monocular
protocol of VGGT-Long and evo's `evo_ape -as` (checked against evo in tests).
kitti_relative_errors(): port of the KITTI odometry devkit (segments 100..800 m, every 10th start
frame) applied after the same global Sim(3) alignment, reported as t_rel [%] and r_rel [deg/100 m].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import fit_similarity, transform_camera_poses, transform_points


def assemble(window_poses, window_frames, window_c2w, n_frames):
    out = np.full((n_frames, 4, 4), np.nan)
    for s, frames, c2w in zip(window_poses, window_frames, window_c2w):
        out[frames] = transform_camera_poses(s, c2w)
    if np.isnan(out).any():
        raise ValueError("Some frames are not covered by any window")
    return out


def sim3_align(est_xyz, ref_xyz):
    return fit_similarity(np.asarray(est_xyz), np.asarray(ref_xyz))


def ate(est_xyz, ref_xyz):
    t = sim3_align(est_xyz, ref_xyz)
    err = np.linalg.norm(transform_points(t, est_xyz) - ref_xyz, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def ate_vs_length(est_xyz, ref_xyz, lengths):
    """Sim(3)-aligned ATE of each prefix [0, L) separately (drift growth with sequence length)."""
    out = []
    for n in lengths:
        n = int(min(n, len(est_xyz)))
        if n >= 3:
            out.append((n, ate(est_xyz[:n], ref_xyz[:n])))
    return out


def _distances(poses):
    step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.r_[0., np.cumsum(step)]


def kitti_relative_errors(est, gt, lengths=(100, 200, 300, 400, 500, 600, 700, 800), step=10):
    """KITTI devkit segment errors after a global Sim(3) alignment of the estimate."""
    align = sim3_align(est[:, :3, 3], gt[:, :3, 3])
    est = transform_camera_poses(align, est)
    dist = _distances(gt)
    t_err, r_err = [], []
    for first in range(0, len(gt), step):
        for length in lengths:
            last = np.searchsorted(dist, dist[first] + length, side="right")
            if last >= len(gt):
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            e = np.linalg.inv(d_est) @ d_gt
            angle = np.arccos(np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1))
            t_err.append(np.linalg.norm(e[:3, 3]) / length)
            r_err.append(angle / length)
    if not t_err:
        return {"t_rel_percent": float("nan"), "r_rel_deg_per_100m": float("nan"), "segments": 0}
    return {"t_rel_percent": float(100 * np.mean(t_err)),
            "r_rel_deg_per_100m": float(100 * np.degrees(np.mean(r_err))), "segments": len(t_err)}


def write_tum(path, poses, timestamps=None):
    """TUM format: t tx ty tz qx qy qz qw (rotations must be unit)."""
    ts = np.arange(len(poses), dtype=float) if timestamps is None else np.asarray(timestamps, float)
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()  # x, y, z, w
    rows = np.c_[ts, poses[:, :3, 3], q]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.9f")


def evaluate(est, gt, prefix_lengths=None):
    xyz, ref = est[:, :3, 3], gt[:, :3, 3]
    out = {"frames": len(est), "ate_rmse_m": ate(xyz, ref)}
    out.update(kitti_relative_errors(est, gt))
    if prefix_lengths is not None:
        out["ate_vs_length"] = ate_vs_length(xyz, ref, prefix_lengths)
    return out


def evo_ate(est, gt):
    """Cross-check with evo's `evo_ape tum --align --correct_scale` (needs evo installed)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_tum(Path(d) / "gt.txt", gt)
        write_tum(Path(d) / "est.txt", est)
        res = subprocess.run(["evo_ape", "tum", str(Path(d) / "gt.txt"), str(Path(d) / "est.txt"), "--align",
                              "--correct_scale", "--save_results", str(Path(d) / "r.zip")], capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr)
        import zipfile
        with zipfile.ZipFile(Path(d) / "r.zip") as z:
            import json
            return json.loads(z.read("stats.json"))["rmse"]


def main():
    """Score an external trajectory, e.g. VGGT-Long's exps/.../camera_poses.txt (16 numbers/line, C2W)."""
    import argparse
    import json
    from .kitti import gt_cam2_poses
    from .trajectory import read_poses
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--estimate", required=True, help="poses file: 16 (4x4 C2W) or 12 (KITTI 3x4) numbers per line")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--kitti", default="data/kitti")
    ap.add_argument("--evo", action="store_true", help="also compute ATE with evo")
    a = ap.parse_args()
    est = read_poses(a.estimate)
    gt = gt_cam2_poses(a.kitti, a.seq)[:len(est)]
    if len(gt) != len(est):
        raise SystemExit(f"{len(est)} estimated poses but only {len(gt)} GT poses")
    out = evaluate(est, gt, [250, 500, 1000, 2000, 3000, 4000])
    if a.evo:
        out["evo_ate_rmse_m"] = evo_ate(est, gt)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
