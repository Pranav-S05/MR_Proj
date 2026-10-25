"""Ground-truth edge labels: per-window reference gauges and 7-DoF edge errors.

A window's prediction lives in its own arbitrary Sim(3) gauge. We fit A_k, mapping window-k
coordinates into the metric reference world, from the full camera poses of its frames:
  * rotation: chordal L2 mean of R_gt(f) @ R_pred(f)^T over the window's frames (orientations
    pin down the roll about the driving direction, which positions on a straight road cannot);
  * scale and translation: least squares on camera centres given that rotation.
The reference connection is Z_ref_ij = inv(A_i) @ A_j and the edge error is
e_ij = Log(inv(Z_hat_ij) @ Z_ref_ij) = [rotation (rad), upsilon (window-j units), log-scale].
At a graph solution S_i^-1 S_j ~= Z_ref, so e is what the edge residual r_ij measures.

Windows whose GT camera centres barely move (the car is stopped) do not determine scale: they are
flagged degenerate and their edges are excluded from fitting and scoring, never silently kept.
These are operational labels under model distortion, not uniquely observed physical seams.
"""
from __future__ import annotations

import numpy as np

from . import lie


def fit_gauge(pred_c2w, gt_c2w, min_spread_m=1.0):
    """Sim(3) A with gt ~= A o pred for camera poses. Returns (A (4,4), info dict)."""
    m = gt_c2w[:, :3, :3] @ np.transpose(pred_c2w[:, :3, :3], (0, 2, 1))
    u, _, vt = np.linalg.svd(m.sum(0))
    rot = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    q = pred_c2w[:, :3, 3] @ rot.T
    g = gt_c2w[:, :3, 3]
    qc, gc = q - q.mean(0), g - g.mean(0)
    spread = float(np.sqrt((gc ** 2).sum(1).mean()))
    denom = (qc ** 2).sum()
    scale = float((qc * gc).sum() / denom) if denom > 0 else float("nan")
    ok = spread >= min_spread_m and np.isfinite(scale) and scale > 0
    if not ok:
        scale = 1.0 if not (np.isfinite(scale) and scale > 0) else scale
    t = g.mean(0) - scale * q.mean(0)
    a = lie.from_srt(scale, rot, t)[0]
    pos_rms = float(np.sqrt(((scale * q + t - g) ** 2).sum(1).mean()))
    rot_err = np.degrees(np.linalg.norm(lie.log(lie.from_srt(np.ones(len(m)), np.transpose(m, (0, 2, 1)) @ rot, np.zeros((len(m), 3))))[:, :3], axis=1))
    return a, {"valid": bool(ok), "gt_spread_m": spread, "scale_m_per_unit": scale,
               "position_rms_m": pos_rms, "position_rms_rel": pos_rms / max(spread, 1e-9),
               "rotation_rms_deg": float(np.sqrt(np.mean(rot_err ** 2)))}


def window_gauges(window_frames, window_c2w, gt_c2w, min_spread_m=1.0):
    """Gauges for all windows. window_frames[k] (n,) frame ids, window_c2w[k] (n,4,4)."""
    gauges, infos = [], []
    for frames, c2w in zip(window_frames, window_c2w):
        a, info = fit_gauge(c2w, gt_c2w[frames], min_spread_m)
        gauges.append(a)
        infos.append(info)
    return np.stack(gauges), infos


def edge_errors(gauges, gauge_ok, i, j, z):
    """7-DoF errors e = Log(inv(Z_hat) @ inv(A_i) @ A_j) and a validity mask."""
    z_ref = lie.inv(gauges[i]) @ gauges[j]
    e = lie.log(lie.inv(z) @ z_ref)
    return e, np.asarray(gauge_ok)[i] & np.asarray(gauge_ok)[j]


def block_targets(e, char_length):
    """Per-coordinate squared-error targets for the block-isotropic model.

    rotation: ||e_rot||^2 / 3 (rad^2); translation: (||e_upsilon|| / L)^2 / 3, dimensionless
    with L the edge's characteristic length (median depth, window-j units); scale: e_s^2.
    """
    return {"rot": (e[:, :3] ** 2).sum(1) / 3,
            "trans": ((e[:, 3:6] ** 2).sum(1) / char_length ** 2) / 3,
            "scale": e[:, 6] ** 2}
