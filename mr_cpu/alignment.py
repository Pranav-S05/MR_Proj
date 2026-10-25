"""Window-to-window Sim(3) alignment, reproducing VGGT-Long's estimator, plus edge features.

VGGT-Long (loop_utils/sim3utils.py, commit c160869) aligns two windows from the point maps of
their shared frames:
  * keep pixels whose point confidence exceeds 0.1 * min(median(conf1), median(conf2)) in both;
  * initial weights w = sqrt(conf1 * conf2);
  * a weighted Sim(3) estimate whose scale is the ratio of weighted RMS spreads (NOT the
    least-squares Umeyama scale), rotation from a weighted Kabsch SVD;
  * up to `max_iters` IRLS rounds with Huber weights (delta in window units) on the 3D residual.
We copy that logic, including its convergence test, so seams match the baseline. The returned
transform maps window-2 coordinates into window-1 coordinates (= Z_12 in our convention).
"""
from __future__ import annotations

import numpy as np

from . import lie

# log-spaced bins for the per-edge histogram of point weights sqrt(c1*c2) (VGGT conf >= 1)
CONF_HIST_EDGES = np.r_[-np.inf, np.exp(np.linspace(0, np.log(1e3), 47)), np.inf]
CONF_HIST_CENTRES = np.r_[1.0, np.sqrt(CONF_HIST_EDGES[1:-2] * CONF_HIST_EDGES[2:-1]), 1e3]


def vggtlong_sim3(src, tgt, w):
    """Weighted estimate of tgt ~= s * R @ src + t with VGGT-Long's spread-ratio scale."""
    total = w.sum()
    if total < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    nw = w / total
    mu_s, mu_t = nw @ src, nw @ tgt
    xs, xt = src - mu_s, tgt - mu_t
    s = np.sqrt(nw @ (xt * xt).sum(1)) / np.sqrt(nw @ (xs * xs).sum(1))
    sq = np.sqrt(nw)[:, None]
    h = (s * xs * sq).T @ (xt * sq)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[2] *= -1
        r = vt.T @ u.T
    return s, r, mu_t - s * r @ mu_s


def irls_sim3(src, tgt, init_w, delta=0.1, max_iters=5, tol=1e-9):
    """VGGT-Long's robust_weighted_estimate_sim3 (Huber IRLS). Returns s, R, t, final weights."""
    s, r, t = vggtlong_sim3(src, tgt, init_w)
    prev = np.inf
    huber = np.ones(len(src))
    for _ in range(max_iters):
        res = np.linalg.norm(tgt - (s * src @ r.T + t), axis=1)
        huber = np.where(res > delta, delta / np.maximum(res, 1e-300), 1.)
        comb = init_w * huber
        comb = comb / (comb.sum() + 1e-12)
        s2, r2, t2 = vggtlong_sim3(src, tgt, comb)
        change = abs(s2 - s) + np.linalg.norm(t2 - t)
        angle = np.arccos(np.clip((np.trace(r2 @ r.T) - 1) / 2, -1, 1))
        err = np.sum(np.where(res <= delta, .5 * res ** 2, delta * (res - .5 * delta)) * init_w)
        if (change < tol and angle < np.radians(.1)) or abs(prev - err) < tol * prev:
            break
        s, r, t, prev = s2, r2, t2, err
    return s, r, t, huber


def camera_depths(points, c2w):
    """Optical-axis depth of world points (n,h,w,3) in their own cameras c2w (n,4,4)."""
    rel = points - c2w[:, None, None, :3, 3]
    return np.einsum("nhwk,nk->nhw", rel, c2w[:, :3, 2])


def align_windows(p1, c1, p2, c2, c2w2, delta=0.1, max_iters=5, rel_threshold=0.1, max_points=None, seed=0):
    """Align window 2 to window 1 from corresponding shared-frame point maps.

    p1, p2: (n,h,w,3) point maps of the same n frames predicted by windows 1 and 2;
    c1, c2: (n,h,w) point confidences; c2w2: (n,4,4) window-2 cameras (for depth features).
    Returns (Z (4,4) mapping window 2 into window 1, features dict).
    """
    p1, p2 = np.asarray(p1, np.float64), np.asarray(p2, np.float64)
    c1, c2 = np.asarray(c1, np.float64), np.asarray(c2, np.float64)
    thr = min(np.median(c1), np.median(c2)) * rel_threshold
    valid = (c1 > thr) & (c2 > thr) & np.all(np.isfinite(p1), -1) & np.all(np.isfinite(p2), -1)
    n_total = valid.size
    depth2 = camera_depths(p2, c2w2)
    tgt, src = p1[valid], p2[valid]
    w = np.sqrt(c1[valid] * c2[valid])
    if max_points is not None and len(src) > max_points:
        keep = np.random.default_rng(seed).choice(len(src), max_points, replace=False)
        tgt, src, w = tgt[keep], src[keep], w[keep]
    if len(src) < 10:
        raise ValueError(f"Too few valid correspondences ({len(src)})")
    s, r, t, huber = irls_sim3(src, tgt, w, delta, max_iters)
    res = np.linalg.norm(tgt - (s * src @ r.T + t), axis=1)  # window-1 units
    fw = w * huber
    fw = fw / fw.sum()
    xs = src - fw @ src
    eig = np.linalg.eigvalsh((xs * fw[:, None]).T @ xs)
    d2 = depth2[valid]
    d2 = d2[np.isfinite(d2) & (d2 > 0)]
    char_len = float(np.median(d2)) if len(d2) else float("nan")  # window-2 units
    z = lie.from_srt(s, r, t)[0]
    log_z = lie.log(z[None])[0]
    feats = {
        "n_points": len(src), "valid_fraction": float(valid.sum() / n_total),
        "conf_mean": float(w.mean()), "conf_median": float(np.median(w)),
        "conf_logmean": float(np.mean(np.log(w))), "conf_p10": float(np.percentile(w, 10)),
        "inlier_fraction": float(np.mean(res <= delta)),
        # residuals of window-1 points; divide by s to express them in window-2 units like char_len
        "residual_rms": float(np.sqrt(fw @ res ** 2) / s), "residual_median": float(np.median(res) / s),
        "char_length": char_len,
        "residual_rel": float(np.sqrt(fw @ res ** 2) / s / char_len) if char_len > 0 else float("nan"),
        "spread_condition": float(eig[0] / max(eig[-1], 1e-300)),
        "rotation_deg": float(np.degrees(np.linalg.norm(log_z[:3]))),
        "abs_log_scale": float(abs(log_z[6])),
        "translation_rel": float(np.linalg.norm(t) / s / char_len) if char_len > 0 else float("nan"),
        "threshold": float(thr),
        "conf_hist": np.histogram(w, bins=CONF_HIST_EDGES)[0].astype(np.int64),
    }
    return z, feats
