"""Explicit geometry conventions used by the learning lab.

Column vectors: p_destination = S_destination_from_source @ p_source.
A Sim(3) matrix has upper blocks [s*R, t], with s > 0 and det(R) = +1.
Lie coordinates are [rotation radians (3), upsilon (3), log-scale (1)].
Upsilon is the translation coordinate of the Lie algebra, not matrix t.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def finite_array(value, shape=None):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected finite array of shape {shape}, got {a.shape}")
    return a


def skew(v):
    x, y, z = finite_array(v, (3,))
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def similarity(scale=1., rotation=None, translation=None):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity scale must be positive and finite")
    rotation = np.eye(3) if rotation is None else finite_array(rotation, (3, 3))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7):
        raise ValueError("Rotation must be orthonormal with determinant +1")
    result = np.eye(4)
    result[:3, :3] = scale * rotation
    result[:3, 3] = np.zeros(3) if translation is None else finite_array(translation, (3,))
    return result


def decompose(s):
    s = finite_array(s, (4, 4))
    if not np.allclose(s[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("Invalid homogeneous last row")
    scale = float(np.cbrt(np.linalg.det(s[:3, :3])))
    if scale <= 0:
        raise ValueError("Similarity has nonpositive determinant")
    rotation = s[:3, :3] / scale
    similarity(scale, rotation, s[:3, 3])  # validate; do not silently repair a matrix
    return scale, rotation, s[:3, 3].copy()


def inverse(s):
    scale, rotation, translation = decompose(s)
    return similarity(1 / scale, rotation.T, -rotation.T @ translation / scale)


def transform_points(s, points):
    scale, rotation, translation = decompose(s)
    points = finite_array(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be N x 3")
    return scale * points @ rotation.T + translation


def transform_camera_poses(s, poses):
    """Change world gauge of C2W poses; preserve unit camera rotations."""
    scale, rotation, translation = decompose(s)
    poses = validate_camera_poses(poses)
    output = poses.copy()
    output[:, :3, :3] = rotation @ poses[:, :3, :3]
    output[:, :3, 3] = scale * poses[:, :3, 3] @ rotation.T + translation
    return output


def sim3_exp(tangent):
    tangent = finite_array(tangent, (7,))
    algebra = np.zeros((4, 4))
    algebra[:3, :3] = skew(tangent[:3]) + tangent[6] * np.eye(3)
    algebra[:3, 3] = tangent[3:6]
    return expm(algebra)


def sim3_log(s):
    """Principal logarithm; rotations at the pi branch cut need special care."""
    scale, rotation, translation = decompose(s)
    omega = Rotation.from_matrix(rotation).as_rotvec()
    sigma = np.log(scale)
    a = skew(omega) + sigma * np.eye(3)
    # exp([[A,I],[0,0]]) has V = integral_0^1 exp(t*A) dt in its top right.
    block = np.zeros((6, 6))
    block[:3, :3], block[:3, 3:] = a, np.eye(3)
    v = expm(block)[:3, 3:]
    return np.r_[omega, np.linalg.solve(v, translation), sigma]


def edge_residual(measurement, pose_i, pose_j):
    """Z_ij maps chunk j into i; S_i maps chunk i into world."""
    return sim3_log(inverse(measurement) @ inverse(pose_i) @ pose_j)


def fit_similarity(source, target, weights=None, estimate_scale=True):
    """Weighted Umeyama fit target ~= s*R*source + t; no outlier rejection."""
    source, target = finite_array(source), finite_array(target)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or len(source) < 3:
        raise ValueError("Need matching N x 3 point arrays with N >= 3")
    weights = np.ones(len(source)) if weights is None else finite_array(weights, (len(source),))
    if np.any(weights < 0) or np.count_nonzero(weights) < 3:
        raise ValueError("Need at least three positive, nonnegative weights")
    weights = weights / weights.sum()
    mean_source, mean_target = weights @ source, weights @ target
    x, y = source - mean_source, target - mean_target
    if np.linalg.matrix_rank(x * np.sqrt(weights[:, None]), tol=1e-10) < 2 or np.linalg.matrix_rank(y * np.sqrt(weights[:, None]), tol=1e-10) < 2:
        raise ValueError("Collinear or stationary points do not determine this alignment")
    u, singular, vt = np.linalg.svd((y * weights[:, None]).T @ x)
    signs = np.ones(3)
    signs[-1] = 1. if np.linalg.det(u @ vt) > 0 else -1.
    rotation = u @ np.diag(signs) @ vt
    variance = np.sum(weights * np.sum(x * x, axis=1))
    scale = float(singular @ signs / variance) if estimate_scale else 1.
    return similarity(scale, rotation, mean_target - scale * rotation @ mean_source)


def validate_camera_poses(poses):
    poses = finite_array(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("Camera poses must be nonempty N x 4 x 4 C2W matrices")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid camera homogeneous rows")
    r = poses[:, :3, :3]
    if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=2e-5) or not np.allclose(np.linalg.det(r), 1., atol=2e-5):
        raise ValueError("Camera poses must contain unit, proper rotations")
    return poses


def project_lidar(points, camera_from_lidar, intrinsics, image_size):
    """Pinhole projection. Returns pixel floats, optical-axis depth z, validity.

    Caller must supply a transform into the actual rectified input camera.
    Image size is (height, width). This does not resolve occlusion or crop pixels.
    """
    h, w = image_size
    if h <= 0 or w <= 0:
        raise ValueError("Image dimensions must be positive")
    if not np.isclose(decompose(camera_from_lidar)[0], 1.):
        raise ValueError("LiDAR extrinsic must be rigid and expressed in metric units")
    camera = transform_points(camera_from_lidar, points)
    projected = camera @ finite_array(intrinsics, (3, 3)).T
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = projected[:, :2] / projected[:, 2:3]
    valid = (z > 0) & np.all(np.isfinite(pixels), axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    return pixels, z, valid
