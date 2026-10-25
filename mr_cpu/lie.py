"""Batched Sim(3) operations for the pose-graph backend.

Same conventions as geometry.py: a Sim(3) matrix is [[s*R, t], [0, 1]] acting on column
vectors, and the tangent is [rotation (3), upsilon (3), log-scale (1)], where upsilon is
the Lie-algebra translation (not the matrix translation t).

exp(xi) has translation t = V(w, sigma) @ upsilon with
    V = integral_0^1 exp(sigma*t) * R(t*w) dt = c0*I + c1*W + c2*W^2,  W = skew(w).
The three scalar integrals are smooth in (theta, sigma), so a 12-point Gauss-Legendre
rule gives them to ~1e-15 for |sigma| + theta < 8 without any small-angle special cases
(checked against scipy.linalg.expm in tests/test_pipeline.py).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(12)
_T = .5 * (_NODES + 1.)          # nodes on [0, 1]
_WQ = .5 * _WEIGHTS


def skew(w):
    w = np.atleast_2d(w)
    k = np.zeros((len(w), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    return k


def _v_matrix(w, sig):
    theta = np.linalg.norm(w, axis=1)[:, None]
    t = _T[None, :]
    e = np.exp(sig[:, None] * t)
    c0 = (e * _WQ).sum(1)
    c1 = (e * t * np.sinc(theta * t / np.pi) * _WQ).sum(1)
    c2 = (e * .5 * t**2 * np.sinc(theta * t / (2 * np.pi))**2 * _WQ).sum(1)
    k = skew(w)
    return c0[:, None, None] * np.eye(3) + c1[:, None, None] * k + c2[:, None, None] * (k @ k)


def exp(xi):
    """(N,7) -> (N,4,4)."""
    xi = np.atleast_2d(np.asarray(xi, float))
    w, u, sig = xi[:, :3], xi[:, 3:6], xi[:, 6]
    m = np.zeros((len(xi), 4, 4))
    m[:, :3, :3] = np.exp(sig)[:, None, None] * Rotation.from_rotvec(w).as_matrix()
    m[:, :3, 3] = np.einsum("nij,nj->ni", _v_matrix(w, sig), u)
    m[:, 3, 3] = 1.
    return m


def split(m):
    """(N,4,4) -> scale (N,), rotation (N,3,3), translation (N,3)."""
    a = m[:, :3, :3]
    s = np.cbrt(np.linalg.det(a))
    return s, a / s[:, None, None], m[:, :3, 3]


def inv(m):
    s, r, t = split(m)
    out = np.zeros_like(m)
    rt = np.transpose(r, (0, 2, 1))
    out[:, :3, :3] = rt / s[:, None, None]
    out[:, :3, 3] = -np.einsum("nij,nj->ni", rt, t) / s[:, None]
    out[:, 3, 3] = 1.
    return out


def log(m):
    """(N,4,4) -> (N,7). Principal branch (rotation angle < pi)."""
    s, r, t = split(m)
    w = Rotation.from_matrix(r).as_rotvec()
    sig = np.log(s)
    u = np.linalg.solve(_v_matrix(w, sig), t[..., None])[..., 0]
    return np.c_[w, u, sig]


def from_srt(s, r, t):
    """Build (N,4,4) from scale, rotation and translation."""
    s, r, t = np.atleast_1d(s), np.asarray(r).reshape(-1, 3, 3), np.asarray(t).reshape(-1, 3)
    m = np.zeros((len(s), 4, 4))
    m[:, :3, :3] = s[:, None, None] * r
    m[:, :3, 3] = t
    m[:, 3, 3] = 1.
    return m
