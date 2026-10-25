"""Sparse Sim(3) pose-graph solver (Levenberg-Marquardt + IRLS robust kernels).

Nodes S_k map window k into the world; node `anchor` is held fixed (removes the 7-DoF gauge).
Edge (i, j, Z) says S_j = S_i @ Z, i.e. Z maps window j into window i (as in geometry.py).

Residual conventions:
  "ours"     r = Log(inv(Z) @ inv(S_i) @ S_j)   -- error expressed in window j's frame
  "upstream" r = Log(Z @ inv(S_j) @ S_i)        -- VGGT-Long's loop_utils/sim3loop.py residual
Both vanish at the same configuration; they weight errors differently unless Z ~ identity, so
the VGGT-Long baseline uses "upstream" with identity information to match its objective.

Cost = 0.5 * sum_e rho(r_e^T Lambda_e r_e). Kernels act on the whitened squared norm q:
  none:  rho(q) = q
  huber: rho(q) = q if q <= k^2 else 2k*sqrt(q) - k^2               (k on the whitened norm)
  dcs:   rho(q) = q if q <= phi else 3*phi - 4*phi^2/(phi + q)      (Agarwal et al. ICRA'13;
         IRLS weight rho'(q) = min(1, 2*phi/(phi+q))^2, the squared DCS scaling factor)
Updates are right perturbations S <- S @ Exp(delta); Jacobians are central differences.
A chain (tree) graph is already solved by its composition: the solver must then return the
initial poses unchanged, whatever the weights (tested).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from . import lie

_H = 1e-6
_BASIS = np.eye(7)


def kernel(q, name="none", param=None):
    """Returns (rho, weight) for whitened squared residuals q."""
    q = np.asarray(q, float)
    if name == "none" or param is None or not np.isfinite(param):
        return q.copy(), np.ones_like(q)
    if name == "huber":
        k2 = param ** 2
        root = np.sqrt(np.maximum(q, 1e-300))
        big = q > k2
        rho = np.where(big, 2 * param * root - k2, q)
        w = np.where(big, param / root, 1.)
        return rho, w
    if name == "dcs":
        big = q > param
        rho = np.where(big, 3 * param - 4 * param ** 2 / (param + q), q)
        w = np.where(big, (2 * param / (param + q)) ** 2, 1.)
        return rho, w
    raise ValueError(f"Unknown kernel {name}")


def residuals(poses, i, j, z, mode="ours", z_inv=None):
    si, sj = poses[i], poses[j]
    if mode == "ours":
        z_inv = lie.inv(z) if z_inv is None else z_inv
        return lie.log(z_inv @ lie.inv(si) @ sj)
    if mode == "upstream":
        return lie.log(z @ lie.inv(sj) @ si)
    raise ValueError(mode)


def _jacobians(poses, i, j, z, mode, z_inv):
    n = len(i)
    ji, jj = np.zeros((n, 7, 7)), np.zeros((n, 7, 7))
    plus, minus = lie.exp(_H * _BASIS), lie.exp(-_H * _BASIS)
    for d in range(7):
        for target, jac in ((i, ji), (j, jj)):
            si, sj = poses[i], poses[j]
            if target is i:
                rp = _res_pair(si @ plus[d], sj, z, mode, z_inv)
                rm = _res_pair(si @ minus[d], sj, z, mode, z_inv)
            else:
                rp = _res_pair(si, sj @ plus[d], z, mode, z_inv)
                rm = _res_pair(si, sj @ minus[d], z, mode, z_inv)
            jac[:, :, d] = (rp - rm) / (2 * _H)
    return ji, jj


def _res_pair(si, sj, z, mode, z_inv):
    if mode == "ours":
        return lie.log(z_inv @ lie.inv(si) @ sj)
    return lie.log(z @ lie.inv(sj) @ si)


def _cost(poses, edges, info, mode, kern, kparam, kmask, z_inv):
    r = residuals(poses, edges[0], edges[1], edges[2], mode, z_inv)
    q = np.einsum("ni,nij,nj->n", r, info, r)
    rho, w = kernel(q, kern, kparam)
    rho = np.where(kmask, rho, q)
    w = np.where(kmask, w, 1.)
    return .5 * rho.sum(), r, q, w


def solve(initial, i, j, z, info, mode="ours", kern="none", kparam=None, kernel_on=None,
          anchor=0, max_iters=60, lam0=1e-4, tol=1e-12, verbose=False):
    """Optimise node poses.

    initial (N,4,4); i, j (E,) int; z (E,4,4); info (E,7,7) symmetric PSD;
    kernel_on (E,) bool mask of edges that get the robust kernel (default: all).
    Returns (poses, report).
    """
    poses = np.array(initial, float)
    i, j = np.asarray(i, int), np.asarray(j, int)
    z, info = np.asarray(z, float), np.asarray(info, float)
    n_nodes, n_edges = len(poses), len(i)
    if not (len(j) == len(z) == len(info) == n_edges):
        raise ValueError("Edge arrays must have equal length")
    if np.any(i == j) or i.min() < 0 or j.min() < 0 or max(i.max(), j.max()) >= n_nodes:
        raise ValueError("Invalid edge indices")
    kmask = np.ones(n_edges, bool) if kernel_on is None else np.asarray(kernel_on, bool)
    z_inv = lie.inv(z)
    free = np.array([k for k in range(n_nodes) if k != anchor])
    col = -np.ones(n_nodes, int)
    col[free] = np.arange(len(free))
    dim = 7 * len(free)
    edges = (i, j, z)
    cost, r, q, w = _cost(poses, edges, info, mode, kern, kparam, kmask, z_inv)
    lam, history, it = lam0, [cost], 0
    for it in range(1, max_iters + 1):
        ji, jj = _jacobians(poses, i, j, z, mode, z_inv)
        wl = w[:, None, None] * info
        rows, cols, vals = [], [], []
        g = np.zeros(dim)
        blocks = ((i, ji), (j, jj))
        for a_nodes, ja in blocks:
            ga = np.einsum("nki,nkl,nl->ni", ja, wl, r)
            ok = col[a_nodes] >= 0
            np.add.at(g, (7 * col[a_nodes[ok]])[:, None] + np.arange(7), ga[ok])
            for b_nodes, jb in blocks:
                hab = np.einsum("nki,nkl,nlj->nij", ja, wl, jb)
                okb = ok & (col[b_nodes] >= 0)
                r0 = (7 * col[a_nodes[okb]])[:, None, None] + np.arange(7)[None, :, None]
                c0 = (7 * col[b_nodes[okb]])[:, None, None] + np.arange(7)[None, None, :]
                rows.append(np.broadcast_to(r0, hab[okb].shape).ravel())
                cols.append(np.broadcast_to(c0, hab[okb].shape).ravel())
                vals.append(hab[okb].ravel())
        h = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)).tocsc()
        diag = np.maximum(h.diagonal(), 1e-12)
        accepted = False
        for _ in range(12):
            step = spsolve((h + diags(lam * diag + 1e-12)).tocsc(), -g)
            if not np.all(np.isfinite(step)):
                lam *= 10
                continue
            trial = poses.copy()
            trial[free] = poses[free] @ lie.exp(step.reshape(-1, 7))
            new_cost, nr, nq, nw = _cost(trial, edges, info, mode, kern, kparam, kmask, z_inv)
            if new_cost <= cost:
                rel = (cost - new_cost) / max(cost, 1e-300)
                poses, cost, r, q, w = trial, new_cost, nr, nq, nw
                lam = max(lam / 3, 1e-12)
                accepted = True
                break
            lam *= 4
        history.append(cost)
        if verbose:
            print(f"  iter {it}: cost {cost:.6e} lambda {lam:.1e}")
        if not accepted or rel < tol or np.abs(step).max() < 1e-12:
            break
    chi2 = float(q.sum())
    return poses, {"iterations": it, "cost": float(cost), "initial_cost": float(history[0]),
                   "chi2_unweighted": chi2, "edge_q": q, "edge_weight": w, "edge_residual": r}
