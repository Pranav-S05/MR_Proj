"""Small reference solvers, not an installed or reproduced VGGT-Long backend."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

from .geometry import decompose, edge_residual, sim3_exp


def solve_translation_graph(node_count, edges, weights):
    """Exact 1D least squares with node 0 fixed to 0.

    Edge tuples are (i, j, measured_x_j_minus_x_i). Weight = inverse variance.
    This is a translation-only special case, useful for understanding the graph.
    """
    weights = np.asarray(weights, float)
    if node_count < 2 or len(edges) != len(weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Expected valid graph with one positive weight per edge")
    a, b = np.zeros((len(edges), node_count - 1)), np.zeros(len(edges))
    for k, ((i, j, measured), weight) in enumerate(zip(edges, weights)):
        if not (0 <= i < node_count and 0 <= j < node_count) or i == j or not np.isfinite(measured):
            raise ValueError("Invalid translation edge")
        root = np.sqrt(weight)
        if i:
            a[k, i - 1] = -root
        if j:
            a[k, j - 1] = root
        b[k] = root * measured
    result, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank != node_count - 1:
        raise ValueError("Graph is disconnected from anchor 0")
    return np.r_[0., result]


@dataclass
class Edge:
    i: int
    j: int
    measurement: np.ndarray
    information: np.ndarray


def solve_sim3_graph(initial_poses, edges, max_evaluations=150):
    """Anchored, dense finite-difference Sim(3) reference optimizer for tiny graphs.

    Uses SciPy trust-region least squares, quadratic edge costs, float64.
    No robust kernels, sparse analytic Jacobians, or upstream parity claim.
    Initial poses and measurements use the geometry.py conventions.
    """
    initial_poses = np.asarray(initial_poses, float)
    if initial_poses.ndim != 3 or initial_poses.shape[1:] != (4, 4) or len(initial_poses) < 2:
        raise ValueError("Need at least two initial Sim(3) poses")
    for pose in initial_poses:
        decompose(pose)
    roots, adjacency = [], {i: set() for i in range(len(initial_poses))}
    for edge in edges:
        if not (0 <= edge.i < len(initial_poses) and 0 <= edge.j < len(initial_poses)) or edge.i == edge.j:
            raise ValueError("Invalid edge indices")
        decompose(edge.measurement)
        information = np.asarray(edge.information, float)
        if information.shape != (7, 7) or not np.all(np.isfinite(information)) or not np.allclose(information, information.T):
            raise ValueError("Information must be a symmetric, finite 7 x 7 matrix")
        try:
            roots.append(np.linalg.cholesky(information).T)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Information must be positive definite") from exc
        adjacency[edge.i].add(edge.j)
        adjacency[edge.j].add(edge.i)
    reached, pending = {0}, [0]
    while pending:
        for other in adjacency[pending.pop()]:
            if other not in reached:
                reached.add(other)
                pending.append(other)
    if len(reached) != len(initial_poses):
        raise ValueError("Graph must be connected to fixed pose 0")

    def decode(parameters):
        poses = [initial_poses[0]]
        for initial, delta in zip(initial_poses[1:], parameters.reshape(-1, 7)):
            poses.append(initial @ sim3_exp(delta))
        return np.stack(poses)

    def residual(parameters):
        poses = decode(parameters)
        return np.concatenate([root @ edge_residual(e.measurement, poses[e.i], poses[e.j]) for e, root in zip(edges, roots)])

    solution = least_squares(residual, np.zeros(7 * (len(initial_poses) - 1)), method="trf", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_evaluations)
    if not solution.success:
        raise RuntimeError(f"Reference graph solver did not converge: {solution.message}")
    return decode(solution.x), {"success": bool(solution.success), "sum_squared_whitened_residual": float(2 * solution.cost), "function_evaluations_reported_by_scipy": int(solution.nfev)}
