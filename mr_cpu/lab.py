"""Run: python -m mr_cpu.lab. All experiments here are explicitly synthetic."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from .calibration import MonotoneVariance, scalar_scores
from .geometry import decompose, fit_similarity, inverse, similarity, transform_points
from .graph import solve_translation_graph
from .trajectory import ate, read_poses


def save_plot(path, title, xlabel, ylabel, curves):
    """Dependency-light scientific SVG plot; each curve is (label, x, y, color)."""
    all_x = np.concatenate([np.asarray(c[1]) for c in curves])
    all_y = np.concatenate([np.asarray(c[2]) for c in curves])
    xmin, xmax = float(all_x.min()), float(all_x.max())
    ymin, ymax = float(all_y.min()), float(all_y.max())
    dx, dy = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    ymin, ymax = ymin - .08 * dy, ymax + .08 * dy
    def xy(x, y):
        return 85 + (x - xmin) / dx * 530, 340 - (y - ymin) / (ymax - ymin) * 250
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="860" height="425" viewBox="0 0 860 425" role="img">', f'<title>{escape(title)}</title>', '<rect width="860" height="425" fill="#ffffff"/>', '<g font-family="Arial,sans-serif" fill="#172033">', f'<text x="38" y="30" font-size="19">{escape(title)}</text>', '<text x="38" y="52" font-size="12">Synthetic teaching experiment — no model or dataset performance claim</text>']
    for tick in np.linspace(0, 1, 6):
        x, _ = xy(xmin + tick * dx, ymin)
        _, y = xy(xmin, ymin + tick * (ymax - ymin))
        parts += [f'<path d="M{x:.2f},90 V340 M85,{y:.2f} H615" fill="none" stroke="#e2e6ee"/>', f'<text x="{x:.2f}" y="360" text-anchor="middle" font-size="11">{xmin + tick * dx:.2g}</text>', f'<text x="75" y="{y+4:.2f}" text-anchor="end" font-size="11">{ymin + tick * (ymax-ymin):.2g}</text>']
    for index, (label, x, y, color) in enumerate(curves):
        points = [xy(float(a), float(b)) for a, b in zip(x, y)]
        coords = " ".join(f"{a:.2f},{b:.2f}" for a, b in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts += [f'<circle cx="{a:.2f}" cy="{b:.2f}" r="3" fill="{color}"/>' for a, b in points]
        parts.append(f'<text x="640" y="{110+index*28}" fill="{color}" font-size="12">{escape(label)}</text>')
    parts += [f'<text x="350" y="398" text-anchor="middle" font-size="14">{escape(xlabel)}</text>', f'<text x="20" y="220" transform="rotate(-90 20 220)" text-anchor="middle" font-size="14">{escape(ylabel)}</text>', '</g></svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/week1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    points = rng.normal(size=(80, 3))
    true_transform = similarity(2.4, Rotation.from_euler("xyz", [.15, -.3, .4]).as_matrix(), [3., -2., 5.])
    local_points = transform_points(inverse(true_transform), points)
    recovered = fit_similarity(local_points, points)
    geometry = {"true_scale": 2.4, "recovered_scale": decompose(recovered)[0], "point_rmse_before_alignment": float(np.sqrt(np.mean(np.sum((local_points - points)**2, axis=1)))), "point_rmse_after_alignment": ate(local_points, points)["ate_rmse"]}

    chain = [(0, 1, 1.), (1, 2, 2.)]
    cycle = chain + [(0, 2, 2.)]
    truth = np.arange(3, dtype=float)
    solutions = {"truth": truth, "chain_uniform": solve_translation_graph(3, chain, [1., 1.]), "chain_changed_weights": solve_translation_graph(3, chain, [100., .01]), "loop_uniform": solve_translation_graph(3, cycle, [1., 1., 1.]), "loop_downweight_bad_seam": solve_translation_graph(3, cycle, [1., .01, 1.])}
    graph = {name: {"positions": value.tolist(), "anchored_position_rmse": float(np.sqrt(np.mean((value - truth)**2)))} for name, value in solutions.items()}
    graph["note"] = "Weights were specified by hand using the synthetic design's known faulty seam. No deployable predictor or VGGT result is demonstrated."
    save_plot(args.output / "graph_weights.svg", "Why loop closures make edge weights matter", "Node index", "Estimated 1D position", [(name, np.arange(3), solutions[name], color) for name, color in [("truth", "#263238"), ("chain_uniform", "#d55e00"), ("loop_uniform", "#0072b2"), ("loop_downweight_bad_seam", "#009e73")]])

    # Fit and evaluation draws are independent; confidence is positive, not a probability.
    fit_c = np.exp(rng.uniform(np.log(1.01), np.log(50.), 6000))
    test_c = np.exp(rng.uniform(np.log(1.01), np.log(50.), 4000))
    def sigma(c):
        return .07 + .45 / np.sqrt(c)
    fit_e = rng.normal(size=len(fit_c)) * sigma(fit_c)
    test_e = rng.normal(size=len(test_c)) * sigma(test_c)
    model = MonotoneVariance.fit(fit_c, fit_e**2, floor=1e-4)
    raw_variance = 1. / test_c**2
    calibrated_variance = model.predict(test_c)
    calibration = {"fit_samples": len(fit_c), "held_out_samples": len(test_c), "generating_sigma": "0.07 + 0.45 / sqrt(confidence), scalar zero-mean Gaussian", "raw_assumed_variance": "1 / confidence**2", "raw": scalar_scores(test_e, raw_variance), "calibrated": scalar_scores(test_e, calibrated_variance), "known_generating_variance": scalar_scores(test_e, sigma(test_c)**2)}
    # Even perfectly calibrated individual errors do not justify independence.
    correlation_rng = np.random.default_rng(131)
    shared_sigma, independent_sigma, trials = .2, .1, 3000
    correlation = []
    for count in [1, 10, 100]:
        shared = correlation_rng.normal(0, shared_sigma, (trials, 1))
        independent = correlation_rng.normal(0, independent_sigma, (trials, count))
        mean_error = (shared + independent).mean(axis=1)
        marginal_variance = shared_sigma**2 + independent_sigma**2
        naive_variance = marginal_variance / count
        true_variance = shared_sigma**2 + independent_sigma**2 / count
        correlation.append({"points_per_estimate": count, "trials": trials, "calibrated_individual_point_variance": marginal_variance, "naive_independent_mean_variance": naive_variance, "true_mean_variance": true_variance, "empirical_mean_squared_error": float(np.mean(mean_error**2)), "expected_overconfidence_variance_factor": true_variance / naive_variance})
    bins = np.array_split(np.argsort(test_c), 10)
    rows = [{"bin": i, "count": len(ids), "mean_confidence": float(np.mean(test_c[ids])), "empirical_rms_error": float(np.sqrt(np.mean(test_e[ids]**2))), "raw_predicted_rms": float(np.sqrt(np.mean(raw_variance[ids]))), "calibrated_predicted_rms": float(np.sqrt(np.mean(calibrated_variance[ids])))} for i, ids in enumerate(bins)]
    with (args.output / "calibration_bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plot(args.output / "calibration.svg", "Confidence ranking and uncertainty magnitude differ", "Mean confidence in held-out bin", "RMS error / predicted RMS", [(label, [r["mean_confidence"] for r in rows], [r[key] for r in rows], color) for label, key, color in [("Measured error", "empirical_rms_error", "#263238"), ("Raw uncertainty", "raw_predicted_rms", "#d55e00"), ("Fitted uncertainty", "calibrated_predicted_rms", "#009e73")]])

    kitti = []
    pose_dir = Path(__file__).resolve().parents[1] / "data/kitti/poses"
    for path in sorted(pose_dir.glob("*.txt")):
        poses = read_poses(path)
        xyz = poses[:, :3, 3]
        # A manufactured change of coordinates of ground truth checks evaluation only.
        manufactured = transform_points(inverse(true_transform), xyz)
        kitti.append({"sequence": path.stem, "frames": len(poses), "path_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()), "manufactured_gauge_change_ate_m": ate(manufactured, xyz)["ate_rmse"], "warning": "GT-only evaluator sanity check; not SLAM inference or a baseline result"})
    result = {"status": "SYNTHETIC LEARNING LAB + OPTIONAL REAL GT-ONLY PREFLIGHT; NO VGGT RESULTS", "seed": 20261016, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "geometry": geometry, "graph": graph, "calibration": calibration, "correlated_measurement_demo": correlation, "kitti_pose_preflight": kitti}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
