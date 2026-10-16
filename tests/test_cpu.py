"""Geometry and evaluation checks that protect the later research conclusions."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from mr_cpu.calibration import MonotoneVariance, scalar_scores
from mr_cpu.geometry import (decompose, edge_residual, fit_similarity, inverse, project_lidar, sim3_exp, sim3_log, similarity, transform_camera_poses, transform_points)
from mr_cpu.graph import Edge, solve_sim3_graph, solve_translation_graph
from mr_cpu.trajectory import ate, read_poses


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(47)
        self.points = self.rng.normal(size=(50, 3))
        self.s = similarity(2.7, Rotation.from_euler("xyz", [.3, -.2, .6]).as_matrix(), [2, -5, 1])

    def test_inverse_and_direction(self):
        assert_allclose(inverse(self.s) @ self.s, np.eye(4), atol=1e-12)
        assert_allclose(transform_points(inverse(self.s), transform_points(self.s, self.points)), self.points, atol=1e-12)

    def test_weighted_umeyama_known_transform_and_outlier(self):
        target = transform_points(self.s, self.points)
        target[-1] += 100
        weights = np.ones(len(target))
        weights[-1] = 0
        assert_allclose(fit_similarity(self.points, target, weights), self.s, atol=1e-11)

    def test_collinear_points_are_rejected(self):
        points = np.c_[np.arange(5), np.zeros((5, 2))]
        with self.assertRaises(ValueError):
            fit_similarity(points, points)

    def test_reflection_is_not_a_rotation(self):
        target = self.points * [-1, 1, 1]
        fit = fit_similarity(self.points, target)
        self.assertAlmostEqual(np.linalg.det(decompose(fit)[1]), 1.)
        self.assertGreater(np.linalg.norm(transform_points(fit, self.points) - target), 1.)

    def test_lie_round_trip_including_zero(self):
        for tangent in [np.zeros(7), np.r_[np.zeros(3), [1., 2., 3.], .4], np.r_[[3.1, 0, 0], [1., -1., 2.], -.2]] + [self.rng.normal(0, .25, 7) for _ in range(10)]:
            assert_allclose(sim3_log(sim3_exp(tangent)), tangent, atol=1e-10)

    def test_lie_translation_is_not_matrix_translation(self):
        tangent = np.array([.2, -.3, .4, 1., 2., 3., .5])
        self.assertGreater(np.linalg.norm(sim3_exp(tangent)[:3, 3] - tangent[3:6]), .1)

    def test_edge_zero_with_correct_transform_convention(self):
        other = sim3_exp(self.rng.normal(0, .2, 7))
        measurement = inverse(self.s) @ other
        assert_allclose(edge_residual(measurement, self.s, other), 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(edge_residual(inverse(measurement), self.s, other)), .1)

    def test_camera_rotations_stay_unit_after_gauge_change(self):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, :3, 3] = self.points[:3]
        changed = transform_camera_poses(self.s, poses)
        assert_allclose(np.linalg.det(changed[:, :3, :3]), 1., atol=1e-12)
        assert_allclose(changed[:, :3, 3], transform_points(self.s, poses[:, :3, 3]))

    def test_invalid_similarities_rejected(self):
        for value in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                similarity(value)
        with self.assertRaises(ValueError):
            similarity(rotation=np.diag([-1., 1., 1.]))

    def test_lidar_depth_is_optical_z_and_checks_visibility(self):
        k = np.array([[100., 0, 50], [0, 100, 40], [0, 0, 1]])
        points = np.array([[0., 0, 2], [1, 0, 2], [0, 0, -1], [0, 0, 0]])
        uv, depth, valid = project_lidar(points, np.eye(4), k, (80, 120))
        assert_allclose(uv[:2], [[50, 40], [100, 40]])
        assert_allclose(depth, [2, 2, -1, 0])
        self.assertEqual(valid.tolist(), [True, True, False, False])

    def test_resize_and_crop_intrinsics(self):
        k = np.array([[100., 0, 50], [0, 100, 40], [0, 0, 1]])
        image_transform = np.array([[.5, 0, -3], [0, .5, -2], [0, 0, 1]])
        points = np.array([[0., 0, 2], [.3, .1, 2]])
        before, _, _ = project_lidar(points, np.eye(4), k, (80, 120))
        after, _, _ = project_lidar(points, np.eye(4), image_transform @ k, (40, 60))
        assert_allclose(after, before * .5 - [3, 2])


class EvaluationTests(unittest.TestCase):
    def test_global_scale_alignment_and_se3_difference(self):
        points = np.random.default_rng(9).normal(size=(50, 3))
        prediction = points * 3 + [1, -4, 7]
        self.assertLess(ate(prediction, points, "sim3")["ate_rmse"], 1e-10)
        self.assertGreater(ate(prediction, points, "se3")["ate_rmse"], 1.)

    def test_frame_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ate(np.zeros((4, 3)), np.zeros((5, 3)))

    def test_pose_file_formats_and_invalid_rotation(self):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, 0, 3] = [0, 1, 2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.txt"
            for rows in [poses[:, :3].reshape(3, 12), poses.reshape(3, 16)]:
                np.savetxt(path, rows)
                assert_allclose(read_poses(path), poses)
            poses[0, 0, 0] = 2
            np.savetxt(path, poses.reshape(3, 16))
            with self.assertRaises(ValueError):
                read_poses(path)


class CalibrationTests(unittest.TestCase):
    def test_tied_confidences_share_estimate_and_direction(self):
        model = MonotoneVariance.fit([1, 2, 2, 3], [9, 2, 6, 1])
        assert_allclose(model.predict([1, 2, 3]), [9, 4, 1])
        query = np.linspace(.1, 5, 100)
        self.assertTrue(np.all(np.diff(model.predict(query)) <= 1e-12))
        self.assertTrue(np.all(model.predict(query) > 0))

    def test_pav_pools_violating_blocks(self):
        model = MonotoneVariance.fit([1, 2, 3], [1, 4, 2])
        assert_allclose(model.predict([1, 2, 3]), [2.5, 2.5, 2.])

    def test_held_out_known_noise_second_moment(self):
        rng = np.random.default_rng(44)
        fit_c = rng.choice([1., 2., 4.], 24000)
        test_c = rng.choice([1., 2., 4.], 12000)
        fit_e, test_e = rng.normal(size=len(fit_c)) * 3 / fit_c, rng.normal(size=len(test_c)) * 3 / test_c
        variance = MonotoneVariance.fit(fit_c, fit_e**2).predict(test_c)
        score = scalar_scores(test_e, variance)
        self.assertTrue(.9 < score["mean_normalized_squared_error_target_1"] < 1.1)
        self.assertTrue(.93 < score["coverage_approximately_95_percent"] < .97)

    def test_invalid_targets_and_floor(self):
        for c, y, floor in [([1, 2], [1, -1], 1e-8), ([0, 2], [1, 1], 1e-8), ([1, 2], [1, 1], 0)]:
            with self.assertRaises(ValueError):
                MonotoneVariance.fit(c, y, floor)


class GraphTests(unittest.TestCase):
    def test_tree_cannot_improve_by_reweighting(self):
        edges = [(0, 1, 1.1), (1, 2, .8), (2, 3, 1.4)]
        a = solve_translation_graph(4, edges, [1, 1, 1])
        b = solve_translation_graph(4, edges, [.02, 300, 7])
        assert_allclose(a, b, atol=1e-10)

    def test_conflicting_loop_has_analytic_solution(self):
        edges = [(0, 1, 1.), (1, 2, 2.), (0, 2, 2.)]
        assert_allclose(solve_translation_graph(3, edges, [1, 1, 1]), [0, 2/3, 7/3], atol=1e-12)
        changed = solve_translation_graph(3, edges, [1, .01, 1])
        self.assertLess(np.linalg.norm(changed - [0, 1, 2]), .02)

    def test_global_weight_scaling_leaves_quadratic_solution(self):
        edges = [(0, 1, 1.), (1, 2, 2.), (0, 2, 2.)]
        assert_allclose(solve_translation_graph(3, edges, [1, 3, 5]), solve_translation_graph(3, edges, [17, 51, 85]), atol=1e-12)

    def test_disconnected_scalar_graph_rejected(self):
        with self.assertRaises(ValueError):
            solve_translation_graph(3, [(0, 1, 1)], [1])

    def test_full_sim3_reference_solver_recovers_known_cycle(self):
        true = np.stack([np.eye(4), sim3_exp([.1, -.1, .2, 1., .3, -.2, .1]), sim3_exp([-.15, .12, .08, 2., -.3, .4, -.1])])
        edges = [Edge(i, j, inverse(true[i]) @ true[j], np.eye(7)) for i, j in [(0, 1), (1, 2), (0, 2)]]
        initial = true.copy()
        initial[1] = true[1] @ sim3_exp([.02, -.03, .01, .1, .2, -.1, .03])
        initial[2] = true[2] @ sim3_exp([-.04, .02, .01, -.1, -.2, .1, -.04])
        recovered, status = solve_sim3_graph(initial, edges)
        assert_allclose(recovered, true, atol=2e-6)
        assert_allclose(recovered[0], initial[0], atol=0)
        self.assertLess(status["sum_squared_whitened_residual"], 1e-12)

    def test_full_sim3_solver_rejects_invalid_information(self):
        initial = np.stack([np.eye(4), np.eye(4)])
        with self.assertRaises(ValueError):
            solve_sim3_graph(initial, [Edge(0, 1, np.eye(4), -np.eye(7))])
        with self.assertRaises(ValueError):
            solve_sim3_graph(initial, [])


if __name__ == "__main__":
    unittest.main()
