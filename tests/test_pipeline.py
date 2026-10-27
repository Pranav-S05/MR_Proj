"""Checks for the experiment pipeline (CPU). Run: python -m unittest discover -s tests"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from mr_cpu import backend, cache, evaluation, geometry, kitti, labels, lie
from mr_cpu.alignment import align_windows, irls_sim3, vggtlong_sim3
from mr_cpu.weighting import BLOCKS, Fitted

ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "data" / "kitti"
HAVE_KITTI = (KITTI / "poses" / "07.txt").exists() and (KITTI / "sequences" / "07" / "calib.txt").exists()


def circle_graph(n=60, seed=0, scale_noise=True):
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    gt = lie.exp(np.c_[np.zeros((n, 1)), np.zeros((n, 1)), th, np.zeros((n, 4))])
    gt[:, :3, 3] = np.c_[30 * np.cos(th), 30 * np.sin(th), np.zeros(n)]
    if scale_noise:
        gt[:, :3, :3] *= np.exp(rng.normal(0, .3, n))[:, None, None]
    return gt, rng


class LieTests(unittest.TestCase):
    def test_batched_exp_log_match_expm(self):
        rng = np.random.default_rng(1)
        xi = rng.normal(0, .6, (200, 7))
        xi[:10, :3] = 0
        xi[10:20, 6] = 0
        xi[20:30, :3] *= 3.0 / np.linalg.norm(xi[20:30, :3], axis=1, keepdims=True)  # near pi
        m = lie.exp(xi)
        for k in range(0, 200, 7):
            assert_allclose(m[k], geometry.sim3_exp(xi[k]), atol=1e-10)
        assert_allclose(lie.log(m), xi, atol=1e-10)
        assert_allclose(lie.inv(m) @ m, np.tile(np.eye(4), (200, 1, 1)), atol=1e-12)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.gt, rng = circle_graph()
        n = len(self.gt)
        self.i, self.j = np.arange(n - 1), np.arange(1, n)
        z = lie.inv(self.gt[self.i]) @ self.gt[self.j]
        self.z = z @ lie.exp(np.c_[rng.normal(0, .004, (n - 1, 3)), rng.normal(0, .05, (n - 1, 3)), rng.normal(0, .004, n - 1)])
        init = [self.gt[0]]
        for k in range(n - 1):
            init.append(init[-1] @ self.z[k])
        self.init = np.stack(init)

    def test_chain_is_unchanged_by_any_weights(self):
        info = np.einsum("n,ij->nij", np.exp(np.random.default_rng(2).normal(0, 2, len(self.i))), np.eye(7))
        poses, rep = backend.solve(self.init, self.i, self.j, self.z, info)
        assert_allclose(poses, self.init, atol=1e-12)

    def test_loops_reduce_error_in_both_residual_modes(self):
        li, lj = np.array([0, 5, 12]), np.array([59, 40, 45])
        zl = lie.inv(self.gt[li]) @ self.gt[lj]
        i, j, z = np.r_[self.i, li], np.r_[self.j, lj], np.concatenate([self.z, zl])
        info = np.tile(np.eye(7), (len(i), 1, 1))
        before = evaluation.ate(self.init[:, :3, 3], self.gt[:, :3, 3])
        for mode in ("ours", "upstream"):
            poses, _ = backend.solve(self.init, i, j, z, info, mode=mode)
            self.assertLess(evaluation.ate(poses[:, :3, 3], self.gt[:, :3, 3]), .5 * before)

    def test_consistent_graph_is_solved_exactly(self):
        li, lj = np.array([0, 10]), np.array([59, 30])
        z_true = lie.inv(self.gt[np.r_[self.i, li]]) @ self.gt[np.r_[self.j, lj]]
        poses, rep = backend.solve(self.init, np.r_[self.i, li], np.r_[self.j, lj], z_true, np.tile(np.eye(7), (len(z_true), 1, 1)))
        self.assertLess(rep["cost"], 1e-12)
        assert_allclose(poses[:, :3, 3], self.gt[:, :3, 3], atol=1e-5)

    def test_dcs_rejects_a_strong_false_loop(self):
        bad = lie.exp(np.array([[0, 0, .6, 2., 0, 0, 0]]))
        i, j, z = np.r_[self.i, 10], np.r_[self.j, 40], np.concatenate([self.z, bad])
        info = np.tile(np.eye(7), (len(i), 1, 1))
        info[-1] *= 100
        mask = np.r_[np.zeros(len(self.i), bool), True]
        folded, _ = backend.solve(self.init, i, j, z, info, kernel_on=mask)
        robust, rep = backend.solve(self.init, i, j, z, info, kern="dcs", kparam=1., kernel_on=mask)
        self.assertLess(rep["edge_weight"][-1], 1e-3)
        self.assertLess(evaluation.ate(robust[:, :3, 3], self.gt[:, :3, 3]), evaluation.ate(folded[:, :3, 3], self.gt[:, :3, 3]))

    def test_kernel_weights_are_derivatives_of_costs(self):
        q = np.linspace(.01, 50, 200)
        for name, p in (("huber", 2.), ("dcs", 3.)):
            rho, w = backend.kernel(q, name, p)
            assert_allclose(np.gradient(rho, q)[1:-1], w[1:-1], rtol=2e-2, atol=1e-3)


class AlignmentTests(unittest.TestCase):
    def test_irls_recovers_transform_despite_outliers(self):
        rng = np.random.default_rng(3)
        src = rng.normal(0, 1, (5000, 3))
        r = Rotation.random(random_state=4).as_matrix()
        tgt = 1.7 * src @ r.T + [1, -2, .5] + rng.normal(0, .005, src.shape)
        tgt[:500] += rng.normal(0, 3, (500, 3))
        s, r_hat, t, _ = irls_sim3(src, tgt, np.ones(len(src)))
        # VGGT-Long's spread-ratio scale is slightly biased upward by outliers/noise (~0.3% here)
        self.assertLess(abs(s / 1.7 - 1), .01)
        self.assertLess(np.abs(r_hat - r).max(), 1e-2)

    def test_spread_ratio_scale_like_vggt_long(self):
        rng = np.random.default_rng(5)
        src = rng.normal(0, 1, (1000, 3))
        s, _, _ = vggtlong_sim3(src, 2.5 * src, np.ones(1000))
        self.assertAlmostEqual(s, 2.5, places=10)

    def test_align_windows_maps_window2_into_window1(self):
        rng = np.random.default_rng(6)
        p1 = rng.normal(0, 1, (3, 8, 10, 3)) + [0, 0, 5]
        z_true = lie.from_srt(.8, Rotation.random(random_state=7).as_matrix(), [.3, -.2, .1])[0]
        p2 = geometry.transform_points(geometry.inverse(z_true), p1.reshape(-1, 3)).reshape(p1.shape)
        conf = np.full(p1.shape[:3], 5.)
        c2w = np.tile(np.eye(4), (3, 1, 1))
        z, f = align_windows(p1, conf, p2, conf, c2w)
        assert_allclose(z, z_true, atol=1e-9)
        self.assertEqual(f["n_points"], 240)


class CacheTests(unittest.TestCase):
    def test_window_layout_matches_vggt_long(self):
        for n, w, o in ((100, 32, 16), (1101, 32, 16), (4541, 75, 30), (20, 32, 16)):
            r = cache.window_ranges(n, w, o)
            self.assertEqual(r[0][0], 0)
            self.assertEqual(r[-1][1], n)
            for a, b in zip(r, r[1:]):
                self.assertEqual(a[1] - b[0], o)  # consecutive windows share `overlap` frames

    def test_loop_retrieval_finds_revisit_and_skips_neighbours(self):
        rng = np.random.default_rng(8)
        pos = np.r_[np.arange(100), np.arange(100)[::-1][:30]].astype(float)  # drive out, come back
        omega, phase = rng.normal(0, .3, 64), rng.uniform(0, 6.28, 64)
        desc = np.cos(pos[:, None] * omega + phase)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True)
        pairs = cache.find_loop_pairs(desc, threshold=.9)
        self.assertTrue(pairs)
        for later, earlier, s in pairs:
            self.assertGreater(later - earlier, 10)
            self.assertLess(abs(pos[later] - pos[earlier]), 3)
        ranges = cache.window_ranges(130, 32, 16)
        loops = cache.loop_windows(ranges, pairs)
        self.assertTrue(loops)
        for lw in loops:  # each half lies inside its window; at most 20 frames (fewer in a short last window)
            for key, chunk in (("range_a", lw["chunk_a"]), ("range_b", lw["chunk_b"])):
                a, b = lw[key]
                self.assertTrue(ranges[chunk][0] <= a < b <= ranges[chunk][1])
                self.assertLessEqual(b - a, 20)


class LabelTests(unittest.TestCase):
    def test_gauge_fit_recovers_known_similarity(self):
        pred =np.stack([lie.from_srt(1., Rotation.from_euler("y", .03 * k).as_matrix(), [.1 * k, 0, .5 * k])[0] for k in range(16)])
        a_true = lie.from_srt(14., Rotation.random(random_state=10).as_matrix(), [5, 1, -3])[0]
        gt = a_true @ pred
        gt[:, :3, :3] /= 14.
        a, info = labels.fit_gauge(pred, gt)
        assert_allclose(a, a_true, atol=1e-9)
        self.assertTrue(info["valid"])
        stopped = np.tile(gt[:1], (16, 1, 1))
        self.assertFalse(labels.fit_gauge(pred, stopped)[1]["valid"])

    def test_edge_error_zero_for_exact_measurement(self):
        g = lie.exp(np.random.default_rng(11).normal(0, .3, (3, 7)))
        z = lie.inv(g[[0, 1]]) @ g[[1, 2]]
        e, ok = labels.edge_errors(g, np.ones(3, bool), np.array([0, 1]), np.array([1, 2]), z)
        assert_allclose(e, 0, atol=1e-12)


class EvaluationTests(unittest.TestCase):
    def test_last_window_wins_like_vggt_long(self):
        c2w = np.tile(np.eye(4), (4, 1, 1))
        a, b = np.eye(4), lie.from_srt(2., np.eye(3), [1, 0, 0])[0]
        out = evaluation.assemble([a, b], [np.arange(0, 3), np.arange(2, 4)], [c2w[:3], c2w[:2]], 4)
        assert_allclose(out[2, :3, 3], [1, 0, 0])

    def test_relative_errors_vanish_for_scaled_truth(self):
        gt, _ = circle_graph(400, scale_noise=False)
        gt[:, :3, 3] *= 10
        est = gt.copy()
        est[:, :3, 3] = est[:, :3, 3] * .1 + 3
        r = evaluation.kitti_relative_errors(est, gt)
        self.assertLess(r["t_rel_percent"], 1e-6)
        self.assertLess(evaluation.ate(est[:, :3, 3], gt[:, :3, 3]), 1e-9)

    @unittest.skipUnless(shutil.which("evo_ape"), "evo not installed")
    def test_ate_matches_evo(self):
        gt, rng = circle_graph(300, scale_noise=False)
        est = gt.copy()
        est[:, :3, 3] = .5 * est[:, :3, 3] + np.cumsum(rng.normal(0, .05, (300, 3)), 0)
        self.assertAlmostEqual(evaluation.ate(est[:, :3, 3], gt[:, :3, 3]), evaluation.evo_ate(est, gt), places=6)


@unittest.skipUnless(HAVE_KITTI, "KITTI poses/calibration not downloaded")
class KittiAndEndToEndTests(unittest.TestCase):
    def test_cam2_offset_and_poses(self):
        calib = kitti.read_calib(KITTI / "sequences" / "07" / "calib.txt")
        b = kitti.cam2_offset(calib)
        self.assertTrue(.05 < b[0] < .07)
        gt = kitti.gt_cam2_poses(KITTI, "07")
        self.assertEqual(len(gt), 1101)
        self.assertEqual(kitti.model_input_size(370, 1226), (154, 518))
        self.assertEqual(kitti.model_input_size(376, 1241), (154, 518))

    def test_synthetic_end_to_end_small(self):
        from mr_cpu import edges, synthetic
        tmp = Path(tempfile.mkdtemp())
        try:
            synthetic.make_cache(KITTI, "07", tmp / "07", window=16, overlap=8, stride=8, n_frames=160, log=None)
            table, frames, poses = edges.build(tmp / "07", log=None)
            self.assertEqual(int((table["kind"] == 0).sum()), len(frames) - 1)
            gt = kitti.gt_cam2_poses(KITTI, "07")[:160]
            gauges, infos = labels.window_gauges(frames, poses, gt)
            err, ok = labels.edge_errors(gauges, [i["valid"] for i in infos], table["i"], table["j"], table["z"])
            self.assertLess(np.degrees(np.median(np.linalg.norm(err[ok, :3], axis=1))), 2.)
            fitted = Fitted.fit([table], [err], [ok])
            info = fitted.information("calibrated", table)
            seam = table["kind"] == 0
            ref = fitted.information("uniform", table)
            self.assertAlmostEqual(np.trace(info[seam], axis1=1, axis2=2).mean() / np.trace(ref[seam], axis1=1, axis2=2).mean(), 1, places=6)
            for b in BLOCKS:
                self.assertTrue(np.all(np.isfinite(fitted.variances("power_law", table)[b])))
            from mr_cpu.pipeline import _chain
            est = evaluation.assemble(_chain(table), frames, poses, 160)
            self.assertLess(evaluation.ate(est[:, :3, 3], gt[:, :3, 3]), 5.)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class AuditAndUpstreamFormatTests(unittest.TestCase):
    def test_audit_keeps_every_lidar_pixel_unless_capped(self):
        from mr_cpu import depth_audit, synthetic
        tmp = Path(tempfile.mkdtemp())
        try:
            synthetic.make_cache(KITTI, "07", tmp / "07", window=16, overlap=8, stride=8, n_frames=48, log=None)
            fn = depth_audit.cache_lidar_fn(tmp / "07")
            full = depth_audit.collect(tmp / "07", fn, None, log=None)
            capped = depth_audit.collect(tmp / "07", fn, 5, log=None)
            n_frames = len(np.unique(np.c_[full["window"], full["frame"]], axis=0))
            self.assertEqual(len(capped["lidar"]), 5 * n_frames)
            self.assertGreater(len(full["lidar"]), len(capped["lidar"]))
            self.assertEqual(full["d_conf"].dtype, np.float16)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reads_vggt_long_ply(self):
        from mr_cpu.qualitative import read_ply
        xyz = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        rgb = np.random.default_rng(1).integers(0, 256, (50, 3)).astype(np.uint8)
        tmp = Path(tempfile.mkdtemp())
        try:
            header = ["ply", "format binary_little_endian 1.0", "element vertex 50", "property float x",
                      "property float y", "property float z", "property uchar red", "property uchar green",
                      "property uchar blue", "end_header"]  # VGGT-Long's write_ply_header
            rec = np.empty(50, np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
            rec["xyz"], rec["rgb"] = xyz, rgb
            (tmp / "p.ply").write_bytes("\n".join(header).encode() + b"\n" + rec.tobytes())
            x, c = read_ply(tmp / "p.ply")
            assert_allclose(x, xyz)
            np.testing.assert_array_equal(c, rgb)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
