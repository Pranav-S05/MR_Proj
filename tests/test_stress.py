"""Checks for the final-submission false-loop tests (mr_cpu.stress)."""
import unittest

import numpy as np

from mr_cpu.cache import find_loop_pairs
from mr_cpu.stress import _with_fakes, fake_loops


class LoopRetrievalTests(unittest.TestCase):
    def test_neighbours_fill_the_top_k_unless_skipped(self):
        # descriptors that change slowly along the drive (random anchors every 10 frames, linearly blended: a frame's
        # neighbours are the most similar frames, distant places nearly unrelated); frames 75-85 revisit 5-15 with
        # noise, so a revisit is similar (~0.9) but less similar than the frame's own neighbours
        rng = np.random.default_rng(0)
        anchors = rng.normal(size=(11, 64))
        t = np.arange(100) / 10
        walk = anchors[t.astype(int)] * (1 - t % 1)[:, None] + anchors[t.astype(int) + 1] * (t % 1)[:, None]
        offset = rng.normal(size=64)  # one change for the whole revisit (other light, other parked cars)
        walk[75:86] = walk[5:16] + .6 * np.linalg.norm(walk[5:16], axis=1, keepdims=True) * offset / np.linalg.norm(offset)
        d = walk / np.linalg.norm(walk, axis=1, keepdims=True)
        revisit_sims = [float(d[k] @ d[k + 70]) for k in range(5, 16)]
        self.assertTrue(min(revisit_sims) > .8)
        thr = min(revisit_sims) - .01
        def revisit(pairs, lo, hi):
            return [(a, b) for a, b, _ in pairs if lo <= a <= hi and 5 <= b <= 15]
        # frames whose 3 neighbours on both sides are inside the revisit never retrieve it with the stock rule
        # (edge frames can: one side's neighbours are unrelated, which frees a top-5 slot)
        self.assertEqual(revisit(find_loop_pairs(d, threshold=thr), 78, 82), [])
        skipped = find_loop_pairs(d, threshold=thr, skip_neighbours=True)
        self.assertTrue(revisit(skipped, 75, 85))
        self.assertTrue(all(a - b > 10 for a, b, _ in skipped))


class FakeLoopTests(unittest.TestCase):
    def test_fake_loops_are_distant_and_reproducible(self):
        pos = np.c_[np.arange(60) * 5.0, np.zeros(60), np.zeros(60)]  # windows 5 m apart on a line
        a = fake_loops(pos, 8, np.random.default_rng(3))
        b = fake_loops(pos, 8, np.random.default_rng(3))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        i, j, z = a
        self.assertTrue(np.all(j - i >= 10))
        self.assertTrue(np.all(np.linalg.norm(pos[i] - pos[j], axis=1) >= 50))
        self.assertEqual(z.shape, (8, 4, 4))

    def test_fakes_get_the_most_trusted_real_loop_information(self):
        table = {"i": np.array([0, 1, 0, 2]), "j": np.array([1, 2, 2, 3]), "z": np.tile(np.eye(4), (4, 1, 1)),
                 "kind": np.array([0, 0, 1, 1])}
        info = np.stack([np.eye(7) * w for w in (100., 100., 3., 7.)])
        i, j, z, inf, kind, fake = _with_fakes(table, info, np.array([0]), np.array([3]), np.eye(4)[None])
        np.testing.assert_allclose(inf[-1], np.eye(7) * 7.)
        self.assertEqual(kind[-1], 1)
        self.assertEqual(fake.tolist(), [False] * 4 + [True])


if __name__ == "__main__":
    unittest.main()
