"""On-disk cache of frontend outputs, window layout, and VGGT-Long's loop-candidate logic.

Cache layout (written by mr_gpu/run_vggt.py or mr_cpu/synthetic.py):
  <cache>/manifest.json
  <cache>/windows/w0000.npz ...   one file per sliding window
  <cache>/loops/l0000.npz ...     one file per loop window (frames of range_a + range_b)
  <cache>/descriptors.npy         (n_frames, D) L2-normalised global image descriptors
  <cache>/loop_pairs.json         retrieved loop pairs (idx_later, idx_earlier, similarity)
Each window/loop npz holds:
  frame_ids (n,) int; c2w (n,4,4) float64 camera-to-window poses (unit rotations);
  intrinsics (n,3,3) in model-input pixels; points (n,h,w,3) float32 point-head map in window
  coordinates; points_conf, depth, depth_conf (n,h,w) float16; stride (pixel subsampling of the
  model-input grid: cached pixel (r, c) = model pixel (stride*r, stride*c)).
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("frame_ids", "c2w", "intrinsics", "points", "points_conf", "depth", "depth_conf")


def window_ranges(n_frames, window, overlap):
    """VGGT-Long's chunking: [(start, end)), step = window - overlap, last window may be short."""
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    if n_frames <= window:
        return [(0, n_frames)]
    step = window - overlap
    count = (n_frames - overlap + step - 1) // step
    return [(k * step, min(k * step + window, n_frames)) for k in range(count)]


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


class Cache:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.ranges = [tuple(r) for r in self.manifest["windows"]]
        self.loop_meta = self.manifest.get("loop_windows", [])

    def window(self, k):
        return load_npz(self.root / "windows" / f"w{k:04d}.npz")

    def loop(self, k):
        return load_npz(self.root / "loops" / f"l{k:04d}.npz")

    def window_cameras(self, k):
        """Only the small arrays (frame ids and poses) of window k."""
        with np.load(self.root / "windows" / f"w{k:04d}.npz", allow_pickle=False) as d:
            return d["frame_ids"], d["c2w"]


# ---------------- loop candidates (port of VGGT-Long LoopModels/LoopModel.py) ----------------

def find_loop_pairs(descriptors, threshold=0.85, top_k=5, min_gap=10, nms=25):
    """Top-k inner-product retrieval + threshold + NMS, as in VGGT-Long's LoopDetector.

    descriptors: (n, D) L2-normalised. Returns [(later_idx, earlier_idx, similarity)].
    """
    d = np.asarray(descriptors, np.float32)
    sims = d @ d.T
    np.fill_diagonal(sims, -np.inf)
    k = min(top_k, len(d) - 1)
    order = np.argsort(-sims, axis=1)[:, :k]
    pairs = set()
    for a in range(len(d)):
        for b in order[a]:
            s = float(sims[a, b])
            if s > threshold and abs(a - int(b)) > min_gap:
                pairs.add((min(a, int(b)), max(a, int(b)), round(s, 6)))
    pairs = sorted(pairs, key=lambda p: p[2], reverse=True)
    if nms > 0 and pairs:
        pairs = _nms(pairs, nms)
    return [(b, a, s) for a, b, s in pairs]  # VGGT-Long returns (max, min, score)


def _nms(loops, radius):
    max_frame = max(max(a, b) for a, b, _ in loops)
    kept, suppressed = [], set()
    for a, b, s in sorted(loops, key=lambda x: x[2], reverse=True):
        if a in suppressed or b in suppressed:
            continue
        kept.append((a, b, s))
        suppressed.update(range(max(0, a - radius), min(a + radius + 1, b)))
        suppressed.update(range(max(a + 1, b - radius), min(b + radius + 1, max_frame + 1)))
    return kept


def _chunk_of(ranges, idx):
    starts = [r[0] for r in ranges]
    pos = bisect.bisect_right(starts, idx) - 1
    if pos < 0 or not (ranges[pos][0] <= idx <= ranges[pos][1]):
        raise ValueError(f"Index {idx} not in any window")
    return pos


def _frame_range(chunk, idx, half):
    begin, end = chunk
    size = 2 * half
    if idx - half < begin:
        return begin, min(end, begin + size)
    if idx + half > end:
        return max(begin, end - size), end
    return idx - half, idx + half


def loop_windows(ranges, pairs, half_window=10):
    """VGGT-Long's process_loop_list + remove_duplicates.

    Returns [{chunk_a, range_a, chunk_b, range_b, similarity}], a = window of the later frame.
    The loop window's images are frames range_a followed by frames range_b.
    """
    out, seen = [], set()
    for later, earlier, sim in pairs:
        try:
            ca = _chunk_of(ranges, later)
            cb = _chunk_of(ranges, earlier)
        except ValueError:
            continue
        if ca == cb or (ca, cb) in seen:
            continue
        seen.add((ca, cb))
        out.append({"chunk_a": ca, "range_a": list(_frame_range(ranges[ca], later, half_window)),
                    "chunk_b": cb, "range_b": list(_frame_range(ranges[cb], earlier, half_window)),
                    "pair": [int(later), int(earlier)], "similarity": float(sim)})
    return out
