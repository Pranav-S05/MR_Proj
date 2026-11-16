"""Scale study, GPU pass: VGGT over overlapping windows of one KITTI drive, compact outputs only.

Unlike mr_gpu/run_vggt.py nothing big is cached: seams and loops are aligned in memory with the same code
as the main project (mr_cpu.alignment.align_windows, VGGT-Long's estimator), so the edge table equals the
one from `pipeline --steps edges` when --ref first.

  python scripts/gpu_run.py --seq 07 --ref first  --out results/final/runs/first_w30_o15/07
  python scripts/gpu_run.py --seq 07 --ref middle --out results/final/runs/middle_w30_o15/07

--ref picks the frame that VGGT treats as its reference (its first input): the window's first frame
(VGGT-Long), its middle frame or its last frame. Frames are given to the model as [ref, others in order] and
all outputs are put back into time order, so everything downstream is unchanged.

Outputs in --out:
  edges.npz     same format as mr_cpu.edges.save (seams + loops, features, window frames and cameras)
  extras.npz    per-window intrinsics and median depths; per-seam depth-ratio statistics and Umeyama scale
  overlaps/sNNNN.npz   pixel-subsampled maps of both windows on the shared frames (for trying other estimators)
  manifest.json config, timing, loops
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mr_cpu import edges as edges_mod, kitti, lie  # noqa: E402
from mr_cpu.alignment import align_windows, camera_depths, irls_sim3  # noqa: E402
from mr_cpu.cache import find_loop_pairs, loop_windows, window_ranges  # noqa: E402
from mr_gpu import run_vggt  # noqa: E402

KEYS = ("w2c", "intrinsics", "points", "points_conf", "depth", "depth_conf")


def infer_ref(model, cls, paths, device, dtype, ref):
    """VGGT with a chosen reference frame; outputs returned in the given (time) order."""
    n = len(paths)
    r = {"first": 0, "middle": n // 2, "last": n - 1}[ref]
    order = [r] + [k for k in range(n) if k != r]
    out, dt, peak = run_vggt.infer(model, cls, [paths[k] for k in order], device, dtype)
    back = np.argsort(order)
    for key in KEYS:
        out[key] = out[key][back]
    out["c2w"] = np.linalg.inv(out["w2c"])
    out["ref_index"] = r
    return out, dt, peak


def subset(out, idx):
    return {k: out[k][idx] for k in ("points", "points_conf", "depth", "depth_conf", "c2w")}


def _terciles(lr, d, ok):
    """Median log ratio overall and in the near / middle / far third of the valid pixels, and its MAD."""
    if ok.sum() < 30:
        return [np.nan] * 5
    x, dd = lr[ok], d[ok]
    q1, q2 = np.quantile(dd, [1 / 3, 2 / 3])
    med = np.median(x)
    return [med, np.median(x[dd <= q1]), np.median(x[(dd > q1) & (dd <= q2)]), np.median(x[dd > q2]),
            np.median(np.abs(x - med))]


def seam_job(a, b, seed, sub_path, stride, overlap):
    """a, b: the two windows' maps on the shared frames. Returns (Z, features, extras)."""
    z, f = align_windows(a["points"], a["points_conf"], b["points"], b["points_conf"], b["c2w"], seed=seed)
    n = len(a["points"])
    ex = {"depth_lr": np.full((overlap, 5), np.nan), "point_lr": np.full((overlap, 5), np.nan)}
    # log ratio window-1 depth / window-2 depth of the same pixel, per shared frame (depth head, then point head)
    thr = min(np.median(a["depth_conf"]), np.median(b["depth_conf"])) * .1
    pd1, pd2 = camera_depths(a["points"], a["c2w"]), camera_depths(b["points"], b["c2w"])
    pthr = min(np.median(a["points_conf"]), np.median(b["points_conf"])) * .1
    for k in range(n):
        d1, d2 = a["depth"][k].astype(np.float64), b["depth"][k].astype(np.float64)
        ok = (a["depth_conf"][k] > thr) & (b["depth_conf"][k] > thr) & (d1 > 0) & (d2 > 0)
        ex["depth_lr"][k] = _terciles(np.log(np.where(ok, d1 / np.where(ok, d2, 1), 1)), d2, ok)
        ok = (a["points_conf"][k] > pthr) & (b["points_conf"][k] > pthr) & (pd1[k] > 0) & (pd2[k] > 0)
        ex["point_lr"][k] = _terciles(np.log(np.where(ok, pd1[k] / np.where(ok, pd2[k], 1), 1)), pd2[k], ok)
    # least-squares (Umeyama) scale with VGGT-Long's final robust weights, for comparison with its spread ratio
    c1, c2 = a["points_conf"].astype(np.float64), b["points_conf"].astype(np.float64)
    valid = (c1 > pthr) & (c2 > pthr)
    tgt, src = a["points"][valid].astype(np.float64), b["points"][valid].astype(np.float64)
    w = np.sqrt(c1[valid] * c2[valid])
    s, r, t, huber = irls_sim3(src, tgt, w)
    fw = w * huber
    fw = fw / fw.sum()
    xs, xt = src - fw @ src, tgt - fw @ tgt
    ex["log_s_spread"] = float(np.log(s))
    ex["log_s_umeyama"] = float(np.log((fw * ((xs @ r.T) * xt).sum(1)).sum() / (fw * (xs * xs).sum(1)).sum()))
    if sub_path is not None:
        q = slice(None, None, stride)
        np.savez(sub_path, **{f"{name}{m}": w_[key][:, q, q].astype(dt) for m, w_ in ((1, a), (2, b))
                              for name, key, dt in (("p", "points", np.float32), ("c", "points_conf", np.float16),
                                                    ("d", "depth", np.float16), ("dc", "depth_conf", np.float16))},
                 c2w1=a["c2w"], c2w2=b["c2w"])
    return z, f, ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ref", choices=["first", "middle", "last"], default="first")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kitti", default=str(ROOT / "data" / "kitti"))
    ap.add_argument("--vggt-long", default=str(ROOT / "external" / "VGGT-Long"))
    ap.add_argument("--weights")
    ap.add_argument("--desc-dir", default=str(ROOT / "results" / "final" / "runs" / "descriptors"), help="SALAD descriptors, shared by variants")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-loops", action="store_true")
    ap.add_argument("--sub-stride", type=int, default=4, help="pixel stride of the saved overlap maps (0 = do not save)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import torch
    out_dir = Path(a.out)
    if (out_dir / "edges.npz").exists():
        print(f"{out_dir} already done")
        return
    (out_dir / "overlaps").mkdir(parents=True, exist_ok=True)
    vl_root = run_vggt.setup_vggt_long(a.vggt_long)
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    paths = kitti.image_paths(a.kitti, a.seq)
    if a.max_frames:
        paths = paths[:a.max_frames]
    n = len(paths)
    if not n:
        sys.exit(f"No images for sequence {a.seq}")
    ranges = window_ranges(n, a.window, a.overlap)
    t_start = time.time()

    # loops first, so the main pass knows which window parts the loop alignments will need
    loops = []
    if not a.no_loops:
        dpath = Path(a.desc_dir) / f"{a.seq}{'' if not a.max_frames else '_' + str(n)}.npy"
        if not dpath.exists():
            dpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(dpath, run_vggt.salad_descriptors(vl_root, paths, device).astype(np.float16))
            torch.cuda.empty_cache()
        desc = np.load(dpath).astype(np.float32)
        pairs = find_loop_pairs(desc, threshold=0.85, top_k=5, min_gap=10, nms=25)
        loops = loop_windows(ranges, pairs, half_window=10)
        print(f"{len(pairs)} loop pairs -> {len(loops)} loop windows", flush=True)
    need = {}  # window index -> list of (loop index, 'a' | 'b', frame range)
    for m, lw in enumerate(loops):
        need.setdefault(lw["chunk_a"], []).append((m, "a", lw["range_a"]))
        need.setdefault(lw["chunk_b"], []).append((m, "b", lw["range_b"]))
    stash = {}

    model, cls_store, weights = run_vggt.load_model(vl_root, a.weights, device)
    pool = ThreadPoolExecutor(a.workers)
    jobs, frames, poses, intr, med_depth, times, peaks = [], [], [], [], [], [], []
    prev = None
    for k, (s, e) in enumerate(ranges):
        out, dt, peak = infer_ref(model, cls_store, paths[s:e], device, dtype, a.ref)
        times.append(dt)
        peaks.append(peak)
        fid = np.arange(s, e)
        frames.append(fid)
        poses.append(out["c2w"])
        intr.append(out["intrinsics"].astype(np.float64))
        keep = out["depth_conf"] > np.median(out["depth_conf"])
        med_depth.append(np.array([np.median(out["depth"][i][keep[i]]) if keep[i].any() else np.nan for i in range(len(fid))]))
        for m, side, (r0, r1) in need.get(k, []):
            stash[(m, side)] = subset(out, np.arange(r0, r1) - s)
        if prev is not None:
            ps, pe, pout = prev
            lo, hi = max(ps, s), min(pe, e)
            if hi <= lo:
                sys.exit(f"windows {k - 1} and {k} do not overlap")
            sub = out_dir / "overlaps" / f"s{k - 1:04d}.npz" if a.sub_stride else None
            jobs.append(pool.submit(seam_job, subset(pout, np.arange(lo, hi) - ps), subset(out, np.arange(lo, hi) - s),
                                    k - 1, sub, a.sub_stride, a.overlap))
            while sum(not j.done() for j in jobs) > 2 * a.workers:  # do not let maps pile up in memory
                time.sleep(.2)
        prev = (s, e, out)
        if k % 20 == 0 or k == len(ranges) - 1:
            print(f"window {k + 1}/{len(ranges)} frames {s}-{e - 1}: {dt:.2f}s, peak {peak:.2f} GB", flush=True)
    del prev
    rows, extras = [], []
    for k, j in enumerate(jobs):
        z, f, ex = j.result()
        f.update(kind=0, i=k, j=k + 1, overlap=len(set(frames[k].tolist()) & set(frames[k + 1].tolist())))
        rows.append((z, f))
        extras.append(ex)
    print(f"{len(rows)} seams aligned ({time.time() - t_start:.0f} s so far)", flush=True)

    for m, lw in enumerate(loops):  # loop windows keep VGGT-Long's first-frame reference in every variant
        ra, rb = range(*lw["range_a"]), range(*lw["range_b"])
        ids = np.r_[np.arange(*lw["range_a"]), np.arange(*lw["range_b"])]
        out, dt, _ = infer_ref(model, cls_store, [paths[f] for f in ids], device, dtype, "first")
        na = len(ra)
        wa, wb = stash.pop((m, "a")), stash.pop((m, "b"))
        t_al, fa = align_windows(wa["points"], wa["points_conf"], out["points"][:na], out["points_conf"][:na],
                                 out["c2w"][:na], seed=10_000 + m)
        t_bl, fb = align_windows(wb["points"], wb["points_conf"], out["points"][na:], out["points_conf"][na:],
                                 out["c2w"][na:], seed=20_000 + m)
        z = t_bl @ lie.inv(t_al[None])[0]
        f = edges_mod._combine(fa, fb)
        f["conf_hist"], f["conf_hist2"] = fa["conf_hist"], fb["conf_hist"]
        f["char_length"] = float(np.median(np.abs(edges_mod._depths(wa))))
        f.update(kind=1, i=lw["chunk_b"], j=lw["chunk_a"], overlap=len(ra) + len(rb), similarity=lw.get("similarity", np.nan))
        rows.append((z, f))
        print(f"loop {m + 1}/{len(loops)}: frames {lw['range_a']} + {lw['range_b']}", flush=True)

    table = {"z": np.stack([r[0] for r in rows]),
             "i": np.array([r[1]["i"] for r in rows]), "j": np.array([r[1]["j"] for r in rows]),
             "kind": np.array([r[1]["kind"] for r in rows]), "overlap": np.array([r[1]["overlap"] for r in rows]),
             "similarity": np.array([r[1].get("similarity", np.nan) for r in rows])}
    for key in edges_mod.FEATURES:
        table["f_" + key] = np.array([r[1][key] for r in rows], float)
    table["conf_hist"] = np.stack([r[1]["conf_hist"] for r in rows])
    table["conf_hist2"] = np.stack([r[1].get("conf_hist2", np.zeros_like(r[1]["conf_hist"])) for r in rows])
    np.savez(out_dir / "extras.npz", window_intrinsics=np.concatenate(intr), window_median_depth=np.concatenate(med_depth),
             seam_depth_lr=np.stack([x["depth_lr"] for x in extras]), seam_point_lr=np.stack([x["point_lr"] for x in extras]),
             seam_log_s_spread=np.array([x["log_s_spread"] for x in extras]),
             seam_log_s_umeyama=np.array([x["log_s_umeyama"] for x in extras]))
    manifest = {"sequence": a.seq, "ref": a.ref, "window": a.window, "overlap": a.overlap, "n_frames": n,
                "windows": [list(r) for r in ranges], "loop_windows": loops, "dtype": str(dtype),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "checkpoint": str(weights),
                "sec_per_window": float(np.mean(times)), "peak_gpu_mem_gb": float(np.max(peaks)),
                "wall_seconds": time.time() - t_start, "sub_stride": a.sub_stride}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    edges_mod.save(out_dir / "edges.npz", table, frames, poses)  # written last: marks the run as complete
    print(f"done: {out_dir} ({manifest['wall_seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
