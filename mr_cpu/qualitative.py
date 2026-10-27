"""Qualitative material: what VGGT predicts, where it fails, and what the pose graph makes of it.

Two phases, because full-resolution caches are too large to keep for every sequence:
  extract(cfg, seq)  (pipeline step `qual`) needs the cache, the images and the LiDAR of one sequence.
                     Small arrays go to <run>/<seq>/qual/, figures to <run>/figures/qualitative/,
                     the per-loop table to <run>/tables/loops_<seq>.csv.
  render(cfg)        (called by the report, after `solve`) point clouds placed with the solved window
                     poses, per-frame trajectory error, upstream VGGT-Long's own point cloud.
  python -m mr_cpu.qualitative ply2npz in.ply out.npz   subsample an upstream VGGT-Long PLY.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import edges, kitti
from .cache import Cache
from .evaluation import sim3_align
from .geometry import transform_points
from .labels import fit_gauge

# VGGT-Long's point-cloud export: keep points with conf >= 0.75 * mean(conf) of the window, then sample.
EXPORT_CONF_COEF = 0.75
SAMPLE_RATIO = 0.004   # of the confident points per window (VGGT-Long writes 0.015; we only plot them)
MAX_PLOT_POINTS = 1_500_000


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rgb_on_grid(path, model_hw, stride):
    """Image as VGGT sees it: resized to the model input (KITTI: no crop), then the cache's pixel stride."""
    from PIL import Image
    h, w = model_hw
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB").resize((w, h), Image.BICUBIC))
    return rgb[::stride, ::stride]


def _window_with_frame_centred(ranges, f):
    best = min((abs((s + e - 1) / 2 - f), k) for k, (s, e) in enumerate(ranges) if s <= f < e)
    return best[1]


def _qdir(cfg, seq):
    d = Path(cfg["run_dir"]) / seq / "qual"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _figdir(cfg):
    d = Path(cfg["run_dir"]) / "figures" / "qualitative"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ extract (needs cache, images, LiDAR)

def extract(cfg, seq, seed=0):
    cache = Cache(Path(cfg["cache_root"]) / seq)
    man = cache.manifest
    if man.get("engine") == "synthetic" or not (Path(cfg["kitti_root"]) / "sequences" / seq / "image_2").exists():
        print(f"  [{seq}] no images (or synthetic cache); skipping qualitative extraction")
        return
    paths = kitti.image_paths(cfg["kitti_root"], seq)
    stride, model_hw = int(man["stride"]), tuple(man["model_hw"])
    rng = np.random.default_rng(seed)
    _points(cache, paths, stride, model_hw, rng, _qdir(cfg, seq) / "points.npz")
    _depth_vs_lidar(cfg, seq, cache, paths, stride, model_hw)
    table, frames, poses = edges.load(Path(cfg["run_dir"]) / seq / "edges.npz")
    with np.load(Path(cfg["run_dir"]) / seq / "labels.npz") as d:
        lab = {k: d[k] for k in d.files}
    _loops(cfg, seq, cache, paths, table, lab)
    _seams(cfg, seq, cache, paths, stride, model_hw, table, lab)
    print(f"  [{seq}] qualitative material extracted")


def _points(cache, paths, stride, model_hw, rng, out):
    """Confident, sampled points of every window in window coordinates, with their colours."""
    xyz, rgb, win = [], [], []
    for k in range(len(cache.ranges)):
        w = cache.window(k)
        conf = w["points_conf"].astype(np.float32)
        keep = np.flatnonzero(((conf >= EXPORT_CONF_COEF * conf.mean()) & (conf > 1e-5)).ravel())
        keep = rng.choice(keep, max(1, int(len(keep) * SAMPLE_RATIO)), replace=False)
        n, h, wd = conf.shape
        fi, rest = np.divmod(keep, h * wd)
        r, c = np.divmod(rest, wd)
        colours = np.zeros((len(keep), 3), np.uint8)
        for f in np.unique(fi):
            img = _rgb_on_grid(paths[int(w["frame_ids"][f])], model_hw, stride)
            m = fi == f
            colours[m] = img[r[m], c[m]]
        xyz.append(w["points"][fi, r, c].astype(np.float32))
        rgb.append(colours)
        win.append(np.full(len(keep), k, np.int16))
    np.savez_compressed(out, xyz=np.concatenate(xyz), rgb=np.concatenate(rgb), window=np.concatenate(win))


def _depth_vs_lidar(cfg, seq, cache, paths, stride, model_hw):
    """Three frames: image, depth head, its confidence, and the scale-aligned relative error at LiDAR pixels."""
    plt = _plt()
    man = cache.manifest
    velo = Path(cfg["kitti_root"]) / "sequences" / seq / "velodyne"
    calib = kitti.read_calib(Path(cfg["kitti_root"]) / "sequences" / seq / "calib.txt")
    picks = [int(q * (man["n_frames"] - 1)) for q in (.2, .5, .8)]
    fig, axes = plt.subplots(len(picks), 4, figsize=(18, 2.3 * len(picks)), squeeze=False)
    saved = {}
    for row, f in enumerate(picks):
        k = _window_with_frame_centred(cache.ranges, f)
        w = cache.window(k)
        n = int(np.flatnonzero(w["frame_ids"] == f)[0])
        rgb = _rgb_on_grid(paths[f], model_hw, stride)
        depth, dconf = w["depth"][n].astype(np.float32), w["depth_conf"][n].astype(np.float32)
        lid = None
        if (velo / f"{f:06d}.bin").exists():
            lid = kitti.lidar_depth_on_grid(velo / f"{f:06d}.bin", calib, model_hw, tuple(man["orig_hw"]), stride)
        ax = axes[row]
        ax[0].imshow(rgb)
        ax[0].set_title(f"KITTI {seq} frame {f} (window {k})", fontsize=9)
        im = ax[1].imshow(depth, cmap="magma_r")
        ax[1].set_title("VGGT depth head (window units)", fontsize=9)
        fig.colorbar(im, ax=ax[1], fraction=.025)
        im = ax[2].imshow(np.log10(dconf), cmap="viridis")
        ax[2].set_title("depth confidence (log10)", fontsize=9)
        fig.colorbar(im, ax=ax[2], fraction=.025)
        ax[3].imshow(rgb.mean(2), cmap="gray", vmin=0, vmax=255)
        if lid is not None and len(lid[2]):
            r, c, d = lid
            pred = depth[r, c]
            ok = pred > 0
            s = np.median(d[ok] / pred[ok])
            err = np.abs(s * pred - d) / d
            sc = ax[3].scatter(c, r, c=err, s=1.5, cmap="turbo", vmin=0, vmax=.3)
            fig.colorbar(sc, ax=ax[3], fraction=.025)
            ax[3].set_title(f"|scaled depth - LiDAR| / LiDAR (median {np.median(err):.3f})", fontsize=9)
            saved.update({f"f{f}_lidar_r": r, f"f{f}_lidar_c": c, f"f{f}_lidar_d": d.astype(np.float32)})
        saved.update({f"f{f}_depth": depth.astype(np.float16), f"f{f}_depth_conf": dconf.astype(np.float16)})
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(_figdir(cfg) / f"depth_lidar_{seq}.png", dpi=110)
    plt.close(fig)
    np.savez_compressed(_qdir(cfg, seq) / "frames.npz", **saved)


def _loops(cfg, seq, cache, paths, table, lab):
    """Per loop: does the loop window place the two visits where they really are?"""
    plt = _plt()
    gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)
    loop_edge = {(int(i), int(j)): e for e, (i, j, kd) in enumerate(zip(table["i"], table["j"], table["kind"])) if kd == 1}
    rows, keep = [], {}
    for k, meta in enumerate(cache.loop_meta):
        lw = cache.loop(k)
        f, c2w = lw["frame_ids"], lw["c2w"].astype(float)
        na = meta["range_a"][1] - meta["range_a"][0]
        a, b = meta["pair"]
        A, info = fit_gauge(c2w, gt[f])
        pos = transform_points(A, c2w[:, :3, 3])
        ia, ib = int(np.flatnonzero(f == a)[0]), int(np.flatnonzero(f == b)[0])
        row = {"loop": k, "frame_later": a, "frame_earlier": b, "similarity": meta["similarity"],
               "window_later": meta["chunk_a"], "window_earlier": meta["chunk_b"],
               "gt_distance_m": float(np.linalg.norm(gt[a][:3, 3] - gt[b][:3, 3])),
               "loop_window_distance_m": float(np.linalg.norm(pos[ia] - pos[ib])),
               "joint_fit_rms_m": info["position_rms_m"],
               "visit_later_alone_rms_m": fit_gauge(c2w[:na], gt[f[:na]])[1]["position_rms_m"],
               "visit_earlier_alone_rms_m": fit_gauge(c2w[na:], gt[f[na:]])[1]["position_rms_m"]}
        e = loop_edge.get((meta["chunk_b"], meta["chunk_a"]))
        if e is not None:
            err = lab["errors"][e]
            row.update(edge_error_trans_m=float(np.linalg.norm(err[3:6]) * lab["gauge_scale"][meta["chunk_a"]]),
                       edge_error_rot_deg=float(np.degrees(np.linalg.norm(err[:3]))), edge_error_log_scale=float(err[6]),
                       edge_valid=bool(lab["valid"][e]))
        rows.append(row)
        keep[k] = (pos, gt[f][:, :3, 3], na)
    if not rows:
        return
    tabs = Path(cfg["run_dir"]) / "tables"
    tabs.mkdir(exist_ok=True)
    with open(tabs / f"loops_{seq}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader()
        wr.writerows(rows)
    scored = [r for r in rows if r.get("edge_valid")] or rows
    key = "edge_error_trans_m" if "edge_error_trans_m" in scored[0] else "joint_fit_rms_m"
    picks = [max(scored, key=lambda r: r[key])]
    if len(scored) > 1:
        picks.append(min(scored, key=lambda r: r[key]))
    fig, axes = plt.subplots(len(picks), 3, figsize=(17, 3.3 * len(picks)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.4, 2.4, 1]})
    for ax, r, tag in zip(axes, picks, ("largest", "smallest")):
        pos, ref, na = keep[r["loop"]]
        for a_, fr, name in ((ax[0], r["frame_later"], "later visit"), (ax[1], r["frame_earlier"], "earlier visit")):
            from PIL import Image
            with Image.open(paths[fr]) as im:
                a_.imshow(np.asarray(im.convert("RGB")))
            a_.set_title(f"{name}: frame {fr}", fontsize=9), a_.set_xticks([]), a_.set_yticks([])
        ax[2].plot(ref[:na, 0], ref[:na, 2], "k-", lw=2, label="GT later")
        ax[2].plot(ref[na:, 0], ref[na:, 2], "k--", lw=2, label="GT earlier")
        ax[2].plot(pos[:na, 0], pos[:na, 2], "-", color="#d55e00", label="VGGT later")
        ax[2].plot(pos[na:, 0], pos[na:, 2], "--", color="#0072b2", label="VGGT earlier")
        ax[2].set_aspect("equal", adjustable="datalim"), ax[2].grid(alpha=.3), ax[2].legend(fontsize=6)
        ax[2].set_title("loop window cameras (top view, m)", fontsize=8)
        ax[0].set_ylabel(f"{tag} loop error", fontsize=9)
        ax[0].set_xlabel(f"loop {r['loop']}: visits {r['gt_distance_m']:.2f} m apart in GT, "
                         f"{r['loop_window_distance_m']:.2f} m apart in the loop window", fontsize=8)
        ax[1].set_xlabel(f"each visit alone fits GT to {r['visit_later_alone_rms_m']:.2f} / "
                         f"{r['visit_earlier_alone_rms_m']:.2f} m"
                         + (f"; loop-edge error {r['edge_error_trans_m']:.2f} m, {r['edge_error_rot_deg']:.2f} deg"
                            if "edge_error_trans_m" in r else ""), fontsize=8)
    fig.suptitle(f"KITTI {seq}: loop windows with the largest and smallest error", fontsize=10)
    fig.tight_layout()
    fig.savefig(_figdir(cfg) / f"loops_{seq}.png", dpi=110)
    plt.close(fig)


def _seams(cfg, seq, cache, paths, stride, model_hw, table, lab):
    """Seams with the largest and smallest true translation error: the shared frame and both windows' confidence."""
    plt = _plt()
    seam = np.flatnonzero((table["kind"] == 0) & lab["valid"])
    if len(seam) < 2:
        return
    terr = np.linalg.norm(lab["errors"][seam, 3:6], axis=1) * lab["gauge_scale"][table["j"][seam]]
    order = np.argsort(terr)
    picks = [(seam[order[-1]], terr[order[-1]], "largest"), (seam[order[-2]], terr[order[-2]], "2nd largest"),
             (seam[order[1]], terr[order[1]], "2nd smallest"), (seam[order[0]], terr[order[0]], "smallest")]
    fig, axes = plt.subplots(len(picks), 3, figsize=(16, 2.2 * len(picks)), squeeze=False)
    for ax, (e, t, tag) in zip(axes, picks):
        i, j = int(table["i"][e]), int(table["j"][e])
        f = cache.ranges[j][0]
        wi, wj = cache.window(i), cache.window(j)
        ni = int(np.flatnonzero(wi["frame_ids"] == f)[0])
        conf_i, conf_j = wi["points_conf"][ni].astype(np.float32), wj["points_conf"][0].astype(np.float32)
        hi = float(np.log10(max(conf_i.max(), conf_j.max())))
        ax[0].imshow(_rgb_on_grid(paths[f], model_hw, stride))
        ax[0].set_title(f"{tag} error: seam {i}-{j}, shared frame {f}: {t:.2f} m, "
                        f"{np.degrees(np.linalg.norm(lab['errors'][e, :3])):.2f} deg", fontsize=8)
        ax[1].imshow(np.log10(conf_i), cmap="viridis", vmin=0, vmax=hi)
        ax[1].set_title(f"window {i} point confidence (log10), seam mean conf {table['f_conf_mean'][e]:.2f}", fontsize=8)
        ax[2].imshow(np.log10(conf_j), cmap="viridis", vmin=0, vmax=hi)
        ax[2].set_title(f"window {j} point confidence; alignment residual_rel {table['f_residual_rel'][e]:.3f}", fontsize=8)
        for a in ax:
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(_figdir(cfg) / f"seams_{seq}.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ render (after solve; no cache needed)

def _bev(ax, xyz, rgb, gt, title):
    lo, hi = np.percentile(xyz[:, [0, 2]], [.5, 99.5], axis=0)
    pad = .05 * (hi - lo)
    m = np.all((xyz[:, [0, 2]] >= lo - pad) & (xyz[:, [0, 2]] <= hi + pad), axis=1)
    m &= np.abs(xyz[:, 1] - np.median(xyz[m, 1])) < 3 * np.std(xyz[m, 1])  # drop sky / far outliers
    ax.scatter(xyz[m, 0], xyz[m, 2], c=rgb[m] / 255., s=.03, lw=0, rasterized=True)
    ax.plot(gt[:, 0], gt[:, 2], "-", color="#d62728", lw=.8, label="GT trajectory")
    ax.set_aspect("equal"), ax.set_title(title, fontsize=9), ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
    ax.legend(fontsize=7, loc="upper right")


def _thin(n, rng):
    return np.sort(rng.choice(n, MAX_PLOT_POINTS, replace=False)) if n > MAX_PLOT_POINTS else np.arange(n)


def render(cfg, labels):
    plt = _plt()
    run = Path(cfg["run_dir"])
    rng = np.random.default_rng(0)
    written = []
    for seq in list(dict.fromkeys(cfg["test_sequences"] + cfg["fit_sequences"])):
        tp, qp = run / "trajectories" / f"{seq}.npz", run / seq / "qual" / "points.npz"
        if not (tp.exists() and qp.exists()):
            continue
        with np.load(tp) as d:
            tr = {k: d[k] for k in d.files}
        with np.load(qp) as d:
            pts = {k: d[k] for k in d.files}
        sel = _thin(len(pts["xyz"]), rng)
        xyz, rgb, win = pts["xyz"][sel].astype(np.float64), pts["rgb"][sel], pts["window"][sel].astype(int)
        keys = [k for k in ("vggt_long_chain", "vggt_long_loops", "calibrated_loops") if k + "_nodes" in tr]
        if not keys:
            continue
        fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 6.4), squeeze=False)
        for ax, key in zip(axes[0], keys):
            nodes = tr[key + "_nodes"]
            world = np.einsum("nij,nj->ni", nodes[win][:, :3, :3], xyz) + nodes[win][:, :3, 3]
            t = sim3_align(tr[key], tr["gt"])
            scheme = key.rsplit("_", 1)[0]
            _bev(ax, transform_points(t, world), rgb, tr["gt"],
                 labels[scheme] + (" without loops" if key.endswith("chain") else " with loops"))
        fig.suptitle(f"KITTI {seq}: VGGT points placed by the solved window poses (Sim(3)-aligned to GT)", fontsize=10)
        fig.tight_layout()
        fig.savefig(_figdir(cfg) / f"pointcloud_{seq}.png", dpi=150)
        plt.close(fig)
        written.append(f"pointcloud_{seq}.png")
        # per-frame position error along the trajectory
        keys = [k for k in ("vggt_long_loops", "calibrated_loops") if k in tr]
        fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 5.6), squeeze=False)
        errs = {k: np.linalg.norm(transform_points(sim3_align(tr[k], tr["gt"]), tr[k]) - tr["gt"], axis=1) for k in keys}
        vmax = max(np.percentile(e, 99) for e in errs.values())
        for ax, key in zip(axes[0], keys):
            al = transform_points(sim3_align(tr[key], tr["gt"]), tr[key])
            ax.plot(tr["gt"][:, 0], tr["gt"][:, 2], "-", color="#bbbbbb", lw=3, label="GT")
            sc = ax.scatter(al[:, 0], al[:, 2], c=errs[key], s=2, cmap="turbo", vmin=0, vmax=vmax)
            fig.colorbar(sc, ax=ax, fraction=.04, label="position error [m]")
            ax.set_aspect("equal"), ax.grid(alpha=.3), ax.legend(fontsize=7)
            ax.set_title(f"{labels[key.rsplit('_', 1)[0]]}: ATE {np.sqrt(np.mean(errs[key] ** 2)):.2f} m", fontsize=9)
        fig.suptitle(f"KITTI {seq}: where the error is (loops on, Sim(3)-aligned)", fontsize=10)
        fig.tight_layout()
        fig.savefig(_figdir(cfg) / f"trajectory_error_{seq}.png", dpi=130)
        plt.close(fig)
        written.append(f"trajectory_error_{seq}.png")
    # upstream VGGT-Long's own exported point cloud
    for d in sorted((run / "upstream").glob("*")) if (run / "upstream").exists() else []:
        pc, cam = d / "pointcloud_sample.npz", d / "camera_poses.txt"
        if not (pc.exists() and cam.exists()):
            continue
        seq = d.name.split("_")[0]
        from .trajectory import read_poses
        est = read_poses(cam)[:, :3, 3]
        gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)[:len(est), :3, 3]
        with np.load(pc) as z:
            xyz, rgb = z["xyz"].astype(np.float64), z["rgb"]
        fig, ax = plt.subplots(figsize=(7, 6.6))
        _bev(ax, transform_points(sim3_align(est, gt), xyz), rgb, gt,
             f"upstream VGGT-Long ({d.name}), its own point export, Sim(3)-aligned")
        fig.tight_layout()
        fig.savefig(_figdir(cfg) / f"upstream_pointcloud_{d.name}.png", dpi=150)
        plt.close(fig)
        written.append(f"upstream_pointcloud_{d.name}.png")
    return written


# ------------------------------------------------------------------ upstream PLY -> small npz

def read_ply(path):
    """VGGT-Long's binary PLY (float x, y, z; uchar r, g, b)."""
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        data = np.fromfile(f, dtype=np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]), count=n)
    return data["xyz"], data["rgb"]


def ply2npz(ply, out, n=MAX_PLOT_POINTS, seed=0):
    xyz, rgb = read_ply(ply)
    sel = _thin(len(xyz), np.random.default_rng(seed)) if len(xyz) > n else np.arange(len(xyz))
    np.savez_compressed(out, xyz=xyz[sel], rgb=rgb[sel], n_total=len(xyz))
    print(f"{ply}: {len(xyz)} points, kept {len(sel)} -> {out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "ply2npz":
        ply2npz(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
