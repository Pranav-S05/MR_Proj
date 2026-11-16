"""Figures for the final submission: false-loop stress tests, the MASt3R-SLAM baseline, qualitative examples.

  python -m mr_cpu.final_figures --config configs/kitti_mid.json

Reads results/final/stress/*.csv (mr_cpu.stress), results/final/mast3r_slam/<seq>_<mode>.txt (MASt3R-SLAM
keyframes, TUM format, t = frame / 30) and the SALAD descriptors in cache/kitti_realistic/<seq>/. Image panels
need the KITTI frames under <kitti_root>/sequences/<seq>/image_2; panels whose images are missing are skipped
(`--list-images` prints the frames they need). Writes results/final/figures/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from . import backend, cache, kitti
from .evaluation import assemble, sim3_align
from .geometry import transform_points
from .pipeline import _chain, load_config
from .report import COLOR, LABEL
from .stress import _setup, _with_fakes

OUT = Path("results/final/figures")
STRESS = Path("results/final/stress")
MAST3R = Path("results/final/mast3r_slam")
SEQS = ("00", "02", "05", "07")
# four schemes only: their colours pass the colour-blind checks as a set (gray stays for references)
SCHEMES = ("vggt_long", "raw_conf", "geometry", "calibrated")
MARKER = {"vggt_long": "s", "raw_conf": "^", "geometry": "D", "calibrated": "o"}
REF = "#9a9a9a"
K_VALUES = (0, 1, 2, 4, 8, 16)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "axes.labelcolor": "#333333", "xtick.color": "#555555",
                         "ytick.color": "#555555", "grid.color": "#dddddd", "grid.linewidth": .6})
    return plt


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _test_label(seq, cfg):
    return f"KITTI {seq} ({'held-out' if seq in cfg['test_sequences'] else 'fitting'})"


def _bottom_legend(fig, ax):
    """One legend under the panels (series + the survival band), so no panel's data is covered."""
    from matplotlib.patches import Patch
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#eef4ea", edgecolor=REF, linestyle="--", label="survives: within 1.25x of clean"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)


# ------------------------------------------------------------------ adversarial false loops
def stress_ratio(cfg):
    """Worst-seed ATE (DCS kernel) relative to the clean graph without a kernel, against the number of fake loops."""
    rows = _rows(STRESS / "adversarial.csv")
    clean = {(r["sequence"], r["scheme"]): float(r["ate_rmse_m"]) for r in rows if r["k"] == "0" and r["kernel"] == "none"}
    worst = {}
    for r in rows:
        if r["kernel"] == "dcs":
            key = (r["sequence"], r["scheme"], int(r["k"]))
            worst[key] = max(worst.get(key, 0), float(r["ate_rmse_m"]))
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    x = np.arange(len(K_VALUES))
    for ax, seq in zip(axes, SEQS):
        ax.axhspan(0, 1.25, color="#eef4ea", zorder=0)
        ax.axhline(1.25, color=REF, lw=1, ls="--", zorder=1)
        for scheme in SCHEMES:
            y = [worst[seq, scheme, k] / clean[seq, scheme] for k in K_VALUES]
            ax.plot(x, y, "-", marker=MARKER[scheme], ms=5, lw=1.6, color=COLOR[scheme], label=LABEL[scheme],
                    markeredgecolor="white", markeredgewidth=.8, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x, [str(k) for k in K_VALUES])
        ax.set_xlabel("fake loops added")
        ax.set_title(_test_label(seq, cfg), fontsize=9)
        ax.grid(axis="y", which="both", alpha=.5)
    axes[0].set_ylabel("worst-seed ATE / clean ATE (no kernel)")
    _bottom_legend(fig, axes[0])
    fig.suptitle("DCS kernel against adversarial fake loops (20 seeds; same fake loops for every scheme)", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT / "stress_ratio_vs_fake_loops.png", dpi=150)
    plt.close(fig)


def dcs_sweep(cfg):
    """Identity weights: no DCS threshold keeps the true loops AND rejects the fake ones on every drive."""
    rows = [r for r in _rows(STRESS / "dcs_sweep_wide.csv") if r["scheme"] == "vggt_long"]
    clean = {r["sequence"]: float(r["ate_rmse_m"]) for r in rows if r["kernel"] == "none"}
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    styles = {0: ("no fake loops", "#222222", "o"), 1: ("1 fake loop (worst of 5 seeds)", COLOR["vggt_long"], "^"),
              16: ("16 fake loops (worst of 5 seeds)", "#8c510a", "s")}
    for ax, seq in zip(axes, SEQS):
        ax.axhspan(0, 1.25, color="#eef4ea", zorder=0)
        ax.axhline(1.25, color=REF, lw=1, ls="--")
        tuned = None
        for k, (lab, col, mk) in styles.items():
            sub = [r for r in rows if r["sequence"] == seq and r["kernel"] == "dcs" and int(r["k"]) == k]
            phis = sorted({float(r["phi"]) for r in sub})
            y = [max(float(r["ate_rmse_m"]) for r in sub if float(r["phi"]) == p) / clean[seq] for p in phis]
            ax.plot(phis, y, "-", marker=mk, ms=4, lw=1.4, color=col, label=lab, markeredgecolor="white",
                    markeredgewidth=.6)
            tuned = phis
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(10, color=REF, lw=.8, ls=":")
        ax.text(10, .98, " tuned on 05/07", transform=ax.get_xaxis_transform(), fontsize=6.5, color="#666666", va="top")
        ax.set_xlabel("DCS threshold phi")
        ax.set_title(_test_label(seq, cfg), fontsize=9)
        ax.grid(which="major", alpha=.5)
        ax.set_xlim(min(tuned) / 2, max(tuned) * 2)
    axes[0].set_ylabel("ATE / clean ATE (no kernel)")
    _bottom_legend(fig, axes[0])
    fig.suptitle("(a) VGGT-Long identity weights: every DCS threshold either drops the true loops or keeps the fake ones "
                 "(5-seed screen; 20 seeds in dcs_sweep_fine.md)", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUT / "dcs_sweep_identity.png", dpi=150)
    plt.close(fig)


def stress_trajectories(cfg, seq, seed=0, ks=(1, 16)):
    """Top view: ground truth, the DCS solution with k fake loops, and the fake loops themselves."""
    import json
    kernels = json.loads((Path(cfg["run_dir"]) / "kernels.json").read_text(encoding="utf-8"))
    plt = _plt()
    shown = ("vggt_long", "raw_conf", "calibrated")
    fig, axes = plt.subplots(len(ks), len(shown), figsize=(4 * len(shown), 3.9 * len(ks)))
    for row, k in enumerate(ks):
        table, frames, poses, lab, gt, n, fitted, fakes = _setup(cfg, seq, k, seed)
        init = _chain(table)
        first = gt[[int(f[0]) for f in frames], :3, 3]
        for col, scheme in enumerate(shown):
            ax = axes[row, col]
            edges = _with_fakes(table, fitted.information(scheme, table, lab["errors"]), *fakes)
            mode = "upstream" if scheme == "vggt_long" else "ours"
            nodes, rep = backend.solve(init, *edges[:4], mode, "dcs", kernels[scheme]["dcs"]["best"], edges[4] == 1)
            est = assemble(nodes, frames, poses, n)[:, :3, 3]
            al = transform_points(sim3_align(est, gt[:, :3, 3]), est)
            ate = float(np.sqrt(np.mean(np.sum((al - gt[:, :3, 3]) ** 2, 1))))
            ax.plot(gt[:, 0, 3], gt[:, 2, 3], color="#c8c8c8", lw=3.2, label="ground truth", zorder=1)
            ax.plot(al[:, 0], al[:, 2], color=COLOR[scheme], lw=1.3, label="estimate", zorder=3)
            rejected = rep["edge_weight"][edges[5]] < .5
            for (a, b), rej in zip(zip(fakes[0], fakes[1]), rejected):
                ax.plot(first[[a, b], 0], first[[a, b], 2], ls=":" if rej else "--", lw=.9,
                        color=REF if rej else "#b2182b", zorder=2)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(f"{LABEL[scheme]}, {k} fake loop{'s' * (k > 1)}\nATE {ate:.1f} m, "
                         f"{int(rejected.sum())}/{k} fake rejected", fontsize=8.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="#c8c8c8", lw=3.2, label="ground truth"),
                        Line2D([], [], color="#333333", lw=1.3, label="DCS solution (scheme colour)"),
                        Line2D([], [], color="#b2182b", ls="--", lw=.9, label="fake loop kept"),
                        Line2D([], [], color=REF, ls=":", lw=.9, label="fake loop rejected")],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{_test_label(seq, cfg)}: fake loops of seed {seed} and the DCS solution, Sim(3)-aligned top view\n"
                 "(one seed for illustration; survival over all 20 seeds is in adversarial_survival.md)", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95), h_pad=2.5)
    fig.savefig(OUT / f"stress_trajectories_{seq}.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ realistic (SALAD) examples
def salad_pairs(cfg, seq):
    """(similarity, frame a, frame b, GT distance) of the best true revisit loop and of the most similar pair of
    frames more than 50 m apart (the closest SALAD comes to a false loop)."""
    d = np.load(Path("cache/kitti_realistic") / seq / "descriptors.npy").astype(np.float32)
    pos = kitti.gt_cam2_poses(cfg["kitti_root"], seq)[:, :3, 3]
    loops = [(s, b, a) for a, b, s in cache.find_loop_pairs(d, 0.85) if a - b > 100]
    s, b, a = max(loops)
    true = (s, b, a, float(np.linalg.norm(pos[a] - pos[b])))
    sims = d @ d.T
    far = np.linalg.norm(pos[:, None] - pos[None], axis=-1) > 50
    sims[~far] = -np.inf
    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    i, j = sorted((int(i), int(j)))
    false = (float(sims[i, j]), i, j, float(np.linalg.norm(pos[i] - pos[j])))
    return true, false


def _image(cfg, seq, frame):
    p = Path(cfg["kitti_root"]) / "sequences" / seq / "image_2" / f"{frame:06d}.png"
    if not p.exists():
        return None
    from PIL import Image
    return np.asarray(Image.open(p))


def salad_figure(cfg, seq):
    true, false = salad_pairs(cfg, seq)
    imgs = [[_image(cfg, seq, f) for f in pair[1:3]] for pair in (true, false)]
    if any(im is None for row in imgs for im in row):
        return False
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(10, 3.6))
    for r, (pair, title) in enumerate(((true, "true revisit (loop SALAD accepts at 0.85)"),
                                       (false, "most similar frames > 50 m apart (no SALAD loop even at 0.5)"))):
        for c in range(2):
            axes[r, c].imshow(imgs[r][c])
            axes[r, c].set_axis_off()
            axes[r, c].set_title(f"frame {pair[1 + c]}", fontsize=8)
        axes[r, 0].text(-0.02, .5, f"{title}\nsimilarity {pair[0]:.2f}, {pair[3]:.1f} m apart",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(f"KITTI {seq}: SALAD place recognition", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"salad_pairs_{seq}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------------ MASt3R-SLAM baseline
def _mast3r(seq, mode):
    p = MAST3R / f"{seq}_{mode}.txt"
    if not p.exists():
        return None
    rows = np.loadtxt(p, ndmin=2)
    return np.rint(rows[:, 0] * 30).astype(int), rows[:, 1:4]


def _lost_after(frames, n_frames, max_gap=30):
    """First keyframe followed by a gap of more than max_gap frames (also up to the end of the drive)."""
    ends = np.r_[frames, n_frames - 1]
    gaps = np.diff(ends)
    return int(ends[:-1][gaps > max_gap][0]) if np.any(gaps > max_gap) else None


def mast3r_figure(cfg, seq):
    runs = {m: _mast3r(seq, m) for m in ("nocalib", "calib")}
    runs = {m: r for m, r in runs.items() if r is not None}
    if not runs:
        return False
    gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)[:, :3, 3]
    plt = _plt()
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), squeeze=False)
    for ax, (mode, (frames, xyz)) in zip(axes[0], runs.items()):
        lost = _lost_after(frames, len(gt))
        ax.plot(gt[:, 0], gt[:, 2], color="#c8c8c8", lw=3, label="ground truth (whole drive)")
        ok = np.isfinite(xyz).all(1)
        if ok.sum() >= 3:
            al = transform_points(sim3_align(xyz[ok], gt[frames[ok]]), xyz[ok])
            ax.scatter(al[:, 0], al[:, 2], s=9, color="#0072b2", zorder=3, label=f"MASt3R-SLAM keyframes ({ok.sum()})")
        # where the keyframes were taken (ground-truth positions), also when the estimated poses are unusable
        ax.scatter(gt[frames, 0], gt[frames, 2], s=22, facecolor="none", edgecolor="#e69f00", lw=.8, zorder=2,
                   label="keyframe locations (ground truth)")
        if ok.sum() < len(ok):
            ax.text(.02, .02, f"{(~ok).sum()} of {len(ok)} keyframe poses are NaN\n(optimiser diverged)",
                    transform=ax.transAxes, fontsize=8, color="#b2182b")
        if lost is not None:
            ax.scatter(gt[lost, 0], gt[lost, 2], s=90, marker="X", color="#b2182b", zorder=4,
                       label=f"tracking lost after frame {lost}")
        ax.scatter(gt[0, 0], gt[0, 2], s=40, marker="o", facecolor="white", edgecolor="#222222", zorder=4, label="start")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]")
        ax.grid(alpha=.4)
        ax.set_title(f"{'uncalibrated' if mode == 'nocalib' else 'calibrated (KITTI intrinsics)'}", fontsize=9)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    fig.suptitle(f"KITTI {seq}: MASt3R-SLAM keyframes, Sim(3)-aligned on the keyframes", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"mast3r_slam_{seq}.png", dpi=150)
    plt.close(fig)
    return True


def mast3r_lost_frames(cfg, seq, offsets=(-8, -4, 0, 4, 8)):
    run = _mast3r(seq, "nocalib")
    lost = None if run is None else _lost_after(run[0], len(kitti.gt_cam2_poses(cfg["kitti_root"], seq)))
    if lost is None:
        return []
    return [lost + o for o in offsets]


def mast3r_lost_figure(cfg, seq):
    frames = mast3r_lost_frames(cfg, seq)
    imgs = [_image(cfg, seq, f) for f in frames]
    if not frames or any(im is None for im in imgs):
        return False
    plt = _plt()
    fig, axes = plt.subplots(len(frames), 1, figsize=(7, 1.45 * len(frames)))
    lost = frames[len(frames) // 2]
    for ax, f, im in zip(axes, frames, imgs):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(f"frame {f}" + ("  (last keyframe before tracking is lost)" if f == lost else ""), fontsize=8)
    fig.suptitle(f"KITTI {seq}: where MASt3R-SLAM (uncalibrated) loses tracking", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"mast3r_slam_lost_{seq}.png", dpi=150)
    plt.close(fig)
    return True


def mast3r_table(cfg, max_gap=30):
    """MASt3R-SLAM on KITTI next to our pipeline (whole drive), plus the TUM fr1/desk install check."""
    import re
    from .evaluation import ate
    mid = {(r["sequence"], r["scheme"]): float(r["ate_rmse_m"]) for r in _rows(Path(cfg["run_dir"]) / "metrics.csv")
           if r["kernel"] == "none" and r["loops"] == "True"}
    lines = ["| sequence | mode | keyframes | tracking lost after frame | drive tracked | ATE over tracked keyframes | "
             "ours (a) / (e), whole drive |", "|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        for mode in ("nocalib", "calib"):
            run = _mast3r(seq, mode)
            if run is None:
                continue
            frames, xyz = run
            gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)
            ends = np.r_[frames, len(gt) - 1]
            gaps = np.diff(ends)
            dist = np.r_[0., np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))]
            tracked = (dist[ends[1:]] - dist[ends[:-1]])[gaps <= max_gap].sum() / dist[-1]
            ok = np.isfinite(xyz).all(1)
            a = f"{ate(xyz[ok], gt[frames[ok], :3, 3]):.2f} m" if ok.sum() >= 3 else f"none ({(~ok).sum()} NaN poses)"
            lines.append(f"| {seq} | {'uncalibrated' if mode == 'nocalib' else 'calibrated'} | {len(frames)} | "
                         f"{_lost_after(frames, len(gt), max_gap)} | {100 * tracked:.0f}% of {dist[-1] / 1000:.1f} km | {a} | "
                         f"{mid[seq, 'vggt_long']:.2f} / {mid[seq, 'calibrated']:.2f} m |")
    tum = ""
    log = MAST3R / "run_info" / "more.log"
    if log.exists():
        got = dict(re.findall(r"TUM desk (\w+):\s+rmse\s+([\d.]+)", log.read_text(encoding="utf-8")))
        if got:
            tum = (f"\nInstall check on TUM RGB-D fr1/desk with its own eval configs and scoring (evo_ape -as): calibrated "
                   f"{float(got.get('calib', 'nan')):.3f} m (paper 0.016), uncalibrated {float(got.get('no_calib', 'nan')):.3f} m "
                   "(paper 0.035; Murai et al., CVPR 2025, Table 1).\n")
    (MAST3R / "summary.md").write_text(
        "MASt3R-SLAM (github.com/rmurai0610/MASt3R-SLAM, unmodified), every KITTI frame, headless, single-threaded.\n"
        f"Tracked = parts of the drive between keyframes at most {max_gap} frames apart (a relocalisation later still leaves\n"
        "the gap unmapped). ATE is Sim(3)-aligned over the tracked keyframes only, so it is not comparable with a\n"
        "whole-drive ATE.\n\n" + "\n".join(lines) + "\n" + tum, encoding="utf-8")
    print("\n".join(lines) + tum)


def speed_memory_table(cfg, mid_run=Path("results/mid/run_info/cache_manifests")):
    """Runtime and GPU memory: our VGGT front end (mid-run manifests) and MASt3R-SLAM (its run logs), same GPU type.
    MASt3R-SLAM memory is the nvidia-smi peak of its main process sampled every 30 s; its backend process is not
    included (0.8 GB seen on 00, 5.4 GB on 02), so its total is higher. Wall time includes model loading and saving."""
    import json
    import re
    lines = ["| system | sequence | frames | wall time | frames/s | peak GPU memory | GPU |",
             "|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        p = mid_run / seq / "manifest.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            t = m["timing"]
            wall = m["n_frames"] / t["unique_frames_per_sec"]
            lines.append(f"| VGGT front end, 32/16 sliding windows (loop windows and CPU back end not included) | {seq} | {m['n_frames']} | {wall / 60:.1f} min | "
                         f"{t['unique_frames_per_sec']:.1f} | {t['peak_gpu_mem_gb']:.1f} GB | {m['gpu']} |")
    for seq in SEQS:
        for mode in ("nocalib", "calib"):
            log = MAST3R / "run_info" / f"run_{seq}_{mode}.log"
            mem = MAST3R / "run_info" / f"mem_{seq}_{mode}.log"
            if not log.exists():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            el = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text)
            if not el:
                continue
            parts = [float(x) for x in el.group(1).split(":")]
            sec = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
            n = len(kitti.gt_cam2_poses(cfg["kitti_root"], seq))
            peak = max((int(x.split(",")[1].split()[0]) for x in mem.read_text().splitlines() if "," in x), default=0) \
                if mem.exists() else 0
            lines.append(f"| MASt3R-SLAM ({'uncalibrated' if mode == 'nocalib' else 'calibrated'}) | {seq} | {n} | "
                         f"{sec / 60:.1f} min | {n / sec:.1f} | "
                         f"{f'{peak / 1024:.1f} GB (main process)' if peak else 'not recorded'} | NVIDIA RTX A4000 |")
    (Path("results/final") / "speed_memory.md").write_text(
        "Runtime and peak GPU memory on the same GPU type (Phoenix RTX A4000, shared machine).\n" +
        speed_memory_table.__doc__.split("\n", 1)[1].replace("    ", "") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def needed_images(cfg):
    out = []
    for seq in cfg["test_sequences"]:
        true, false = salad_pairs(cfg, seq)
        out += [(seq, f) for f in (true[1], true[2], false[1], false[2])]
    for seq in SEQS:
        out += [(seq, f) for f in mast3r_lost_frames(cfg, seq)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kitti_mid.json")
    ap.add_argument("--list-images", action="store_true", help="print the KITTI frames the image panels need")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.list_images:
        for seq, f in needed_images(cfg):
            print(f"{seq} {f:06d}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    stress_ratio(cfg)
    dcs_sweep(cfg)
    for seq in cfg["test_sequences"]:
        stress_trajectories(cfg, seq)
        print(f"salad_pairs_{seq}:", "written" if salad_figure(cfg, seq) else "skipped (images missing)")
    for seq in SEQS:
        print(f"mast3r_slam_{seq}:", "written" if mast3r_figure(cfg, seq) else "skipped (no run)")
        print(f"mast3r_slam_lost_{seq}:", "written" if mast3r_lost_figure(cfg, seq) else "skipped")
    speed_memory_table(cfg)
    mast3r_table(cfg)
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
