"""Figures for the final report (run from scale_scripts/):
  results/report_mechanism.pdf  (a) in-window step profile at 30/15, 60/30, 75/30; (b) seam bias against travel / depth
  results/report_maps.pdf       upstream VGGT-Long at 30/15 on drives 02 and 08, plain and calibrated (top view)
Also prints the numbers quoted in the report.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import lib  # noqa: E402
from mr_cpu import evaluation, kitti  # noqa: E402
from mr_cpu.geometry import transform_points  # noqa: E402
from mr_cpu.trajectory import read_poses  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
SET = [("first_w30_o15", "30/15", "#2f6fdb"), ("first_w60_o30", "60/30", "#d97a00"), ("first_w75_o30", "75/30", "#7a4fc9")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.family": "serif"})


def mechanism():
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for v, name, col in SET:
        prof, ys, sp = [], [], []
        for s in lib.ALL:
            d = lib.load(v, s)
            p = lib.step_profile(d)
            if len(p):
                prof.append(p)
            f = lib.seam_features(d)
            k = d["seam"]
            ok = d["valid"][k] & np.isfinite(f["speed"])
            ys.append(d["err"][k, 6][ok]); sp.append(f["speed"][ok])
        prof = np.concatenate(prof)
        x = np.linspace(0, 100, prof.shape[1])
        med = np.exp(np.mean(np.log(prof), 0))
        ax[0].plot(x, med, color=col, lw=1.2, label=name)
        y, s_ = np.concatenate(ys), np.concatenate(sp)
        q = np.quantile(s_, [0, .2, .4, .6, .8, 1])
        idx = np.clip(np.searchsorted(q, s_, side="right") - 1, 0, 4)
        mids = [np.median(np.exp(s_[idx == i])) for i in range(5)]
        mean = [y[idx == i].mean() for i in range(5)]
        se = [y[idx == i].std() / np.sqrt((idx == i).sum()) for i in range(5)]
        ax[1].errorbar(mids, mean, yerr=se, color=col, lw=1.0, marker="o", ms=2.5, capsize=1.5)
        half = prof[:, : prof.shape[1] // 2].prod(1) ** (1 / (prof.shape[1] // 2)) / \
            prof[:, prof.shape[1] - prof.shape[1] // 2:].prod(1) ** (1 / (prof.shape[1] // 2))
        print(f"{name}: windows {len(prof)}, first step {med[0]:.3f}, last {med[-1]:.3f}, half ratio median "
              f"{np.median(half):.3f}; bias by travel quintile " + " ".join(f"{m:+.4f}+-{e:.4f}" for m, e in zip(mean, se))
              + " at travel/depth " + " ".join(f"{m:.2f}" for m in mids))
    ax[0].axhline(1, color=GRID, lw=0.6, zorder=0)
    ax[0].set_xlabel("position in window (%)")
    ax[0].set_ylabel("predicted / true step")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.2)
    ax[1].axhline(0, color=GRID, lw=0.6, zorder=0)
    ax[1].set_xlabel("travel / scene depth")
    ax[1].set_ylabel("seam log-scale error")
    for a, t in zip(ax, ("(a)", "(b)")):
        a.set_title(t, fontsize=8, color=INK, pad=2)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(lib.FIG / "report_mechanism.pdf")
    fig.savefig(lib.FIG / "report_mechanism.png", dpi=250)


def maps():
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.15))
    for c, (s, tag, name) in enumerate((("02", "plain", "VGGT-Long"), ("02", "C4", "calibrated scale"),
                                        ("08", "plain", "VGGT-Long"), ("08", "C4", "calibrated scale"))):
        gt = kitti.gt_cam2_poses(lib.KITTI, s)[:, :3, 3]
        d = lib.RUNS / "upstream" / f"{tag}_w30_o15" / s
        est = read_poses(d / "camera_poses.txt")[:, :3, 3]
        x = np.load(d / "pointcloud_sample.npz")
        t = evaluation.sim3_align(est, gt)
        pts, rgb = transform_points(t, x["xyz"].astype(np.float64)), x["rgb"]
        sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 400_000), replace=False)
        pts, rgb = pts[sel], rgb[sel]
        p = transform_points(t, est)
        both = np.r_[gt[:, [0, 2]], p[:, [0, 2]]]
        lo, hi = both.min(0), both.max(0)
        pad = 0.04 * (hi - lo).max()
        m = np.all((pts[:, [0, 2]] > lo - pad) & (pts[:, [0, 2]] < hi + pad), 1)
        m &= np.abs(pts[:, 1] - np.median(pts[m, 1])) < 3 * np.std(pts[m, 1])
        a = ax[c]
        a.scatter(pts[m, 0], pts[m, 2], c=rgb[m] / 255.0, s=0.02, lw=0, rasterized=True)
        a.plot(gt[:, 0], gt[:, 2], color=INK, lw=0.7, label="ground truth")
        a.plot(p[:, 0], p[:, 2], color="#d1242f", lw=0.7, label="estimate")
        a.set_aspect("equal")
        a.set_xlim(lo[0] - pad, hi[0] + pad); a.set_ylim(lo[1] - pad, hi[1] + pad)
        a.set_title(f"{s}, {name}: {evaluation.ate(est, gt):.1f} m", fontsize=7.5, color=INK, pad=2)
        a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_color(GRID)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7, handlelength=1.5)
    fig.tight_layout(pad=0.2, w_pad=0.3, rect=(0, 0.07, 1, 1))
    fig.savefig(lib.FIG / "report_maps.pdf", dpi=300)
    fig.savefig(lib.FIG / "report_maps.png", dpi=250)


if __name__ == "__main__":
    mechanism()
    maps()
