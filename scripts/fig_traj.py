"""python scripts/fig_traj.py --variant first_w30_o15   -> results/fig_traj_<variant>.png / .pdf
Top view of every drive: ground truth, VGGT-Long, VGGT-Long + calibrated seam scale (the `auto` pick), with loops."""
import argparse, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lib
from exp_main import Scorer, load_variant
from mr_cpu import evaluation
from mr_cpu.geometry import transform_points
GT, BASE, OURS, MUTED = "#1f2328", "#d1242f", "#2f6fdb", "#59636e"
def aligned(est, gt):
    return transform_points(evaluation.sim3_align(est[:, :3, 3], gt[:, :3, 3]), est[:, :3, 3])
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); a = ap.parse_args()
seqs, data, feats, y, ok = load_variant(a.variant)
sc = Scorer(data, feats, y, ok)
picks = {r["seq"]: r["auto_pick"] for r in csv.DictReader(open(lib.RES / f"{a.variant}.csv")) if r["method"] == "auto"}
fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
for ax, s in zip(axes.flat, seqs):
    d = data[s]; gt = d["gt"][:, :3, 3]
    a0, _, e0 = lib.solve(d, None, loops=True, return_nodes=True)
    z = lib.rescale_seams(d, sc.corr(s, picks[s], [o for o in seqs if o != s]))
    a1, _, e1 = lib.solve(d, z, loops=True, return_nodes=True)
    p0, p1 = aligned(e0, d["gt"]), aligned(e1, d["gt"])
    ax.plot(gt[:, 0], gt[:, 2], color=GT, lw=1.6, label="ground truth")
    ax.plot(p0[:, 0], p0[:, 2], color=BASE, lw=1.0, label="VGGT-Long")
    ax.plot(p1[:, 0], p1[:, 2], color=OURS, lw=1.0, label="+ calibrated seam scale")
    ax.set_aspect("equal", adjustable="datalim"); ax.set_title(f"{s}: {a0:.1f} m -> {a1:.1f} m", fontsize=10)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ax in axes.flat[len(seqs):]:
    ax.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=10)
m = a.variant.split("_")
fig.suptitle(f"KITTI 00-10, chunk {m[1][1:]} / overlap {m[2][1:]}: top view (m), ATE with loops", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(lib.FIG / f"fig_traj_{a.variant}.{ext}", dpi=200)
