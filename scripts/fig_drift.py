"""python scripts/fig_drift.py --variant first_w30_o15   -> results/fig_drift_<variant>.png / .pdf
Scale of the reconstruction along each drive without loop closure: VGGT-Long and VGGT-Long + calibrated seam scale
(the `auto` pick). Step lengths summed over a sliding window of 100 frames, estimate / ground truth, normalised to the
start of the drive."""
import argparse, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lib
from exp_main import Scorer, load_variant
BASE, OURS, MUTED, GRID = "#d1242f", "#2f6fdb", "#59636e", "#bbbbbb"
WIN = 100


def scale_curve(est, gt):
    se = np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1)
    sg = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
    ce, cg = np.convolve(se, np.ones(WIN), "valid"), np.convolve(sg, np.ones(WIN), "valid")
    r = np.where(cg > 1.0, ce / np.where(cg > 1.0, cg, 1.0), np.nan)
    fin = r[np.isfinite(r)]
    return r / np.median(fin[:WIN])


ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); a = ap.parse_args()
seqs, data, feats, y, ok = load_variant(a.variant)
sc = Scorer(data, feats, y, ok)
picks = {r["seq"]: r["auto_pick"] for r in csv.DictReader(open(lib.RES / f"{a.variant}.csv")) if r["method"] == "auto"}
fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
for ax, s in zip(axes.flat, seqs):
    d = data[s]
    _, _, e0 = lib.solve(d, None, loops=False, return_nodes=True)
    z = lib.rescale_seams(d, sc.corr(s, picks[s], [o for o in seqs if o != s]))
    _, _, e1 = lib.solve(d, z, loops=False, return_nodes=True)
    x = np.arange(WIN // 2, WIN // 2 + len(d["gt"]) - WIN)
    ax.axhline(1.0, color=GRID, lw=0.6)
    c0, c1 = scale_curve(e0, d["gt"]), scale_curve(e1, d["gt"])
    ax.plot(x, c0, color=BASE, lw=1.0, label="VGGT-Long")
    ax.plot(x, c1, color=OURS, lw=1.0, label="+ calibrated seam scale")
    print(s, "%.3f %.3f" % tuple(np.median(c[np.isfinite(c)][-100:]) for c in (c0, c1)))
    ax.set_yscale("log"); ax.set_title(s, fontsize=10)
    ax.tick_params(which="both", labelsize=7, colors=MUTED)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("map scale relative to the start", fontsize=8, color=MUTED)
for ax in axes.flat[len(seqs):]:
    ax.axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, l, loc="center", frameon=False, fontsize=10)
m = a.variant.split("_")
fig.suptitle(f"Scale of the reconstruction along the drive, without loop closure (chunk {m[1][1:]} / overlap {m[2][1:]})",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(lib.FIG / f"fig_drift_{a.variant}.{ext}", dpi=200)
