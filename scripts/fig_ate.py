"""Figure: ATE with loops per drive, VGGT-Long against VGGT-Long + calibrated seam scale, one panel per setting.

  python scripts/fig_ate.py results/first_w30_o15.csv results/first_w60_o30_from_first_w30_o15.csv ...
Bars: VGGT-Long (grey) and the calibrated seam scale C4 (blue); tick: true-scale bound. Log x axis.
Writes results/final/figures/fig_ate.png and results/fig_ate.pdf.
"""
import csv
from pathlib import Path
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
BASE, OURS = "#9aa4ae", "#2f6fdb"


def read(path):
    rows = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        rows[r["method"]][r["seq"]] = float(r["ate_loops"])
    return rows


def main():
    paths = sys.argv[1:]
    fig, axes = plt.subplots(1, len(paths), figsize=(max(5.2, 3.6 * len(paths)), 4.6), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], paths):
        rows = read(p)
        seqs = sorted(rows["none"])
        y = np.arange(len(seqs))
        base = np.array([rows["none"][s] for s in seqs])
        ours = np.array([rows["C4_fuse_depth+speed"][s] for s in seqs])
        orc = np.array([rows["oracle"][s] for s in seqs])
        ax.barh(y - 0.19, base, 0.34, color=BASE, label="VGGT-Long")
        ax.barh(y + 0.19, ours, 0.34, color=OURS, label="+ calibrated seam scale (C4)")
        ax.scatter(orc, y + 0.19, marker="|", s=90, color=INK, linewidths=1.5, label="true-scale bound", zorder=3)
        ax.set_xscale("log")
        ax.set_yticks(y, seqs)
        ax.invert_yaxis()
        m = re.search(r"w(\d+)_o(\d+)", p)
        ax.set_title(f"chunk {m.group(1)} / overlap {m.group(2)}\nmean {base.mean():.1f} m -> {ours.mean():.1f} m",
                     fontsize=10, color=INK)
        ax.set_xlabel("ATE with loops (m, log scale)", color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0][0].set_ylabel("KITTI drive", color=MUTED)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(Path(__file__).resolve().parents[1] / "results" / "final" / "figures" / f"fig_ate.{ext}", dpi=200)
    print("wrote results/final/figures/fig_ate.png / .pdf")


if __name__ == "__main__":
    main()
