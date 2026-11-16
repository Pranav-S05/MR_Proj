"""KITTI relative errors (t_rel %, r_rel deg / 100 m) as a secondary metric (final_PLAN.md 4d item 1).

  python scripts/rel_errors.py   ->  results/rel_errors.csv
Upstream VGGT-Long 30/15 (plain, C4) from the saved camera_poses.txt; our pipeline at 30/15, 60/30, 75/30 (none, C4,
auto) from the same solves as scripts/fig_traj.py (with loops, corrections fitted on the other drives).
"""
import csv

import lib
from exp_main import Scorer, load_variant
from mr_cpu import evaluation, kitti
from mr_cpu.trajectory import read_poses

C4 = "C4_fuse_depth+speed"
rows = []
for tag, name in (("plain", "none"), ("C4", C4)):
    for s in lib.ALL:
        est, gt = read_poses(lib.RUNS / "upstream" / f"{tag}_w30_o15" / s / "camera_poses.txt"), kitti.gt_cam2_poses(lib.KITTI, s)
        r = evaluation.kitti_relative_errors(est, gt)
        rows.append(["upstream", "30/15", name, s, r["t_rel_percent"], r["r_rel_deg_per_100m"]])
for v, setting in (("first_w30_o15", "30/15"), ("first_w60_o30", "60/30"), ("first_w75_o30", "75/30")):
    seqs, data, feats, y, ok = load_variant(v)
    sc = Scorer(data, feats, y, ok)
    picks = {r["seq"]: r["auto_pick"] for r in csv.DictReader(open(lib.RES / f"{v}.csv")) if r["method"] == "auto"}
    for s in seqs:
        d = data[s]
        fitting = [o for o in seqs if o != s]
        for name, method in (("none", None), (C4, C4), ("auto", picks[s])):
            z = None if method is None else lib.rescale_seams(d, sc.corr(s, method, fitting))
            _, _, est = lib.solve(d, z, loops=True, return_nodes=True)
            r = evaluation.kitti_relative_errors(est, d["gt"])
            rows.append(["ours", setting, name, s, r["t_rel_percent"], r["r_rel_deg_per_100m"]])
    print(setting, "done")
with open(lib.RES / "rel_errors.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["source", "setting", "method", "seq", "t_rel_percent", "r_rel_deg_per_100m"])
    w.writerows([r[:4] + [f"{r[4]:.4f}", f"{r[5]:.4f}"] for r in rows])
print("wrote results/rel_errors.csv,", len(rows), "rows")
