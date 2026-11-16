"""python scripts/score_upstream.py --w 30 --o 15  ->  results/upstream_w30_o15.csv (read by scripts/stats.py)"""
import argparse, csv
import lib
from mr_cpu import evaluation, kitti
from mr_cpu.trajectory import read_poses
NAMES = {"plain": "none", "C1": "C1_const", "C4": "C4_fuse_depth+speed"}
ap = argparse.ArgumentParser(); ap.add_argument("--w", type=int, required=True); ap.add_argument("--o", type=int, required=True)
a = ap.parse_args()
rows = []
for tag, name in NAMES.items():
    for s in lib.ALL:
        f = lib.RUNS / "upstream" / f"{tag}_w{a.w}_o{a.o}" / s / "camera_poses.txt"
        if not f.exists():
            continue
        est, gt = read_poses(f), kitti.gt_cam2_poses(lib.KITTI, s)
        assert len(est) == len(gt), (f, len(est), len(gt))
        ate = evaluation.ate(est[:, :3, 3], gt[:, :3, 3])
        rows.append([name, s, f"{ate:.4f}", f"{ate:.4f}", "", "", ""])
        print(tag, s, f"{ate:.2f}")
with open(lib.RES / f"upstream_w{a.w}_o{a.o}.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["method", "seq", "ate_chain", "ate_loops", "resid_mean", "resid_std", "auto_pick"]); w.writerows(rows)
