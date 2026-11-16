"""Independent check of the KITTI ground-truth height problem: KITTI odometry poses against the SemanticKITTI poses
(LiDAR SLAM, SuMa with loop closure) at places the car visits twice.

For every frame a we take the frame b (more than 300 frames away) that is closest according to the SemanticKITTI
poses, keep pairs closer than 3 m, and express b's position in a's camera frame (x right, y down, z forward) under
both pose sets. If the KITTI poses were consistent the two would agree.
  python scripts/kitti_gt_vs_suma.py
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "data" / "kitti" / "poses"
SEM = ROOT / "data" / "semkitti" / "dataset" / "sequences"


def read(path):
    m = np.loadtxt(path).reshape(-1, 3, 4)
    out = np.tile(np.eye(4), (len(m), 1, 1))
    out[:, :3] = m
    return out


def rel(p, a, b):
    return np.einsum("nij,nj->ni", np.transpose(p[a, :3, :3], (0, 2, 1)), p[b, :3, 3] - p[a, :3, 3])


print("seq  frames  step-agree(m)  revisit pairs | KITTI GT gap: |vertical| median / max, |sideways| median / max"
      " | SemanticKITTI gap: |vertical| median / max")
for s in [f"{k:02d}" for k in range(11)]:
    g, m = read(KITTI / f"{s}.txt"), read(SEM / s / "poses.txt")
    n = min(len(g), len(m))
    g, m = g[:n], m[:n]
    idx = np.arange(n - 1)
    step = np.abs(rel(g, idx, idx + 1) - rel(m, idx, idx + 1)).max(1)  # same convention -> small
    xyz = m[:, :3, 3]
    a_list, b_list = [], []
    for a in range(0, n, 5):
        d = np.linalg.norm(xyz - xyz[a], axis=1)
        d[max(0, a - 300):a + 301] = np.inf
        b = int(np.argmin(d))
        if d[b] < 3.0 and b < a:
            a_list.append(a)
            b_list.append(b)
    if not a_list:
        print(f"{s}  {n:5d}  {np.median(step):.3f}          0 (no revisit)")
        continue
    a, b = np.array(a_list), np.array(b_list)
    rg, rm = rel(g, a, b), rel(m, a, b)
    print(f"{s}  {n:5d}  {np.median(step):.3f}        {len(a):5d}        | "
          f"{np.median(np.abs(rg[:, 1])):5.2f} / {np.abs(rg[:, 1]).max():5.2f} m, "
          f"{np.median(np.abs(rg[:, 0])):5.2f} / {np.abs(rg[:, 0]).max():5.2f} m | "
          f"{np.median(np.abs(rm[:, 1])):5.2f} / {np.abs(rm[:, 1]).max():5.2f} m")
