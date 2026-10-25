"""KITTI odometry: calibration, colour-camera ground truth, LiDAR depth on the model pixel grid.

Coordinate facts (KITTI odometry devkit, calib.txt):
  * Ground-truth poses/SS.txt are camera-0 (left grey, rectified) to world, T_w_cam0.
  * We feed VGGT the left COLOUR images, image_2. In rectified coordinates camera 2 has the same
    orientation as camera 0 and is shifted: P2 = K [I | K^-1 p4], so x_cam2 = x_cam0 + b with
    b = K^-1 P2[:, 3] (about 6 cm along x). Hence T_w_cam2 = T_w_cam0 @ [I, -b].
  * Tr maps Velodyne points into rectified camera 0. A LiDAR point projects into image_2 at
    P2 @ [X_cam0; 1], and its optical-axis depth in camera 2 is the third homogeneous coordinate.
  * Image sizes differ by sequence: 00-02 are 1241x376, 03 is 1242x375, 04-10 are 1226x370.
    Always read the size from the images (image_size); never assume one.
  * VGGT's load_and_preprocess_images("crop") resizes W x H to 518 x round(H*518/W/14)*14 (154 for
    all KITTI sizes) with PIL's pixel-centre convention: x' = (x + 0.5) * 518/W - 0.5, and likewise
    for y. No crop happens because 154 < 518.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .trajectory import read_poses


def read_calib(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, values = line.split(":", 1)
            out[key.strip()] = np.array([float(v) for v in values.split()])
    calib = {k: out[k].reshape(3, 4) for k in ("P0", "P1", "P2", "P3")}
    tr = np.eye(4)
    tr[:3] = out["Tr"].reshape(3, 4)
    calib["Tr"] = tr
    return calib


def cam2_offset(calib):
    """b such that x_cam2 = x_cam0 + b (rectified frames)."""
    p2 = calib["P2"]
    return np.linalg.solve(p2[:, :3], p2[:, 3])


def gt_cam2_poses(root, seq):
    """(N,4,4) camera-2-to-world ground truth."""
    root = Path(root)
    t_w_c0 = read_poses(root / "poses" / f"{seq}.txt")
    calib = read_calib(root / "sequences" / seq / "calib.txt")
    t_c0_c2 = np.eye(4)
    t_c0_c2[:3, 3] = -cam2_offset(calib)
    return t_w_c0 @ t_c0_c2


def image_paths(root, seq):
    return sorted((Path(root) / "sequences" / seq / "image_2").glob("*.png"))


def image_size(root, seq):
    """(height, width) of the sequence's colour images."""
    from PIL import Image
    with Image.open(image_paths(root, seq)[0]) as im:
        return im.size[1], im.size[0]


def model_input_size(height, width, target=518, patch=14):
    """(height, width) produced by VGGT's crop-mode preprocessing (no crop for KITTI)."""
    new_h = int(round(height * (target / width) / patch) * patch)
    return min(new_h, target), target


def lidar_depth_on_grid(velo_path, calib, model_hw, orig_hw, stride=1):
    """Project one Velodyne scan to the cached model grid.

    Returns (rows, cols, depth) of grid cells that a LiDAR point falls into (nearest cell centre
    within half a model pixel), keeping the nearest return per cell (simple occlusion handling).
    """
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    homog = np.c_[pts, np.ones(len(pts))]
    cam0 = (calib["Tr"] @ homog.T).T  # rectified camera-0 coordinates
    uvw = (calib["P2"] @ cam0.T).T
    depth = uvw[:, 2]
    ok = depth > .5
    u, v, depth = uvw[ok, 0] / depth[ok], uvw[ok, 1] / depth[ok], depth[ok]
    (oh, ow), (mh, mw) = orig_hw, model_hw
    x = (u + .5) * mw / ow - .5
    y = (v + .5) * mh / oh - .5
    c, r = np.rint(x / stride).astype(int), np.rint(y / stride).astype(int)
    near = (np.abs(x - stride * c) <= .5) & (np.abs(y - stride * r) <= .5)
    gh, gw = -(-mh // stride), -(-mw // stride)
    inside = near & (r >= 0) & (r < gh) & (c >= 0) & (c < gw) & (stride * r < mh) & (stride * c < mw)
    r, c, depth = r[inside], c[inside], depth[inside]
    order = np.argsort(depth)[::-1]  # write far first so the nearest return wins
    grid = np.full((gh, gw), np.nan)
    grid[r[order], c[order]] = depth[order]
    rr, cc = np.nonzero(np.isfinite(grid))
    return rr, cc, grid[rr, cc]
