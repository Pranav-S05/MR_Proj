"""Small KITTI/C2W reader and global-alignment ATE preflight.

This is not the official KITTI segment-error evaluator or a replacement for evo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .geometry import fit_similarity, transform_points, validate_camera_poses


def read_poses(path):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] == 12:
        poses = np.tile(np.eye(4), (len(rows), 1, 1))
        poses[:, :3] = rows.reshape(-1, 3, 4)
    elif rows.shape[1] == 16:
        poses = rows.reshape(-1, 4, 4)
    else:
        raise ValueError("Expected 12 KITTI values or 16 C2W values per row; TUM timestamps are not supported here")
    return validate_camera_poses(poses)


def ate(estimated_xyz, reference_xyz, alignment="sim3"):
    estimated_xyz, reference_xyz = np.asarray(estimated_xyz, float), np.asarray(reference_xyz, float)
    if estimated_xyz.ndim != 2 or estimated_xyz.shape[1] != 3 or estimated_xyz.shape != reference_xyz.shape or len(estimated_xyz) == 0:
        raise ValueError("ATE requires matched, nonempty N x 3 positions; frame association is your responsibility")
    if not np.all(np.isfinite(estimated_xyz)) or not np.all(np.isfinite(reference_xyz)):
        raise ValueError("ATE does not accept missing or nonfinite positions")
    if alignment not in ("none", "se3", "sim3"):
        raise ValueError("Alignment must be none, se3 or sim3")
    transform = np.eye(4) if alignment == "none" else fit_similarity(estimated_xyz, reference_xyz, estimate_scale=alignment == "sim3")
    aligned = transform_points(transform, estimated_xyz)
    errors = np.linalg.norm(aligned - reference_xyz, axis=1)
    return {"frames": len(errors), "alignment": alignment, "ate_rmse": float(np.sqrt(np.mean(errors**2))), "alignment_scale": float(np.cbrt(np.linalg.det(transform[:3, :3])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--estimate", type=Path)
    parser.add_argument("--alignment", choices=["none", "se3", "sim3"], default="sim3")
    args = parser.parse_args()
    reference = read_poses(args.reference)
    xyz = reference[:, :3, 3]
    result = {"reference": str(args.reference), "frames": len(xyz), "path_length_reference_units": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()), "status": "Reference inspection only; no SLAM estimate supplied"}
    if args.estimate is not None:
        estimate = read_poses(args.estimate)
        result["evaluation"] = ate(estimate[:, :3, 3], xyz, args.alignment)
        result["status"] = "Position ATE; assumes identical frame IDs, camera convention and ordering"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
