"""Shared helpers of the scale study (CPU). Run scripts from the repository root: python scripts/<script>.py

A "variant" is a folder results/final/runs/<variant>/<seq>/ written by scripts/gpu_run.py; the special variant "mid" reads the main
project's mid-submission results (first-frame reference, window 32, overlap 16; drives 00, 02, 05, 07).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repository root
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "results" / "final" / "runs"      # small result files of every run (written by scripts/gpu_run.py)
RES = ROOT / "results" / "final" / "tables"     # tables, statistics, SUMMARY.md
FIG = ROOT / "results" / "final" / "figures"    # figures of the final report
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
from mr_cpu import backend, edges, evaluation, kitti, labels, lie  # noqa: E402

KITTI = ROOT / "data" / "kitti"
ALL = [f"{k:02d}" for k in range(11)]


def run_dir(variant, seq):
    return ROOT / "results" / "mid" / seq if variant == "mid" else RUNS / variant / seq


def available(variant):
    return [s for s in ALL if (run_dir(variant, s) / "edges.npz").exists()]


def load(variant, seq):
    """Edge table, cameras, ground truth and GT-based labels of one drive."""
    d = run_dir(variant, seq)
    table, frames, poses = edges.load(d / "edges.npz")
    gt = kitti.gt_cam2_poses(KITTI, seq)
    n = max(int(f.max()) for f in frames) + 1
    gt = gt[:n]
    gauges, infos = labels.window_gauges(frames, poses, gt)
    gok = np.array([i["valid"] for i in infos])
    err, valid = labels.edge_errors(gauges, gok, table["i"], table["j"], table["z"])
    seam = np.flatnonzero(table["kind"] == 0)
    seam = seam[np.argsort(table["i"][seam])]
    out = dict(seq=seq, variant=variant, table=table, frames=frames, poses=poses, gt=gt, gauges=gauges, gauge_ok=gok,
               err=err, valid=valid, seam=seam, n_loops=int((table["kind"] == 1).sum()))
    if (d / "extras.npz").exists():
        with np.load(d / "extras.npz") as x:
            out["extras"] = {k: x[k] for k in x.files}
    return out


def chain(table):
    seam = np.flatnonzero(table["kind"] == 0)
    nodes = [np.eye(4)]
    for k in seam[np.argsort(table["i"][seam])]:
        nodes.append(nodes[-1] @ table["z"][k])
    return np.stack(nodes)


def solve(d, z_seam=None, loops=True, info=None, mode="ours", return_nodes=False):
    """ATE (m) of the chain (loops=False) or of the solved pose graph. z_seam replaces the seam measurements."""
    t = {k: v.copy() for k, v in d["table"].items()}
    if z_seam is not None:
        t["z"][d["seam"]] = z_seam
    nodes = chain(t)
    if loops and d["n_loops"]:
        inf = np.tile(np.eye(7), (len(t["i"]), 1, 1)) if info is None else info
        nodes, _ = backend.solve(nodes, t["i"], t["j"], t["z"], inf, mode, "none", None, t["kind"] == 1)
    est = evaluation.assemble(nodes, d["frames"], d["poses"], len(d["gt"]))
    a = evaluation.ate(est[:, :3, 3], d["gt"][:, :3, 3])
    return (a, nodes, est) if return_nodes else a


def overlap_cameras(d, e):
    """Cameras of the shared frames of seam e as seen by window i and by window j: (c_i, c_j), each (n,4,4)."""
    i, j = int(d["table"]["i"][e]), int(d["table"]["j"][e])
    _, ii, jj = np.intersect1d(d["frames"][i], d["frames"][j], return_indices=True)
    return d["poses"][i][ii], d["poses"][j][jj]


def rescale(z, centre_j, factor):
    """Multiply the scale of Sim(3) z (maps j -> i) by `factor`, keeping the image of point centre_j fixed."""
    s, r, _ = lie.split(z[None])
    target = (z @ np.r_[centre_j, 1.0])[:3]
    s1 = s * factor
    return lie.from_srt(s1, r, target[None] - s1 * (r[0] @ centre_j))[0]


def rescale_seams(d, log_factor):
    """Seam measurements with scale multiplied by exp(log_factor[k]) about window j's overlap camera centroid."""
    out = []
    for n, e in enumerate(d["seam"]):
        _, cj = overlap_cameras(d, e)
        out.append(rescale(d["table"]["z"][e], cj[:, :3, 3].mean(0), float(np.exp(log_factor[n]))))
    return np.stack(out)


def seam_features(d):
    """Deployable (no ground truth) per-seam features, in seam order. NaN where the car hardly moves."""
    t, rows = d["table"], []
    ex = d.get("extras")
    for n, e in enumerate(d["seam"]):
        ci, cj = overlap_cameras(d, e)
        f = {}
        si = np.linalg.norm(np.diff(ci[:, :3, 3], axis=0), axis=1)
        sj = np.linalg.norm(np.diff(cj[:, :3, 3], axis=0), axis=1)
        s_pt = float(lie.split(t["z"][e][None])[0][0])
        f["log_s_point"] = np.log(s_pt)
        moving = len(si) >= 4 and si.sum() / t["f_char_length"][e] / s_pt > 0.3 and si.min() > 0 and sj.min() > 0
        if moving:
            lr = np.log(si / sj)  # log scale j -> i from each shared step
            q = np.arange(len(lr)) - (len(lr) - 1) / 2
            f["pose_minus_point"] = float(np.median(lr)) - f["log_s_point"]
            f["pose_slope"] = float((q * (lr - lr.mean())).sum() / (q * q).sum())
            f["speed"] = float(np.log(sj.sum() / t["f_char_length"][e]))
        else:
            f["pose_minus_point"] = f["pose_slope"] = f["speed"] = np.nan
        f["residual_rel"] = float(np.log(t["f_residual_rel"][e]))
        f["log_conf"] = float(np.log(t["f_conf_mean"][e]))
        f["turn_deg"] = float(t["f_rotation_deg"][e])
        if ex is not None:
            dl = ex["seam_depth_lr"][n]  # (overlap, 5): median, near, middle, far, MAD of log(depth_i / depth_j)
            ok = np.isfinite(dl[:, 0])
            if ok.sum() >= 4:
                q = np.arange(len(dl))[ok] - np.arange(len(dl))[ok].mean()
                f["depth_minus_point"] = float(np.median(dl[ok, 0])) - f["log_s_point"]
                f["depth_slope"] = float((q * (dl[ok, 0] - dl[ok, 0].mean())).sum() / (q * q).sum())
                f["depth_near_far"] = float(np.median(dl[ok, 1] - dl[ok, 3]))
                f["depth_mad"] = float(np.median(dl[ok, 4]))
            else:
                f["depth_minus_point"] = f["depth_slope"] = f["depth_near_far"] = f["depth_mad"] = np.nan
            f["umeyama_minus_spread"] = float(ex["seam_log_s_umeyama"][n] - ex["seam_log_s_spread"][n])
        rows.append(f)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def step_profile(d, min_step_m=0.3):
    """Per full window: predicted step length / true step length for each step, normalised to geometric mean 1."""
    out = []
    full = max(len(f) for f in d["frames"])
    for f, c in zip(d["frames"], d["poses"]):
        if len(f) != full:
            continue
        ps = np.linalg.norm(np.diff(c[:, :3, 3], axis=0), axis=1)
        gs = np.linalg.norm(np.diff(d["gt"][f][:, :3, 3], axis=0), axis=1)
        if gs.min() < min_step_m or ps.min() <= 0:
            continue
        r = ps / gs
        out.append(r / np.exp(np.mean(np.log(r))))
    return np.array(out)
