"""End-to-end CPU experiment runner: everything after frontend (VGGT) inference.

  python -m mr_cpu.pipeline --config configs/kitti_mid.json                 # all steps
  python -m mr_cpu.pipeline --config configs/kitti_mid.json --steps solve report

Steps (each reads earlier outputs from the run directory):
  edges   cache -> <run>/<seq>/edges.npz      deployable measurements + features (no GT)
  labels  edges + GT -> <run>/<seq>/labels.npz reference gauges and 7-DoF edge errors
  audit   cache + LiDAR -> <run>/<seq>/depth_audit.npz (skipped if a sequence has no LiDAR)
  qual    cache + images + LiDAR -> qualitative arrays (<run>/<seq>/qual/) and figures (mr_cpu/qualitative.py)
  fit     fitting sequences -> <run>/fitted.json (edge calibration maps, power law, pixel maps)
  tune    fitting sequences -> <run>/kernels.json (robust-kernel threshold per scheme)
  solve   all sequences -> <run>/metrics.csv, <run>/trajectories/*.npz (positions + solved window poses)
  upstream  <run>/upstream/<seq>_w<W>_o<O>/camera_poses.txt (VGGT-Long's own code) -> <run>/upstream.csv
  report  -> <run>/summary.md, <run>/tables/*.csv, <run>/figures/*.png
The per-sequence steps only need the cache of their sequence, so a disk-limited run can do
edges..qual for one sequence, delete its cache windows (keep manifest.json) and move on.
Ground truth of the held-out (test) sequences is used only by `labels` (for scoring), `solve`
(ATE) and the oracle scheme (f), which is a hindsight diagnostic, not a method.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import backend, depth_audit, edges, evaluation, kitti, labels
from .cache import Cache
from .calibration import Isotonic, equal_count_edges, fit_power_law, loglog_slope
from .weighting import SCHEMES, Fitted

DEFAULTS = {
    "kitti_root": "data/kitti", "cache_root": "cache/kitti", "run_dir": "results/mid",
    "fit_sequences": ["05", "07"], "test_sequences": ["00", "02"],
    "alignment": {"delta": 0.1, "max_iters": 5, "rel_threshold": 0.1, "max_points": None},  # all points, as VGGT-Long
    "schemes": list(SCHEMES.values()), "conf_feature": "conf_mean", "geo_feature": "residual_rel",
    "kernel_grid": {"huber": [0.5, 1, 2, 3, 5, 10, 20], "dcs": [1, 3, 10, 30, 100, 300, 1000]},
    "kernel_on": "loops", "prefix_lengths": [250, 500, 1000, 1500, 2000, 3000, 4000, 5000],
    "audit_max_per_frame": None,  # every LiDAR pixel
}
STEPS = ("edges", "labels", "audit", "qual", "fit", "tune", "solve", "upstream", "report")
PER_SEQUENCE = ("edges", "labels", "audit", "qual")


def load_config(path=None, **overrides):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _run(cfg):
    return Path(cfg["run_dir"])


def _cache_dir(cfg, seq):
    return Path(cfg["cache_root"]) / seq


def _seqs(cfg):
    return list(dict.fromkeys(cfg["fit_sequences"] + cfg["test_sequences"]))


def _gt(cfg, seq):
    n = Cache(_cache_dir(cfg, seq)).manifest["n_frames"]
    return kitti.gt_cam2_poses(cfg["kitti_root"], seq)[:n]


def _load_seq(cfg, seq):
    table, frames, poses = edges.load(_run(cfg) / seq / "edges.npz")
    with np.load(_run(cfg) / seq / "labels.npz") as d:
        lab = {k: d[k] for k in d.files}
    return table, frames, poses, lab


# ------------------------------------------------------------------ steps

def step_edges(cfg, seq):
    out = _run(cfg) / seq
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    table, frames, poses = edges.build(_cache_dir(cfg, seq), **cfg["alignment"])
    edges.save(out / "edges.npz", table, frames, poses)
    print(f"  [{seq}] {int((table['kind'] == 0).sum())} seams, {int((table['kind'] == 1).sum())} loops "
          f"({time.time() - t0:.0f}s)")


def step_labels(cfg, seq):
    table, frames, poses = edges.load(_run(cfg) / seq / "edges.npz")
    gauges, infos = labels.window_gauges(frames, poses, _gt(cfg, seq))
    ok = np.array([i["valid"] for i in infos])
    err, valid = labels.edge_errors(gauges, ok, table["i"], table["j"], table["z"])
    np.savez(_run(cfg) / seq / "labels.npz", errors=err, valid=valid, gauges=gauges, gauge_valid=ok,
             gauge_spread_m=[i["gt_spread_m"] for i in infos], gauge_pos_rms_rel=[i["position_rms_rel"] for i in infos],
             gauge_rot_rms_deg=[i["rotation_rms_deg"] for i in infos], gauge_scale=[i["scale_m_per_unit"] for i in infos])
    print(f"  [{seq}] labels: {valid.sum()}/{len(valid)} edges usable; "
          f"{(~ok).sum()} degenerate windows (car nearly stopped)")


def step_audit(cfg, seq):
    cache = Cache(_cache_dir(cfg, seq))
    if cache.manifest.get("engine") == "synthetic":
        fn = depth_audit.cache_lidar_fn(_cache_dir(cfg, seq))
    else:
        if not (Path(cfg["kitti_root"]) / "sequences" / seq / "velodyne").exists():
            print(f"  [{seq}] no LiDAR downloaded; skipping depth audit")
            return
        fn = depth_audit.kitti_lidar_fn(cfg["kitti_root"], seq, cache.manifest["model_hw"], cache.manifest["orig_hw"])
    data = depth_audit.collect(_cache_dir(cfg, seq), fn, cfg["audit_max_per_frame"], log=None)
    if not len(data["lidar"]):
        print(f"  [{seq}] no LiDAR pixels matched any window frame; skipping")
        return
    np.savez_compressed(_run(cfg) / seq / "depth_audit.npz", **data)
    print(f"  [{seq}] depth audit: {len(data['lidar'])} LiDAR pixels")


def step_qual(cfg, seq):
    from .qualitative import extract
    extract(cfg, seq)


def _audit(cfg, seq):
    p = _run(cfg) / seq / "depth_audit.npz"
    if not p.exists():
        return None
    with np.load(p) as d:  # confidences are stored as float16 (their cache precision); compute in float32
        return {k: d[k].astype(np.float32) if d[k].dtype == np.float16 else d[k] for k in d.files}


def step_fit(cfg):
    tables, errs, valid, pc, pe = [], [], [], [], []
    for seq in cfg["fit_sequences"]:
        table, _, _, lab = _load_seq(cfg, seq)
        tables.append(table)
        errs.append(lab["errors"])
        valid.append(lab["valid"])
        a = _audit(cfg, seq)
        if a is not None:
            ok = np.isfinite(a["p_err"]) & (a["p_err"] > 0)
            pc.append(a["p_conf"][ok])
            pe.append(a["p_err"][ok])
    pc = np.concatenate(pc) if pc else None
    pe = np.concatenate(pe) if pe else None
    fitted = Fitted.fit(tables, errs, valid, pc, pe, cfg["conf_feature"], cfg["geo_feature"])
    d = fitted.d
    d["pixel"] = {}
    for head in ("d", "p"):
        cs, es = [], []
        for seq in cfg["fit_sequences"]:
            a = _audit(cfg, seq)
            if a is not None:
                ok = np.isfinite(a[head + "_err"]) & (a[head + "_err"] > 0)
                cs.append(a[head + "_conf"][ok])
                es.append(a[head + "_err"][ok])
        if not cs:
            continue
        c, e = np.concatenate(cs), np.concatenate(es)
        k, p = fit_power_law(c, e)
        d["pixel"][head] = {"n": int(len(c)), "raw_kappa": float(np.sqrt(np.mean((e * c) ** 2))),
                            "power_law": {"K": k, "p": p}, "loglog_slope": loglog_slope(c, e)[0],
                            "isotonic": Isotonic.fit(c, e ** 2, decreasing=True, floor=1e-8).to_dict(),
                            "bin_edges": equal_count_edges(c, 10).tolist()}
    (_run(cfg) / "fitted.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"  fitted on {d['n_seams']} seams / {d['n_loops']} loops from {cfg['fit_sequences']}; "
          f"pixel power law p = {d['power_law']['p']:.3f}")


def _chain(table):
    seam = np.flatnonzero(table["kind"] == 0)
    n = int(max(table["i"].max(), table["j"].max())) + 1
    poses = [np.eye(4)]
    for k in seam[np.argsort(table["i"][seam])]:
        poses.append(poses[-1] @ table["z"][k])
    if len(poses) != n:
        raise ValueError("Seams do not form a chain over all windows")
    return np.stack(poses)


def solve_one(table, frames, poses, gt, fitted, scheme, kern="none", kparam=None, loops=True,
              errors=None, kernel_on="loops", prefix=None):
    use = np.ones(len(table["kind"]), bool) if loops else table["kind"] == 0
    sub = {k: v[use] for k, v in table.items()}
    info = fitted.information(scheme, sub, None if errors is None else errors[use])
    mode = "upstream" if scheme == "vggt_long" else "ours"
    kmask = sub["kind"] == 1 if kernel_on == "loops" else np.ones(len(sub["kind"]), bool)
    init = _chain(table)
    t0 = time.time()
    if loops and (sub["kind"] == 1).any():
        nodes, rep = backend.solve(init, sub["i"], sub["j"], sub["z"], info, mode, kern, kparam, kmask)
    else:
        nodes, rep = init, {"iterations": 0, "chi2_unweighted": 0.0, "edge_weight": np.ones(len(sub["kind"]))}
    est = evaluation.assemble(nodes, frames, poses, len(gt))
    m = evaluation.evaluate(est, gt, prefix)
    m.update(iterations=rep["iterations"], solve_s=time.time() - t0,
             loops_used=int((sub["kind"] == 1).sum()),
             loops_downweighted=int(np.sum(rep["edge_weight"][sub["kind"] == 1] < .5)) if loops else 0)
    return est, m, nodes


def step_tune(cfg):
    fitted = Fitted(json.loads((_run(cfg) / "fitted.json").read_text(encoding="utf-8")))
    data = {seq: _load_seq(cfg, seq) + (_gt(cfg, seq),) for seq in cfg["fit_sequences"]}
    out = {}
    for scheme in cfg["schemes"]:
        out[scheme] = {}
        for kern, grid in cfg["kernel_grid"].items():
            scores = []
            for p in grid:
                ates = [solve_one(t, f, po, gt, fitted, scheme, kern, p, True, lab["errors"], cfg["kernel_on"])[1]["ate_rmse_m"]
                        for (t, f, po, lab, gt) in data.values()]
                scores.append(float(np.mean(ates)))
            best = int(np.argmin(scores))
            out[scheme][kern] = {"best": grid[best], "grid": grid, "mean_fit_ate": scores}
        print(f"  {scheme}: huber k={out[scheme]['huber']['best']}, dcs phi={out[scheme]['dcs']['best']}")
    (_run(cfg) / "kernels.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def step_solve(cfg):
    fitted = Fitted(json.loads((_run(cfg) / "fitted.json").read_text(encoding="utf-8")))
    kp = _run(cfg) / "kernels.json"
    kernels = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else {}
    rows, curves = [], []
    traj_dir = _run(cfg) / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for seq in _seqs(cfg):
        table, frames, poses, lab = _load_seq(cfg, seq)
        gt = _gt(cfg, seq)
        split = "fit" if seq in cfg["fit_sequences"] else "test"
        keep = {"gt": gt[:, :3, 3]}
        for scheme in cfg["schemes"]:
            for kern in ("none", "huber", "dcs"):
                kparam = None if kern == "none" else kernels.get(scheme, {}).get(kern, {}).get("best")
                if kern != "none" and kparam is None:
                    continue
                for loops in (False, True):
                    if not loops and kern != "none":
                        continue  # kernels only act on loop edges
                    est, m, nodes = solve_one(table, frames, poses, gt, fitted, scheme, kern, kparam, loops,
                                              lab["errors"], cfg["kernel_on"], cfg["prefix_lengths"])
                    rows.append({"sequence": seq, "split": split, "scheme": scheme, "kernel": kern,
                                 "kernel_param": kparam, "loops": loops,
                                 **{k: v for k, v in m.items() if k != "ate_vs_length"}})
                    for n, a in m["ate_vs_length"]:
                        curves.append({"sequence": seq, "scheme": scheme, "kernel": kern, "loops": loops,
                                       "prefix_frames": n, "ate_rmse_m": a})
                    if kern == "none":
                        keep[f"{scheme}_{'loops' if loops else 'chain'}"] = est[:, :3, 3]
                        keep[f"{scheme}_{'loops' if loops else 'chain'}_nodes"] = nodes  # window -> world Sim(3)
            mine = {(r["kernel"], r["loops"]): r["ate_rmse_m"] for r in rows if r["sequence"] == seq and r["scheme"] == scheme}
            print(f"  [{seq}] {scheme:10s} ATE [m]: chain {mine[('none', False)]:.2f} | loops {mine[('none', True)]:.2f}"
                  + "".join(f" | {k} {mine[(k, True)]:.2f}" for k in ("huber", "dcs") if (k, True) in mine), flush=True)
        np.savez(traj_dir / f"{seq}.npz", **keep)
    _write_csv(_run(cfg) / "metrics.csv", rows)
    _write_csv(_run(cfg) / "ate_vs_length.csv", curves)


def step_upstream(cfg):
    """Score upstream VGGT-Long runs (its own code) copied to <run>/upstream/<seq>_w<W>_o<O>/camera_poses.txt."""
    from .trajectory import read_poses
    rows = []
    for d in sorted((_run(cfg) / "upstream").glob("*_w*_o*")):
        if not (d / "camera_poses.txt").exists():
            continue
        seq, w, o = d.name.split("_")
        est = read_poses(d / "camera_poses.txt")
        gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)
        if len(est) != len(gt):
            raise ValueError(f"{d}: {len(est)} poses for {len(gt)} frames")
        m = evaluation.evaluate(est, gt, cfg["prefix_lengths"])
        np.save(d / "positions.npy", est[:, :3, 3])
        rows.append({"sequence": seq, "window": int(w[1:]), "overlap": int(o[1:]), "run": d.name,
                     **{k: v for k, v in m.items() if k != "ate_vs_length"},
                     **{f"ate_first_{n}": a for n, a in m["ate_vs_length"]}})
        print(f"  upstream {d.name}: ATE {m['ate_rmse_m']:.2f} m, t_rel {m['t_rel_percent']:.2f} %")
    _write_csv(_run(cfg) / "upstream.csv", rows)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))  # rows may differ (e.g. per-sequence prefix lengths)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def step_report(cfg):
    from .report import write_report
    write_report(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    ap.add_argument("--cache-root")
    ap.add_argument("--run-dir")
    ap.add_argument("--fit", nargs="+", dest="fit_sequences")
    ap.add_argument("--test", nargs="+", dest="test_sequences")
    a = ap.parse_args()
    cfg = load_config(a.config, cache_root=a.cache_root, run_dir=a.run_dir,
                      fit_sequences=a.fit_sequences, test_sequences=a.test_sequences)
    _run(cfg).mkdir(parents=True, exist_ok=True)
    (_run(cfg) / "config_used.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    for step in STEPS:
        if step not in a.steps:
            continue
        print(f"== {step}", flush=True)
        if step in PER_SEQUENCE:
            for seq in _seqs(cfg):
                globals()[f"step_{step}"](cfg, seq)
        else:
            globals()[f"step_{step}"](cfg)


if __name__ == "__main__":
    main()
