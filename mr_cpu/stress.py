"""False-loop stress test (proposal Step 6, following DCS, Agarwal et al. 2013).

Adversarial version: add k in {0, 1, 2, 4, 8, 16} wrong loop edges between distant windows, for 20
random seeds, and solve every scheme x kernel. Every fake loop gets the same high weight: the
information matrix of the most trusted real loop of that sequence under that scheme. So the weights
cannot reject them; only the robust kernel can. A run "survives" if its ATE stays within 1.25x of the
same scheme and kernel with no fake loops.

A fake loop claims that window j's first camera sits where window i's first camera is (the identity
Sim(3)), with a small random perturbation, which is what a wrong place-recognition match between two
look-alike streets would assert. Pairs are at least `min_gap` windows apart in time and at least
`min_dist` metres apart in the ground truth, so every fake loop is truly wrong. The same pairs (per
seed) are used for every scheme and kernel.

  python -m mr_cpu.stress --config configs/kitti_mid.json --out results/final/stress
  python -m mr_cpu.stress --config configs/kitti_mid.json --out results/final/stress --summary-only
  python -m mr_cpu.stress --config configs/kitti_realistic.json --realistic --out results/final/stress
  python -m mr_cpu.stress --config configs/kitti_mid.json --dcs-sweep wide --out results/final/stress
  python -m mr_cpu.stress --config configs/kitti_mid.json --dcs-sweep fine --phis 0.0562,0.1,0.178 --out results/final/stress

Realistic version: would SALAD place recognition itself ever propose a wrong loop if its threshold
were lowered? See `realistic` below; it needs only the per-frame SALAD descriptors (descriptors.npy).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import backend, cache, evaluation, kitti, lie
from .pipeline import _chain, _load_seq, _run, load_config
from .weighting import Fitted

K_VALUES = (0, 1, 2, 4, 8, 16)
KERNELS = ("none", "huber", "dcs")
SURVIVE = 1.25


def window_positions(frames, gt):
    """Ground-truth position of the first frame of every window (frames: list of per-window frame ids)."""
    return gt[[int(f[0]) for f in frames], :3, 3]


def fake_loops(pos, k, rng, min_gap=10, min_dist=50.0, rot_deg=5.0, trans=0.1, log_scale=0.05):
    """k wrong loop edges (i < j) between distant windows, with perturbed identity measurements."""
    n = len(pos)
    ii, jj = np.triu_indices(n, min_gap)
    far = np.linalg.norm(pos[ii] - pos[jj], axis=1) >= min_dist
    ii, jj = ii[far], jj[far]
    if k > len(ii):
        raise ValueError(f"Only {len(ii)} distant window pairs, cannot place {k} fake loops")
    pick = rng.choice(len(ii), k, replace=False)
    xi = np.zeros((k, 7))
    axis = rng.normal(size=(k, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    xi[:, :3] = axis * np.radians(rng.normal(0, rot_deg, (k, 1)))
    xi[:, 3:6] = rng.normal(0, trans, (k, 3))
    xi[:, 6] = rng.normal(0, log_scale, k)
    return ii[pick], jj[pick], lie.exp(xi)


def _with_fakes(table, info, fi, fj, fz):
    """Append fake loop edges; each gets the information of the most trusted real loop."""
    loop = np.flatnonzero(table["kind"] == 1)
    best = loop[np.argmax(np.trace(info[loop], axis1=1, axis2=2))]
    k = len(fi)
    i = np.r_[table["i"], fi]
    j = np.r_[table["j"], fj]
    z = np.concatenate([table["z"], fz])
    inf = np.concatenate([info, np.repeat(info[best][None], k, 0)])
    kind = np.r_[table["kind"], np.ones(k, int)]
    fake = np.r_[np.zeros(len(table["kind"]), bool), np.ones(k, bool)]
    return i, j, z, inf, kind, fake


def _setup(cfg, seq, k, seed):
    """What every solve of one (sequence, k, seed) shares: graph, ground truth, fitted maps, fake loops."""
    table, frames, poses, lab = _load_seq(cfg, seq)
    gt = kitti.gt_cam2_poses(cfg["kitti_root"], seq)
    n = max(int(f.max()) for f in frames) + 1
    if n != len(gt):
        raise ValueError(f"{seq}: windows cover {n} frames, ground truth has {len(gt)}")
    fitted = Fitted(json.loads((_run(cfg) / "fitted.json").read_text(encoding="utf-8")))
    rng = np.random.default_rng(1000 * seed + k)
    fakes = fake_loops(window_positions(frames, gt), k, rng) if k else (
        np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4, 4)))
    return table, frames, poses, lab, gt, n, fitted, fakes


def _solve(init, frames, poses, n, gt, edges, mode, kern, kparam):
    """One graph solve; ATE and how many fake / real loops the kernel switched off (weight < 0.5)."""
    i, j, z, inf, kind, fake = edges
    t0 = time.time()
    nodes, rep = backend.solve(init, i, j, z, inf, mode, kern, kparam, kind == 1)
    m = evaluation.evaluate(evaluation.assemble(nodes, frames, poses, n), gt)
    w = rep["edge_weight"]
    return {"ate_rmse_m": m["ate_rmse_m"], "t_rel_percent": m["t_rel_percent"],
            "fake_rejected": int(np.sum(w[fake] < .5)), "real_loops_rejected": int(np.sum(w[(kind == 1) & ~fake] < .5)),
            "iterations": rep["iterations"], "solve_s": round(time.time() - t0, 3)}


def _task(args):
    """All schemes x kernels for one (sequence, k, seed)."""
    cfg, seq, k, seed = args
    table, frames, poses, lab, gt, n, fitted, fakes = _setup(cfg, seq, k, seed)
    kernels = json.loads((_run(cfg) / "kernels.json").read_text(encoding="utf-8"))
    init = _chain(table)
    rows = []
    for scheme in cfg["schemes"]:
        edges = _with_fakes(table, fitted.information(scheme, table, lab["errors"]), *fakes)
        mode = "upstream" if scheme == "vggt_long" else "ours"
        for kern in KERNELS:
            kparam = None if kern == "none" else kernels[scheme][kern]["best"]
            rows.append({"sequence": seq, "scheme": scheme, "kernel": kern, "kernel_param": kparam, "k": k, "seed": seed,
                         **_solve(init, frames, poses, n, gt, edges, mode, kern, kparam)})
    return rows


def run(cfg, out, seeds, sequences, workers):
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(cfg, s, 0, 0) for s in sequences]
    tasks += [(cfg, s, k, seed) for s in sequences for k in K_VALUES if k for seed in range(seeds)]
    path = out / "adversarial.csv"
    t0, done, header = time.time(), 0, None
    with open(path, "w", newline="", encoding="utf-8") as f, Pool(workers) as pool:
        for rows in pool.imap_unordered(_task, tasks):
            if header is None:
                header = list(rows[0])
                wr = csv.DictWriter(f, header)
                wr.writeheader()
            wr.writerows(rows)
            f.flush()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
    return path


def summarise(path, out):
    """Two survival rules: ATE within 1.25x of (1) the same scheme and kernel with no fake loops, and
    (2, strict) the same scheme with no fake loops and no kernel, so a kernel that already throws away
    true loops on the clean graph (DCS on 02) does not count as surviving."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    base = {(r["sequence"], r["scheme"], r["kernel"]): float(r["ate_rmse_m"]) for r in rows if r["k"] == "0"}
    agg = {}
    for r in rows:
        key = (r["sequence"], r["scheme"], r["kernel"], int(r["k"]))
        a = float(r["ate_rmse_m"])
        agg.setdefault(key, []).append((a, a <= SURVIVE * base[key[:3]], a <= SURVIVE * base[key[0], key[1], "none"],
                                        int(r["fake_rejected"]), int(r["real_loops_rejected"])))
    table = []
    for (seq, scheme, kern, k), v in sorted(agg.items()):
        a = np.array([x[0] for x in v])
        table.append({"sequence": seq, "scheme": scheme, "kernel": kern, "k": k, "runs": len(v),
                      "ate_median_m": round(float(np.median(a)), 2),
                      "ate_p25_m": round(float(np.percentile(a, 25)), 2),
                      "ate_p75_m": round(float(np.percentile(a, 75)), 2),
                      "survival": round(float(np.mean([x[1] for x in v])), 2),
                      "survival_strict": round(float(np.mean([x[2] for x in v])), 2),
                      "fake_rejected_mean": round(float(np.mean([x[3] for x in v])), 2),
                      "real_rejected_mean": round(float(np.mean([x[4] for x in v])), 2)})
    with open(out / "adversarial_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(table[0]))
        wr.writeheader()
        wr.writerows(table)

    def largest(seq, scheme, kern, col):
        """Largest k survived in every seed (k = 0 included: '-' if even the clean graph fails the rule)."""
        best = "-"
        for r in sorted((r for r in table if (r["sequence"], r["scheme"], r["kernel"]) == (seq, scheme, kern)),
                        key=lambda r: r["k"]):
            if r[col] < 1.0:
                break
            best = str(r["k"])
        return best

    lines = ["| seq | scheme | none | huber | dcs | dcs (strict) |", "|---|---|---|---|---|---|"]
    for seq in sorted({r["sequence"] for r in table}):
        for scheme in dict.fromkeys(r["scheme"] for r in table):
            cells = [largest(seq, scheme, kern, "survival") for kern in KERNELS] + [largest(seq, scheme, "dcs", "survival_strict")]
            lines.append(f"| {seq} | {scheme} | " + " | ".join(cells) + " |")
    (out / "adversarial_survival.md").write_text(
        "Largest number of fake loops survived in all 20 seeds. none/huber/dcs: ATE within 1.25x of the same\n"
        "scheme and kernel with no fake loops. dcs (strict): within 1.25x of the same scheme with no fake loops\n"
        "and no kernel ('-': DCS already fails this on the clean graph).\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ DCS threshold sweep
# The DCS threshold was tuned on the clean fitting sequences on a grid (1 to 1000) that suits weighted
# schemes, whose residuals are in standard deviations. With identity weights every grid value gave the
# same ATE. Is there ANY threshold, even one picked with hindsight on the test sequences, with which
# identity weights reject the fake loops and keep the true ones? "wide": 0.000001 to 1000 in half
# decades, 5 seeds, k in {1, 16}; "fine": the given thresholds, all seeds and k. Same fake loops as above.
SWEEP_SCHEMES = ("vggt_long", "equal")
SWEEP_WIDE = tuple(float(f"{x:.3g}") for x in 10 ** np.arange(-6, 3.01, 0.5))


def _sweep_task(args):
    cfg, seq, scheme, k, seed, phis = args
    table, frames, poses, lab, gt, n, fitted, fakes = _setup(cfg, seq, k, seed)
    edges = _with_fakes(table, fitted.information(scheme, table, lab["errors"]), *fakes)
    mode = "upstream" if scheme == "vggt_long" else "ours"
    init = _chain(table)
    runs = [("dcs", phi) for phi in phis] + ([("none", None)] if k == 0 else [])
    return [{"sequence": seq, "scheme": scheme, "kernel": kern, "phi": phi, "k": k, "seed": seed,
             **_solve(init, frames, poses, n, gt, edges, mode, kern, phi)} for kern, phi in runs]


def dcs_sweep(cfg, out, sequences, workers, name, phis, seeds, ks):
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(cfg, s, sc, 0, 0, phis) for s in sequences for sc in SWEEP_SCHEMES]
    tasks += [(cfg, s, sc, k, sd, phis) for s in sequences for sc in SWEEP_SCHEMES for k in ks for sd in range(seeds)]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for d, r in enumerate(pool.imap_unordered(_sweep_task, tasks), 1):
            rows += r
            if d % 20 == 0 or d == len(tasks):
                print(f"  {d}/{len(tasks)} tasks, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["sequence"], r["scheme"], r["kernel"], r["phi"] or 0, r["k"], r["seed"]))
    with open(out / f"dcs_sweep_{name}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    # strict survival: every seed within 1.25x of the same scheme with no fake loops and no kernel
    clean = {(r["sequence"], r["scheme"]): r["ate_rmse_m"] for r in rows if r["kernel"] == "none"}
    lines = ["| scheme | phi | " + " | ".join(sequences) + " |", "|---|---|" + "---|" * len(sequences)]
    for scheme in SWEEP_SCHEMES:
        for phi in phis:
            cells = []
            for seq in sequences:
                best = "-"
                for k in (0,) + tuple(ks):
                    a = [r["ate_rmse_m"] for r in rows if (r["sequence"], r["scheme"], r["phi"], r["k"]) == (seq, scheme, phi, k)]
                    if not a or max(a) > SURVIVE * clean[seq, scheme]:
                        break
                    best = str(k)
                cells.append(best)
            lines.append(f"| {scheme} | {phi:g} | " + " | ".join(cells) + " |")
    (out / f"dcs_sweep_{name}.md").write_text(
        f"DCS threshold sweep ({name}), identity weights. Largest number of fake loops (of k = {', '.join(map(str, ks))}) "
        f"survived in all {seeds} seeds: ATE within 1.25x of the same scheme with no fake loops and no kernel "
        "('-': the kernel already fails this on the clean graph).\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ realistic version
# Lower SALAD's similarity threshold and check what it returns, from the per-frame SALAD descriptors
# alone (no VGGT run is needed to find out whether a false loop would be proposed). Two retrieval rules:
#   vggt_long      VGGT-Long's rule: top 5 most similar frames, then drop pairs <= 10 frames apart.
#                  The top 5 are usually the frame's own neighbours, so the threshold barely matters.
#   non_neighbour  drop pairs <= 10 frames apart first, then take the top 5.
# A revisit loop joins frames > 100 frames (10 s) apart; the rest are same-pass loops (11-100 frames
# apart on the same stretch of road, looking at the same scene ahead: valid matches, even when the
# frames are 20 m apart). A revisit loop is false if its frames are more than 20 m apart in the ground
# truth (true revisits are at most ~4 m apart, mid report Table 4).
THRESHOLDS = (0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)
FALSE_LOOP_M = 20.0
REVISIT_GAP = 100


def realistic(cfg, out, sequences):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in sequences:
        desc = np.load(Path(cfg["cache_root"]) / seq / "descriptors.npy").astype(np.float32)
        pos = kitti.gt_cam2_poses(cfg["kitti_root"], seq)[:, :3, 3].astype(np.float32)
        if len(pos) != len(desc):
            raise ValueError(f"{seq}: {len(desc)} descriptors but {len(pos)} poses")
        far = np.linalg.norm(pos[:, None] - pos[None], axis=-1) > 50.0
        best_far = float((desc @ desc.T)[far].max())  # most similar pair of frames > 50 m apart
        for rule in ("vggt_long", "non_neighbour"):
            for thr in THRESHOLDS:
                pairs = cache.find_loop_pairs(desc, thr, skip_neighbours=rule == "non_neighbour")
                dist = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b, _ in pairs])
                gap = np.array([a - b for a, b, _ in pairs])
                rev = gap > REVISIT_GAP
                rows.append({"sequence": seq, "rule": rule, "threshold": thr, "loops": len(pairs),
                             "same_pass_loops": int((~rev).sum()), "revisit_loops": int(rev.sum()),
                             "false_loops": int((rev & (dist > FALSE_LOOP_M)).sum()),
                             "max_revisit_gt_dist_m": round(float(dist[rev].max()), 2) if rev.any() else None,
                             "max_same_pass_gt_dist_m": round(float(dist[~rev].max()), 2) if (~rev).any() else None,
                             "max_similarity_over_50m": round(best_far, 4)})
                print(rows[-1])
    with open(out / "realistic.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out / 'realistic.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", default="results/final/stress")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sequences", nargs="*")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--realistic", action="store_true",
                    help="realistic version: SALAD loops at lowered thresholds (needs <cache_root>/<seq>/descriptors.npy)")
    ap.add_argument("--dcs-sweep", choices=["wide", "fine"],
                    help="DCS threshold sweep for identity weights (wide: 5 seeds, k 1 and 16; fine: --phis, all seeds and k)")
    ap.add_argument("--phis", help="comma-separated DCS thresholds for --dcs-sweep fine")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seqs = a.sequences or list(dict.fromkeys(cfg["test_sequences"] + cfg["fit_sequences"]))
    out = Path(a.out)
    if a.realistic:
        realistic(cfg, out, seqs)
        return
    if a.dcs_sweep == "wide":
        dcs_sweep(cfg, out, seqs, a.workers, "wide", SWEEP_WIDE, 5, (1, 16))
        return
    if a.dcs_sweep == "fine":
        dcs_sweep(cfg, out, seqs, a.workers, "fine", tuple(float(x) for x in a.phis.split(",")), a.seeds,
                  tuple(k for k in K_VALUES if k))
        return
    path = out / "adversarial.csv" if a.summary_only else run(cfg, out, a.seeds, seqs, a.workers)
    summarise(path, out)


if __name__ == "__main__":
    main()
