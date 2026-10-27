"""Tables and figures from a pipeline run directory (called by `pipeline --steps report`)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, spearmanr

from . import edges
from .calibration import Isotonic, calibration_error, equal_count_edges, gaussian_nll, loglog_slope, reliability
from .cache import Cache
from .evaluation import sim3_align
from .geometry import transform_points
from .labels import block_targets
from .weighting import BLOCKS, Fitted, info_from_var

# VGGT-Long paper (Deng et al., arXiv:2507.16443v2), Table 1: KITTI ATE RMSE [m], overlap = chunk size / 2
PAPER_ATE = {30: {"00": 9.06, "01": 96.20, "02": 99.95, "03": 19.76, "04": 11.36, "05": 10.20, "06": 10.04,
                  "07": 4.00, "08": 139.79, "09": 50.53, "10": 40.94},
             60: {"00": 8.06, "01": 96.96, "02": 34.16, "03": 6.83, "04": 4.16, "05": 9.15, "06": 4.68,
                  "07": 2.68, "08": 63.15, "09": 32.24, "10": 27.87},
             90: {"00": 11.97, "01": 40.51, "02": 49.85, "03": 5.41, "04": 2.86, "05": 9.88, "06": 6.07,
                  "07": 3.47, "08": 66.27, "09": 32.27, "10": 21.33},
             120: {"00": 16.13, "01": 53.43, "02": 51.98, "03": 4.37, "04": 2.15, "05": 12.69, "06": 11.33,
                   "07": 3.603, "08": 70.29, "09": 34.55, "10": 21.05}}
LABEL = {"vggt_long": "(a) VGGT-Long (identity)", "uniform": "(b) uniform", "raw_conf": "(c) raw confidence",
         "geometry": "(d) geometry-only", "calibrated": "(e) calibrated (ours)", "oracle": "(f) oracle*",
         "power_law": "(g) power law"}
COLOR = {"vggt_long": "#d55e00", "uniform": "#7f7f7f", "raw_conf": "#cc79a7", "geometry": "#e69f00",
         "calibrated": "#0072b2", "oracle": "#009e73", "power_law": "#56b4e9", "gt": "#222222"}


def _read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows):
    if rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fmt(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def _load(run, seq):
    table, frames, poses = edges.load(run / seq / "edges.npz")
    with np.load(run / seq / "labels.npz") as d:
        lab = {k: d[k] for k in d.files}
    return table, lab


def _audit(run, seq):
    p = run / seq / "depth_audit.npz"
    if not p.exists():
        return None
    with np.load(p) as d:  # confidences are stored as float16 (their cache precision); compute in float32
        return {k: d[k].astype(np.float32) if d[k].dtype == np.float16 else d[k] for k in d.files}


def write_report(cfg):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(cfg["run_dir"])
    figs, tabs = run / "figures", run / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)
    fitted = Fitted(json.loads((run / "fitted.json").read_text(encoding="utf-8")))
    metrics = _read_csv(run / "metrics.csv")
    fit_s, test_s = cfg["fit_sequences"], cfg["test_sequences"]
    engine = Cache(Path(cfg["cache_root"]) / test_s[0]).manifest.get("engine", "vggt")
    md = [f"# Results: {run.name}", ""]
    if engine == "synthetic":
        md += ["> **SYNTHETIC FRONTEND.** These numbers come from `mr_cpu/synthetic.py` (real KITTI trajectories, "
               "simulated VGGT outputs). They validate the pipeline, not VGGT or VGGT-Long.", ""]
    md += [f"Fitting sequences: {', '.join(fit_s)}. Held-out sequences: {', '.join(test_s)}. "
           f"Window {Cache(Path(cfg['cache_root']) / test_s[0]).manifest['window']}, overlap "
           f"{Cache(Path(cfg['cache_root']) / test_s[0]).manifest['overlap']}.", ""]

    # ---------------- trajectory tables
    def ate(seq, scheme, kern, loops):
        for r in metrics:
            if r["sequence"] == seq and r["scheme"] == scheme and r["kernel"] == kern and r["loops"] == str(loops):
                return r
        return None

    for split, seqs in (("Held-out", test_s), ("Fitting (in-sample)", fit_s)):
        rows, csv_rows = [], []
        chain = [ate(s, "vggt_long", "none", False) for s in seqs]
        rows.append(["no loops (chain; identical for every scheme)", "-"] +
                    [f"{_fmt(c['ate_rmse_m'])} / {_fmt(c['t_rel_percent'])}" if c else "-" for c in chain])
        for scheme in cfg["schemes"]:
            for kern in ("none", "huber", "dcs"):
                rs = [ate(s, scheme, kern, True) for s in seqs]
                if not any(rs):
                    continue
                param = rs[0]["kernel_param"] if rs[0] else ""
                rows.append([LABEL[scheme], kern if kern == "none" else f"{kern} ({param})"] +
                            [f"{_fmt(r['ate_rmse_m'])} / {_fmt(r['t_rel_percent'])}" if r else "-" for r in rs])
                for s, r in zip(seqs, rs):
                    if r:
                        csv_rows.append({"split": split, "sequence": s, "scheme": scheme, "kernel": kern,
                                         "kernel_param": param, "ate_rmse_m": r["ate_rmse_m"],
                                         "t_rel_percent": r["t_rel_percent"], "r_rel_deg_per_100m": r["r_rel_deg_per_100m"],
                                         "loops_used": r["loops_used"], "loops_downweighted": r["loops_downweighted"]})
        md += [f"## {split} trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)", "",
               _md_table(["weighting", "kernel"] + [f"KITTI {s}" for s in seqs], rows), ""]
        _write_csv(tabs / f"trajectory_{'test' if split == 'Held-out' else 'fit'}.csv", csv_rows)
    md += ["\\* The oracle uses the true edge errors of that sequence (hindsight diagnostic, not a method). "
           "Kernel thresholds were chosen on the fitting sequences only.", ""]

    # ---------------- edge calibration on held-out edges
    calib_rows, fig_rel = [], {}
    fit_tabs = [_load(run, s) for s in fit_s]
    test_tabs = [_load(run, s) for s in test_s]
    cat = lambda pairs, key: np.concatenate([p[0][key] for p in pairs])
    err_t = np.concatenate([p[1]["errors"] for p in test_tabs])
    ok_t = np.concatenate([p[1]["valid"] for p in test_tabs])
    tab_t = {k: cat(test_tabs, k) for k in test_tabs[0][0]}
    tab_f = {k: cat(fit_tabs, k) for k in fit_tabs[0][0]}
    tg_t = block_targets(err_t, tab_t["f_char_length"])
    for kind_name, kind in (("seams", 0), ("loops", 1)):
        m = ok_t & (tab_t["kind"] == kind)
        if m.sum() < 3:
            continue
        sub = {k: v[m] for k, v in tab_t.items()}
        for scheme in ("uniform", "raw_conf", "geometry", "calibrated", "power_law"):
            var_t = fitted.variances(scheme, sub)
            var_f = fitted.variances(scheme, {k: v[tab_f["kind"] == kind] for k, v in tab_f.items()})
            row = {"edges": kind_name, "scheme": scheme, "n": int(m.sum())}
            info = info_from_var(var_t, sub["f_char_length"])
            nees = np.einsum("ni,nij,nj->n", err_t[m], info, err_t[m])
            row["nees_mean_target7"] = float(np.mean(nees))
            row["nees_cover95_target0.95"] = float(np.mean(nees <= chi2.ppf(.95, 7)))
            for b in BLOCKS:
                edges_b = equal_count_edges(var_f[b], 5) if len(np.unique(var_f[b])) > 5 else np.array([-np.inf, np.inf])
                rel = reliability(var_t[b], tg_t[b][m], var_t[b], edges_b)
                row[f"{b}_msq_ratio_target1"] = float(np.mean(tg_t[b][m]) / np.mean(var_t[b]))
                row[f"{b}_calib_err"] = calibration_error(rel)
                row[f"{b}_nll"] = gaussian_nll(tg_t[b][m], var_t[b])
                if kind == 0:
                    fig_rel.setdefault(b, {})[scheme] = rel
            calib_rows.append(row)
    _write_csv(tabs / "edge_calibration_test.csv", calib_rows)
    if calib_rows:
        md += ["## Edge (seam/loop) uncertainty on held-out edges", "",
               "Per-coordinate second moments before any graph rescaling. `msq ratio` = mean(error^2)/mean(predicted "
               "variance) (1 is calibrated); `calib err` = count-weighted mean |ratio - 1| over bins of predicted variance "
               "fixed on the fitting edges; NEES uses all 7 DoF (7 expected, 95% inside the chi-square 95% quantile).", "",
               _md_table(["edges", "scheme", "n", "rot ratio", "trans ratio", "scale ratio", "rot err", "trans err",
                          "scale err", "NEES", "cover95"],
                         [[r["edges"], LABEL[r["scheme"]], r["n"], _fmt(r["rot_msq_ratio_target1"]),
                           _fmt(r["trans_msq_ratio_target1"]), _fmt(r["scale_msq_ratio_target1"]), _fmt(r["rot_calib_err"]),
                           _fmt(r["trans_calib_err"]), _fmt(r["scale_calib_err"]), _fmt(r["nees_mean_target7"], 1),
                           _fmt(r["nees_cover95_target0.95"])] for r in calib_rows]), ""]
    # ranking ability of features (Spearman of block error vs feature), held-out seams
    m = ok_t & (tab_t["kind"] == 0)
    rank_rows = []
    for feat in ("conf_mean", "conf_median", "conf_p10", "residual_rel", "inlier_fraction", "spread_condition", "valid_fraction"):
        if "f_" + feat not in tab_t:
            continue
        x = tab_t["f_" + feat][m]
        rank_rows.append([feat] + (["-"] * len(BLOCKS) if np.ptp(x) == 0 else  # constant feature: undefined
                                   [_fmt(spearmanr(x, tg_t[b][m])[0]) for b in BLOCKS]))
    md += ["### Does the feature rank seam errors? (Spearman, held-out seams)", "",
           _md_table(["feature", "rotation", "translation", "scale"], rank_rows), ""]

    # ---------------- pixel audit
    pix = fitted.d.get("pixel", {})
    pix_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, head, title in zip(axes, ("d", "p"), ("depth head (depth_conf)", "point head (world_points_conf)")):
        if head not in pix:
            ax.set_visible(False)
            continue
        audits = [a for a in (_audit(run, s) for s in test_s) if a is not None]
        scope = "held-out"
        if not audits:
            audits, scope = [a for a in (_audit(run, s) for s in fit_s) if a is not None], "fitting (in-sample)"
        c = np.concatenate([a[head + "_conf"] for a in audits])
        e = np.concatenate([a[head + "_err"] for a in audits])
        ok = np.isfinite(e) & (e > 0)
        c, e = c[ok], e[ok]
        p = pix[head]
        bins = np.array(p["bin_edges"])
        preds = {"raw  kappa/C": (p["raw_kappa"] / c) ** 2,
                 "power law": (p["power_law"]["K"] * c ** -p["power_law"]["p"]) ** 2,
                 "isotonic": Isotonic.from_dict(p["isotonic"]).predict(c)}
        row = {"head": title, "scope": scope, "n_pixels": len(c), "loglog_slope_fit": p["loglog_slope"],
               "loglog_slope_eval": loglog_slope(c, e)[0], "power_law_p": p["power_law"]["p"],
               "spearman_conf_err": float(spearmanr(c[::10], e[::10])[0])}
        for name, var in preds.items():
            rel = reliability(c, e ** 2, var, bins)
            row["calib_err_" + name.split()[0]] = calibration_error(rel)
            ax.plot([r["mean_x"] for r in rel], [r["predicted_rms"] for r in rel], "o--", label=f"predicted: {name}")
        rel = reliability(c, e ** 2, preds["isotonic"], bins)
        ax.plot([r["mean_x"] for r in rel], [r["empirical_rms"] for r in rel], "ko-", lw=2, label=f"measured ({scope})")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("confidence (bin mean)"), ax.set_ylabel("RMS relative depth error")
        ax.set_title(title, fontsize=10), ax.legend(fontsize=8), ax.grid(alpha=.3, which="both")
        pix_rows.append(row)
    fig.suptitle("Pixel confidence vs LiDAR depth error (per-window scale-aligned)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "pixel_calibration.png", dpi=130)
    plt.close(fig)
    _write_csv(tabs / "pixel_audit.csv", pix_rows)
    if pix_rows:
        md += ["## Pixel confidence audit (LiDAR)", "",
               "A slope of -1 is what the loss optimum C = alpha/E[e] would give. Calibration errors use bins fixed on "
               "the fitting sequences; `raw` uses the best single constant kappa/C.", "",
               _md_table(["head", "scope", "pixels", "slope (fit)", "slope (eval)", "Spearman", "calib err raw",
                          "calib err power law", "calib err isotonic"],
                         [[r["head"], r["scope"], r["n_pixels"], _fmt(r["loglog_slope_fit"]), _fmt(r["loglog_slope_eval"]),
                           _fmt(r["spearman_conf_err"]), _fmt(r["calib_err_raw"]), _fmt(r["calib_err_power"]),
                           _fmt(r["calib_err_isotonic"])] for r in pix_rows]), ""]

    # ---------------- figures: edge reliability, trajectories, ATE vs length, error vs confidence
    if fig_rel:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, b in zip(axes, BLOCKS):
            hi = 0
            for scheme, rel in fig_rel.get(b, {}).items():
                x = [r["predicted_rms"] for r in rel]
                y = [r["empirical_rms"] for r in rel]
                ax.plot(x, y, "o-", color=COLOR[scheme], label=LABEL[scheme])
                hi = max(hi, max(x), max(y))
            ax.plot([0, hi], [0, hi], "k:", lw=1)
            unit = {"rot": "rad", "trans": "x char. length", "scale": "log-scale"}[b]
            ax.set_xlabel(f"predicted RMS ({unit})"), ax.set_ylabel("measured RMS")
            ax.set_title({"rot": "rotation", "trans": "translation", "scale": "scale"}[b]), ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("Held-out seam uncertainty: predicted vs measured (diagonal = calibrated)", fontsize=11)
        fig.tight_layout()
        fig.savefig(figs / "edge_calibration.png", dpi=130)
        plt.close(fig)
    # seam error vs confidence
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for split, pairs, col in (("fit", fit_tabs, "#999999"), ("held-out", test_tabs, "#0072b2")):
        t = {k: np.concatenate([p[0][k] for p in pairs]) for k in pairs[0][0]}
        e = np.concatenate([p[1]["errors"] for p in pairs])
        ok = np.concatenate([p[1]["valid"] for p in pairs]) & (t["kind"] == 0)
        tg = block_targets(e, t["f_char_length"])
        for ax, b in zip(axes, BLOCKS):
            ax.scatter(t["f_conf_mean"][ok], np.sqrt(tg[b][ok]), s=4, alpha=.5, color=col, label=split)
    for ax, b in zip(axes, BLOCKS):
        x = np.linspace(tab_f["f_conf_mean"].min(), tab_f["f_conf_mean"].max(), 200)
        ax.plot(x, np.sqrt(Isotonic.from_dict(fitted.d["models"]["calibrated"][b]).predict(x)), "k-", lw=2, label="isotonic fit")
        ax.set_yscale("log"), ax.set_xlabel("seam mean point confidence"), ax.set_ylabel(f"|{b} error| (per coord.)")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "seam_error_vs_confidence.png", dpi=130)
    plt.close(fig)
    # trajectories
    for seq in dict.fromkeys(test_s + fit_s):
        p = run / "trajectories" / f"{seq}.npz"
        if not p.exists():
            continue
        with np.load(p) as d:
            tr = {k: d[k] for k in d.files}
        fig, ax = plt.subplots(figsize=(6.5, 6))
        gt = tr["gt"]
        ax.plot(gt[:, 0], gt[:, 2], color=COLOR["gt"], lw=2.5, label="ground truth")
        for key, style in (("vggt_long_chain", ":"), ("vggt_long_loops", "-"), ("calibrated_loops", "-"), ("uniform_loops", "--")):
            if key in tr:
                al = transform_points(sim3_align(tr[key], gt), tr[key])
                scheme = key.rsplit("_", 1)[0]
                ax.plot(al[:, 0], al[:, 2], style, color=COLOR[scheme], lw=1.4,
                        label=LABEL[scheme] + (" no loops" if key.endswith("chain") else ""))
        ax.set_aspect("equal"), ax.set_xlabel("x [m]"), ax.set_ylabel("z [m]"), ax.grid(alpha=.3)
        ax.set_title(f"KITTI {seq} ({'held-out' if seq in test_s else 'fitting'}), Sim(3)-aligned", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / f"trajectory_{seq}.png", dpi=130)
        plt.close(fig)
    # ATE vs length
    curves = _read_csv(run / "ate_vs_length.csv")
    if curves:
        fig, axes = plt.subplots(1, len(test_s), figsize=(5.5 * len(test_s), 4), squeeze=False)
        for ax, seq in zip(axes[0], test_s):
            for scheme in cfg["schemes"]:
                for loops, style in (("True", "-"),):
                    pts = [(int(r["prefix_frames"]), float(r["ate_rmse_m"])) for r in curves
                           if r["sequence"] == seq and r["scheme"] == scheme and r["kernel"] == "none" and r["loops"] == loops]
                    if pts:
                        x, y = zip(*sorted(pts))
                        ax.plot(x, y, style + "o", ms=3, color=COLOR[scheme], label=LABEL[scheme])
            pts = [(int(r["prefix_frames"]), float(r["ate_rmse_m"])) for r in curves
                   if r["sequence"] == seq and r["scheme"] == "vggt_long" and r["loops"] == "False"]
            if pts:
                x, y = zip(*sorted(pts))
                ax.plot(x, y, ":", color="k", label="no loops")
            ax.set_xlabel("sequence length (frames)"), ax.set_ylabel("ATE RMSE [m]"), ax.set_title(f"KITTI {seq}")
            ax.grid(alpha=.3)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "ate_vs_length.png", dpi=130)
        plt.close(fig)

    # ---------------- upstream VGGT-Long (its own code, scored with our metrics) vs our reproduction
    up = _read_csv(run / "upstream.csv")
    if up:
        settings = sorted({(int(r["window"]), int(r["overlap"])) for r in up}, reverse=True)
        man = Cache(Path(cfg["cache_root"]) / test_s[0]).manifest
        rows, seqs = [], [s for s in dict.fromkeys(fit_s + test_s) if any(r["sequence"] == s for r in up)]
        for seq in seqs:
            u = {(int(r["window"]), int(r["overlap"])): r for r in up if r["sequence"] == seq}
            ours = [ate(seq, "vggt_long", "none", True), ate(seq, "calibrated", "none", True)]
            paper = [PAPER_ATE.get(w, {}).get(seq) if o * 2 == w else None for w, o in settings]
            rows.append([seq] + [f"{_fmt(u[s]['ate_rmse_m'])} / {_fmt(u[s]['t_rel_percent'])}"
                                 + (f" (paper {p:.2f})" if p is not None else "") if s in u else "-"
                                 for s, p in zip(settings, paper)]
                        + [f"{_fmt(r['ate_rmse_m'])} / {_fmt(r['t_rel_percent'])}" if r else "-" for r in ours])
        md += ["## Upstream VGGT-Long (original code) vs our reproduction", "",
               "ATE [m] / KITTI t_rel [%]. Upstream = `vggt_long.py` at the pinned commit with its KITTI config "
               "(only chunk size / overlap changed), scored with our metrics; `paper` = ATE in Table 1 of the "
               "VGGT-Long paper (arXiv:2507.16443v2, overlap = half the chunk). Ours uses window "
               f"{man['window']} / overlap {man['overlap']}; scheme (a) is VGGT-Long's identity weighting.", "",
               _md_table(["seq"] + [f"upstream W{w}/O{o}" for w, o in settings]
                         + [f"ours (a) W{man['window']}/O{man['overlap']}", f"ours (e) W{man['window']}/O{man['overlap']}"],
                         rows), ""]
        fig, axes = plt.subplots(1, len(seqs), figsize=(4.6 * len(seqs), 4.6), squeeze=False)
        for ax, seq in zip(axes[0], seqs):
            p = run / "trajectories" / f"{seq}.npz"
            if not p.exists():
                continue
            with np.load(p) as d:
                gt, ours = d["gt"], d["vggt_long_loops"]
            ax.plot(gt[:, 0], gt[:, 2], color=COLOR["gt"], lw=2.5, label="ground truth")
            styles = [("-", "#009e73"), ("--", "#56b4e9"), (":", "#009e73"), ("-.", "#cc79a7"), ("-", "#e69f00")]
            for (w, o), (style, col) in zip(settings, styles):
                q = run / "upstream" / f"{seq}_w{w}_o{o}" / "positions.npy"
                if q.exists():
                    est = np.load(q)
                    al = transform_points(sim3_align(est, gt), est)
                    ax.plot(al[:, 0], al[:, 2], style, color=col, lw=1.2, label=f"upstream W{w}/O{o}")
            al = transform_points(sim3_align(ours, gt), ours)
            ax.plot(al[:, 0], al[:, 2], "-", color=COLOR["vggt_long"], lw=1, label="ours (a)")
            ax.set_aspect("equal"), ax.grid(alpha=.3), ax.set_title(f"KITTI {seq}", fontsize=10)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figs / "upstream_vs_ours.png", dpi=130)
        plt.close(fig)

    # ---------------- qualitative figures (point clouds need the solved window poses)
    from .qualitative import render
    qual = render(cfg, LABEL)
    qdir = figs / "qualitative"
    if qdir.exists():
        names = sorted(p.name for p in qdir.glob("*.png"))
        md += ["## Qualitative figures (`figures/qualitative/`)", "",
               "* `depth_lidar_<seq>`: image, depth head, its confidence and the scale-aligned error at LiDAR pixels",
               "* `loops_<seq>`: loop windows with the largest / smallest edge error (per-loop numbers in "
               "`tables/loops_<seq>.csv`)",
               "* `seams_<seq>`: seams with the largest / smallest true error and both windows' confidence",
               "* `pointcloud_<seq>`, `trajectory_error_<seq>`: points and trajectories placed by the solved poses",
               "* `upstream_pointcloud_<run>`: upstream VGGT-Long's own point export", "",
               f"{len(names)} files: " + ", ".join(f"`{n}`" for n in names), ""]
        print(f"  qualitative: {len(qual)} rendered after solve, {len(names)} in total")

    # ---------------- systems (GPU runs record timings in the manifest)
    sys_rows = []
    for seq in dict.fromkeys(test_s + fit_s):
        man = Cache(Path(cfg["cache_root"]) / seq).manifest
        t = man.get("timing", {})
        sys_rows.append([seq, man["n_frames"], len(man["windows"]), len(man.get("loop_windows", [])),
                         _fmt(t.get("peak_gpu_mem_gb")), _fmt(t.get("sec_per_window")), _fmt(t.get("frames_per_sec"), 1),
                         man.get("gpu", "-")])
    md += ["## Frontend cost", "", _md_table(["seq", "frames", "windows", "loop windows", "peak GPU mem [GB]",
                                             "s / window", "frames / s", "GPU"], sys_rows), "",
           "Figures: `figures/pixel_calibration.png`, `figures/edge_calibration.png`, "
           "`figures/seam_error_vs_confidence.png`, `figures/ate_vs_length.png`, `figures/trajectory_<seq>.png`"
           + (", `figures/upstream_vs_ours.png`" if up else "") + ".", ""]
    (run / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {run / 'summary.md'} and {len(list(figs.glob('*.png')))} figures")
