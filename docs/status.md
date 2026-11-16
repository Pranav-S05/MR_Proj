# Project status, 26 October 2026

## Done (mid submission, everything at full resolution, nothing subsampled)
- **Data and conventions.** KITTI poses and calibration for 00-10 parsed; ground truth converted from camera 0 to the
  colour camera 2 that VGGT sees (6 cm offset); per-sequence image sizes; VGGT's 518x154 preprocessing mapped exactly.
  LiDAR projection checked visually (`results/checks/lidar_overlay_kitti07.png`).
- **Frontend, RTX A4000.** `mr_gpu/run_vggt.py` on 07, 05, 00, 02, window 32 / overlap 16, bf16, maps cached at full
  model resolution. Memory sweep (`results/memory_sweep.csv`: 9.6 GB peak at 32 frames, ~17 frames/s); smoke test
  4/4 PASS; SALAD loops 3 / 12 / 20 / 8, every one a true revisit (at most 4.1 m apart).
- **VGGT-Long reproduction.** Upstream `vggt_long.py` (pinned commit, its KITTI config) run on all four sequences at
  chunk/overlap 30/15 and 60/30 (the paper's Table 1 rows), 32/16 (our setting) and 75/30 (released config), scored
  with our metrics (`results/mid/upstream.csv`). Its loop detector finds exactly our loop pairs.
- **Experiment pipeline.** Seams and loops (alignment on every point above the confidence threshold), ground-truth
  gauges and 7-DoF edge errors, LiDAR audit of both heads on every LiDAR pixel of every scan (478M pixel samples),
  isotonic calibration on 05/07, power law, seven weighting schemes, Huber/DCS tuned on 05/07, loops on/off,
  ATE / ATE vs length / KITTI t_rel, qualitative figures (`results/mid/figures/qualitative/`).
- **Checks.** 44 unit tests pass. ATE equals evo on all 56 trajectories of our run and all 16 upstream runs
  (max difference 1e-9 m).

## Main numbers (ATE in metres)
| | 05 (fit) | 07 (fit) | 00 (held-out) | 02 (held-out) |
|---|---|---|---|---|
| paper, chunk 30 | 10.20 | 4.00 | 9.06 | 99.95 |
| upstream code, 30/15 | 10.24 | 3.98 | 9.07 | 99.65 |
| paper, chunk 60 | 9.15 | 2.68 | 8.06 | 34.16 |
| upstream code, 60/30 | 9.85 | 2.66 | 8.63 | 52.78 |
| upstream code, 32/16 | 10.04 | 4.09 | 9.13 | 104.61 |
| ours, no loops | 48.70 | 22.36 | 140.49 | 181.52 |
| ours (a) VGGT-Long weighting, 32/16 | 10.02 | 4.06 | 9.18 | 105.07 |
| ours (a') equal weights, our residual (control for b-g), 32/16 | 9.80 | 4.07 | 9.14 | 106.90 |
| ours (e) calibrated, 32/16 | 9.57 | 4.37 | 9.09 | 117.07 |

Pixel audit (held-out 00+02, 338M pixels, depth head): calibration error 0.40 raw, 0.25 power law, 0.18 isotonic;
log-log slope -0.70 (the loss optimum would give -1). Point head: 0.60 / 0.53 / 0.32, slope -0.63.

## Findings
- **Reproduction holds.** Upstream reproduces the paper's chunk-30 row within 0.5% on all four sequences, and our
  re-implementation (a) matches upstream at the same 32/16 setting within 0.5-1%. At chunk 60, 07/00/05 are within
  0-8% of the paper but 02 gives 52.8 m instead of 34.2 m. Checked (`results/mid/upstream_checks/`): a repeat run
  gives exactly 52.78 m again (deterministic), and the older loop threshold 0.7 instead of 0.85 finds the same 8 loops
  and the same 52.78 m (each frame's top-5 retrieval neighbours are mostly its own temporal neighbours, so the
  threshold does not matter on 02). The released code at the pinned commit therefore does not give the paper's
  chunk-60 number on 02. Of the other seven paper numbers, five are matched within 1% and two (05 and 00 at
  60/30) within 8%. Further checks on Nov 4 (`results/mid/upstream_checks/`): VGGT-Long's loop optimiser falls back
  from C++ to Python and always stops at 30 iterations, but 200 iterations give exactly 52.78 m again (cost settles at
  6.04e-5); the pre-refactor commit 97bbc20 (Oct 2025) gives exactly the current code's numbers (02: 52.78 at 60/30 and
  99.65 at 30/15; 05: 9.85; 07: 2.66; its 00 run ran out of GPU memory on a shared card); and the current code on 02 at
  90/45 gives 50.56 m against the paper's 49.85 m. So on 02 we match the paper at windows 30 and 90 but not 60; the
  paper's own 02 column (99.95, 34.16, 49.85, 51.98 for windows 30, 60, 90, 120) has 60 as the outlier. Not reproducible
  with the released code; cause unknown (remaining difference: compiled C++ solver, GPU).
- **Ground truth is inconsistent at the 02 revisits (corrected finding).** The ground-truth gap between the two visits
  of a loop on 02 is 1-4 m and almost all vertical (camera frame: sideways 0.03-1.1 m, vertical 0.85-4.0 m). Placing both
  visits' LiDAR scans with the ground-truth poses puts the two road surfaces 0.8-4.1 m apart in height at the same spot
  (00: up to 1.3 m; 05 and 07: within 0.2 m, two loops of 05 at 0.5 m), so the ground-truth heights are wrong there.
  VGGT putting the visits together is right; most of the measured loop-edge error on 02 is ground-truth error. The
  earlier "different lanes" explanation was wrong. Measured held-out loop squared errors are ~20x the predicted
  variance (about 4.7x in error size), but on 00/02 this is dominated by these label errors. Seam labels are not
  affected (one short window each). Check: `scripts/check_revisit_ground.py`.
- **Seams.** Seam translation errors are hard to rank from confidence (Spearman -0.12); the alignment residual ranks
  them better (0.55). The worst seams are low-texture vegetation and turns with low confidence
  (`figures/qualitative/seams_<seq>.png`).
- **Weighting.** Against the fair control (a'), calibrated weights help slightly on 05 (9.57 vs 9.80 m) and 00
  (9.09 vs 9.14 m), not on 07, and hurt on 02 (cause not isolated; the 02 loop labels used for fitting-based schemes
  are not involved, since fitting uses 05/07, but evaluation on 02 is against inconsistent ground truth). Small windows
  drift more (02: ~100 m at chunk 30 in the paper too). Scheme (a) vs (a') shows the residual convention alone moves
  ATE by up to 2%. The oracle not winning only shows that inverse true-error weighting is not an upper bound here.

## Pending (after the mid submission, as planned in the proposal)
- Waymo subset (download needs the team's Waymo Open Dataset account and licence acceptance) and TUM loaders;
  ORB-SLAM3 (mono); streaming models; VGGT-Omega bonus. (False-loop stress tests and MASt3R-SLAM: done, see the
  last section.)
- Raw LiDAR audit arrays (6.1 GB, the only copy) are kept on the lab server in
  `/scratch2/vpr-team/kavish/mr_audit_arrays/`; everything else of the run is in `results/mid/`.

## Known limitations
- Loop-edge calibration rests on few fitting loops (07 has one usable, 05 ten).
- DCS can reject true loops when the starting chain has drifted a lot (02 with schemes b, c, d, g).
- VGGT-Long's spread-ratio scale estimate is slightly biased by noise/outliers (inherited on purpose for parity).
- Labels are operational (window gauge fitted to GT poses), not physically observed seam errors.

## Final submission work (13 to 14 November 2026)
Code: `mr_cpu/stress.py`, `mr_cpu/final_figures.py`, `--tum-fps` in `mr_cpu/evaluation.py`, `tests/test_stress.py`
(47 tests pass). Results: `results/final/` (commands in README section 4).

- **Adversarial false loops** (`stress/adversarial_survival.md`, `figures/stress_*.png`). k = 0, 1, 2, 4, 8, 16 wrong loops
  between windows at least 10 windows and 50 m apart, 20 seeds, each wrong loop given the weight of the most trusted
  real loop. Without a kernel or with Huber, one wrong loop breaks every scheme. With DCS (threshold tuned on clean
  05/07), largest k survived in all seeds, strict rule (within 1.25x of the clean graph without a kernel):
  00: raw confidence and power law 16, calibrated and geometry 1, uniform and identity 0;
  02: only calibrated survives (16); the other weighted schemes fail already on the clean graph, because DCS throws
  away all 8 true loops (175 m instead of about 113 m); calibrated keeps all 8 (117 m);
  05: all weighted schemes 16 (oracle 8), identity 0; 07: geometry 16, raw confidence 2, the rest 0.
  No scheme wins on every drive. Identity weights (VGGT-Long and our control) never survive one wrong loop.
- **DCS threshold sweep for identity weights** (`stress/dcs_sweep_{wide,fine}.md`, `figures/dcs_sweep_identity.png`).
  Thresholds 1e-6 to 1000: at 0.03 and below DCS drops nearly all true loops (ATE falls back to the no-loop chain);
  at 1 and above it keeps one wrong loop. With 20 seeds no threshold survives one wrong loop on 00, and none
  survives on all four drives. The original grid (1 to 1000) was too high for unwhitened residuals, but no
  threshold works: with identity weights the residual size of true and wrong loops is not measured against any
  uncertainty.
- **Realistic false loops** (`stress/realistic.csv`, `figures/salad_pairs_*.png`). SALAD at thresholds 0.85 down to 0.5,
  with VGGT-Long's retrieval rule and with neighbours excluded before the top 5: no loop between places more than
  20 m apart on 00 or 02. Lowering the threshold only adds same-road matches 11 to 30 frames apart and a few more
  true revisits (at most 3.4 m apart on 00, 9.3 m on 02). The most similar frames more than 50 m apart score 0.62
  (00) and 0.56 (02). With VGGT-Long's rule the threshold hardly matters: a frame's top 5 are its own neighbours.
- **Error bars (side check, negative, code not kept).** Gauss-Newton covariances of the solved graph against true
  relative-pose errors: calibrated weights are not more honest than uniform or power law on 00 or 02; on 02 every
  scheme is 3 to 5 times overconfident over longer distances (errors of nearby seams are correlated, which
  independent per-edge weights cannot express).
- **MASt3R-SLAM baseline** (`mast3r_slam/summary.md`, `figures/mast3r_slam_*.png`, `speed_memory.md`). Unmodified code
  (commit e6f4e3d) and config, headless and single-threaded, every KITTI frame, lab server RTX A4000, without and
  with the KITTI intrinsics. Install check on TUM fr1/desk with its own eval scripts: 0.016 m calibrated and
  0.035 m uncalibrated, exactly the paper's Table 1, so the setup is right. On KITTI it loses tracking on every
  drive, as VGGT-Long reports ("TL"): first lost after frame 188 / 183 (00, uncalibrated / calibrated), 374 / 309
  (02), 61 / 489 (05), 468 / 413 (07); drive tracked (keyframes at most 3 s apart) 7/6%, 9/7%, 19/17%, 45/39% of
  the path. It relocalises only when the car returns to an already mapped place (00: near the start). ATE over the
  tracked keyframes only (not comparable with whole-drive ATE): 00 1.76 m / none (113 of 114 calibrated poses are
  NaN: the optimiser diverged), 02 9.68 / 25.41 m, 05 6.10 / 10.67 m, 07 13.57 / 44.27 m. Tracking is lost at
  junction turns with hard sun and shadow on 00 and 02 (`figures/mast3r_slam_lost_*.png`; an observation, not a
  tested cause). Runtime 3.4 to 5.5 frames/s, 4.7 GB for its main process (its backend process adds 0.8 to
  5.4 GB); our VGGT front end 5.0 to 5.6 frames/s at 9.6 GB.
- **Qualitative figures** (`results/final/figures/`): fake loops and the DCS solution for (a), (c), (e) on 00 and 02
  (seed 0); worst-seed ATE ratio against the number of fake loops; the DCS threshold sweep; SALAD's best true revisit
  and its most similar frames more than 50 m apart; MASt3R-SLAM keyframes on the map and the frames where it loses
  tracking.
- Lab server cleaned on 14 Nov: only `pixi/`, `.cache/pixi`, `mr_audit_arrays/` and the unrelated `repro/` remain.
