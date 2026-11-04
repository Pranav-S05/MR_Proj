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
  stress tests with injected false loops; ORB-SLAM3 (mono); streaming models; VGGT-Omega bonus.
- Raw LiDAR audit arrays (6.1 GB, the only copy) are kept on the lab server in
  `/scratch2/vpr-team/kavish/mr_audit_arrays/`; everything else of the run is in `results/mid/`.

## Known limitations
- Loop-edge calibration rests on few fitting loops (07 has one usable, 05 ten).
- DCS can reject true loops when the starting chain has drifted a lot (02 with schemes b, c, d, g).
- VGGT-Long's spread-ratio scale estimate is slightly biased by noise/outliers (inherited on purpose for parity).
- Labels are operational (window gauge fitted to GT poses), not physically observed seam errors.
