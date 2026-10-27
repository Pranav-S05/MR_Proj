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
  chunk-60 number on 02; the other 15 paper comparisons agree.
- **Loop windows merge revisits.** On 02 the two passes of a revisit are 1-4 m apart (different lanes), but VGGT's
  loop window places them 0.1-1.1 m apart, while each pass alone is reconstructed to 0.1-0.4 m
  (`tables/loops_<seq>.csv`, `figures/qualitative/loops_02.png`). Loop-edge error tracks the true lane offset;
  confidence and inlier ratios do not flag it; loop uncertainty fitted on 05 under-predicts held-out loops ~20x.
- **Seams.** Seam translation errors are hard to rank from confidence (Spearman -0.12); the alignment residual ranks
  them better (0.55). The worst seams are low-texture vegetation and turns with low confidence
  (`figures/qualitative/seams_<seq>.png`).
- **Weighting.** Calibrated weights help slightly on 00 and 05, not on 07, and hurt on 02, where the loop edges
  themselves are biased; small windows also drift more (02: ~100 m at chunk 30 in the paper too).

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
