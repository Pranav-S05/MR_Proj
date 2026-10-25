# Project status, 25 October 2026

## Done
- **Data and conventions.** KITTI poses and calibration for 00-10 parsed; ground truth converted from camera 0 to the
  colour camera 2 that VGGT sees (6 cm offset); per-sequence image sizes (00-02 1241x376, 03 1242x375, 04-10 1226x370);
  VGGT's 518x154 preprocessing mapped exactly. LiDAR projection checked visually on real KITTI 07 frames
  (`results/checks/lidar_overlay_kitti07.png`).
- **Frontend, run on an RTX A4000.** `mr_gpu/run_vggt.py` on 07, 05, 00, 02 with window 32 / overlap 16 (bf16):
  memory sweep (`results/memory_sweep.csv`: 9.6 GB peak at 32 frames, about 18 frames/s), smoke test 4/4 PASS,
  SALAD loops 3 / 12 / 20 / 8, every one a true revisit (ground-truth distance at most 4.1 m).
- **VGGT-Long reproduction (back end).** Our alignment equals VGGT-Long's IRLS code to ~1e-15. Our pose graph in
  "upstream" mode reaches the same optimum as VGGT-Long's pypose optimiser (same ATE to the mm).
- **Experiment pipeline on real data** (`results/mid/summary.md`). Seams and loops with features, ground-truth gauges
  and 7-DoF edge errors (stationary windows excluded), LiDAR audit over every scan of the four sequences
  (20.8M pixels, at most 800 per frame), isotonic calibration on 05/07, power-law baseline, seven weighting schemes,
  Huber/DCS tuned on 05/07, loops on/off, ATE / ATE vs length / KITTI t_rel.
- **Checks.** 42 unit tests pass. Our ATE equals evo on all 56 trajectories of the real run (max difference 1e-9 m).

## Main numbers (ATE in metres, loops on, scheme (a) = VGGT-Long's weighting)
| | 05 (fit) | 07 (fit) | 00 (held-out) | 02 (held-out) |
|---|---|---|---|---|
| no loops | 48.62 | 22.29 | 140.42 | 181.58 |
| (a) VGGT-Long | 10.03 | 4.04 | 9.20 | 105.10 |
| (e) calibrated | 9.56 | 4.33 | 9.11 | 117.16 |

Pixel audit (held-out, depth head): calibration error 0.40 raw, 0.26 power law, 0.18 isotonic; log-log slope about
-0.7 (the loss optimum would give -1).

## Findings to check further
- **Loop windows merge revisits.** On 02 the two passes of a revisit are 1-4 m apart (different lanes), but VGGT's loop
  window places them 0.1-1.1 m apart, while each pass alone is reconstructed to 0.1-0.4 m. Loop-edge translation error
  is close to the true lane offset; confidence and inlier ratios do not flag it. Loop uncertainty fitted on 05 (revisits
  0.1-0.8 m apart) therefore under-predicts held-out loop errors by about 20x.
- Seam translation errors are hard to rank from confidence (Spearman -0.12); the alignment residual ranks them better
  (0.55).
- 02 keeps a large ATE because its loops only connect frames 936-1963 with 4207-4546; the long middle part has none.

## Pending
1. Upstream `vggt_long.py` runs (its own 75/30 setting and our 32/16) scored with `python -m mr_cpu.evaluation`.
2. Rerun without subsampling: full-resolution caches (`--stride 1`) and the audit without the 800 px/frame cap.
3. Qualitative results: trajectory overlays, point clouds, depth / confidence next to LiDAR, loop-window image pairs
   (incl. the 02 lane-offset case).
Items 1-2 wait for space on the lab server's shared disk (~60 GB).

## Known limitations
- Loop-edge calibration rests on few fitting loops (07 has one usable, 05 ten).
- DCS can reject true loops when the starting chain has drifted a lot (seen on 02 with schemes b and c).
- VGGT-Long's spread-ratio scale estimate is slightly biased by noise/outliers (inherited on purpose for parity).
- Labels are operational (window gauge fitted to GT poses), not physically observed seam errors.
- Waymo and TUM loaders, stress tests with injected false loops, streaming models and VGGT-Omega are after Oct 26.
