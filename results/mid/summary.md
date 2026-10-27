# Results: mid

Fitting sequences: 05, 07. Held-out sequences: 00, 02. Window 32, overlap 16.

## Held-out trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 00 | KITTI 02 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 140.49 / 34.47 | 181.52 / 40.90 |
| (a) VGGT-Long (identity) | none | 9.18 / 4.82 | 105.07 / 26.95 |
| (a) VGGT-Long (identity) | huber (5) | 9.18 / 4.82 | 105.07 / 26.95 |
| (a) VGGT-Long (identity) | dcs (10) | 9.18 / 4.82 | 105.07 / 26.95 |
| (b) uniform | none | 10.15 / 4.88 | 113.36 / 29.81 |
| (b) uniform | huber (1) | 10.15 / 4.88 | 113.36 / 29.81 |
| (b) uniform | dcs (100) | 10.15 / 4.88 | 175.65 / 39.15 |
| (c) raw confidence | none | 10.28 / 5.03 | 119.60 / 31.44 |
| (c) raw confidence | huber (2) | 10.28 / 5.03 | 119.60 / 31.44 |
| (c) raw confidence | dcs (100) | 10.28 / 5.03 | 175.68 / 39.18 |
| (d) geometry-only | none | 9.34 / 4.75 | 113.27 / 29.86 |
| (d) geometry-only | huber (0.5) | 9.34 / 4.75 | 113.27 / 29.86 |
| (d) geometry-only | dcs (100) | 9.34 / 4.75 | 174.54 / 38.81 |
| (e) calibrated (ours) | none | 9.09 / 4.68 | 117.07 / 30.59 |
| (e) calibrated (ours) | huber (0.5) | 9.09 / 4.68 | 117.07 / 30.59 |
| (e) calibrated (ours) | dcs (300) | 9.09 / 4.68 | 117.07 / 30.59 |
| (f) oracle* | none | 11.92 / 5.31 | 116.08 / 29.56 |
| (f) oracle* | huber (1) | 11.92 / 5.31 | 116.08 / 29.56 |
| (f) oracle* | dcs (300) | 11.92 / 5.31 | 116.08 / 29.56 |
| (g) power law | none | 9.89 / 4.89 | 119.70 / 31.47 |
| (g) power law | huber (0.5) | 9.85 / 4.89 | 119.70 / 31.47 |
| (g) power law | dcs (30) | 9.89 / 4.89 | 175.69 / 39.16 |

## Fitting (in-sample) trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 05 | KITTI 07 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 48.70 / 13.78 | 22.36 / 13.30 |
| (a) VGGT-Long (identity) | none | 10.02 / 5.11 | 4.06 / 3.94 |
| (a) VGGT-Long (identity) | huber (5) | 10.02 / 5.11 | 4.06 / 3.94 |
| (a) VGGT-Long (identity) | dcs (10) | 10.02 / 5.11 | 4.06 / 3.94 |
| (b) uniform | none | 11.52 / 5.83 | 3.61 / 3.53 |
| (b) uniform | huber (1) | 11.52 / 5.83 | 3.61 / 3.53 |
| (b) uniform | dcs (100) | 11.52 / 5.83 | 3.61 / 3.53 |
| (c) raw confidence | none | 10.05 / 5.12 | 4.55 / 3.84 |
| (c) raw confidence | huber (2) | 10.05 / 5.12 | 4.55 / 3.84 |
| (c) raw confidence | dcs (100) | 10.05 / 5.12 | 4.55 / 3.84 |
| (d) geometry-only | none | 11.10 / 5.49 | 4.49 / 3.88 |
| (d) geometry-only | huber (0.5) | 11.10 / 5.49 | 4.49 / 3.88 |
| (d) geometry-only | dcs (100) | 11.10 / 5.49 | 4.49 / 3.88 |
| (e) calibrated (ours) | none | 9.57 / 4.78 | 4.37 / 3.83 |
| (e) calibrated (ours) | huber (0.5) | 9.57 / 4.78 | 4.37 / 3.83 |
| (e) calibrated (ours) | dcs (300) | 9.57 / 4.78 | 4.37 / 3.83 |
| (f) oracle* | none | 8.94 / 4.30 | 4.59 / 4.17 |
| (f) oracle* | huber (1) | 8.94 / 4.30 | 4.59 / 4.17 |
| (f) oracle* | dcs (300) | 8.94 / 4.30 | 4.59 / 4.17 |
| (g) power law | none | 10.39 / 5.24 | 4.37 / 3.79 |
| (g) power law | huber (0.5) | 10.38 / 5.24 | 4.37 / 3.79 |
| (g) power law | dcs (30) | 10.39 / 5.24 | 4.37 / 3.79 |

\* The oracle uses the true edge errors of that sequence (hindsight diagnostic, not a method). Kernel thresholds were chosen on the fitting sequences only.

## Edge (seam/loop) uncertainty on held-out edges

Per-coordinate second moments before any graph rescaling. `msq ratio` = mean(error^2)/mean(predicted variance) (1 is calibrated); `calib err` = count-weighted mean |ratio - 1| over bins of predicted variance fixed on the fitting edges; NEES uses all 7 DoF (7 expected, 95% inside the chi-square 95% quantile).

| edges | scheme | n | rot ratio | trans ratio | scale ratio | rot err | trans err | scale err | NEES | cover95 |
|---|---|---|---|---|---|---|---|---|---|---|
| seams | (b) uniform | 569 | 1.36 | 1.32 | 1.29 | 0.36 | 0.32 | 0.29 | 9.3 | 0.80 |
| seams | (c) raw confidence | 569 | 0.90 | 0.87 | 0.85 | 0.14 | 0.55 | 0.45 | 7.7 | 0.86 |
| seams | (d) geometry-only | 569 | 1.44 | 1.40 | 1.33 | 0.73 | 0.42 | 0.38 | 11.0 | 0.77 |
| seams | (e) calibrated (ours) | 569 | 1.29 | 1.26 | 1.29 | 0.84 | 0.59 | 0.48 | 11.5 | 0.76 |
| seams | (g) power law | 569 | 1.57 | 1.07 | 1.04 | 0.47 | 0.55 | 0.40 | 9.9 | 0.78 |
| loops | (b) uniform | 27 | 2.00 | 22.36 | 9.78 | 1.00 | 21.36 | 8.78 | 82.9 | 0.11 |
| loops | (c) raw confidence | 27 | 1.72 | 19.20 | 8.40 | 1.92 | 32.63 | 13.22 | 132.0 | 0.15 |
| loops | (d) geometry-only | 27 | 2.02 | 23.55 | 10.29 | 3.00 | 30.89 | 9.29 | 115.1 | 0.11 |
| loops | (e) calibrated (ours) | 27 | 1.58 | 23.27 | 10.70 | 2.84 | 22.27 | 9.70 | 119.8 | 0.15 |
| loops | (g) power law | 27 | 0.26 | 4.63 | 0.65 | 0.64 | 6.20 | 0.37 | 25.3 | 0.63 |

### Does the feature rank seam errors? (Spearman, held-out seams)

| feature | rotation | translation | scale |
|---|---|---|---|
| conf_mean | -0.44 | -0.12 | -0.10 |
| conf_median | -0.26 | -0.13 | -0.03 |
| conf_p10 | 0.02 | 0.08 | 0.14 |
| residual_rel | 0.24 | 0.55 | 0.30 |
| inlier_fraction | -0.17 | -0.25 | -0.20 |
| spread_condition | 0.11 | -0.06 | 0.02 |
| valid_fraction | - | - | - |

## Pixel confidence audit (LiDAR)

A slope of -1 is what the loss optimum C = alpha/E[e] would give. Calibration errors use bins fixed on the fitting sequences; `raw` uses the best single constant kappa/C.

| head | scope | pixels | slope (fit) | slope (eval) | Spearman | calib err raw | calib err power law | calib err isotonic |
|---|---|---|---|---|---|---|---|---|
| depth head (depth_conf) | held-out | 337887623 | -0.77 | -0.70 | -0.34 | 0.40 | 0.25 | 0.18 |
| point head (world_points_conf) | held-out | 337884779 | -0.74 | -0.63 | -0.40 | 0.60 | 0.53 | 0.32 |

## Upstream VGGT-Long (original code) vs our reproduction

ATE [m] / KITTI t_rel [%]. Upstream = `vggt_long.py` at the pinned commit with its KITTI config (only chunk size / overlap changed), scored with our metrics; `paper` = ATE in Table 1 of the VGGT-Long paper (arXiv:2507.16443v2, overlap = half the chunk). Ours uses window 32 / overlap 16; scheme (a) is VGGT-Long's identity weighting.

| seq | upstream W75/O30 | upstream W60/O30 | upstream W32/O16 | upstream W30/O15 | ours (a) W32/O16 | ours (e) W32/O16 |
|---|---|---|---|---|---|---|
| 05 | 10.15 / 5.08 | 9.85 / 4.99 (paper 9.15) | 10.04 / 5.11 | 10.24 / 5.24 (paper 10.20) | 10.02 / 5.11 | 9.57 / 4.78 |
| 07 | 2.84 / 3.71 | 2.66 / 3.27 (paper 2.68) | 4.09 / 3.95 | 3.98 / 4.02 (paper 4.00) | 4.06 / 3.94 | 4.37 / 3.83 |
| 00 | 10.83 / 5.42 | 8.63 / 4.72 (paper 8.06) | 9.13 / 4.81 | 9.07 / 4.96 (paper 9.06) | 9.18 / 4.82 | 9.09 / 4.68 |
| 02 | 55.90 / 15.04 | 52.78 / 13.71 (paper 34.16) | 104.61 / 26.82 | 99.65 / 26.38 (paper 99.95) | 105.07 / 26.95 | 117.07 / 30.59 |

## Qualitative figures (`figures/qualitative/`)

* `depth_lidar_<seq>`: image, depth head, its confidence and the scale-aligned error at LiDAR pixels
* `loops_<seq>`: loop windows with the largest / smallest edge error (per-loop numbers in `tables/loops_<seq>.csv`)
* `seams_<seq>`: seams with the largest / smallest true error and both windows' confidence
* `pointcloud_<seq>`, `trajectory_error_<seq>`: points and trajectories placed by the solved poses
* `upstream_pointcloud_<run>`: upstream VGGT-Long's own point export

36 files: `depth_lidar_00.png`, `depth_lidar_02.png`, `depth_lidar_05.png`, `depth_lidar_07.png`, `loops_00.png`, `loops_02.png`, `loops_05.png`, `loops_07.png`, `pointcloud_00.png`, `pointcloud_02.png`, `pointcloud_05.png`, `pointcloud_07.png`, `seams_00.png`, `seams_02.png`, `seams_05.png`, `seams_07.png`, `trajectory_error_00.png`, `trajectory_error_02.png`, `trajectory_error_05.png`, `trajectory_error_07.png`, `upstream_pointcloud_00_w30_o15.png`, `upstream_pointcloud_00_w32_o16.png`, `upstream_pointcloud_00_w60_o30.png`, `upstream_pointcloud_00_w75_o30.png`, `upstream_pointcloud_02_w30_o15.png`, `upstream_pointcloud_02_w32_o16.png`, `upstream_pointcloud_02_w60_o30.png`, `upstream_pointcloud_02_w75_o30.png`, `upstream_pointcloud_05_w30_o15.png`, `upstream_pointcloud_05_w32_o16.png`, `upstream_pointcloud_05_w60_o30.png`, `upstream_pointcloud_05_w75_o30.png`, `upstream_pointcloud_07_w30_o15.png`, `upstream_pointcloud_07_w32_o16.png`, `upstream_pointcloud_07_w60_o30.png`, `upstream_pointcloud_07_w75_o30.png`

## Frontend cost

| seq | frames | windows | loop windows | peak GPU mem [GB] | s / window | frames / s | GPU |
|---|---|---|---|---|---|---|---|
| 00 | 4541 | 283 | 20 | 9.64 | 1.85 | 17.3 | NVIDIA RTX A4000 |
| 02 | 4661 | 291 | 8 | 9.64 | 1.86 | 17.2 | NVIDIA RTX A4000 |
| 05 | 2761 | 172 | 12 | 9.64 | 1.84 | 17.4 | NVIDIA RTX A4000 |
| 07 | 1101 | 68 | 3 | 9.64 | 1.83 | 17.4 | NVIDIA RTX A4000 |

Figures: `figures/pixel_calibration.png`, `figures/edge_calibration.png`, `figures/seam_error_vs_confidence.png`, `figures/ate_vs_length.png`, `figures/trajectory_<seq>.png`, `figures/upstream_vs_ours.png`.
