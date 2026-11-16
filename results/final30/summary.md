# Results: final30

Fitting sequences: 05, 07. Held-out sequences: 00, 02. Window 30, overlap 15.

## Held-out trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 00 | KITTI 02 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 143.20 / 35.34 | 189.55 / 42.89 |
| (a) VGGT-Long (identity) | none | 9.04 / 4.96 | 108.36 / 27.73 |
| (a) VGGT-Long (identity) | huber (1) | 9.04 / 4.96 | 108.36 / 27.73 |
| (a) VGGT-Long (identity) | dcs (3) | 9.04 / 4.96 | 108.36 / 27.73 |
| (a') equal, our residual | none | 9.02 / 4.88 | 110.18 / 28.22 |
| (a') equal, our residual | huber (5) | 9.02 / 4.88 | 110.18 / 28.22 |
| (a') equal, our residual | dcs (1) | 9.02 / 4.88 | 110.18 / 28.22 |
| (b) uniform | none | 9.78 / 4.88 | 116.01 / 30.42 |
| (b) uniform | huber (0.5) | 9.78 / 4.88 | 116.01 / 30.42 |
| (b) uniform | dcs (100) | 9.78 / 4.88 | 188.59 / 42.61 |
| (c) raw confidence | none | 10.31 / 5.10 | 122.32 / 32.04 |
| (c) raw confidence | huber (0.5) | 10.31 / 5.10 | 122.32 / 32.04 |
| (c) raw confidence | dcs (100) | 10.31 / 5.10 | 187.91 / 42.41 |
| (d) geometry-only | none | 9.67 / 4.79 | 113.56 / 29.79 |
| (d) geometry-only | huber (0.5) | 9.68 / 4.79 | 113.56 / 29.79 |
| (d) geometry-only | dcs (100) | 9.67 / 4.79 | 182.62 / 40.67 |
| (e) calibrated (ours) | none | 9.11 / 4.77 | 118.61 / 30.79 |
| (e) calibrated (ours) | huber (20) | 9.11 / 4.77 | 118.61 / 30.79 |
| (e) calibrated (ours) | dcs (1000) | 9.11 / 4.77 | 118.61 / 30.79 |
| (f) oracle* | none | 12.66 / 5.73 | 116.36 / 29.52 |
| (f) oracle* | huber (10) | 12.66 / 5.73 | 116.36 / 29.52 |
| (f) oracle* | dcs (30) | 12.66 / 5.73 | 116.36 / 29.52 |
| (g) power law | none | 9.89 / 4.96 | 122.31 / 32.07 |
| (g) power law | huber (0.5) | 9.85 / 4.96 | 122.31 / 32.07 |
| (g) power law | dcs (100) | 9.89 / 4.96 | 174.39 / 37.94 |

## Fitting (in-sample) trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 05 | KITTI 07 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 51.72 / 14.87 | 22.13 / 13.01 |
| (a) VGGT-Long (identity) | none | 10.18 / 5.21 | 3.98 / 4.03 |
| (a) VGGT-Long (identity) | huber (1) | 10.18 / 5.21 | 3.98 / 4.03 |
| (a) VGGT-Long (identity) | dcs (3) | 10.18 / 5.21 | 3.98 / 4.03 |
| (a') equal, our residual | none | 10.00 / 5.18 | 3.98 / 4.00 |
| (a') equal, our residual | huber (5) | 10.00 / 5.18 | 3.98 / 4.00 |
| (a') equal, our residual | dcs (1) | 10.00 / 5.18 | 3.98 / 4.00 |
| (b) uniform | none | 11.97 / 6.02 | 3.75 / 3.74 |
| (b) uniform | huber (0.5) | 11.97 / 6.02 | 3.75 / 3.74 |
| (b) uniform | dcs (100) | 11.97 / 6.02 | 3.75 / 3.74 |
| (c) raw confidence | none | 10.63 / 5.40 | 4.56 / 4.05 |
| (c) raw confidence | huber (0.5) | 10.63 / 5.40 | 4.56 / 4.05 |
| (c) raw confidence | dcs (100) | 10.63 / 5.40 | 4.56 / 4.05 |
| (d) geometry-only | none | 11.34 / 5.67 | 4.36 / 3.92 |
| (d) geometry-only | huber (0.5) | 11.32 / 5.66 | 4.36 / 3.92 |
| (d) geometry-only | dcs (100) | 11.34 / 5.67 | 4.36 / 3.92 |
| (e) calibrated (ours) | none | 10.20 / 5.09 | 4.26 / 3.95 |
| (e) calibrated (ours) | huber (20) | 10.20 / 5.09 | 4.26 / 3.95 |
| (e) calibrated (ours) | dcs (1000) | 10.20 / 5.09 | 4.26 / 3.95 |
| (f) oracle* | none | 10.03 / 4.88 | 4.55 / 4.16 |
| (f) oracle* | huber (10) | 10.03 / 4.88 | 4.55 / 4.16 |
| (f) oracle* | dcs (30) | 10.03 / 4.88 | 4.55 / 4.16 |
| (g) power law | none | 10.93 / 5.51 | 4.45 / 4.03 |
| (g) power law | huber (0.5) | 10.80 / 5.44 | 4.45 / 4.03 |
| (g) power law | dcs (100) | 10.93 / 5.51 | 4.45 / 4.03 |

\* The oracle uses the true edge errors of that sequence (hindsight diagnostic, not a method). Kernel thresholds were chosen on the fitting sequences only.

## Edge (seam/loop) uncertainty on held-out edges

Per-coordinate second moments before any graph rescaling. `msq ratio` = mean(error^2)/mean(predicted variance) (1 is calibrated); `calib err` = count-weighted mean |ratio - 1| over bins of predicted variance fixed on the fitting edges; NEES uses all 7 DoF (7 expected, 95% inside the chi-square 95% quantile).

| edges | scheme | n | rot ratio | trans ratio | scale ratio | rot err | trans err | scale err | NEES | cover95 |
|---|---|---|---|---|---|---|---|---|---|---|
| seams | (b) uniform | 607 | 1.38 | 1.28 | 1.32 | 0.38 | 0.28 | 0.32 | 9.3 | 0.80 |
| seams | (c) raw confidence | 607 | 0.92 | 0.85 | 0.87 | 0.15 | 0.53 | 0.42 | 7.7 | 0.85 |
| seams | (d) geometry-only | 607 | 1.45 | 1.36 | 1.37 | 0.59 | 0.35 | 0.38 | 10.2 | 0.78 |
| seams | (e) calibrated (ours) | 607 | 1.44 | 1.26 | 1.30 | 0.62 | 0.40 | 0.38 | 10.7 | 0.75 |
| seams | (g) power law | 607 | 1.48 | 1.05 | 1.07 | 0.45 | 0.52 | 0.37 | 9.6 | 0.79 |
| loops | (b) uniform | 27 | 1.34 | 16.72 | 5.79 | 0.34 | 15.72 | 4.79 | 60.0 | 0.26 |
| loops | (c) raw confidence | 27 | 1.12 | 13.97 | 4.83 | 1.13 | 24.08 | 7.65 | 104.9 | 0.22 |
| loops | (d) geometry-only | 27 | 1.73 | 19.05 | 6.89 | 0.73 | 20.83 | 4.64 | 81.7 | 0.07 |
| loops | (e) calibrated (ours) | 27 | 1.12 | 15.85 | 5.35 | 1.42 | 14.85 | 4.38 | 77.0 | 0.22 |
| loops | (g) power law | 27 | 0.18 | 4.08 | 0.37 | 0.73 | 5.33 | 0.44 | 23.5 | 0.67 |

### Does the feature rank seam errors? (Spearman, held-out seams)

| feature | rotation | translation | scale |
|---|---|---|---|
| conf_mean | -0.42 | -0.15 | -0.09 |
| conf_median | -0.27 | -0.14 | -0.04 |
| conf_p10 | 0.05 | 0.07 | 0.11 |
| residual_rel | 0.23 | 0.53 | 0.29 |
| inlier_fraction | -0.15 | -0.24 | -0.17 |
| spread_condition | 0.09 | -0.06 | 0.01 |
| valid_fraction | - | - | - |

## Pixel confidence audit (LiDAR)

A slope of -1 is what the loss optimum C = alpha/E[e] would give. Calibration errors use bins fixed on the fitting sequences; `raw` uses the best single constant kappa/C.

| head | scope | pixels | slope (fit) | slope (eval) | Spearman | calib err raw | calib err power law | calib err isotonic |
|---|---|---|---|---|---|---|---|---|
| depth head (depth_conf) | held-out | 337845166 | -0.78 | -0.71 | -0.34 | 0.40 | 0.25 | 0.17 |
| point head (world_points_conf) | held-out | 337842539 | -0.74 | -0.63 | -0.40 | 0.58 | 0.52 | 0.31 |

## Frontend cost

| seq | frames | windows | loop windows | peak GPU mem [GB] | s / window | frames / s | GPU |
|---|---|---|---|---|---|---|---|
| 00 | 4541 | 302 | 20 | 9.45 | 1.76 | 17.0 | NVIDIA RTX A4000 |
| 02 | 4661 | 310 | 8 | 9.45 | 1.78 | 16.9 | NVIDIA RTX A4000 |
| 05 | 2761 | 184 | 12 | 9.45 | 1.91 | 24.9 | NVIDIA RTX A4000 |
| 07 | 1101 | 73 | 3 | 9.45 | 1.77 | 16.9 | NVIDIA RTX A4000 |

Figures: `figures/pixel_calibration.png`, `figures/edge_calibration.png`, `figures/seam_error_vs_confidence.png`, `figures/ate_vs_length.png`, `figures/trajectory_<seq>.png`.
