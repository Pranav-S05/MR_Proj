# Results: mid

Fitting sequences: 05, 07. Held-out sequences: 00, 02. Window 32, overlap 16.

## Held-out trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 00 | KITTI 02 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 140.42 / 34.47 | 181.58 / 40.88 |
| (a) VGGT-Long (identity) | none | 9.20 / 4.83 | 105.10 / 26.94 |
| (a) VGGT-Long (identity) | huber (2) | 9.20 / 4.83 | 105.10 / 26.94 |
| (a) VGGT-Long (identity) | dcs (10) | 9.20 / 4.83 | 105.10 / 26.94 |
| (b) uniform | none | 10.15 / 4.88 | 113.41 / 29.82 |
| (b) uniform | huber (0.5) | 10.15 / 4.88 | 113.41 / 29.82 |
| (b) uniform | dcs (100) | 10.15 / 4.88 | 175.78 / 39.15 |
| (c) raw confidence | none | 10.29 / 5.03 | 119.68 / 31.44 |
| (c) raw confidence | huber (3) | 10.29 / 5.03 | 119.68 / 31.44 |
| (c) raw confidence | dcs (100) | 10.29 / 5.03 | 175.82 / 39.18 |
| (d) geometry-only | none | 9.31 / 4.74 | 112.92 / 29.77 |
| (d) geometry-only | huber (0.5) | 9.31 / 4.74 | 112.92 / 29.77 |
| (d) geometry-only | dcs (300) | 9.31 / 4.74 | 112.92 / 29.77 |
| (e) calibrated (ours) | none | 9.11 / 4.69 | 117.16 / 30.58 |
| (e) calibrated (ours) | huber (0.5) | 9.11 / 4.69 | 117.16 / 30.58 |
| (e) calibrated (ours) | dcs (300) | 9.11 / 4.69 | 117.16 / 30.58 |
| (f) oracle* | none | 11.93 / 5.31 | 116.13 / 29.57 |
| (f) oracle* | huber (1) | 11.93 / 5.31 | 116.13 / 29.57 |
| (f) oracle* | dcs (30) | 11.93 / 5.31 | 116.13 / 29.57 |
| (g) power law | none | 9.87 / 4.89 | 119.81 / 31.49 |
| (g) power law | huber (0.5) | 9.83 / 4.88 | 119.80 / 31.49 |
| (g) power law | dcs (100) | 9.87 / 4.89 | 119.80 / 31.49 |

## Fitting (in-sample) trajectories: ATE [m] / KITTI t_rel [%] (loops on unless stated)

| weighting | kernel | KITTI 05 | KITTI 07 |
|---|---|---|---|
| no loops (chain; identical for every scheme) | - | 48.62 / 13.76 | 22.29 / 13.25 |
| (a) VGGT-Long (identity) | none | 10.03 / 5.11 | 4.04 / 3.94 |
| (a) VGGT-Long (identity) | huber (2) | 10.03 / 5.11 | 4.04 / 3.94 |
| (a) VGGT-Long (identity) | dcs (10) | 10.03 / 5.11 | 4.04 / 3.94 |
| (b) uniform | none | 11.51 / 5.82 | 3.59 / 3.53 |
| (b) uniform | huber (0.5) | 11.51 / 5.82 | 3.59 / 3.53 |
| (b) uniform | dcs (100) | 11.51 / 5.82 | 3.59 / 3.53 |
| (c) raw confidence | none | 10.06 / 5.12 | 4.53 / 3.84 |
| (c) raw confidence | huber (3) | 10.06 / 5.12 | 4.53 / 3.84 |
| (c) raw confidence | dcs (100) | 10.06 / 5.12 | 4.53 / 3.84 |
| (d) geometry-only | none | 11.12 / 5.50 | 4.49 / 3.87 |
| (d) geometry-only | huber (0.5) | 11.12 / 5.50 | 4.49 / 3.87 |
| (d) geometry-only | dcs (300) | 11.12 / 5.50 | 4.49 / 3.87 |
| (e) calibrated (ours) | none | 9.56 / 4.77 | 4.33 / 3.82 |
| (e) calibrated (ours) | huber (0.5) | 9.56 / 4.77 | 4.33 / 3.82 |
| (e) calibrated (ours) | dcs (300) | 9.56 / 4.77 | 4.33 / 3.82 |
| (f) oracle* | none | 8.94 / 4.30 | 4.59 / 4.17 |
| (f) oracle* | huber (1) | 8.94 / 4.30 | 4.59 / 4.17 |
| (f) oracle* | dcs (30) | 8.94 / 4.30 | 4.59 / 4.17 |
| (g) power law | none | 10.40 / 5.24 | 4.38 / 3.79 |
| (g) power law | huber (0.5) | 10.39 / 5.24 | 4.38 / 3.79 |
| (g) power law | dcs (100) | 10.40 / 5.24 | 4.38 / 3.79 |

\* The oracle uses the true edge errors of that sequence (hindsight diagnostic, not a method). Kernel thresholds were chosen on the fitting sequences only.

## Edge (seam/loop) uncertainty on held-out edges

Per-coordinate second moments before any graph rescaling. `msq ratio` = mean(error^2)/mean(predicted variance) (1 is calibrated); `calib err` = count-weighted mean |ratio - 1| over bins of predicted variance fixed on the fitting edges; NEES uses all 7 DoF (7 expected, 95% inside the chi-square 95% quantile).

| edges | scheme | n | rot ratio | trans ratio | scale ratio | rot err | trans err | scale err | NEES | cover95 |
|---|---|---|---|---|---|---|---|---|---|---|
| seams | (b) uniform | 569 | 1.36 | 1.32 | 1.28 | 0.36 | 0.32 | 0.28 | 9.3 | 0.80 |
| seams | (c) raw confidence | 569 | 0.90 | 0.87 | 0.85 | 0.14 | 0.55 | 0.45 | 7.7 | 0.86 |
| seams | (d) geometry-only | 569 | 1.44 | 1.40 | 1.34 | 0.74 | 0.42 | 0.37 | 10.9 | 0.77 |
| seams | (e) calibrated (ours) | 569 | 1.29 | 1.26 | 1.29 | 0.84 | 0.58 | 0.47 | 11.5 | 0.76 |
| seams | (g) power law | 569 | 1.57 | 1.07 | 1.04 | 0.47 | 0.54 | 0.40 | 9.9 | 0.78 |
| loops | (b) uniform | 27 | 1.99 | 22.23 | 9.89 | 0.99 | 21.23 | 8.89 | 82.5 | 0.11 |
| loops | (c) raw confidence | 27 | 1.71 | 19.19 | 8.54 | 1.90 | 32.40 | 13.39 | 131.3 | 0.15 |
| loops | (d) geometry-only | 27 | 2.01 | 23.55 | 10.53 | 2.91 | 22.55 | 9.53 | 109.2 | 0.11 |
| loops | (e) calibrated (ours) | 27 | 1.59 | 23.11 | 10.87 | 2.81 | 22.11 | 9.87 | 119.0 | 0.15 |
| loops | (g) power law | 27 | 0.20 | 3.68 | 0.53 | 0.71 | 4.82 | 0.24 | 20.1 | 0.67 |

### Does the feature rank seam errors? (Spearman, held-out seams)

| feature | rotation | translation | scale |
|---|---|---|---|
| conf_mean | -0.44 | -0.12 | -0.10 |
| conf_median | -0.26 | -0.14 | -0.03 |
| conf_p10 | 0.02 | 0.08 | 0.14 |
| residual_rel | 0.24 | 0.55 | 0.30 |
| inlier_fraction | -0.17 | -0.25 | -0.20 |
| spread_condition | 0.11 | -0.05 | 0.02 |
| valid_fraction | - | - | - |

## Pixel confidence audit (LiDAR)

A slope of -1 is what the loss optimum C = alpha/E[e] would give. Calibration errors use bins fixed on the fitting sequences; `raw` uses the best single constant kappa/C.

| head | scope | pixels | slope (fit) | slope (eval) | Spearman | calib err raw | calib err power law | calib err isotonic |
|---|---|---|---|---|---|---|---|---|
| depth head (depth_conf) | held-out | 14683164 | -0.78 | -0.71 | -0.34 | 0.40 | 0.26 | 0.18 |
| point head (world_points_conf) | held-out | 14683162 | -0.75 | -0.63 | -0.40 | 0.60 | 0.54 | 0.33 |

## Frontend cost

| seq | frames | windows | loop windows | peak GPU mem [GB] | s / window | frames / s | GPU |
|---|---|---|---|---|---|---|---|
| 00 | 4541 | 283 | 20 | 9.64 | 1.92 | 16.6 | NVIDIA RTX A4000 |
| 02 | 4661 | 291 | 8 | 9.64 | 1.96 | 16.3 | NVIDIA RTX A4000 |
| 05 | 2761 | 172 | 12 | 9.64 | 1.96 | 18.1 | NVIDIA RTX A4000 |
| 07 | 1101 | 68 | 3 | 9.64 | 1.91 | 16.8 | NVIDIA RTX A4000 |

Figures: `figures/pixel_calibration.png`, `figures/edge_calibration.png`, `figures/seam_error_vs_confidence.png`, `figures/ate_vs_length.png`, `figures/trajectory_<seq>.png`.
