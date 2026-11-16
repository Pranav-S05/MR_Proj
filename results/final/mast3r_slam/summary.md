MASt3R-SLAM (github.com/rmurai0610/MASt3R-SLAM, unmodified), every KITTI frame, headless, single-threaded.
Tracked = parts of the drive between keyframes at most 30 frames apart (a relocalisation later still leaves
the gap unmapped). ATE is Sim(3)-aligned over the tracked keyframes only, so it is not comparable with a
whole-drive ATE.

| sequence | mode | keyframes | tracking lost after frame | drive tracked | ATE over tracked keyframes | ours (a) / (e), whole drive |
|---|---|---|---|---|---|---|
| 00 | uncalibrated | 122 | 188 | 7% of 3.7 km | 1.76 m | 9.18 / 9.09 m |
| 00 | calibrated | 114 | 183 | 6% of 3.7 km | none (113 NaN poses) | 9.18 / 9.09 m |
| 02 | uncalibrated | 254 | 374 | 9% of 5.1 km | 9.68 m | 105.07 / 117.07 m |
| 02 | calibrated | 202 | 309 | 7% of 5.1 km | 25.41 m | 105.07 / 117.07 m |
| 05 | uncalibrated | 211 | 61 | 19% of 2.2 km | 6.10 m | 10.02 / 9.57 m |
| 05 | calibrated | 252 | 489 | 17% of 2.2 km | 10.67 m | 10.02 / 9.57 m |
| 07 | uncalibrated | 169 | 468 | 45% of 0.7 km | 13.57 m | 4.06 / 4.37 m |
| 07 | calibrated | 143 | 413 | 39% of 0.7 km | 44.27 m | 4.06 / 4.37 m |

Install check on TUM RGB-D fr1/desk with its own eval configs and scoring (evo_ape -as): calibrated 0.016 m (paper 0.016), uncalibrated 0.035 m (paper 0.035; Murai et al., CVPR 2025, Table 1).
