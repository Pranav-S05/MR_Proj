# Scale study: running log of findings (started 14 Nov 2026)

Folder: `scale_study/` (side study; the main project `MR_Proj-main/` is not modified). Scripts in `study/`, run from
this folder with the main project's venv: `../MR_Proj-main/.venv/Scripts/python.exe study/<script>.py`.
Phoenix copy: `/scratch2/vpr-team/kavish/scale_study/` (queue in tmux `kavish-scale-run`, logs in `logs/`).

## 1. Diagnosis on the mid-submission data (first-frame reference, window 32 / overlap 16, drives 00 02 05 07)
Scripts: `notes/diag*.py` (first scratch versions), `study/exp_profile.py --variant mid`.
- Replacing one component of every seam by its true value (equal weights, with loops), ATE in m for 05 / 07 / 00 / 02:
  measured 9.80 / 4.07 / 9.14 / 106.90; true rotation 8.65 / 3.50 / 8.46 / 108.77; true translation 9.80 / 3.71 /
  9.34 / 106.41; true scale 3.37 / 2.68 / 6.91 / 31.52; true rotation and scale 2.27 / 2.18 / 3.19 / 10.28.
  Without loops: measured 48.70 / 22.36 / 140.49 / 181.52; true scale 38.08 / 6.20 / 26.72 / 127.16.
- Seam log-scale error (true / measured): mean +0.0057 / +0.0148 / +0.0109 / +0.0136, std 0.058 to 0.071,
  lag-1 autocorrelation 0.06 to 0.29. Mean times number of seams = total drift of the chain: 2.6x (05), 2.5x (07),
  21x (00), 52x (02), always the map shrinking. With loops 02 is still 11x off between start and end.
- In-window profile (predicted step / true step by position in the window, normalised per window): about 1.03 to
  1.08 at the first frames, 0.97 to 0.99 at the last; first half / second half = 1.040 (00), 1.040 (02), 1.038 (05),
  1.054 (07).
- Seam scale from the overlapping camera poses instead of the point clouds: lower noise (std 0.032 to 0.052) but a
  larger bias (+0.034 to +0.049, equal to the log of the half ratio) and a worse trajectory (with loops 13.0 / 7.7 /
  10.4 / 175). Lower variance with more bias is worse: bias decides.

## 2. Corrections fitted on the other drives (leave one drive out), `study/exp_correct.py --variant mid`
ATE with loops, equal weights, 00 / 02 / 05 / 07 / mean:
- none 9.14 / 106.90 / 9.80 / 4.07 / 32.48
- one constant 9.14 / 77.14 / 9.51 / 3.93 / 24.93
- regression on (pose scale minus point scale) 8.62 / 51.80 / 7.71 / 6.09 / 18.55
- true scale (bound) 6.87 / 31.58 / 3.27 / 2.53 / 11.06
Without loops the regression is not better than the constant (mean 95.9 vs 71.1 vs 98.3 uncorrected): its residual
errors are correlated along the drive. Four drives only; to be repeated on all 11.

## 3. Edge weights after removing the bias, `study/exp_weights.py --variant mid`
Schemes fitted on the other three drives. Mean ATE with loops, equal / uniform blocks / raw confidence / geometry /
calibrated / oracle sizes: raw seams 32.5 / 34.6 / 36.1 / 34.8 / 34.6 / 35.3; constant bias removed 24.9 / 27.2 /
27.4 / 26.8 / 26.8 / 27.2. Weighting does not beat equal weights before or after the bias is removed.

## 4. Why equal weights are hard to beat, `study/sim_bias_vs_weights.py` (output in `notes/sim_bias_vs_weights.txt`)
One loop of 280 seams, 30% of them 3x noisier, exact loop closure, score = RMS accumulated error.
- Noisy seams scattered: perfect variance weights give no gain at all (ratio 1.00); with a bias of realistic
  strength (N b / random walk = 3) they are 2% worse.
- Noisy seams together in one stretch: perfect weights are 21% better without bias, equal at strength 1, and 1.55x /
  2.2x WORSE at strength 2 / 3 (real drives: 1.3 to 3.3).
- Theory without bias: extra variance of equal over perfect weights at node m is V (m/N - V_m/V)^2.

## 5. Checks of the two "anomalies"
- KITTI GT height: documented (SemanticKITTI, SuMa; `notes/lit_kitti_gt_height.md`). Own check against the
  SemanticKITTI poses, `study/kitti_gt_vs_suma.py` (output `notes/kitti_gt_vs_semantickitti.txt`): vertical gap of the
  KITTI poses at revisited places, median / max: 00 0.42 / 1.22 m, 02 1.72 / 4.39 m, 05 0.06 / 0.61 m, 06 0.03 /
  0.10 m, 07 0.07 / 0.12 m, 08 2.52 / 6.08 m, 09 3.17 / 3.27 m; SemanticKITTI poses 0.01 to 0.05 m median.
  (Pairs are chosen as close in the SemanticKITTI poses, so its own gap is small by construction; the KITTI column is
  the evidence, and it agrees with our earlier LiDAR road-surface check.)
- VGGT-Long 02 at chunk 60: others get about 52 m too (`notes/lit_vggt_long_02.md`).

## 6. Novelty (see `notes/lit_novelty.md`)
Per-seam scale shrink from first-frame anchoring is published (VGGT-Bridge, Oct 2026; VGGT-Align, Aug 2026).
Not found in print: sections 1 (oracle breakdown, in-window measurement), 2 (learned correction, no extra
inference), 3 and 4 (why weighting cannot help).

## 7. Reference frame test on drive 07 (study/gpu_run.py --ref first|middle|last, window 32 / overlap 16)
`first` reproduces the mid-submission run exactly (chain 22.36 m, loops 4.07 m). Seam log-scale bias: first +0.0148,
middle +0.0210, last +0.0134; half ratio 1.054 / 1.044 / 1.026; chain ATE 22.36 / 30.60 / 18.54 m; loops ATE 4.07 /
4.01 / 3.66 m. The early steps of a window are longer than the late ones with every reference; only the steps next
to the reference change. No sign flip with the reference at the end, no cancellation with it in the middle: on this
drive the shrink follows the time order, not the reference frame. One drive, 49 full windows; `last` on the other
drives is queued.

## 8. More drives and VGGT-Bridge read directly (15 Nov 2026)
- Reference = middle against first on 03 / 04 / 06 / 07: seam scale bias +0.024 / +0.058 / +0.017 / +0.021 against
  +0.015 / +0.049 / +0.010 / +0.015; mean chain ATE 29.0 against 21.6 m; mean loops ATE 13.6 against 10.5 m.
- Seam scale bias against travel per window relative to scene depth (907 seams, seven drives, five equal groups):
  +0.001, +0.006, +0.010, +0.018, +0.026.
- VGGT-Bridge's statistic (mean log s of the seam transforms) on drive 07: first -0.011 +- 0.014, middle -0.018 +-
  0.009, last -0.008 +- 0.011.
- VGGT-Bridge reverses only its coarse chunks; its fine sequential chunks are never reversed (paper Sec. 5.2, code
  `vggt_bridge.py` lines 116 and 351). Our first / middle / last test on fine windows is the experiment they did not
  run. See `notes/vggt_bridge_check.md`.

## 9. Ten drives, first-frame reference, window 32 / overlap 16 (15 Nov 2026, afternoon)
Phoenix queue finished 02:37 on 15 Nov for 00-08 and 10 (first and middle). Drive 09 FAILED: 51 of its 1591 images
on Phoenix give "Input/output error" when read (`data/kitti/sequences/09/image_2/`, first bad one 000010.png); the
other drives ran, so the fault is in the stored copy, not in the code. 09 needs its images fetched again.
`exp_profile.py --variant first_w32_o16`: 00 / 02 / 05 / 07 reproduce the mid-submission numbers exactly (9.14 /
106.90 / 9.80 / 4.07). Seam scale bias is positive on all ten drives (+0.006 to +0.049); half ratio 1.026 to 1.080.
`exp_correct.py --variant first_w32_o16`, each correction fitted on the other nine drives, ATE with loops (m):

| seam scale | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 10 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (VGGT-Long) | 9.14 | 88.65 | 106.90 | 17.79 | 10.49 | 9.80 | 9.67 | 4.07 | 122.29 | 38.52 | 41.73 |
| one constant | 9.14 | 138.01 | 72.77 | 7.29 | 7.79 | 9.51 | 8.81 | 3.92 | 63.94 | 23.99 | 34.52 |
| regression on pose scale | 8.61 | 141.08 | 51.34 | 8.34 | 9.60 | 7.60 | 6.01 | 6.02 | 84.69 | 31.10 | 35.44 |
| regression on depth-ratio scale | 8.63 | 92.48 | 59.07 | 16.29 | 6.86 | 5.89 | 7.17 | 4.31 | 64.17 | 22.06 | 28.69 |
| regression on depth + pose | 8.44 | 55.72 | 52.14 | 11.04 | 8.83 | 7.05 | 6.32 | 6.28 | 85.18 | 29.20 | 27.02 |
| true scale (bound) | 6.87 | 40.97 | 31.58 | 2.96 | 0.85 | 3.27 | 3.11 | 2.53 | 37.67 | 7.99 | 13.78 |

Without loops (mean): none 70.67, constant 58.05, depth-ratio 55.68, depth + pose 61.45, true scale 31.13.
Reading:
- Constant: mean 17% lower, nine drives equal or better, but drive 01 (highway, seam noise std 0.113, twice the
  others) goes from 88.65 to 138.01. Mean without 01: 36.52 to 23.02.
- Depth-ratio regression (feature `depth_minus_point` from `extras.npz`): mean 31% lower, eight drives better, the
  other two worse by 4% (01) and 6% (07). The most even result so far.
- Depth + pose has the best mean (35% lower) but 07 is 54% worse; the larger feature sets are unstable on 01
  (up to 233.78).
- The linear `speed` feature inside the pose set did not help; the isotonic travel/depth curve (C2) alone is still
  untested, so it is no longer the expected main version. The depth-ratio cue (C3) is.
- All of this is at 32 / 16. Nothing yet at VGGT-Long's 60 / 30 setting.

## 10. Fixed in advance for the 60/30 and 75/30 runs (written 15 Nov 16:50, before any of their results were read)
At 32/16 (`study/exp_main.py`, results/first_w32_o16.csv) the nested selection ("auto": pick among C1-C4 by an inner
leave-one-out that never sees the test drive) chose C4 (depth-ratio fusion + isotonic travel curve) for all 10
drives. For 60/30 and 75/30 the evaluation is fixed now: (1) "auto" with the same code, (2) C4 as the single named
method, (3) the models fitted on 32/16 applied unchanged ("T:" rows, `--transfer-from first_w32_o16`). No other
method will be promoted after seeing those results; all rows are reported.

## 11. Final runs under final_PLAN.md revision 2 / 2a (15-16 Nov 2026; running log, completed in F9)
Deviations and checks, in time order:
- F1: old queue stopped 18:43. Drive 09's SALAD descriptors (`runs/descriptors/09.npy`, computed 00:32 on 15 Nov from
  the damaged image copy) were deleted and recomputed in G1 from the re-fetched images.
- 20:39: /scratch2 had 36 GB free (other users; 99 GB at 16:30). The upstream queue's flat 50 GB limit would have
  blocked every run, so it was replaced by: free space > drive's peak need (6 GB per 1000 frames, from
  `MR_Proj-main/scripts/run_upstream.sh`) + 15 GB. The 75/30 and later runs skip the overlap maps (`SUB=0`), which
  the plan never uses at those settings. The 32/16 and 60/30 overlap maps (about 12 GB) were moved to the laptop
  (`scale_study/runs/<variant>/<seq>/overlaps/`) and deleted on Phoenix only after file counts matched.
- F3 checks against upstream VGGT-Long (last column of `results/profile_<V>.txt`): 60/30 all four within 1%
  (00 8.63/8.63, 02 53.17/52.78, 05 9.75/9.85, 07 2.67/2.66). 30/15: 00 9.04/9.07, 05 10.18/10.24, 07 3.98/3.98
  within 1%; 02 108.36 against 99.65 m (8.7% off, beyond the 5% band; recorded, nothing changed; at 32/16 the same
  comparison was 105.07 against 104.61).
- R2 (ours 60/30, `auto`): mean 25.82 -> 25.00 m (-3.2%), geometric-mean ratio 0.822, 8 better / 3 worse, Wilcoxon
  p 0.206, worst 03 x1.73 (01 x1.72): "no reliable gain".
- R1 (ours 30/15, `auto`): mean 45.66 -> 35.35 m (-22.6%), geometric-mean ratio 0.746, 7 better / 3 worse, Wilcoxon
  p 0.083, worst 01 x1.63 (auto picked C1 for 01 only; C4 elsewhere): "improvement". Named method C4 at 30/15:
  28.17 m (-38.3%), 0.700, 8 / 2, p 0.019, worst 00 x1.16 (secondary row).
- 6c at 30/15 (drive 07, upstream VGGT-Long): plain ATE 3.98 m (mid-submission 3.98: pass); C4 4.40 m (our pipeline
  4.37); `check_patch.py` 72 / 72 seams, correlation 1.000, mean difference 0.0000 (pass). Wall time C4 488 s against
  plain 762 s: outside the 5% band, but in the harmless direction; both logs show identical work (73 chunks, 3 loop
  chunks, 76 network calls, 73 alignments), so the difference is load on the shared machine. Judged as meeting the
  condition's purpose (no extra inference); G3b started 21:09. (`check_patch.py` needed `encoding="utf-8"` on Windows.)
- 21:15: /scratch2 down to 27 GB free. Overlap maps that no step of the plan uses were deleted on Phoenix (32/16
  first and middle, 60/30, `last`, `smoke`; 33 folders; copying them over the VPN ran at about 0.1 MB/s, so it was
  abandoned). After F5 the 30/15 overlap maps were deleted too. Result files of every run are kept on both sides.
- F5 at 30/15 (`results/x5_first_w30_o15.txt`), loops+C1 means: base 37.16, calib 36.03, calib_d 32.72 (-11.9%,
  better on 10 of 11), depth_only 32.99. Rule: "pixel weights in the alignment look promising, not part of the
  method, future work". As at 32/16, `depth_only` (no calibration) does about as well as `calib_d`: the gain comes
  from the depth term, not from calibration.
- 22:10: G3b stopped at C4:05 (27 GB free, 31 GB needed). To keep GPU 0 busy, the order was changed (same runs):
  plain upstream for 04 03 06 10 01 09, then G4 (our 75/30, `SUB=0`, almost no disk), then the four long drives of
  G3b as C4 / plain pairs (05 08 00 02). tmux `kavish-g3b2`.
- 00:05 on 16 Nov: G4 done (our 75/30, 11 drives). /scratch2 down to 7 GB free (our whole team folder about 47 GB;
  the rest is other users). The upstream queue got an option `SHM=1`: VGGT-Long's temporary files go to RAM
  (`/dev/shm`), as the mid-submission upstream runs did (`check_version5.sh`), with stricter guards: start only if
  available RAM > peak need + 40 GB; a watchdog stops our run if available RAM falls below 20 GB (writes `ABORTED`);
  the RAM folder is removed when each run ends. Results still go to disk (small). tmux `kavish-g3c`: rest of G3b
  (plain 09; C4 / plain 05 08 00 02), then G5a.
- F3 at 75/30: checks against upstream all within 3% (00 10.83/10.83, 02 55.59/55.90, 05 10.22/10.15, 07
  2.77/2.84). R3 (ours 75/30, `auto`): mean 24.61 -> 25.37 m (+3.1%), geometric-mean ratio 0.843, 9 better / 2
  worse (01 x1.60, 03 x1.92), Wilcoxon p 0.065: "no reliable gain". C4 at 75/30: 23.99 m (-2.5%), 0.857, 8 / 3.
- Upstream 30/15, first seven C4 runs and five plain runs: within about 1% of our pipeline's C4 and VGGT-Long rows
  (for example 10: plain 40.86 / ours 40.96, C4 22.37 / ours 22.39).
- 00:26 on 16 Nov: /scratch2 reached 0 GB free (other users). C4:05 at 30/15 and every later job failed at once
  (nothing written). No process or RAM folder of ours was left. A resume script (`phoenix/resume_when_space.sh`,
  started from RAM in tmux `kavish-resume` because the disk could not take even a script) waits until >= 10 GB are
  free and GPU 0 is idle, then runs: rest of upstream 30/15 (plain 09; C4 / plain 05 08 00 02), G5a (75/30
  plain:07, C4:07) and the RAM-against-disk check (`plainram:04` at 30/15, to compare with the disk run plain:04,
  11.22 m). RAM-mode runs also need >= 5 GB free on disk for their outputs.
- Weights after C4 (`exp_weights.py`, C4 condition), equal against calibrated, mean ATE with loops: 30/15 28.17 /
  26.48 (-6.0%, better on 5 of 11); 60/30 24.22 / 23.47 (-3.1%, 4 of 11); 75/30 24.00 / 23.41 (-2.4%, 4 of 11). F7
  rule: "they still do not" (the mean moves, but on fewer than 7 drives).
- F4 (`results/f4_bias_vs_chunk.txt`): pooled seam bias 30/15 +0.0136, 60/30 +0.0102, 75/30 +0.0086 (32/16
  development +0.0119); per frame of step +0.00091 / +0.00034 / +0.00019; mean drift factor without loops x17.6 /
  x2.4 / x1.7. Per frame, the bias falls about 5x from 30/15 to 75/30.
- Q2 end-of-drive chain scale at 30/15 (`results/drift_first_w30_o15.txt`): 8 of 11 drives end closer to 1 after
  calibration; 01, 03, 06 end further from 1.
- 00:41 on 16 Nov (user's request: store files temporarily elsewhere while /scratch2 is full). `/scratch2/vpr-team/
  kavish` is on the same full disk, `/scratch` and `/home2` are not allowed (Phoenix.md), so the remaining upstream
  runs write only to RAM: working copy of VGGT-Long in `/dev/shm/kavish_vl` (weights stay links to the read-only
  originals), outputs in `/dev/shm/kavish_out`, log `/dev/shm/kavish_ramq.log`, same RAM guards. Results are copied
  to the laptop and then deleted from RAM. Images, weights and coefficient files are only read from /scratch2.
  tmux `kavish-ramq` runs final_PLAN 4b items 1-3a: rest of upstream 30/15, `plainram:04`, 75/30 plain:07 / C4:07.
  `kavish-resume` (disk waiter) stopped.
- Upstream 30/15 so far (plain / C4; ours VGGT-Long / C4): 05 10.24 / 7.19 (10.00 / 7.04); 08 140.03 / 59.82
  (140.54 / 59.82); 09 50.64 / 50.49 (50.35 / 50.29). Drives 00 and 02 held back by the RAM guard (62-63 GB
  available, 67 needed); retried every 5 min until 17:30 in tmux `kavish-ramq2`.
- RAM against disk (plan 4b item 2): `plainram:04` ATE 11.223 m, disk run 11.223 m; largest position difference
  0.0 m over a 394 m drive (identical trajectories). Pass.
- 6c at 75/30 (drive 07): plain 2.84 m (mid-submission 2.84: pass); C4 3.10 m (ours 3.26); `check_patch.py` 23 / 23
  seams, correlation 1.000, mean difference -0.0002 (pass); identical work in both logs (24 chunks, 27 network calls,
  24 alignments), wall C4 385 s against plain 422 s (same reading as at 30/15).
- 02:45: tmux `kavish-ramq2` (all output in RAM): 00 / 02 at 30/15 if RAM allows; 75/30 agreement runs C4 / plain
  for 04 03 06 10; G6 (`middle` at 30/15, SUB=0, outputs in `/dev/shm/kavish_runs`); then retries of 00 / 02.
- 75/30 upstream agreement (plan 4b item 3), upstream plain / ours VGGT-Long, upstream C4 / ours C4: 03 5.58 / 5.61,
  12.26 / 12.11; 04 3.92 / 3.88, 1.99 / 1.92; 06 6.94 / 6.34, 4.48 / 3.82; 07 2.84 / 2.74, 3.10 / 3.26; 10 22.09 /
  22.59, 11.65 / 11.67. Within about 4% except drive 06 (upstream 9% worse plain, 17% worse C4). C4 moves upstream
  in the same direction as our pipeline on all five drives (better on 04, 06, 10; worse on 03, 07).
- G6 done 05:14 (outputs in RAM, copied). Middle against first reference at 30/15, 11 drives: pooled seam bias
  +0.0218 against +0.0136 (se 0.0019 each), larger on 11 of 11 drives; mean ATE with loops 52.43 against 45.66 m,
  chain 95.17 against 77.48 m (middle better only on 01 and 09 with loops). Same conclusion as at 32/16: a middle
  reference does not remove the bias, it adds to it.
- Revision 2c applied 10:45 on 16 Nov: RAM start margin peak + 40 -> peak + 30 GB (watchdog unchanged at 20 GB), in
  `/dev/shm/kavish_run_upstream_queue.sh` (new file by `sed -i`, the running C4:00 not disturbed) and on the laptop.
  C4:00 at 30/15 started 10:24 under the old margin. Summary generator: part 5 shows upstream and our pipeline side
  by side (30/15 interim, 9 of 11: plain 42.55 -> C4 27.88 m, -34.5%, 7 better / 1 worse, Wilcoxon p 0.020; largest
  gap to our pipeline 2%); 75/30 shown as agreement only (no statistics); parts 1, 6, 10 extended as in 4b.
- 12:46 on 16 Nov: all runs done. Upstream 30/15 drive 00: plain 9.07, C4 10.43 (ours 9.02 / 10.48); drive 02:
  plain 99.65 (mid-submission 99.65), C4 47.62 (ours 110.18 / 48.29).
- R4 (upstream VGGT-Long 30/15, C4, 11 drives): mean 44.70 -> 28.09 m (-37.2%), geometric-mean ratio 0.704, 8 better
  / 2 worse (00 x1.15, 07 x1.10; 09 equal), Wilcoxon p 0.019, 95% interval of the mean change [-51%, -15%]:
  "clear improvement". Upstream agrees with our pipeline within 2% on every drive at 30/15.
- Claim line 2 of F7 applies: calibrating the seam scale improves VGGT-Long at its small-chunk setting (30/15), in
  our pipeline and in its own code, with no extra inference; at 60/30 and 75/30 most drives still improve (8 and 9
  of 11) but the mean does not, because drives 01 and 03 get worse; the gain shrinks with chunk size because the
  bias does (F4). Plain VGGT-Long at 75/30 (24.61 m) remains more accurate than calibrated 30/15 (28.09 m upstream).
- Q3 map figure: drives 02, 08 and 00 (worst C4 / plain ratio), `results/fig_maps_upstream_w30_o15.*`.
- Stages not run: G5b in full (reduced to the five-drive agreement check by revision 2b) and G7 (upstream C1,
  dropped by revision 2b). Everything else of final_PLAN.md ran.
- Cleanup: RAM files on Phoenix removed (`ls /dev/shm | grep kavish` empty), no tmux session or process of ours left.
  Logs of the RAM sessions in `scale_study/phoenix_logs/`. `/scratch2/vpr-team/kavish/scale_study/` left in place.
- The 32/16 numbers of sections 9 and 10 are development results; the final results are those of this section.
- Independent re-check (final_PLAN 4d): every drive's upstream coefficients fitted on the ten other drives; plain logs
  contain no calibrated seam, C4 logs one per seam; all 22 upstream 30/15 trajectories rescored with a separately
  written Umeyama alignment give the same ATE; R4 statistics recomputed: same values; upstream plain reproduces the
  mid-submission values to three decimals (00 9.071, 02 99.646, 05 10.236, 07 3.984), which match the VGGT-Long
  paper's chunk-30 numbers quoted in MR_Proj-main/docs/status.md (9.06, 99.95, 10.20, 4.00); summary means = CSVs.
- Robustness of R4: median ATE 19.53 -> 18.04 m; without drives 02 and 08 (the two largest gains) 28.00 -> 22.39 m
  (-20%).
- Secondary metric, KITTI t_rel (`results/rel_errors.csv`, no label, ATE stays the judge), change of the mean against
  VGGT-Long: upstream 30/15 C4 -37%; ours 30/15 C4 -37%, auto -26%; 60/30 C4 -10%, auto -8%; 75/30 C4 -7%, auto -4%.
  r_rel within -4% to +0% everywhere. Upstream plain t_rel equals the mid-submission values (00 4.96, 02 26.38).
- Closing pass done 16 Nov 2026 (final_PLAN.md revision 3). The study is closed; remaining decisions are the team's
  (report text, moving anything into MR_Proj-main, deleting the Phoenix folder, checking quoted numbers of other
  papers against their PDFs, any follow-up).
- 16 Nov, evening: Phoenix folder `/scratch2/vpr-team/kavish/scale_study` deleted at the user's request after checking
  that every result has a copy on the laptop (only the 96-frame smoke run existed on Phoenix alone); its logs are in
  `scale_study/phoenix_logs/scratch2_logs/`. `kavish/pixi`, `kavish/repro`, `kavish/mr_audit_arrays` kept.
- 16 Nov, evening: cited papers checked against their PDFs (`notes/citation_check.md`, PDFs in `papers/pdf_check/`).
  Three corrections to our notes (VGGT-Long's table has four chunk rows; VGGT-Bridge's per-dataset seam scales are in
  its appendix; three summaries reworded). The VGGT-Long paper's chunk-30 row averages 44.713 m; our plain upstream
  runs at 30/15 average 44.70 m, per drive within 1.3%.
- Correction (16 Nov): with all 11 drives, upstream 30/15 and our pipeline agree within 2.4% on every run except plain
  02 (upstream 99.65 m, ours 110.18 m, 9.6%); calibrated 02 agrees within 1.4% (47.62 / 48.29). Earlier statements
  "within 2% on every drive" were made before drive 02 was in and are superseded.

## 12. Mid-submission analysis rerun at window 30/15 (16 Nov 2026), `results/final30/`
Pipeline of sections 3 and 4 with `configs/kitti_final30.json` on 00 02 05 07 (fit 05 07, test 00 02), all LiDAR
pixels, so that the final report uses one window setting throughout.
- Pixel audit, held-out 00 / 02 (337.8 M pixels; 478 M with the fitting drives): depth head log-log slope -0.71,
  Spearman -0.34, calibration error raw 0.40, power law 0.25, isotonic 0.17; point head 0.58 / 0.52 / 0.31.
- Seams (607 held-out): every scheme predicts the error size within a factor of 1.5; mean confidence ranks rotation
  error (Spearman -0.42) but hardly translation (-0.15); alignment residual ranks translation (0.53).
- False loops (20 seeds, DCS tuned on 05 07), largest k survived (strict): equal weights 0 on every drive; calibrated
  1 / 4 / 1 / 0 (00 / 02 / 05 / 07); geometry 8 / - / 1 / 4; raw confidence and power law 8 / - / 1-2 / 0-2. On the clean
  graph of 02, DCS drops the true loops for uniform, raw confidence, geometry and power law (174 to 189 m instead of 114
  to 122 m); calibrated keeps them (118.6 m). DCS threshold sweep (20 seeds) with equal weights: best threshold 0.1
  survives 2 / 2 / 1 / 0. No scheme survives on all four drives.
- /scratch2 returned read errors on 2 freshly fetched images of 05 (fetched again); the run script now checks every
  file of a drive before VGGT and retries once.
