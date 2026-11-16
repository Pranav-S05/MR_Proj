# Viva guide: proposal, report and code

Read `explained_simply.md` first for the basic ideas. This guide connects the three things the examiners can ask
about: what we promised (proposal), what we delivered (mid report), and where it lives in the code.

---

## Part 1. The proposal (`mobile_robotics_2026.pdf`) in plain words

### The three problems we pointed out (S1, S2, S3)

* **S1. All edges get the same weight.** VGGT-Long uses confidence *inside* each seam fit (to weight the 3D points),
  but in the pose graph every seam and loop gets weight 1 (an identity information matrix). Bad seams (hedges, turns,
  moving cars, small overlap) count as much as good ones. Also rotation, position and scale are not equally reliable,
  and scale drift is the classic weakness of one-camera SLAM (Strasdat et al.).
* **S2. Nobody checked confidence at the level a pose graph needs.** VGGT, DUSt3R and MASt3R learn confidence with the
  loss `C·e − α·log C` (Kendall and Gal). That loss is smallest when `C = α / e`, so a perfect model would have
  error proportional to 1/C. Earlier checks (Hillemann et al. on DTU; N. Deng et al.'s calibration audit, which
  includes KITTI) only looked at single pixels. Nobody checked seams and loops, or whether fixing confidence changes
  the trajectory.
* **S3. Pixel confidence does not directly give an edge weight.** A seam uses about 10,000 pixels. If you treat them
  as independent and add up their information, the seam looks far more certain than it is, because all those pixels
  come from one network pass with one shared scale, so their errors move together.

### The three things we promised (C1, C2, C3)

* **C1. Confidence audit** (answers S2): check confidence against real error for pixels (LiDAR) *and* for seams and
  loops (ground-truth poses). Test the "error ∝ 1/C" rule directly.
* **C2. Calibrated edge weights without training** (answers S1 and S3): a monotone lookup (isotonic regression) from a
  seam's confidence statistics to its expected error, separately for rotation, translation and scale, giving a 7x7
  information matrix. Fitted in seconds on a CPU.
* **C3. Systematic evaluation**: seven weighting schemes x three kernels (none, Huber, DCS) x loops on/off, on held-out
  KITTI, Waymo and TUM; false-loop stress test; memory and speed; bonus: CUT3R, StreamVGGT, VGGT-Omega.

### The method steps in the proposal

| Step | What | Status at mid |
|---|---|---|
| 1 | Reproduce VGGT-Long, cache VGGT outputs | Done |
| 2 | Pose graph with `r = Log(Z⁻¹ S_i⁻¹ S_j)`, Levenberg-Marquardt, schemes scaled to the same average weight | Done |
| 3 | True seam/loop errors from ground truth; LiDAR pixel audit; reliability diagrams; log-log slope | Done |
| 4 | Isotonic maps per block, `Λ = diag(σr⁻² I, (σt‖t‖)⁻² I, σs⁻²)`; NEES check | Done |
| 5 | Schemes (a) to (g), kernels, loops on/off | Done (on KITTI) |
| 6 | False-loop stress test (0, 1, 2, 4, 8, 16 fake loops, 20 seeds; "survives" = ATE within 1.25x) | Planned, Nov 6 to 9 |
| 7 | Bonus: streaming models and VGGT-Omega | Planned, Nov 15 to 19 |

Datasets promised: KITTI (fit 05/07, test 00/02), Waymo, TUM RGB-D. Only KITTI is done so far; Waymo and TUM are in the
timeline for Nov 10 to 14.

### The three "possible outcomes" the proposal already allowed

1. Calibrated weights improve ATE or robustness.
2. No ATE gain, but the oracle shows how much room weighting has, and robustness results still stand.
3. Raw confidence already ranks well and only its level is off; then we report where calibration still matters.

**This is important for the viva:** our mid result (small gains on 05 and 00, losses on 07 and 02) is close to outcome
2/3, which the proposal explicitly listed as a valid result. We did not promise a win.

### Where the mid work differs from the proposal (be ready to explain)

| Proposal said | What we did | Why (true reason) |
|---|---|---|
| Seam fit with RANSAC + Umeyama "as in VGGT-Long" | VGGT-Long's actual code uses a confidence-weighted IRLS (Huber) Sim(3) fit, so we copied that | We matched their real code to 1e-15; the proposal described it wrongly |
| Loop retrieval with "a DINOv2-based encoder" | SALAD (which is DINOv2-based), exactly as VGGT-Long | Same thing, named precisely |
| fp16 on Turing GPUs (2080 Ti, T4) | bf16 on an RTX A4000 (16 GB) on the lab server | Better GPU became available |
| Window W from {8, 16, 24, 32} | 32 frames, 16 overlap | Chosen from the memory sweep; top of the proposal's own range |
| Loop edges get "a separate map" | One loop correction factor per scheme on top of the seam map | Only about 13 usable loops on 05/07, too few for a separate map |
| Only scheme (a) as identity | Added (a′): identity in our residual convention | To separate the effect of weights from the residual formula |
| Leave-one-sequence-out, 3 to 5 window offsets | Not yet | Planned for the final |

---

## Part 2. The mid report (`report/mid_report.pdf`), section by section

### Section 1. Problem statement

* Equation (1): find window placements `S_k` that minimise `Σ ρ(rᵀ Λ r)` over all edges.
  `r_ij = Log(Z_ij⁻¹ S_i⁻¹ S_j)` is the 7-number disagreement of edge (i, j); `Λ` is its weight; `ρ` is the optional
  robust kernel. VGGT-Long uses `Λ = I`.
* **Our problem:** learn, without training, a map `g` from an edge's confidence features `φ` to a diagonal `Λ` that
  is *calibrated*: the predicted variance in each block equals the true average squared error.
* **Three questions:** Q1 pixel level, Q2 seam/loop level, Q3 trajectory level.
* **Refinements since the proposal:** (a′) control; ground-truth height problem at revisits; 32/16 windows.

### Section 2. Literature review (one line each)

* **DUSt3R, MASt3R:** point maps with confidence for image pairs. **VGGT:** many views, also cameras.
  **CUT3R, StreamVGGT:** streaming with memory. **VGGT-Omega:** register tokens, less memory.
* **VGGT-Long:** our baseline. **VGGT-SLAM:** SL(4) instead of Sim(3); our weighting would carry over.
  **MASt3R-SLAM:** uses confidence as weights but never checks calibration. **DROID-SLAM:** learns weights (needs
  training). **Strasdat et al.:** Sim(3) pose graphs for scale drift. **ORB-SLAM3:** classical baseline.
* **Hillemann et al.:** VGGT uncertainty on DTU. **N. Deng et al.:** calibration audit, power-law fix (our scheme g).
  **GeoCond (WACV 2027):** closest work; finds pose failures follow geometry (overlap, parallax, rotation), trains an
  adapter. Difference: we train nothing, we target Sim(3) seams and loops of a kilometre-scale pipeline. It also
  motivates scheme (d).
* **Kuleshov et al.:** reliability diagrams + isotonic regression (our method). **DCS (Agarwal et al.):** robust
  pose graphs against wrong loops. **Umeyama:** Sim(3) fit. **evo:** standard ATE tool.

### Section 3. Progress

**Table 1 (reproduction).** Paper vs original code vs our copy. Key sentence: original code matches the paper within
0.5% at 30/15; our copy matches the original within 0.7% at 32/16. One unexplained number: 02 at 60/30.

**Table 2 (Q1, pixels).** On 00/02, 338 million pixels.
* *Slope*: of log(error) vs log(confidence). −1 would match error ∝ 1/C. We get −0.70 (depth head), −0.63 (point head).
* *Spearman*: rank correlation between confidence and error (negative = higher confidence, lower error). −0.34 / −0.40.
  It measures ordering only, so calibration cannot change it.
* *Calibration error*: 0.40 raw, 0.25 power law, 0.18 isotonic (depth head). See `explained_simply.md` 4.4.
* Two heads: VGGT has a **depth head** (depth per pixel) and a **point head** (3D point per pixel), each with its own
  confidence. Seams are built from the point head, which is why both are audited.

**Figure 2.** Black = measured error per confidence bin; coloured = what each predictor expected. The green
(isotonic) line sits closest to black.

**Table 3 (Q2, seams and loops)** on 569 held-out seams and 27 loops.
* *Ratio* (rot, trans, scale): real mean squared error ÷ predicted. 1 is perfect. Seams: all schemes 0.85 to 1.57,
  i.e. "within about 1.5x".
* *NEES*: `eᵀ Λ e` averaged; 7 is perfect for 7 numbers. Seams 7.7 to 11.5.
* *Coverage*: share of edges whose real error falls inside the predicted 95% range. 0.95 is perfect. Seams 0.76 to 0.86.
* Loops: ratios 10 to 23 and NEES 83 to 120: loops far worse than predicted (explained by Table 4).
* *Right side*: Spearman of each seam feature with the true error. Mean confidence: −0.44 rotation, −0.12 translation.
  Alignment residual: 0.55 for translation (the best clue for position).

**Table 4 (loops).** For each drive: ground-truth gap between the two visits, its vertical part, the LiDAR road-height
gap, and the loop edge error. On 02 the vertical part (1.94 m median, 4.01 max) is almost the whole gap, and the LiDAR
road gap (2.02, 4.06) matches it: ground-truth height is wrong at revisits.

**Table 5 (Q3, trajectories).** ATE / t_rel. Loops: 140 → 9 m on 00. Ours (e) vs control (a′): better on 05 and 00,
worse on 07 and 02. Every weighted scheme is worse than equal weights on 02. DCS switched off all 8 true loops on 02 for
(b), (c), (d), (g). The oracle is not the best.

**Figure 3.** On 00 loops pull the drifting chain onto the truth. On 02 all loops join the start to the end, so the
middle keeps its drift.

**Figure 4.** Position error along drive 00; the biggest error (about 20 m) is on a detour that no loop closes.

**Section 4. Timeline**: false-loop test (Nov 6 to 9), Waymo + TUM + ORB-SLAM3 (Nov 10 to 14), bonus models and
memory/speed table (Nov 15 to 19), MASt3R-SLAM if time and final figures (Nov 20 to 22), submission Nov 23, demo Nov 26.

**Appendix.** Fig 5: depth, confidence and LiDAR error per pixel (errors at object borders). Fig 6: worst and best
seams of 07 (worst ones: vegetation, low confidence). Fig 7: largest/smallest loop on 02 (3.09 m apart in height in
the ground truth, and 3.09 m in LiDAR road height).

---

## Part 3. The code

### The big picture: two halves joined by a cache

```
GPU machine (Phoenix)                         CPU (any machine)
mr_gpu/run_vggt.py  ──writes──>  cache/kitti/<seq>/  ──read by──>  mr_cpu/pipeline.py
 (VGGT on windows and                manifest.json                  edges → labels → audit → qual
  loop windows, SALAD)               windows/*.npz                  → fit → tune → solve → upstream → report
                                     loops/*.npz                                      │
                                                                                      ▼
                                                                              results/mid/
```

The GPU part runs the network **once** and saves everything. Every experiment after that runs on the CPU from the
saved files. This is the proposal's design ("all outputs are cached ... every experiment runs on the CPU").

### Every file, what it does, main functions

**GPU side**

| File | Purpose | Main functions |
|---|---|---|
| `mr_gpu/run_vggt.py` | Runs VGGT on all windows and loop windows, writes the cache | `load_model` (VGGT + a hook to grab DINOv2 features), `infer` (one window through VGGT), `window_checks` (the 4 convention checks), `to_cache` (save poses, depth, points, confidence), `salad_descriptors` (place-recognition fingerprints), `cmd_run` (whole sequence), `cmd_memory` (memory sweep) |

**CPU side, in the order data flows**

| File | Purpose | Main functions |
|---|---|---|
| `cache.py` | Reads the cache; VGGT-Long's window and loop layout | `window_ranges` (window k = frames 16k..16k+31), `Cache` (load window/loop files), `find_loop_pairs` (similarity > 0.85, gap > 10, NMS 25, top 5), `loop_windows` (20 frames around each visit) |
| `kitti.py` | KITTI specifics | `read_calib`, `gt_cam2_poses` (ground truth moved from camera 0 to camera 2), `model_input_size` (518-wide grid), `lidar_depth_on_grid` (project LiDAR onto VGGT's pixel grid) |
| `geometry.py` | Basic 3D maths | `similarity`, `inverse`, `transform_points`, `fit_similarity` (Umeyama), `project_lidar` |
| `lie.py` | Sim(3) maths, batched | `exp`, `log` (7 numbers ↔ 4x4 matrix), `inv`, `from_srt` |
| `alignment.py` | VGGT-Long's seam fit, copied exactly | `vggtlong_sim3` (one weighted fit), `irls_sim3` (repeat with Huber reweighting, 5 rounds), `align_windows` (keep confident points, fit, and compute features: mean confidence, residual, inlier fraction, depth) |
| `edges.py` | Builds the table of all seams and loops | `build` (for each neighbour pair and each loop window: align, store Z and features), `save`, `load` |
| `labels.py` | True errors from ground truth | `fit_gauge` (rotate/resize/shift a window's cameras onto the true cameras), `window_gauges` (all windows), `edge_errors` (`Log(Z⁻¹ A_i⁻¹ A_j)`), `block_targets` (split into rotation, translation, scale squared errors) |
| `depth_audit.py` | Pixel check against LiDAR | `collect` (per window: median scale, relative error and confidence for every LiDAR pixel), `kitti_lidar_fn` |
| `calibration.py` | Calibration tools | `_pav` (pool adjacent violators), `Isotonic` (fit/predict/save), `reliability` (10 bins), `calibration_error`, `gaussian_nll`, `coverage_curve`, `loglog_slope`, `fit_power_law`, `shrunk_inflation` (loop factor pulled towards 1) |
| `weighting.py` | The schemes | `Fitted.fit` (everything learned on 05/07), `Fitted.variances` (predicted error per edge for a scheme), `Fitted.information` (turn into 7x7 weights; identity for (a) and (a′)), `info_from_var` |
| `backend.py` | The pose graph solver | `residuals` (ours or upstream convention), `_jacobians`, `kernel` (none, Huber, DCS), `solve` (sparse Levenberg-Marquardt, first window fixed) |
| `evaluation.py` | Scoring | `assemble` (window placements → one pose per frame), `sim3_align`, `ate`, `ate_vs_length`, `kitti_relative_errors` (t_rel, r_rel), `evo_ate` (cross-check), `main` (score any pose file) |
| `qualitative.py` | Pictures | depth vs LiDAR, worst/best seams, worst/best loops, top-view maps |
| `report.py` | Tables and figures | `write_report` → `summary.md`, `tables/*.csv`, `figures/*.png` |
| `pipeline.py` | Runs everything in order | `step_edges`, `step_labels`, `step_audit`, `step_qual`, `step_fit`, `step_tune` (Huber/DCS thresholds on 05/07), `step_solve` (all schemes x kernels x loops), `step_upstream` (score original VGGT-Long), `step_report`; `solve_one` does one run |
| `synthetic.py` | Fake VGGT outputs built from real KITTI paths | `make_cache` (same file format), so the whole CPU pipeline can be tested without a GPU |
| `trajectory.py` | Small helpers: read pose files, simple ATE | `read_poses`, `ate` |
| `graph.py`, `lab.py` | Earlier teaching/foundation code, not used by the pipeline | (left over; not in the Mid_Submission folder) |

**Scripts and config**

* `scripts/run_gpu_mid.sh`: the GPU run (memory sweep, smoke test, VGGT on 07, 05, 00, 02, original VGGT-Long at four
  window sizes). `scripts/run_upstream.sh`: one original VGGT-Long run. `scripts/check_revisit_ground.py`: the LiDAR
  road-height check.
* `configs/kitti_mid.json`: fit = 05, 07; test = 00, 02; window 32/16; schemes; kernels.
* `tests/`: automated checks (maths, conventions, file formats, alignment equals VGGT-Long's, ATE equals evo).

### One seam's journey through the code (good to tell at the viva)

1. `run_vggt.py` runs VGGT on window 200 and window 201 and saves their point maps and confidence.
2. `edges.build` takes their 16 shared frames and calls `alignment.align_windows` → measured seam `Z` + features.
3. `labels.fit_gauge` places both windows onto the ground truth → `labels.edge_errors` gives the true error.
4. On 05/07, `weighting.Fitted.fit` learns "mean confidence → expected error" with `calibration.Isotonic`.
5. On 00/02, `Fitted.information` turns the seam's confidence into a 7x7 weight.
6. `backend.solve` finds all window placements; `evaluation.assemble` and `evaluation.ate` give the score.

### Is the code modular? (honest answer)

**Yes, in these ways:**
* **GPU and CPU are separated by a file format** (the cache). The CPU side never imports torch or VGGT. Any other
  network (CUT3R, VGGT-Omega) can be added by writing the same cache, without touching the CPU code. `synthetic.py`
  proves this: it writes the same format and the whole pipeline runs on it.
* **One file per job**: alignment, edges, labels, calibration, weighting, solver, evaluation, report.
* **Steps are independent and restartable**: each pipeline step reads files and writes files, so you can run one step
  alone (`--steps fit tune solve`), or run the heavy steps one drive at a time and delete the big cache after.
* **Schemes are data, not code paths**: adding a scheme means adding a branch in `Fitted.variances`; the solver and the
  evaluation do not change.
* **Low-level maths is shared**: `lie.py` and `geometry.py` are used everywhere instead of re-deriving formulas.
* **Configuration in JSON**, not hard-coded (`configs/kitti_mid.json`).
* **Tested**: unit tests on the maths and conventions, plus equality tests against VGGT-Long's code and evo.

**Weak spots (say them before they are pointed out):**
* `graph.py` and `lab.py` are older code still in the full repository (removed in the submission folder).
* Two small overlaps: `trajectory.ate` and `evaluation.ate`; `geometry.py` and `lie.py` both have Sim(3) helpers
  (one single-matrix, one batched).
* `pipeline.py` and `report.py` are the largest files (about 320 and 400 lines) because they orchestrate everything.

---

## Part 4. Likely viva questions, short answers

* **What is your contribution in one sentence?** Testing whether VGGT's confidence, calibrated without training,
  gives better edge weights for VGGT-Long's pose graph, measured from pixels up to whole trajectories.
* **Did it work?** Partly. Pixel confidence calibrates well (0.40 → 0.18). At seam level it predicts rotation error
  but not position error. Trajectory gains are small and not consistent. The proposal allowed this outcome.
* **Why fit on 05/07 and test on 00/02?** To keep learning and grading separate. 05/07 have loops; 00/02 are the long
  drives with loops. The split was fixed before any experiment.
* **Why is 02 so bad?** Its loops only join the start and end; the long middle has no loop, so its drift stays. The
  paper has the same ~100 m at small windows.
* **Why did DCS fail on 02?** The drifted chain disagrees so much with the true loops that DCS treats them as wrong and
  switches them off.
* **Why is the oracle not the best?** It uses one sample's exact error, not the typical error; lucky seams get huge
  weights and force corrections into other seams. 1/E² is right for expected error, not for one realisation.
* **What is NEES?** `eᵀΛe`: the error measured in units of its predicted size. If predictions are honest, its average
  is the number of values, 7.
* **What is Huber vs DCS?** Both weaken loop edges that disagree a lot. Huber weakens gradually; DCS can switch them
  off almost completely.
* **Why Sim(3) and not SE(3)?** One camera has unknown scale, and each window has its own scale, so every alignment
  needs a scale factor: rotation + translation + scale = 7 numbers.
* **What is next?** False-loop stress test, Waymo and TUM, ORB-SLAM3, better seam features (confidence plus
  alignment residual), cleaning loop labels at inconsistent revisits, and bonus models.
