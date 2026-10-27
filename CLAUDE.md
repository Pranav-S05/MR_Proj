# CLAUDE.md: project memory for continuing work

Mobile Robotics course project (IIIT Hyderabad, Monsoon 2026). Team: Kavish Vora, Pranav Srivastava, Mehrish Khan.
Proposal: `mobile_robotics_2026.pdf`. Mentor brief: `Project_Context.md`. Plain-language overview:
`docs/explained_simply.md`. Current state: `docs/status.md`. How to run: `README.md`.

## State (2026-10-26): full mid-submission run DONE
Everything ran on Phoenix (logs + driver scripts in `results/mid/run_info/`): upstream VGGT-Long at 30/15, 60/30,
32/16, 75/30 on 07/05/00/02; our VGGT at full model resolution; alignment on all points; audit on all LiDAR pixels
(478M); qual; fit/tune/solve/upstream/report. Results copied to `results/mid/` (core 107 MB; archive
`../phoenix_backup/full_run/results_core.tgz`; first-pass stride-2 results in `../phoenix_backup/phoenix_results.tgz`).
evo matches all 56 + 16 trajectories (1e-9 m). 44 tests pass. docs/status.md + README updated.
Phoenix cleaned 2026-10-27: only `/scratch2/vpr-team/kavish/mr_audit_arrays/` (6.1 GB raw audit arrays, the only
copy, README inside) and `kavish/pixi/` remain (plus the unrelated `repro/`). 02 chunk-60 check done: repeat run and
loop threshold 0.7 both give 52.78 m with the same 8 loops (deterministic; threshold irrelevant on 02); paper 34.16 m
not reproducible with the released code. Results in `results/mid/upstream_checks/`.
MID SUBMISSION COMPLETE. Next (after Oct 26, per proposal): Waymo (needs team account/licence), TUM, false-loop stress
test, ORB-SLAM3, streaming models, VGGT-Omega.

Results: paper chunk 30 (00/02/05/07) 9.06/99.95/10.20/4.00; upstream 30/15 9.07/99.65/10.24/3.98; upstream 32/16
9.13/104.61/10.04/4.09; ours (a) 32/16 9.18/105.07/10.02/4.06; ours (e) 9.09/117.07/9.57/4.37; chain
140.49/181.52/48.70/22.36. Pixel depth head calib err 0.40 raw / 0.25 power / 0.18 isotonic, slope -0.70.

## Findings
- VGGT collapses two visits of the same place: in 02 the passes are 1-4 m apart (different lanes) but the loop
  window puts them 0.1-1.1 m apart, while each visit alone is fine (0.1-0.4 m rms). Loop-edge translation error ~=
  true revisit offset; confidence/inliers do not flag it; loop calibration from 05 under-predicts 00/02 loops ~20x.
- 02's ~100 m ATE at small windows is also in the paper (chunk 30: 99.95 m); larger chunks help (60: 34 m).
- Worst seams: low-texture vegetation / turning, low confidence; best seams: high confidence (figures/qualitative).

## User instructions (must follow)
- GPU work runs on Phoenix (lab server) following `../Phoenix.md`; use pixi there. Do not subsample data to save time.
- **No GPU on this laptop** (Intel Ultra 7 155U, 15 GB RAM, ~219 GB free disk, CPU-only torch). GPU runs go to Phoenix.
  Do not download the 5 GB VGGT weights here.
- **Keep the codebase lean**: one-off helpers (downloaders etc.) do not stay in the repo; README gives wget/unzip.
  (Removed: `scripts/download_kitti.py`, `scripts/fetch_reference_material.py`.)
- Course rule 3: AI may not write report *text*; code is fine.
- Plain wording, no em dashes in course documents.

## Environment
- Project root `c:\Users\kavis\Desktop\MR\Project\MR_Proj-main` (not a git repo). CPU venv `.venv` (Python 3.13,
  numpy 2.2.6, scipy 1.15.3, matplotlib, pillow, evo, pyflakes). Tests: `./.venv/Scripts/python.exe -m unittest discover -s tests`
  (42 pass; put `.venv/Scripts` on PATH for the evo test).
- `data/kitti/` (gitignored): poses + calib for 00–10, seq 07 images 0–199 and velodyne every 5th of 0–199.
- `external/` and `cache/` are gitignored and regenerable (VGGT-Long clone at c160869...; synthetic caches via
  `python -m mr_cpu.synthetic --seq SS --out cache/synthetic/SS --seed N`).

## Conventions (source of truth: docstrings of mr_cpu/geometry.py, lie.py, backend.py, kitti.py, cache.py)
- Sim(3) [[sR,t],[0,1]], tangent [rot(3), upsilon(3), log-scale]. Node S_k: window k → world. Edge (i,j,Z): S_j = S_i Z,
  Z maps j → i. Residual "ours" Log(Z⁻¹S_i⁻¹S_j); "upstream" (VGGT-Long) Log(Z S_j⁻¹ S_i).
- VGGT extrinsic = W2C; window frame = first camera; scene normalised (depth ~1 unit). VGGT-Long aligns with the point
  head; spread-ratio scale (not Umeyama), Huber IRLS δ=0.1, 5 iters, conf threshold 0.1·min median.
- KITTI GT is camera 0; we use camera 2 (b = K⁻¹P2[:,3]). Image sizes differ per sequence; read them (manifest `orig_hw`).
- Loop edge (i=b earlier window, j=a later window), Z_ba = T_bL T_aL⁻¹.

## Key verified facts
- IRLS alignment == VGGT-Long numpy code (~1e-15). Backend upstream+identity == VGGT-Long pypose optimizer (same ATE).
- Our ATE == evo `--align --correct_scale`. LiDAR overlay correct (`results/checks/lidar_overlay_kitti07.png`).
- GPU convention self-checks validated with VGGT's own unprojection: right conventions ~1e-8, swapped 50–100%.
- Synthetic end-to-end (`results/synthetic_check/summary.md`): loops cut ATE strongly, weighting differences modest.

## Lessons / pitfalls found
- Synthetic gauges must use the first-camera frame, else fake translation errors.
- Scheme rescaling constants must be fixed on fitting seams (all seams, not only labelled ones), never per sequence.
- DCS from a drifted chain can reject all true loops; loop calibration from ~13 fitting loops is weak.
- VGGT-Long's spread-ratio scale is slightly biased by noise/outliers (kept for parity).
