# CLAUDE.md: project memory for continuing work

Mobile Robotics course project (IIIT Hyderabad, Monsoon 2026). Team: Kavish Vora, Pranav Srivastava, Mehrish Khan.
Proposal: `mobile_robotics_2026.pdf`. Mentor brief: `Project_Context.md`. Plain-language overview:
`docs/explained_simply.md`. Current state: `docs/status.md`. How to run: `README.md`.

## State (2026-10-25)
Mid-submission run done on real data (Phoenix A4000, see below and `docs/status.md`); results in `results/mid/`.
Remaining: the Pending list below. After Oct 26: Waymo/TUM loaders, false-loop stress test, streaming models,
VGGT-Omega bonus.

## Phoenix run (2026-10-24, in progress)
- Ran on Phoenix (see `../Phoenix.md`; pixi env from `pixi.toml`), workspace `/scratch2/vpr-team/kavish/mr/`, GPU 1.
  Memory sweep (A4000, bf16): W8 7.7 GB, W16 8.4, W24 9.0, W32 9.6, W48 10.9, W64 12.2; ~18 frames/s. Smoke: 4/4 PASS.
  GPU part done for 07/05/00/02 at W32/O16 (3/12/20/8 SALAD loops, all true revisits <= 4 m apart).
- User asked for ALL LiDAR scans (no subsampling): range-fetching all 13064 scans from KITTI S3 (~26 GB, slow);
  `kavish-mr-final` tmux reruns `--steps audit fit tune solve report` when done. Then copy `results/mid/` here and
  delete the Phoenix project folder (keep `kavish/pixi/`).
- First full pass (LiDAR every 10th frame) ATE loops-on, VGGT-Long weighting: 05 10.0, 07 4.0, 00 9.2, 02 105 m.
- **Finding:** VGGT collapses two visits of the same place into ~the same position: in 02 the passes are 1-4 m apart
  (different lanes) but the loop window puts them 0.1-1.1 m apart, while each visit alone is fine (0.1-0.4 m rms).
  So loop-edge translation error ~= true revisit offset; confidence/inliers do not flag it; loop calibration fitted on
  05 (revisits 0.1-0.8 m apart) under-predicts 00/02 loop errors ~20x. Not a bug in our edge code (checked).

## Pending (2026-10-25; user said wait, do nothing until they say continue)
- Phoenix /scratch2 hit 100% (12 GB free, filled by someone else). Blocked until ~60 GB free:
  upstream `vggt_long.py` reproduction runs (deps already in pixi.toml), full-resolution caches (`--stride 1`, ~35 GB),
  audit without the 800 px/frame cap. Then rerun the CPU pipeline.
- **Qualitative results** (user request, do together with the upstream task): e.g. trajectory overlays (est vs GT,
  chain vs loops, per scheme), point-cloud renders, depth / confidence maps next to LiDAR, loop-window image pairs incl.
  the 02 lane-offset failure, seam-error vs confidence examples.
- Results of the 2026-10-24 Phoenix run are now in `results/mid/` (logs + per-sequence manifests in
  `results/mid/run_info/`), `results/memory_sweep.csv`, `pixi.lock`. Archive copy: `../phoenix_backup/phoenix_results.tgz`.
  evo check done (56 trajectories, max diff 1e-9 m); README + docs/status.md updated with real numbers.
  Caches (9.7 GB) stay on Phoenix (copy runs at 0.1-0.5 MB/s; regenerable). Phoenix project folder NOT deleted yet:
  it is needed for the pending runs. Delete at the very end (keep kavish/pixi).
- Local cleanup done 2026-10-25: removed cache/ (synthetic), results/synthetic_check, results/week1, the 07 image/LiDAR
  subset and external/vggt. data/kitti keeps poses + calib (tests need them).

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
