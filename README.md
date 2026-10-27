# Calibrated confidence for kilometre-scale feed-forward SLAM

VGGT reconstructs a few dozen frames in one forward pass. VGGT-Long scales it to whole KITTI drives by running it on
overlapping windows, joining neighbouring windows with Sim(3) "seams", closing loops, and optimising a pose graph in
which every edge gets the same weight. We measure whether VGGT's per-pixel confidence predicts the real error of those
seams and loops, turn it into calibrated per-edge uncertainties (7x7 information matrices), and compare the resulting
trajectories against equal weights, raw confidence, geometry-only weights, a power-law baseline and VGGT-Long itself.
Proposal: [mobile_robotics_2026.pdf](mobile_robotics_2026.pdf). Plain-language explanation: [docs/explained_simply.md](docs/explained_simply.md).

## Status (26 Oct 2026)

| Mid-submission item | State |
|---|---|
| Data, conventions, LiDAR projection check | Done ([overlay](results/checks/lidar_overlay_kitti07.png)); camera-2 ground truth and per-sequence image sizes handled |
| VGGT window inference + cache, memory per window size | Done on an RTX A4000 for 07, 05, 00, 02 (window 32, overlap 16, full model resolution); smoke test 4/4 PASS; [memory sweep](results/memory_sweep.csv) |
| VGGT-Long reproduction | Upstream `vggt_long.py` run at the paper's chunk 30 and 60, at our 32/16 and at its released 75/30 on all four sequences, scored with our metrics |
| Seams, true errors, pixel and seam confidence audit, calibration on 05/07 | Done; alignment on every point, LiDAR audit on every LiDAR pixel of every scan (478M samples) |
| Weighting schemes (a)-(g), kernels, loops on/off on 00/02, tables and figures | Done, plus qualitative figures ([results/mid/summary.md](results/mid/summary.md)) |

Reproduction (ATE, m; 00 / 02 / 05 / 07): VGGT-Long paper at chunk 30: 9.06 / 99.95 / 10.20 / 4.00; upstream code at
30/15 on our setup: 9.07 / 99.65 / 10.24 / 3.98; upstream at 32/16: 9.13 / 104.61 / 10.04 / 4.09; our
re-implementation (scheme a) at 32/16: 9.18 / 105.07 / 10.02 / 4.06. On 02 the loop windows place the two passes of a
revisit 0.1-1.1 m apart while they are 1-4 m apart in reality, so loop edges carry an error close to the true lane
offset that confidence does not flag. Details: [docs/status.md](docs/status.md); run logs and per-sequence settings in
`results/mid/run_info/`.

Verified:
* our ATE equals `evo_ape --align --correct_scale` on all 56 trajectories of our run and all 16 upstream runs
  (max difference 1e-9 m);
* window alignment is identical to VGGT-Long's IRLS code (differences ~1e-15);
* our pose graph in "upstream" mode reaches the same optimum as VGGT-Long's own pypose `Sim3LoopOptimizer`
  (same ATE to the millimetre on three sequences);
* the whole pipeline runs end to end on a **synthetic** frontend built from real KITTI trajectories
  (regenerate with the commands in section 3). Those numbers test the code, not VGGT.

## Code map

| Path | What it does |
|---|---|
| `mr_gpu/run_vggt.py` | GPU: VGGT on overlapping windows + VGGT-Long loop windows, SALAD / VGGT-encoder descriptors, memory sweep, convention self-checks. Writes `cache/kitti/<seq>/` |
| `mr_cpu/pipeline.py` | CPU: everything after inference: `edges, labels, audit, fit, tune, solve, report` |
| `mr_cpu/alignment.py` | VGGT-Long's window alignment (IRLS weighted Sim(3)) + per-edge features |
| `mr_cpu/edges.py`, `labels.py` | seam/loop edge table; ground-truth window gauges and 7-DoF edge errors |
| `mr_cpu/depth_audit.py`, `kitti.py` | LiDAR vs depth head and point head; KITTI calibration, camera-2 GT, LiDAR projection |
| `mr_cpu/calibration.py`, `weighting.py` | isotonic maps, reliability, power law; schemes (a)-(g) |
| `mr_cpu/backend.py`, `lie.py` | sparse Levenberg-Marquardt Sim(3) pose graph with Huber/DCS; batched Sim(3) maths |
| `mr_cpu/evaluation.py` | trajectory assembly, Sim(3) ATE, ATE vs length, KITTI t_rel/r_rel, evo check; CLI to score VGGT-Long's `camera_poses.txt` |
| `mr_cpu/report.py` | `summary.md`, `tables/*.csv`, `figures/*.png` |
| `mr_cpu/synthetic.py` | fake frontend (same cache format) for CPU testing |
| `mr_cpu/geometry.py`, `trajectory.py`, `graph.py`, `lab.py` | earlier CPU foundations and teaching lab |
| `configs/kitti_mid.json`, `configs/synthetic.json` | experiment configs (fit 05/07, test 00/02) |
| `scripts/run_gpu_mid.sh` | the full GPU sequence for the mid submission |
| `tests/` | 42 checks: `python -m unittest discover -s tests` |

Weighting schemes: (a) VGGT-Long identity, (b) uniform per-DoF constants, (c) raw confidence (mean conf^2),
(d) geometry-only isotonic (alignment residual), (e) calibrated isotonic (mean confidence, ours), (f) oracle with true
errors (diagnostic, not a method), (g) Deng et al. power law per pixel. Each with no kernel, Huber and DCS (on loop
edges; thresholds tuned on 05/07), loops off and on. Conventions are documented at the top of `mr_cpu/geometry.py`,
`mr_cpu/backend.py` and `mr_cpu/kitti.py`.

## 1. Data (both machines need `data/kitti`)

```bash
mkdir -p data && cd data
B=https://s3.eu-central-1.amazonaws.com/avg-kitti
wget $B/data_odometry_calib.zip $B/data_odometry_poses.zip
wget $B/data_odometry_color.zip        # 65 GB, all sequences
wget $B/data_odometry_velodyne.zip     # 80 GB, only needed for the depth audit
unzip -q data_odometry_calib.zip && unzip -q data_odometry_poses.zip
unzip -q data_odometry_color.zip    'dataset/sequences/0[0257]/image_2/*'
unzip -q data_odometry_velodyne.zip 'dataset/sequences/0[0257]/velodyne/*'
mv dataset kitti && cd ..
```
Result: `data/kitti/poses/SS.txt` and `data/kitti/sequences/SS/{image_2,velodyne,calib.txt,times.txt}`.
The CPU machine needs calib, poses and velodyne (not the images); the GPU machine needs calib, poses and images.

## 2. GPU machine (VGGT inference)

```bash
git clone https://github.com/DengKaiCQ/VGGT-Long.git external/VGGT-Long
git -C external/VGGT-Long checkout c160869d1d99c96bb227f414afb3bc68c29c9a76
pixi install                           # environment from pixi.toml / pixi.lock (Linux, CUDA 12 driver)
mkdir -p external/VGGT-Long/weights && cd external/VGGT-Long/weights
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt                          # 5 GB
wget https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
cd ../../..
pixi run gpu-mid                       # memory sweep, smoke test (PASS/FAIL), then 07 05 00 02
```
The CPU pipeline also runs inside this environment (`pixi run pipeline`, `pixi run test`); that is how
`results/mid/` was produced (RTX A4000, 24 Oct 2026, about 2 s per 32-frame window).
Set `WINDOW=... OVERLAP=...` before the script after looking at `results/memory_sweep.csv` (default 32/16; VGGT-Long
uses 75/30 on a 24 GB GPU). The smoke test must print PASS for every check before the long runs are trusted. Disk:
about 45 MB per window at full model resolution (`--stride 1`, the default; 3-14 GB per sequence). The script then
runs upstream VGGT-Long itself (`scripts/run_upstream.sh`) at the paper's chunk sizes 30 and 60, at our window and at
its released 75/30 config. With little disk, the CPU steps that need a sequence's cache (`edges labels audit qual`) can
run per sequence (`--fit SS --test SS --steps edges labels audit qual`), after which that cache's `windows/` and
`loops/` can be deleted; everything later only reads `manifest.json`.

## 3. CPU machine (everything else)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-cpu.txt   # Linux: .venv/bin/python
# copy cache/kitti/ from the GPU machine into this folder, then
.venv/Scripts/python -m mr_cpu.pipeline --config configs/kitti_mid.json
```
Outputs in `results/mid/`: `summary.md` (tables), `tables/*.csv`, `figures/*.png` (qualitative ones in
`figures/qualitative/`), `metrics.csv`, `ate_vs_length.csv`, `upstream.csv`, `fitted.json`, `kernels.json`.
Steps: `edges labels audit qual fit tune solve upstream report` (single steps with `--steps`). Alignment uses every
point above the confidence threshold and the LiDAR audit every LiDAR pixel (no subsampling).
To check the code without a GPU: generate synthetic caches with
`python -m mr_cpu.synthetic --seq 07 --out cache/synthetic/07` (likewise 05, 00, 02) and run
`python -m mr_cpu.pipeline --config configs/synthetic.json`.

Score any external trajectory with the same metrics:
`python -m mr_cpu.evaluation --estimate <camera_poses.txt> --seq 07 --evo`.

## Background documents
[docs/explained_simply.md](docs/explained_simply.md) (start here), [docs/proposal_review.md](docs/proposal_review.md)
(corrections and related work, incl. GeoCond), [docs/plan_and_protocol.md](docs/plan_and_protocol.md),
[docs/status.md](docs/status.md), [docs/start_here.md](docs/start_here.md), [CLAUDE.md](CLAUDE.md) (working notes).
