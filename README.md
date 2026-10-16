# Calibrated confidence for long-sequence reconstruction

We are studying whether the reliability of connections between VGGT reconstruction windows can be estimated well enough to improve a long camera trajectory. This is the project described in the original [proposal](mobile_robotics_2026.pdf) and [mentor context](Project_Context.md).

Current state, October 16, 2026: **CPU foundations implemented and checked. No VGGT inference, reproduced baseline, or real confidence-calibration result yet.** The original proposal is preserved.

Start with [the plain-language explanation](docs/start_here.md). Then use [the first practical lesson](docs/week1_workbook.md). The [proposal review](docs/proposal_review.md) explains the corrections and the important overlap with recent work. The [revised plan and experiment protocol](docs/plan_and_protocol.md) separates the essential study from optional extensions. [Current status](docs/status.md) records what is actually finished.

The isolated CPU environment is already installed in `.venv`. From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m mr_cpu.lab
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m mr_cpu.trajectory --reference data/kitti/poses/05.txt
```

The lab generates [a summary](results/week1/summary.json), [a graph illustration](results/week1/graph_weights.svg), and [a calibration plot](results/week1/calibration.svg). **The graph and calibration demonstrations use synthetic data. KITTI is used only for ground-truth pose inspection and evaluator checks.**

To recreate the CPU environment on another machine:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

`python scripts/fetch_reference_material.py` downloads small public source snapshots and the KITTI pose-only archive. It does not download images, LiDAR, or model weights. The downloaded source is for inspection and is never executed. Its commit IDs, URLs and file hashes are recorded in `reference_material/manifest.json`; a compact copy is kept in [the source record](docs/source_manifest.json).

`mr_cpu/graph.py` includes a small anchored 3D Sim(3) reference solver. It is suitable for tiny synthetic checks; it has no claim of parity with VGGT-Long, real-sequence scalability, or robust-loop handling. Real baseline integration is a later step.

## Where the code lives

The project root is this `MR_Proj` folder. Our own implementation is in `mr_cpu/`; downloaded research source in `reference_material/` is separate and is not imported by our code.

| Path | Purpose |
| --- | --- |
| `mr_cpu/geometry.py` | Camera/point transforms, Sim(3) exp/log and residual conventions, weighted alignment, synthetic projection helper. |
| `mr_cpu/graph.py` | Exact 1D teaching solver and a small anchored 3D Sim(3) reference solver. |
| `mr_cpu/calibration.py` | One-feature monotone variance fitting and scalar uncertainty scores. |
| `mr_cpu/trajectory.py` | KITTI/4-by-4 C2W pose loading and global position-ATE preflight. |
| `mr_cpu/lab.py` | Reproducible synthetic demonstrations and reference-pose checks; generates the example outputs. |
| `tests/test_cpu.py` | The initial 24 correctness checks. |
| `scripts/fetch_reference_material.py` | Downloads small reference source snapshots and KITTI reference poses. |
| `configs/experiment_plan.json` | Draft experiment choices, data split and conventions; not a completed experiment runner. |
| `results/week1/` | Small example outputs; synthetic results and reference-only checks are explicitly labelled. |
| `docs/` | Background, proposal review, learning workbook, sources, status and experiment plan. |

For scientific concerns, start with [the proposal review](docs/proposal_review.md), especially “Overall assessment,” “Existing work changes the novelty boundary,” and “The highest-value possible upgrade.” For completed versus pending work, use [the status record](docs/status.md).

For sharing the existing folder, teammate setup, and the remaining server tasks, use [the team handoff](docs/team_handoff.md).
