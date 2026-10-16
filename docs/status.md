# Project status, October 16, 2026

**Current conclusion:** useful CPU foundations are complete. The original week-1 milestone is only partially complete because it also requires real images/LiDAR checks and GPU inference.

Completed in this session:

- Read the mentor/course context and all seven proposal pages.
- Checked key primary literature and inspected pinned source snapshots; identified the recent GeoCond overlap and proposal corrections.
- Installed an isolated CPU environment: Python 3.11.9, NumPy 2.2.6, SciPy 1.15.3.
- Implemented explicit Sim(3) transforms, exp/log, weighted point alignment, camera-gauge conversion, a pinhole LiDAR projection helper, a small C2W/KITTI reader and position-ATE preflight.
- Implemented one-feature monotone second-moment calibration, an exact 1D graph teaching solver and a tiny anchored 3D Sim(3) reference optimizer.
- Ran 24 meaningful correctness checks; all passed.
- Downloaded all 11 public KITTI odometry reference pose files: 23,201 poses in total. Parsed and checked camera rotations and manufactured global-transform alignment for each.
- Ran the deterministic synthetic learning lab. Recovered known scale 2.4 to floating-point precision; verified tree invariance and the effect of a competing loop.
- Demonstrated shared-error correlation: calibrated individual-point variances can still underestimate the variance of a 100-point average by a factor of 80.2 under the stated synthetic model.
- Evaluated synthetic calibration on separate fit/test draws: nominal 95% interval coverage changed from 69.25% to 94.125%. These are simulated errors, not VGGT outputs.
- Created the explanations, proposal review, revised plan and machine-readable draft protocol.

Still pending:

- GPU/server confirmation, writable storage and a real inference memory pilot.
- Image and LiDAR samples, actual sensor/colour-camera conversion and a visual projection check.
- Installation and reproduction of the pinned VGGT-Long baseline.
- Verified conversion between our residual/covariance conventions and upstream's.
- Pose-aware reference chunk gauges and real seam/loop error labels.
- Real calibration fitting, geometry-only comparison and held-out trajectory experiments.
- evo parity, official KITTI relative errors, robustness experiments and any paper-level claim.

Source inspection does not mean reproduction. Passing synthetic tests does not mean the method improves SLAM. The very small KITTI alignment errors in `results/week1/summary.json` are reference-coordinate sanity checks and must never be reported as a reconstruction result.

The immediate learning sequence is `start_here.md` followed by `week1_workbook.md`. Once compute is confirmed, the next implementation milestone is one short fitting-sequence baseline run with verified camera conventions and retained outputs.
