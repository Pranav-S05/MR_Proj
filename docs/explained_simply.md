# The project, explained simply

## The one-sentence version
A neural network (VGGT) builds 3D maps from short video clips and says how sure it is about every pixel; we check
whether those "how sure" numbers are honest, fix them if not, and use them to decide how much to trust each joint when
we glue hundreds of clips into one long, kilometre-scale map of a drive.

## 1. The robotics problem
A car drives through a town with one camera. We want two things: where the car was at every moment (its
**trajectory**) and what the surroundings look like (the **map**). Doing both at once is **SLAM**. We judge a method by
comparing its trajectory with the true one (KITTI provides it, from GPS/IMU). The main score is **ATE**: line the two
trajectories up as well as possible, then take the average distance between them, in metres. Lower is better.

One camera cannot measure metres: a toy car close up looks like a real car far away. So every reconstruction has an
unknown size. ATE handles this by also choosing the best overall scale when lining up.

## 2. What VGGT is
VGGT is a large pretrained network (Meta/Oxford, CVPR 2025). Give it, say, 32 photos and in one pass it returns:
* where each photo was taken (a **camera pose**: position + orientation),
* a **depth map** (distance of every pixel),
* a **point map** (a 3D point for every pixel),
* a **confidence** for every pixel, for both the depth and the point map.

Think of it as a very fast surveyor who glances at a stack of photos and sketches the street and the camera spots. We
never train it ("frozen"); we only use its outputs. It cannot take 4,000 photos at once: memory grows too fast. That
is why we cut the drive into pieces.

## 3. Windows, seams and drift
We split the drive into **windows** of W frames (default 32), each overlapping the next by half (16 frames). Each
window gets its own sketch, but each sketch has its own position, rotation and **scale** (different rulers).
The overlapping frames appear in both sketches, so we can find the transform that best lays one sketch on top of the
other: rotate, move and resize. That 7-number transform (3 rotation, 3 translation, 1 scale) is a **Sim(3)**, and the
join between two windows is a **seam**. VGGT-Long computes it with a confidence-weighted least-squares fit; our code
reproduces its maths exactly.

Each seam is slightly wrong. Chaining 300 slightly wrong seams makes the map bend and stretch: this is **drift**.

## 4. Loops: where your VPR knowledge comes in
When the car returns to a place it has been before, we can tie the two moments together. That is **visual place
recognition**: VGGT-Long describes every image with a SALAD descriptor (DINOv2 features), finds image pairs whose
descriptors are very similar (cosine similarity above 0.85, at least 10 frames apart, keeping only the best pair in each
neighbourhood), and then runs VGGT on a small "loop window" made of 20 frames from each visit to measure how the two
places line up. That gives a **loop edge**. A correct loop removes drift. A wrong loop (two different streets that look
alike) folds the map, which is why we also test robust methods that can ignore bad loops.

## 5. The pose graph: a network of springs
Picture every window as a block and every seam or loop as a spring between two blocks. The optimiser (the **pose
graph** solver) moves the blocks until the springs are as relaxed as possible overall. Each spring's stiffness is its
**weight**. Formally it is a 7x7 **information matrix**: stiff = "I trust this measurement a lot".
* Without loops, the springs form a chain and every spring can be perfectly relaxed: weights change nothing.
  (Our code checks this.)
* With loops, the springs disagree, and the weights decide who gives way. That is where weighting matters.
* Only relative stiffness matters: doubling every spring changes nothing.

**VGGT-Long gives every spring the same stiffness** (identity matrix), even though some seams are clearly worse (blurry,
sharp turns, moving cars, little overlap), and even though VGGT tells us how confident it is. That is our opening.

## 6. Confidence vs calibration
A weather forecaster may always pick the rainier day correctly (good **ranking**) but say "90%" when it rains 30% of
the time (bad **calibration**). VGGT's confidence could be the same: it may rank good and bad pixels well but be wrong
about how big the errors are. Recent work (Deng et al. 2026) found exactly that for pixels. Nobody has checked it for
what a SLAM back end needs: whole seams and loops.

To **calibrate**, we fit a curve that maps "confidence" to "typical size of the error". We use **isotonic regression**:
the curve is only allowed to go one way (more confidence never means more error), and otherwise follows the data.
It is fitted on sequences 05 and 07 (where we know the truth) and then frozen and tested on 00 and 02.

## 7. How we know the true errors
* **Pixels:** KITTI has a laser scanner (LiDAR). We project its points into the image (we checked the alignment
  visually) and compare with VGGT's depth. A per-window scale is fitted first, because VGGT has no metric scale.
* **Seams and loops:** for every window we find the transform that puts VGGT's cameras onto the true KITTI cameras
  (its "gauge"). The true seam between two windows follows from their gauges; the difference from VGGT-Long's measured
  seam is the seam's true 7-number error. Windows where the car is standing still are skipped (scale is undefined).

## 8. The seven ways of setting the springs
| Scheme | Spring stiffness comes from | What it tells us |
|---|---|---|
| (a) VGGT-Long | identity (all equal) | the baseline we must beat |
| (b) uniform | one fitted value each for rotation, translation, scale | does balancing the 7 numbers alone help? |
| (c) raw confidence | VGGT confidence squared, uncalibrated | is the raw signal enough? |
| (d) geometry-only | how badly the two windows' points fit after alignment | does classical geometry do as well as the network? |
| (e) **calibrated (ours)** | VGGT confidence through the fitted isotonic curve | our method |
| (f) oracle | the true error (cheating, only as a reference) | roughly how much any weighting could gain |
| (g) power law | Deng et al.'s pixel calibration, summed over pixels | the most recent published idea |

Each is run with no robust kernel, with **Huber** (softens loops that disagree a lot) and with **DCS** (can switch off
loops that disagree a lot), with loops off and on. Kernel settings are tuned on 05/07 only.

## 9. Fair-test rules
Same VGGT outputs, same seams, same loop candidates, same window size and solver for every scheme; only the spring
stiffness (and separately the kernel) changes. Nothing from 00/02 ground truth is used for fitting, except by the oracle,
which is marked as a reference, not a method.

## 10. What the results could look like
1. Calibrated weights beat equal weights and survive bad loops better. Best case.
2. No clear gain in ATE, but the calibration curves and the oracle show how much weighting can matter, and where
   it fails. Still a solid finding.
3. VGGT's confidence ranks errors well and only its scale is off, so (b) does about as well as (e). Also reportable.

The synthetic test run (fake VGGT, real KITTI routes) behaved sensibly: loops cut the error a lot (sequence 00: about
13 m to about 2 m), and the weighting schemes differed by a few percent to tens of percent. Real VGGT outputs are needed
before any claim.

## 11. Where things stand and what to run
* Done: all CPU code, checked against VGGT-Long's own code and against evo, plus 42 automatic tests.
* To do: run `bash scripts/run_gpu_mid.sh` on a GPU laptop (see README section 2). It first prints a PASS/FAIL smoke
  test; all must pass. Then copy `cache/kitti/` back and run
  `python -m mr_cpu.pipeline --config configs/kitti_mid.json`. Read `results/mid/summary.md` and the figures.

The pipeline steps, in order: `edges` (measure every seam and loop) → `labels` (true errors) → `audit` (pixel check
with LiDAR) → `fit` (calibration curves on 05/07) → `tune` (kernel settings on 05/07) → `solve` (every scheme on every
sequence) → `report` (tables and figures).

## Glossary
* **Pose:** position + orientation of the camera. **Trajectory:** poses over time.
* **Sim(3):** rotate + move + resize (7 numbers). **SE(3):** rotate + move (6 numbers).
* **Seam:** the measured Sim(3) between neighbouring windows. **Loop edge:** the same between two visits of a place.
* **Information matrix:** 7x7 spring stiffness of an edge; the inverse of its covariance (expected error spread).
* **Robust kernel (Huber, DCS):** a rule that reduces a spring's pull when it disagrees too much.
* **ATE:** average distance between estimated and true trajectories after the best global alignment.
* **t_rel:** KITTI's drift measure: % position error over 100-800 m stretches.
* **NEES:** checks whether stated uncertainties match actual errors (about 7 for a 7-number edge if honest).
