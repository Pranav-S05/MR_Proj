# The whole project, explained from zero

This document assumes you know nothing about robotics, computer vision or this project. Every idea is built up from
the very basics, one step at a time, and every technical word is explained the first time it appears (and again in
the glossary at the end). Read it top to bottom once; after that, use the headings to jump around.

**Contents**
1. The goal, in one paragraph
2. The basic ideas you need (photos, positions, 3D, drift, loops, datasets, neural networks)
3. The two existing tools we build on: VGGT and VGGT-Long
4. Our idea and what we planned to do
5. What we have actually done so far, and how
6. What it all means
7. What comes next
8. Glossary
9. Where things are in the project folder

---

## 1. The goal, in one paragraph

A car drives through a city with a single ordinary camera on its roof. From the video alone, we want to work out
**where the car was at every moment** and **what the street around it looks like in 3D**. A recent neural network
called VGGT can do this for short pieces of video, and a method called VGGT-Long glues many short pieces together to
cover a whole drive of several kilometres. VGGT-Long treats every glue joint as equally trustworthy, even though some
joints are clearly worse than others. VGGT also reports, for every pixel, how confident it is. **Our project asks:
is that confidence honest, and if we use it to decide how much to trust each glue joint, does the final path of the car
become more accurate?**

The rest of this document explains every word in that paragraph.

---

## 2. The basic ideas you need

### 2.1 Photos, pixels and video

A digital photo is a grid of tiny coloured squares called **pixels**. A KITTI photo is 1241 pixels wide and 376 pixels
tall, so about 470,000 pixels. Each pixel only stores a colour; it does not know how far away the thing it shows is.

A video is just many photos taken quickly one after another. Each photo in a video is called a **frame**. The KITTI car
records 10 frames per second, so a 7-minute drive gives about 4,500 frames.

### 2.2 Where is the camera? (pose and trajectory)

To describe where a camera is at one moment you need two things:
* its **position**: where it is in space (3 numbers: left/right, up/down, forward/back);
* its **orientation**: which way it is pointing (3 more numbers: turned left/right, tilted up/down, rolled sideways).

Position plus orientation together is called the camera's **pose** (6 numbers). The list of poses for every frame of
the drive is the **trajectory**: the path the car took, plus which way it was facing along the way.

### 2.3 Why do we want this?

Self-driving cars, delivery robots, drones and phones (for augmented reality) all need to know where they are and what
is around them. GPS is not precise enough (several metres off) and does not work indoors, in tunnels or between tall
buildings. Cameras are cheap and everywhere, so working out the trajectory from a camera alone is very useful. Doing it
from one camera is called **monocular visual odometry**. When the system also builds a map and uses it to correct its
own past mistakes, it is called **SLAM** (Simultaneous Localisation And Mapping).

### 2.4 Depth and 3D points

If you knew, for every pixel, **how far away** the thing in it is (its **depth**), you could place every pixel as a dot
in 3D space. Do that for many photos and you get a 3D **point cloud**: a map of the street made of millions of dots.
A grid that stores one depth per pixel is a **depth map**. A grid that stores one 3D position per pixel is a **point
map**.

### 2.5 The size problem with one camera (scale)

With one camera you cannot tell the true size of things. A toy car 1 metre away and a real car 10 metres away can look
identical in a photo. So anything reconstructed from one camera has the right *shape* but an unknown *size*: it could
be the real street or a perfect miniature model of it. This unknown overall size is called the **scale**. (Two cameras
side by side, like our two eyes, would fix it; we deliberately use only one.)

Because of this, whenever we compare a reconstructed path with the true path, we are allowed to resize it first, as
well as rotate and shift it, to line it up as well as possible. Only what remains after that counts as error.

### 2.6 Small errors add up (drift)

Imagine walking through a building with your eyes closed, counting steps and turns. Each step estimate is only a tiny
bit wrong, but after 500 steps you can be far from where you think you are. Every new estimate builds on the previous
ones, so the small errors pile up. This slow build-up of error is called **drift**. It is the main enemy of any system
that estimates its path step by step.

### 2.7 Coming back to the same place (loop closure)

Now imagine that, still walking blind, you touch a door you know you touched 10 minutes ago. Instantly you know where
you are, and you can correct your whole mental map of the walk. Recognising that you are back at a place you have
already been is called **loop closure**, and the pair of moments (earlier visit, later visit) is a **loop**. Recognising
a place from its appearance is called **place recognition**. Loops are the strongest tool against drift.

A wrong loop (two different streets that just look alike) is dangerous: it tells the system "these two places are the
same" when they are not, and can fold the map in half. Methods that can ignore loops that do not fit are called
**robust**.

### 2.8 How do we know the true answer? (ground truth and KITTI)

To measure how wrong an estimate is, you need the correct answer. The correct answer is called **ground truth**.

**KITTI** is a famous public dataset recorded in Karlsruhe, Germany. A car was fitted with:
* cameras (we use one colour camera, called camera 2);
* a **LiDAR**: a spinning laser scanner on the roof that measures the exact distance to about 100,000 points around the
  car, ten times a second (so it gives true depth, but only at those points);
* a very precise GPS combined with motion sensors, which gives the true pose of the car for every frame.

The drives are numbered **sequences 00 to 10**. We use four of them:
* **05 and 07** for *fitting*: anything our method needs to learn, it learns here, using their ground truth;
* **00 and 02** for *testing*: we never learn anything from them; we only use their ground truth to grade our results.

Keeping learning and testing separate is what makes the test fair. It is like studying with past exam papers and then
being graded on a new exam: if you were graded on the same papers you studied, a good score would mean nothing.

### 2.9 Neural networks, GPUs and memory

A **neural network** is a very large mathematical function with millions or billions of adjustable numbers. It is
**trained** by showing it many examples with known answers and nudging the numbers until its answers are good. After
training it can be used on new inputs. We never train anything ourselves; we use already trained networks as they are
(**frozen**) and only look at their outputs.

Neural networks run on a **GPU** (graphics card), a chip that does millions of calculations in parallel. A GPU has a
fixed amount of memory (ours: 16 GB). The more photos you give a network at once, the more memory it needs. That memory
limit is the reason for most of the design of VGGT-Long.

---

## 3. The two existing tools we build on

### 3.1 VGGT: one network that does the whole job for a few photos

**VGGT** (Visual Geometry Grounded Transformer) is a neural network published by Meta and the University of Oxford in
2025. You give it a handful of photos (say 32) and, in one go, it returns for every photo:
* the camera **pose** (where the photo was taken from and which way it was pointing);
* a **depth map** (how far away every pixel is);
* a **point map** (a 3D position for every pixel);
* a **confidence map**: one number per pixel saying how sure the network is about that pixel's depth or 3D point.
  It is always at least 1, and bigger means "more sure".

Classical methods need many hand-designed steps to get this (finding corners in images, matching them between photos,
solving geometry equations). VGGT does it all in a fraction of a second. But its results have the scale problem from
2.5: each batch of photos comes back in its own made-up units.

**Confidence** is the key ingredient of our project. In principle it tells you which pixels to trust. Whether it does
that *honestly* is exactly what we test.

### 3.2 Why VGGT cannot do a whole drive

A drive has up to 4,661 frames. VGGT's memory use grows quickly with the number of photos, so a 16 GB GPU can only take
about 30 to 100 photos at once. So the drive has to be cut into pieces.

### 3.3 VGGT-Long: cut, glue, close loops, fix

**VGGT-Long** (published in 2025 by a different group) turns VGGT into a system for long drives in four steps:

**Step 1, cut into windows.** The video is cut into short overlapping pieces called **windows**. In our runs a window is
32 frames, and each window shares its last 16 frames with the next window. (Frames 0 to 31 are window 0, frames 16 to 47
are window 1, frames 32 to 63 are window 2, and so on.) VGGT is run on each window separately. Each window comes back
as its own little 3D sketch of a short stretch of road, in its own position, rotation and scale.

**Step 2, glue neighbouring windows (seams).** Because neighbouring windows share 16 frames, both sketches contain 3D
points for the same pixels. Imagine two transparent sheets with drawings of the same street corner, drawn at different
sizes and angles. You can rotate, slide and enlarge one sheet until the shared parts lie exactly on top of each other.
That "rotate, slide, resize" instruction is 7 numbers (3 for rotation, 3 for sliding, 1 for resizing) and is called a
**similarity transform**, written **Sim(3)**. The glue joint between two neighbouring windows is what we call a **seam**.
VGGT-Long computes each seam with a careful fit that uses only confident pixels, gives more say to more confident pixels,
and ignores pixels that clearly disagree. A drive of 4,661 frames has about 290 seams.

Each seam is slightly wrong, and chaining 290 slightly wrong seams gives **drift** (2.6): the path slowly bends and
stretches.

**Step 3, close loops.** Every frame gets a compact "fingerprint" describing what the scene looks like (computed by a
place-recognition network called SALAD). When two frames far apart in time have very similar fingerprints, the car is
probably back at the same place. VGGT-Long then takes 20 frames around each of the two moments, puts all 40 frames
together into one special **loop window**, and runs VGGT on it. Since both visits are now in the same sketch, this tells
us exactly how the two moments relate. That gives a **loop edge**: a glue joint between two windows that are far apart
in time.

**A worked example of Step 3 (drive 02).** Window k covers frames 16k to 16k+31, so window 120 covers frames 1920 to
1951 and window 281 covers frames 4496 to 4527. SALAD says frame 4507 looks almost exactly like frame 1923 (similarity
0.956), although they are 2,584 frames apart in time: the car is back on the same street. VGGT-Long builds a loop window
from frames 4497 to 4516 (later visit) and 1920 to 1939 (earlier visit) and runs VGGT on all 40 together. Those frames
also lie inside window 281 and window 120, so the result says directly "window 281 sits like *this* relative to
window 120". Drive 02 has 8 such loop edges.

**Step 4, fix the whole path (the pose graph).** Now there are two different answers for where window 281 is:
* **through the seams:** start at window 120, apply the seam to 121, then to 122, and so on up to 281. That is 161 seams,
  each slightly wrong, so the errors add up;
* **through the loop edge:** directly from window 120, in one measurement.

These two answers disagree, often by tens of metres. The program then moves all 291 window positions at once until the
total disagreement with all 290 seams and all loop edges is as small as possible. In practice window 281 moves most of
the way to where the loop says, and the correction is shared out over windows 121 to 280, a little each, so no single
seam has to take all of it.

The list of windows (**nodes**) and the measurements between them (seams and loop edges, together called **edges**) is
the **pose graph**. The program that finds the best positions is the **optimiser** or **solver**.

Each edge also has a **weight**: how much the optimiser should trust it. If the seam between windows 200 and 201 has a
high weight, the optimiser tries hard to keep that seam exactly as measured, and pushes the correction into other seams.
If it has a low weight, more of the correction goes into that seam. Technically the weight is a 7x7 table of numbers
called the **information matrix** (separate trust for rotation, sliding and resizing, and how they interact).

Two useful facts:
* If there are no loops, there is nothing to disagree with: every seam can be kept exactly as measured. Then the
  weights do not matter at all.
* With loops, the seams and the loops disagree, and the weights decide which edges absorb the correction. That is where
  weights matter.

This also explains why drive 02 stays bad even with loops: its 8 loops only connect frames around 940 to 1960 with
frames around 4200 to 4550. Nothing ties the long stretch in between (roughly frames 2000 to 4200) to anything else, so
its drift is never corrected.

### 3.4 The weakness we target

**VGGT-Long gives every edge exactly the same weight.** But some seams are obviously worse than others: a window
that sees only a hedge while the car is turning is much harder than one that looks down a straight street full of
buildings. And VGGT already tells us, for every pixel, how confident it is. Using that confidence to set the weight
of each edge is the opening our project explores.

---

## 4. Our idea and what we planned to do

### 4.1 The two questions

1. **Is VGGT's confidence honest?** First for single pixels, then for whole seams and loops.
2. **If we turn confidence into an honest estimate of error and use it as the edge weight, does the car's path
   become more accurate?**

### 4.2 What "honest" means: ranking versus calibration

VGGT gives every pixel a confidence number. Take three pixels as an example (the numbers are made up to show the idea):

| Pixel | VGGT confidence | True error (measured with LiDAR) |
|---|---|---|
| A | 2 | 1.0 m |
| B | 4 | 0.7 m |
| C | 8 | 0.5 m |

**Check 1: is the order right? (ranking)** C has the highest confidence and the smallest error; A has the lowest
confidence and the biggest error. So more confidence really means less error. VGGT passes this check.

**Check 2: are the amounts right? (calibration)** VGGT's confidence roughly says "double the confidence, half the
error". If that were true, going from A (2) to B (4) should halve the error to 0.5 m, and going to C (8) should halve it
again to 0.25 m. In reality the errors are 0.7 m and 0.5 m: the error shrinks, but much more slowly than the confidence
suggests. So VGGT is right about *which* pixel is better, but wrong about *how much* better. VGGT fails this check.

**Why the amounts matter.** We turn the expected error E of an edge into its weight with the rule weight = 1 / E²
(a standard rule from statistics). If we believed C's error is 0.25 m, its weight would be 16 times A's. With the real
0.5 m, it is only 4 times A's. Taking VGGT at its word would make us trust confident edges far too much.

Fixing the amounts, while keeping the order, is called **calibrating**.

### 4.3 How you calibrate

1. On the fitting drives 05 and 07, collect millions of (confidence, true error) pairs, using LiDAR as the truth.
2. Build a lookup table from them, for example "confidence 2 means about 1.0 m of error, confidence 4 about 0.7 m,
   confidence 8 about 0.5 m".
3. The only rule the table must obey is that higher confidence may never give a bigger error; apart from that it simply
   follows the data. This kind of table is called **isotonic regression**.
4. Freeze the table and use it, unchanged, on the test drives 00 and 02. Nothing is learned from those two drives, so
   the test stays fair.

### 4.4 How we measure whether it worked (calibration error)

To give one number for "how wrong are the predicted error sizes", we do this on the test drives:
1. Sort all pixels by confidence and split them into 10 equal-size groups (lowest confidence to highest).
2. In each group, compare the average squared error that was **predicted** with the average squared error that
   **really happened** (from LiDAR). Divide one by the other. If the prediction is perfect, this ratio is 1.
3. Note how far each ratio is from 1 (a ratio of 1.3 is 0.3 off; a ratio of 0.6 is 0.4 off), and average these
   over the 10 groups.

That average is the **calibration error**. 0 means the predicted sizes are exactly right; 0.40 means that, on average,
the predicted squared error is about 40% too big or too small.

Our result on drives 00 and 02: **0.40** if VGGT's confidence is taken at face value, **0.25** with the power-law formula
from the 2026 paper by N. Deng et al., and **0.18** with our lookup table. The table, learned only on 05 and 07, more
than halves the mistake.

### 4.5 The seven ways of setting edge weights we compare

To know whether *our* idea helps, we have to compare it with sensible alternatives, all tested in exactly the same way.

| Name | Where the weight comes from | Why it is in the comparison |
|---|---|---|
| (a) VGGT-Long | all edges weighted equally | what we have to beat |
| (b) uniform | one fitted weight for rotation, one for sliding, one for resizing, same for every edge | does just balancing those three help? |
| (c) raw confidence | VGGT's confidence, squared, no calibration | is the raw signal good enough on its own? |
| (d) geometry only | how badly the two windows' points still disagree after gluing | the classical choice, without the network's confidence |
| (e) **calibrated (ours)** | VGGT's confidence passed through the calibration curve | our method |
| (f) oracle | the true error of each edge (cheating) | not a method; shows the most any weighting could possibly gain |
| (g) power law | a calibration formula from a 2026 paper by N. Deng et al., refitted on 05/07 | the most recent published idea |

On top of this, each one is run three ways for the loops: plain; with **Huber**, which trusts a loop edge less when it
disagrees unusually strongly with the rest; and with **DCS**, which can ignore such loop edges almost completely. These two are called **robust
kernels**. How hard is "unusually hard" is chosen on 05 and 07 only. And everything is also run without loops at all.

### 4.6 How we grade a result

**ATE** (Absolute Trajectory Error) is the main score. Take the estimated path and the true path, rotate, shift and
resize the estimate so it lines up with the truth as well as possible (allowed because of 2.5), then measure the
distance between them at every frame and average it (as a root-mean-square). It is in metres; lower is better.
For example, an ATE of 9 m on a 3.7 km drive means the estimated path is, on average, 9 m away from the truth.

We also report KITTI's own **drift** numbers (**t_rel**: how many % of the distance travelled the path is off, over
stretches of 100 to 800 m) and, for the uncertainty itself, whether the predicted errors match the real ones on average.

### 4.7 The plan over time (from our proposal)

| Dates | Work | By |
|---|---|---|
| Oct 4 to 25 | data and set-up; run VGGT and reproduce VGGT-Long; measure true errors; check confidence; calibrate on 05/07; compare the seven schemes on 00/02; first tables and figures | **mid submission, Oct 26** |
| Oct 26 to Nov 22 | test with deliberately wrong loops; more datasets (Waymo, TUM); a classical method (ORB-SLAM3) for comparison; bonus: newer networks (CUT3R, StreamVGGT, VGGT-Omega) | final submission Nov 23, demo Nov 26 |

Everything in the mid-submission row is done (section 5).

---

## 5. What we have actually done so far, and how

### 5.1 The practical set-up

* **Code.** Everything is written in Python. The project folder has two parts: `mr_gpu/` runs VGGT on the GPU, and
  `mr_cpu/` does everything else (gluing, measuring errors, calibrating, the pose graph, grading, tables and
  figures). There are 44 automatic tests in `tests/` that check the maths and the conventions.
* **Computer.** The laptop has no suitable GPU, so the heavy work ran on the lab's server, **Phoenix**, which has three
  NVIDIA RTX A4000 graphics cards with 16 GB each. We used one card at a time, as the lab rules require.
* **Software environment.** On Phoenix all Python packages were installed with a tool called **pixi**, which keeps a
  project's packages in one folder and records exact versions (`pixi.toml`, `pixi.lock`), so the set-up can be
  rebuilt exactly.
* **Data.** The KITTI photos for sequences 00, 02, 05 and 07 came from a copy already on Phoenix; the true poses and
  calibration files from KITTI's website; all 13,064 LiDAR scans of these four drives were downloaded piece by piece from
  KITTI's server. The VGGT network weights (the trained numbers) were found on Phoenix and checked to be byte-identical to
  the official release.

### 5.2 Step 1: getting the details right

Before anything else we checked the small details that silently ruin results if they are wrong:
* KITTI's true poses are for a grey camera (camera 0), but VGGT sees the colour camera (camera 2), which sits 6 cm to the
  side. We converted the truth to camera 2.
* VGGT shrinks every photo to 518 x 154 pixels. We made sure LiDAR points are placed on exactly that grid.
* We drew LiDAR points on top of photos to see that they land on the right objects (they do).

### 5.3 Step 2: running VGGT

* **How much fits on the GPU.** We measured memory for windows of 8 to 64 frames: a 32-frame window needs 9.6 GB and
  takes 1.8 seconds; even 64 frames fits (12.2 GB). We chose **32-frame windows with 16 frames overlap**: close to the
  smallest setting in the VGGT-Long paper, and it gives many seams to learn from.
* **A "smoke test"** first ran 96 frames and checked four things automatically: VGGT's depth and its 3D points agree; the
  first camera of every window sits at the origin; VGGT's cameras match the true cameras after one rotate-shift-resize;
  and we read VGGT's poses the right way round. All four passed.
* **The real run** covered all four drives: 814 windows, about 17 frames per second, plus 43 loop windows.
  Everything was kept at VGGT's full resolution, without throwing away any pixels.

### 5.4 Step 3: reproducing VGGT-Long

**Reproducing** means: before claiming you improved a method, show that you can get the method's own published results.
Otherwise you do not know whether a difference comes from your idea or from a mistake in your set-up.

We had to rewrite VGGT-Long's gluing and pose graph inside our own code, because the original code does not let you
change the edge weights. So we checked two things:
1. **Our copy behaves like the original.** Piece by piece, our gluing gives the same numbers as theirs (to 15 decimal
   places), our pose graph solver ends in the same place, and our loop finder finds exactly the same loops as theirs on all
   four drives.
2. **The original reproduces its paper, and our copy matches the original.** We ran the original VGGT-Long program on all
   four drives at four window sizes and graded everything the same way:

| ATE in metres | 05 | 07 | 00 | 02 |
|---|---|---|---|---|
| VGGT-Long paper, 30-frame windows | 10.20 | 4.00 | 9.06 | 99.95 |
| original program, run by us, 30-frame windows | 10.24 | 3.98 | 9.07 | 99.65 |
| original program, 32-frame windows | 10.04 | 4.09 | 9.13 | 104.61 |
| **our copy, 32-frame windows, equal weights** | **10.02** | **4.06** | **9.18** | **105.07** |
| VGGT-Long paper, 60-frame windows | 9.15 | 2.68 | 8.06 | 34.16 |
| original program, 60-frame windows | 9.85 | 2.66 | 8.63 | 52.78 |

So the original program gives the paper's numbers (within 0.5% at 30 frames), and our copy gives the original program's
numbers (within 0.7%). One number does not match: drive 02 with 60-frame windows (52.8 m instead of 34.2 m). We checked
it every way we could: a second run (identical, so not random), an older loop setting (no change), an October 2025
version of their code (identical), letting their optimiser run 200 steps instead of 30 (identical), and the same drive
with 90-frame windows (50.6 m against the paper's 49.9 m, a match). The paper's own numbers for drive 02 go 100, 34, 50,
52 m for windows of 30, 60, 90, 120 frames, so 34 is the odd one out even in their table. We could not reproduce that
one number with their published code; everything else matches.

Note that drive 02 really does end up about 100 m off with small windows; the paper reports the same. Bigger windows help
there (about 50 m), because there are fewer seams to drift.

### 5.5 Step 4: measuring how wrong every seam and loop really is

To judge whether confidence predicts error, we need the true error of every seam and loop. The idea: for every window,
use the ground truth to work out where that window *should* be placed (rotate, shift and resize VGGT's cameras of that
window onto the true cameras). Then the true seam between two windows is simply "where window B should be, seen from
where window A should be". Compare that with what the seam actually measured: the difference is the seam's true error,
again 7 numbers (3 rotation, 3 sliding, 1 resizing).

Windows where the car is standing still are skipped, because when nothing moves you cannot tell the scale. That left
834 usable edges out of 853.

### 5.6 Step 5: is pixel confidence honest? (checked with LiDAR)

For every frame we projected the LiDAR scan into the photo and compared the true depth (from the laser) with VGGT's depth
at the same pixel, after fixing each window's size with one number. That gave **478 million** pairs of (confidence, true
error), counting every LiDAR point, nothing thrown away.

Findings, on the test drives 00 and 02 (338 million pixels):
* **Ranking works**: higher confidence really means smaller error.
* **Size is off**: VGGT's training suggests error should shrink in direct proportion to confidence (double the
  confidence, half the error). In reality it shrinks more slowly.
* **Calibration fixes most of it.** The calibration error (0 would be perfect) is 0.40 if you take confidence at face
  value, 0.25 with the 2026 paper's formula, and **0.18 with our calibration curve**, learned on 05/07 and tested on
  00/02.
* The biggest pixel errors are at the edges of objects, where depth jumps suddenly (for example the outline of a parked
  car), and those pixels also get low confidence.

### 5.7 Step 6: does confidence predict how wrong a seam is?

Here the answer is "partly":
* Averaged over many seams, all the weighting schemes predict roughly the right size of error (within a factor of about
  1.5).
* A seam's average confidence predicts its **rotation** error reasonably well, but hardly its **sliding** (position)
  error. For the sliding error, how badly the two windows still disagree after gluing is a much better clue.
* The pictures agree: the worst seams look at hedges and trees or are on a sharp turn, where confidence is low almost
  everywhere; the best seams look at buildings and road, where confidence is high.

In short: confidence is very informative per pixel, but averaging it over a whole seam loses much of what matters for
the seam's position.

### 5.8 Step 7: the surprising problem with loops (corrected after a review)

All 43 loops VGGT-Long found are real: in every case the car really was back at the same place. But measured against
the ground truth, the loop edges looked much more wrong than any scheme expected (their squared errors about 20 times
the prediction, which means about 4.7 times in plain error size).

Our first explanation was that VGGT squashes two visits in different lanes onto each other. **That was wrong**, and a
friend's review caught it. When we split the ground-truth gap between the two visits into sideways, up/down and
forward parts, almost all of it on drive 02 is **up/down**: the ground truth says that, on the same street, the car was
up to 4 m higher or lower the second time. A car cannot drive 4 m above the road it drove on before.

We checked this with the laser scanner. We took the LiDAR scan of each visit, placed both with the ground-truth poses,
and compared the height of the road surface at the same spot. If the ground truth were right, the two road surfaces
would line up. On drive 02 they are 0.8 to 4.1 m apart, almost exactly the ground truth's own height gap; on drive 00 up
to 1.3 m; on 05 and 07 they line up (within 0.2 m, two loops of 05 at 0.5 m). So **KITTI's ground-truth heights are
wrong at the revisits of 02** (the GPS height drifts), VGGT is right to put the two visits together, and most of what we
measured as loop error on 02 is error in the ground truth, not in VGGT. Our seam errors are not affected, because each
seam only uses the ground truth over a few seconds of driving.

### 5.9 Step 8: the final paths

ATE in metres (lower is better), loops on, 32-frame windows:

| | 05 (fit) | 07 (fit) | 00 (test) | 02 (test) |
|---|---|---|---|---|
| no loops at all | 48.70 | 22.36 | 140.49 | 181.52 |
| (a) VGGT-Long, equal weights | 10.02 | 4.06 | 9.18 | **105.07** |
| (a') equal weights, our way of measuring mismatch (the fair comparison for b to g) | 9.80 | 4.07 | 9.14 | 106.90 |
| (b) uniform | 11.52 | **3.61** | 10.15 | 113.36 |
| (c) raw confidence | 10.05 | 4.55 | 10.28 | 119.60 |
| (d) geometry only | 11.10 | 4.49 | 9.34 | 113.27 |
| (e) calibrated (ours) | 9.57 | 4.37 | **9.09** | 117.07 |
| (f) oracle (cheating) | **8.94** | 4.59 | 11.92 | 116.08 |
| (g) power law | 10.39 | 4.37 | 9.89 | 119.70 |

What this says, in plain words:
* **Loops matter enormously.** On drive 00, closing loops takes the error from 140 m to about 9 m.
* **Edge weights matter much less.** Compared with the fair version of equal weights (a'), our calibrated weights
  are better on 05 (9.57 against 9.80) and test drive 00 (9.09 against 9.14), but worse on 07 and on test drive 02. The
  schemes differ by 10 to 25% on the same drive. Just changing how an edge's disagreement is measured ((a) against (a'))
  moves the result by up to 2%.
* **On drive 02, every scheme is worse than equal weights.** We have not yet pinned down why. Grading on 02 is also
  done against ground truth that is inconsistent at the revisits (5.8).
* **DCS can do harm.** On 02 it switched off all 8 (correct) loops for four of the schemes, leaving about 175 m of error.
  Huber never did that.
* **Even cheating does not win.** Using each edge's measured error as its weight is not the best on 00 or 07. That only
  shows this particular rule is not the best possible one, not that per-edge weights cannot help; on 02 its loop
  "true errors" also include the ground-truth height problem.

### 5.10 The pictures we made

In `results/mid/figures/` and `results/mid/figures/qualitative/` (36 qualitative pictures):
* the estimated paths drawn over the true paths for every drive, and the original program against our copy;
* photos next to VGGT's depth, its confidence and the error measured with LiDAR;
* the worst and best seams with their confidence maps;
* the worst and best loops, with both visits' photos and a top view of where VGGT put them;
* the paths coloured by how wrong they are at each point;
* the 3D maps (point clouds) seen from above, with and without loops, from our copy and from the original program.

### 5.11 How we know our numbers are right

* 44 automatic tests of the maths, the conventions and the file formats.
* The four smoke-test checks before the long runs.
* Our ATE was compared with **evo**, a standard tool other researchers use, on all 72 paths: identical to 0.000000001 m.
* The original VGGT-Long program, run by us, reproduces its own paper, and our copy reproduces the original program.
* Every loop was compared with the ground truth and with the original program's loop finder.
* A repeated run of the original program gave identical results.

### 5.12 How the run was organised

The lab server's shared disk was almost full, so the run went one drive at a time: run the original VGGT-Long at four
window sizes, run our VGGT, measure seams, true errors and LiDAR, make the pictures, then delete that drive's big
temporary files before starting the next. At the end all drives were combined for calibration, the seven schemes and the
report. The whole thing took about six hours of computing. Results came back to the laptop and to GitHub; the 6 GB of raw
LiDAR comparison data stay on the server (`/scratch2/vpr-team/kavish/mr_audit_arrays/`) because they were too big to copy.

---

## 6. What it all means

* **We have a trustworthy baseline.** The original VGGT-Long gives its published numbers on our set-up, and our version
  gives the same paths. Every comparison is against the real thing.
* **VGGT's pixel confidence is useful, once calibrated.** It sorts pixels from good to bad correctly, and a simple
  calibration curve learned on other drives predicts the size of the error well.
* **Turning pixel confidence into seam trust is harder than expected.** Averaging confidence over a seam keeps the
  information about rotation but loses most of it about position.
* **The ground truth itself has a problem at the loops of drive 02.** Its heights at the revisits are off by up to 4 m
  (checked with LiDAR), so loop errors measured there mostly measure the ground truth, not VGGT.
* **So far, calibrated weights give small gains on some drives and losses on others.** That is an honest mid-term
  result; why they lose on 02 is still open.

---

## 7. What comes next (until the final submission)

* Deliberately add wrong loops to see which schemes survive them (the "false-loop stress test").
* Try other driving data (Waymo) and indoor data (TUM), and a classical method (ORB-SLAM3) for comparison.
* Use better clues for seams (confidence *and* how badly the windows disagree after gluing).
* Clean the loop labels: correct or leave out revisits where the ground truth is inconsistent, then re-check loops.
* Check that both pose graph solvers (ours and VGGT-Long's) really finish their job on every drive.
* Bonus: newer networks that read the video as one stream (CUT3R, StreamVGGT) and the newest VGGT (VGGT-Omega).

---

## 8. Glossary

* **ATE (Absolute Trajectory Error):** average distance between estimated and true path after the best rotate, shift and
  resize; in metres; lower is better.
* **Calibration (of uncertainty):** making stated uncertainty match real error sizes ("70% rain" means rain 70% of the time).
* **Camera 0 / camera 2:** KITTI's grey and colour cameras; ground truth is for 0, we use 2.
* **Confidence:** VGGT's per-pixel "how sure am I" number, at least 1, bigger is more sure.
* **DCS (Dynamic Covariance Scaling):** a robust kernel that can almost ignore loop edges that disagree a lot.
* **Depth / depth map:** distance of a pixel's content from the camera / one such distance per pixel.
* **Drift:** slow build-up of small errors along a path.
* **Edge:** a measurement between two windows in the pose graph: a seam or a loop edge.
* **evo:** a standard program for grading trajectories, used to double-check our ATE.
* **Fit / test sequences:** drives used to learn (05, 07) / drives used only for grading (00, 02).
* **Frame:** one photo of a video.
* **Frozen network:** a trained network used as is, without any further training.
* **GPU:** graphics card; the chip that runs neural networks; ours has 16 GB of memory.
* **Ground truth:** the correct answer, used for grading (KITTI's precise GPS poses and LiDAR depths).
* **Huber:** a robust kernel that trusts edges less when they disagree unusually strongly.
* **Information matrix:** the 7x7 table that sets an edge's weight; about 1 / (expected error)².
* **Isotonic regression:** a curve fit that only requires "more confidence never means more error".
* **KITTI:** a public driving dataset with cameras, LiDAR and precise ground truth; drives numbered 00 to 10.
* **LiDAR:** a spinning laser scanner that measures exact distances to many points.
* **Loop / loop closure:** recognising that you are back at a place you visited before, and using it to correct drift.
* **Loop window:** 20 frames from each of the two visits, given to VGGT together.
* **Monocular:** using a single camera.
* **Node:** a window in the pose graph: one window's placement in the world.
* **Optimiser / solver:** the program that finds the window positions that best agree with all edges.
* **Oracle:** a "cheating" scheme that uses the true errors, only to show the best possible gain.
* **Pixel:** one tiny coloured square of a digital photo.
* **Place recognition:** recognising a place from its appearance (here with the SALAD network).
* **Point cloud / point map:** 3D map made of dots / one 3D position per pixel.
* **Pose:** position plus orientation of the camera (6 numbers).
* **Pose graph:** the windows (nodes) and the seams and loop edges between them (edges), solved together for the final path.
* **Ranking:** whether a score puts things in the right order (better before worse).
* **Reproduction:** getting a published method's own numbers before claiming to improve it.
* **Robust kernel:** a rule that weakens edges that disagree too much (Huber, DCS).
* **Scale:** the unknown overall size of anything reconstructed from one camera.
* **Seam:** the glue joint (a Sim(3)) between two neighbouring windows.
* **Sim(3) / similarity transform:** rotate, shift and resize; 7 numbers.
* **SLAM:** Simultaneous Localisation And Mapping: estimating your path and a map together, correcting past mistakes.
* **Smoke test:** a quick check on a small piece of data before a long run.
* **Weight:** how much an edge is trusted by the optimiser.
* **Trajectory:** the list of camera poses over a whole drive.
* **t_rel:** KITTI's drift measure: % of distance travelled that the path is off, over 100 to 800 m stretches.
* **VGGT:** the neural network that turns a few photos into poses, depth, 3D points and confidence in one go.
* **VGGT-Long:** the method that runs VGGT on windows, glues them, closes loops and solves the pose graph.
* **Window:** a short overlapping piece of the video (here 32 frames, overlapping the next by 16).

---

## 9. Where things are in the project folder

| Path | What it is |
|---|---|
| `report/mid_report.pdf` | the mid-submission report |
| `results/mid/summary.md` | every result table in one file |
| `results/mid/figures/` | result figures; `qualitative/` has the pictures from 5.10 |
| `results/mid/upstream/` | the original VGGT-Long runs (paths, loops, settings) |
| `results/mid/run_info/` | logs and scripts of the server run |
| `mr_gpu/run_vggt.py` | runs VGGT on windows and loop windows |
| `mr_cpu/` | everything else: `alignment.py` (gluing), `edges.py` (seams and loops), `labels.py` (true errors), `depth_audit.py` (LiDAR check), `calibration.py` and `weighting.py` (the seven schemes), `backend.py` (pose graph solver), `evaluation.py` (ATE), `qualitative.py` (pictures), `report.py` (tables and figures), `pipeline.py` (runs all steps in order) |
| `tests/` | the 44 automatic tests |
| `docs/status.md` | short technical status with all numbers |
| `README.md` | how to install and run everything |
