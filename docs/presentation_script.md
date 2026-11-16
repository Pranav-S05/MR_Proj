# Mid report presentation script (about 10 to 12 minutes)

Speak slowly. Lines in [brackets] are what to point at.

---

## 1. Opening (about 1 minute)

Good morning. I am Kavish, and with Pranav and Mehrish our project is "Calibrated Confidence for Kilometre-Scale
Feed-Forward SLAM".

The question we ask is simple. A recent neural network called VGGT can look at a few dozen photos and, in one pass,
tell you where the camera was, how far away everything is, and how confident it is about every pixel. A method called
VGGT-Long stretches this to drives of several kilometres. But when VGGT-Long puts the pieces together, it trusts every
joint equally, even though some joints are clearly worse than others, and even though VGGT already tells us how
confident it is.

So our project asks: is VGGT's confidence honest, and if we turn it into weights for those joints, does the final path
of the car become more accurate? Importantly, we train nothing. VGGT stays frozen. We only learn a small calibration
on some drives and test it on others.

## 2. The problem in more detail (about 1.5 minutes)

[Equation 1 in the report]

VGGT can only take about 30 to 100 images at a time on a 16 GB GPU. So VGGT-Long cuts the video into overlapping
windows. We use windows of 32 frames, overlapping by 16. Each window is reconstructed separately, in its own position,
rotation and scale, because one camera cannot know true size.

Neighbouring windows share 16 frames, so they can be glued: we find the rotation, shift and resize that line up their
shared 3D points. We call that joint a seam. A drive like KITTI 02, with 4,661 frames, has about 290 seams. Each seam
is slightly wrong, and the errors add up. That is drift.

When the car comes back to a place it has seen, a place-recognition network finds it, and we get a loop edge between
two windows far apart in time. Then a pose graph finds the window placements that best agree with all seams and loops.
That is Equation 1. Each edge has a 7 by 7 weight, Lambda. VGGT-Long sets Lambda to the identity for every edge.

Our problem is to learn, without training, a map from simple features of an edge, mainly its average confidence, to a
calibrated Lambda, meaning that the error we predict for an edge matches the error it really has, on average. We split
this into three questions. Q1: is pixel confidence honest? Q2: does it predict the error of seams and loops? Q3: do
calibrated weights give a better trajectory? Everything is fitted on KITTI 05 and 07, and tested on 00 and 02, which
are never used for fitting.

## 3. Related work (about 1 minute)

[Section 2]

Briefly, where we stand. DUSt3R, MASt3R and VGGT all learn confidence with the same kind of loss, confidence times
error minus alpha log confidence. That loss is minimised when error equals alpha over confidence, which gives us a
concrete rule to test.

Two works checked this confidence before: Hillemann et al. on the small-object DTU dataset, and the calibration audit
of N. Deng et al. on 13 datasets including KITTI. They found that confidence ranks errors well but gets their size
wrong, and they proposed a power-law fix, which we use as a baseline. Both work only per pixel. The closest work to ours
is GeoCond, which trains a small network to predict pose reliability. We differ in that we train nothing, and we work
at the level of seams and loops of a kilometre-scale pose graph, and measure the final trajectory. For calibration we
follow Kuleshov et al.: reliability diagrams and isotonic regression.

## 4. Setup and reproduction (about 1.5 minutes)

[Section 3, first paragraph, then Table 1 and Figure 1]

Everything ran on one RTX A4000 with 16 GB on the lab server. We used KITTI drives 00, 02, 05 and 07. We converted the
ground truth from camera 0 to camera 2, which is the colour camera VGGT sees, and projected all 13,064 LiDAR scans onto
VGGT's input grid. VGGT ran on 814 windows plus 43 loop windows, and four automatic convention checks passed before the
long runs.

Before claiming any improvement, we had to show that our baseline is really VGGT-Long. We rewrote VGGT-Long's back end
so that edge weights can be changed, and checked it piece by piece against their code: same alignment to 10 to the
minus 15, same loops, same optimum.

[Table 1] Then we ran the original VGGT-Long code ourselves. At the paper's 30-frame setting, it matches the paper
within 0.5 percent on all four drives. At our 32-frame setting, our re-implementation matches the original code within
0.7 percent. So our baseline is VGGT-Long itself.

One honest note: on drive 02 with 60-frame windows, the paper reports 34 metres and we get 53. We repeated the run,
tried an older version of their code, more optimiser iterations and a different loop threshold, and always got the
same number, while 90-frame windows match the paper again. So we report that we could not reproduce this one number.

[Figure 1] Here you can see our re-implementation lying on top of the original code.

## 5. Q1: pixel confidence (about 1.5 minutes)

[Table 2 and Figure 2]

For Q1 we compared VGGT's depth with LiDAR at every LiDAR pixel, after one scale factor per window. That gave 478
million pairs of confidence and real error; 338 million are on the test drives.

Three findings. First, the ranking works: higher confidence does mean lower error, the Spearman correlation is
negative. Second, the size is off. If error were alpha over confidence, the log-log slope would be minus one. We
measure about minus 0.7, so doubling the confidence cuts the error by about 38 percent, not 50.

[Figure 2] You can see it here. The black line is the real error per confidence bin. The blue line, taking confidence
literally, is too steep: too pessimistic at low confidence and over-confident at high confidence.

Third, calibration fixes most of it. The calibration error, where zero is perfect, is 0.40 for raw confidence, 0.25
with the power law, and 0.18 with our isotonic map, which was learned only on 05 and 07. The green line follows the
black one best. In the appendix, Figure 5 shows the largest errors are on object borders, and those pixels also have
low confidence.

## 6. Q2: seams and loops (about 1.5 minutes)

[Table 3]

To answer Q2 we need the true error of every seam. For each window we line up VGGT's cameras with the ground-truth
cameras, which gives where the window should be. The true seam follows from two such placements, and comparing it with
the measured seam gives its error in rotation, translation and scale. Windows where the car barely moves are skipped,
because their scale cannot be determined. 834 of 853 edges are usable.

On held-out seams, all schemes predict the size of seam errors within about a factor of 1.5. But the right side of the
table is the important part. A seam's mean confidence ranks its rotation error reasonably, with Spearman minus 0.44,
but hardly its translation error, minus 0.12. The alignment residual, how badly the two windows still disagree after
gluing, is a much better clue for translation, 0.55. This agrees with GeoCond. [Figure 6] The worst seams are on
vegetation and turns, where confidence is low almost everywhere.

[Table 4 and Figure 7] Loops were a surprise. All 43 loops are real revisits, yet their measured errors were about 20
times larger than predicted in squared terms. We found that on drive 02 almost all of the ground-truth gap between the
two visits is vertical, up to 4 metres. A car cannot drive 4 metres above the same road. So we placed both visits'
LiDAR scans with the ground truth and compared the road heights. On 02 they differ by 0.8 to 4.1 metres, matching the
ground truth's own height gap; on 05 and 07 they agree. So KITTI's ground-truth height is inconsistent at the 02
revisits, and those loop errors mostly measure the ground truth, not VGGT. Seam errors are not affected, because a
seam covers only a few seconds of driving.

## 7. Q3: trajectories (about 1.5 minutes)

[Table 5]

Finally the trajectories. Two clear findings and one honest one.

First, loops matter enormously. On drive 00, closing loops takes the error from 140 metres to about 9.

Second, weights matter much less. Against the fair control, (a prime), which uses equal weights but the same residual
formula as our schemes, our calibrated weights are better on 05, 9.57 against 9.80, and on test drive 00, 9.09 against
9.14, but worse on 07 and on 02. On 02 every weighted scheme is worse than equal weights, and we have not yet isolated
why. We also saw that DCS, a robust kernel, switched off all eight correct loops on 02 for four schemes, while Huber
never did.

[Figure 3] This figure explains 02. On 00, loops pull the drifting chain onto the truth. On 02, all loops join the
start of the drive to its end, so the long middle section is never corrected, which is why 02 stays around 100 metres,
as in the VGGT-Long paper.

[Figure 4] And on 00, the largest remaining error is on a detour that no loop closes. Changing the weights barely
changes where the error is.

## 8. Summary and next steps (about 1 minute)

[Summary paragraph and Table 6]

To summarise. VGGT-Long is reproduced. Pixel confidence ranks depth error and is well predicted after calibration.
Averaging it over a seam keeps the information about rotation but loses most of it about translation. KITTI's ground
truth is inconsistent in height at the 02 revisits, which affects loop labels. And calibrated edge weights so far give
small gains on some drives and losses on others. Our proposal listed this as a possible outcome, and it tells us where
the signal is missing.

For the rest of the project, as in our proposal: the false-loop stress test, where we add wrong loops and see which
schemes survive; Waymo and TUM to test other domains; ORB-SLAM3 as a classical reference; and, if time allows,
streaming models and VGGT-Omega. Final submission is November 23 and the demo November 26.

Thank you. We are happy to take questions.

---

## If you need to stretch or cut

* **To stretch:** explain the 02 reproduction checks in more detail; walk through one seam's journey (window 200 and
  201, features, lookup table, weight); explain Huber versus DCS.
* **To cut:** skip section 3 (related work) to one sentence, and skip Figure 4.
