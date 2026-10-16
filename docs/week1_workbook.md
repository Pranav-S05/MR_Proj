# First practical session: three things to understand

Estimated study time is about 45–60 minutes, depending on how often you pause. You are not expected to understand the entire proposal today. First read sections 1–6 of `start_here.md`.

Open PowerShell in this project directory. You do not need to activate the environment:

```powershell
.\.venv\Scripts\python.exe -m mr_cpu.lab
```

This uses small arrays and writes the same reproducible outputs under `results/week1`. It never loads VGGT. The seed is fixed so you can relate the explanation to your own output.

## Experiment 1: two map fragments can describe the same scene

Open `results/week1/summary.json`, under `geometry`. We generated one set of 3D points and expressed them in a second coordinate system, using a rotation, translation and scale 2.4.

Before alignment the point RMSE is about **2.722** in synthetic coordinate units. The estimated scale is **2.4000000000000004**. After alignment, the error is about **2e-15**, effectively floating-point zero.

Why do this? Later we will compare predictions for the same shared pixels in two real windows. This experiment checks that our transform direction and scale solver work in a case where we know the answer. It does not establish that VGGT's real predictions will be accurate or mutually consistent.

The relevant function is `fit_similarity` in `mr_cpu/geometry.py`. Its source and destination arguments matter: switching them estimates the inverse transform. Camera rotations also need special handling when a world gauge changes; multiplying a camera matrix by a scaled transform without renormalizing its rotation gives an invalid camera pose.

Explain to yourself: why can two reconstructions disagree numerically while describing exactly the same geometry? Answer: they can use different origins, axes and rulers.

## Experiment 2: weights only matter when measurements compete

Open [the graph plot](../results/week1/graph_weights.svg). The true 1D node positions are `[0, 1, 2]`. Node 0 is fixed. One edge measures a move of 1; the next incorrectly measures 2. Chaining them gives `[0, 1, 3]`.

Changing the two edge weights cannot fix this chain. Both measurements can be satisfied exactly. The changed-weight result in the JSON is the same up to roundoff.

Now add an independent edge from node 0 to node 2 measuring 2. This is analogous to a loop constraint. We can no longer satisfy all three measurements. Equal weights give about `[0, 0.667, 2.333]`. Giving the faulty seam a small weight yields about `[0, 0.990, 2.010]`.

Here the weights were chosen by hand to illustrate the mechanism. We did not discover the bad real-world measurement. Learning a reliable way to do that is the research problem.

Try changing the last loop measurement from `2.` to `0.` in a separate copy of the example. This represents a false loop: even strong sequential measurements can be dragged toward the wrong relationship. You can reason about this before writing code. Calibration must be evaluated under this failure, not presumed to solve it.

The simple solver is `solve_translation_graph` in `mr_cpu/graph.py`. The same file also has a small 3D Sim(3) solver whose consistency is checked by the tests. Understanding the 1D example comes first.

## Experiment 3: good ordering is different from good uncertainty

Open [the calibration plot](../results/week1/calibration.svg). Its horizontal axis is a simulated confidence score. The model errors were manufactured with a known, decreasing noise scale. The raw uncertainty assumes a different relationship, `variance = 1/confidence²`.

We fit a monotone curve on **6,000 synthetic examples** and evaluate it on **4,000 independent examples**. Measured results from the fixed seed:

| Quantity | Raw assumption | Fitted curve |
| --- | ---: | ---: |
| Mean squared error divided by predicted variance; target 1 | 8.104 | 1.055 |
| Fraction inside nominal 95% Gaussian intervals | 69.25% | 94.125% |

The raw uncertainty is too small overall, so many errors fall outside its intervals. The fitted curve improves the magnitude for this deliberately learnable distribution. These are teaching results, not evidence about VGGT. The synthetic construction was chosen to illustrate miscalibration, and a different distribution can defeat the fit.

The relevant file is `mr_cpu/calibration.py`. Its fit target is squared error. Predicted variance is positive; the standard deviation is its square root. A larger variance gives a smaller graph information weight. Keep those directions straight.

Explain to yourself: can a monotone fit discover a hidden distinction between two equally confident but geometrically different cases? No. We may need geometric features, another model, or an honest statement that the output is insufficient.

## A small real-data check

Before the real-data command, inspect `correlated_measurement_demo` in the summary if you want to see the proposal's central statistical issue. Each point error contains a shared random displacement plus independent noise. Individual point variance is exactly known: `0.2² + 0.1² = 0.05`.

For 100 points, assuming independence predicts variance of the mean `0.05/100 = 0.0005`. The correct variance is `0.2² + 0.1²/100 = 0.0401`: **80.2 times larger**. Averaging cannot remove the displacement shared by every point. The lab checks this with 3,000 simulated estimates. Even perfect per-point uncertainty does not automatically produce correct uncertainty for a combined estimate. This is a scalar illustration, not a derivation of a complete Sim(3) covariance.

Run:

```powershell
.\.venv\Scripts\python.exe -m mr_cpu.trajectory --reference data/kitti/poses/05.txt
```

This reports **2,761 reference poses**, spanning about **2,205.58 metres** of reference path. It has not estimated those poses from images; it is reading the dataset's supplied answer.

The lab also applied an artificial global coordinate/scale change to each reference trajectory and checked that global Sim(3) alignment recovers it. Near-zero errors there check the evaluator. They must never appear in a table as our SLAM accuracy.

Finally run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The initial suite has **24 passing checks**, including a known 3D graph cycle, transform inverses, scale recovery, wrong matrix rejection, a synthetic LiDAR projection, an independent-noise calibration check and the chain invariance result. They protect basic assumptions; they do not replace real baseline reproduction.

For a first viva explanation, practise: “We keep VGGT fixed. It makes local reconstructions. We connect them with measured similarity transforms. Our hypothesis is that better estimated uncertainty on those connections can help a graph resolve conflicting evidence. We must show this on held-out sequences and compare against simple geometric alternatives.”
