# Plan from the actual starting date

Planning date: October 16, 2026, Asia/Calcutta. The user asked us to assume the proposal's dates: mid submission October 26, report November 23, demo November 26. GPU availability is expected to be clarified in about a day; access and specifications are unconfirmed. No previous code, data, predictions or results exist outside this setup according to the user.

## What the laptop can support

The machine reports about 7.88 GiB physical RAM and had about 13.12 GiB free disk after the CPU environment was installed. The user reports 2 GB VRAM. Keep CPU jobs small and process cached arrays one chunk at a time. Do not load every dense window into memory.

| Work | Local feasibility | Evidence or prerequisite |
| --- | --- | --- |
| Read papers and inspect baseline source | Yes | Pinned snapshots downloaded and inspected. |
| Learn/test camera transforms, Sim(3), alignment | Yes | Implemented and tested. |
| Small pose graphs and calibration fitting | Yes | Tiny reference solver and independent synthetic fit/evaluation implemented. |
| Inspect KITTI ground-truth trajectories | Yes | All 11 files downloaded and parsed. |
| Confirm real LiDAR projection and image-camera mapping | CPU work, but still pending | Need the correct calibration, images and matching LiDAR samples. Synthetic projection tests do not complete it. |
| Real VGGT/VGGT-Long reconstruction | Server work | Practical model/window memory has not been measured on the available server. |
| Reweight a fixed cached graph | Usually feasible on CPU | Actual graph size and solver timing must be measured. |
| Store all dense KITTI image/LiDAR/output data | Unsuitable here | Official colour and LiDAR archives are approximately 65 GB and 80 GB separately. [KITTI downloads](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) |

The proposal's original first week included a GPU inference smoke test and large dataset setup. Those portions cannot be marked complete on this laptop. We can complete the CPU foundation while access is pending.

## Revised schedule and decision gates

| Dates | Essential work | Concrete exit condition |
| --- | --- | --- |
| Oct 16–17 | CPU learning lab, proposal corrections, data/coordinate conventions, source/version record, GPU and storage confirmation | Explain the pipeline and rerun the lab; obtain server specifications or keep the fallback active. |
| Oct 17–19, conditional on access | Install the pinned baseline on the server; run a short fitting-sequence clip; validate RGB/LiDAR/pose conventions on real samples | Valid input/output hashes, a projected-LiDAR overlay, correct C2W export, memory/time measurements, reproducible pilot output. |
| Oct 19–22 | Run fitting sequences and a development sequence at one feasible window size; retain outputs; extract connection features and operational reference errors | Unit-information backend parity, non-leaking connection labels, calibration and geometry-only comparisons on development data. |
| Oct 22–24 | Freeze the method; run held-out 00/02 if the server budget allows; examine calibration and trajectory effects | A first honest results table, per-sequence plots, failures reported. Short clips remain pilot results and cannot establish kilometre-scale performance. |
| Oct 24–26 | Mid report and viva preparation | Refined problem statement, literature comparison including GeoCond, implemented progress, actual pilot findings or explicit compute limitation, remaining timeline. |
| Oct 27–Nov 8 | Extend the main comparison, investigate failure modes and true/false loops, add one transfer domain if feasible | Enough independent sequence evidence to support or reject the core hypothesis. |
| Nov 9–16 | Focused ablations, uncertainty analysis, resource table, mentor feedback | Explain which ingredient helped and where it failed; no new model families by default. |
| Nov 17–22 | Final validation, figures, writing, reproducibility checks | Freeze code/configurations/results with provenance. |
| Nov 23 / Nov 26 | Report / demo | Match claims to evidence; rehearse explanations of every figure. |

These are targets conditional on access, not promises of runtimes. If the baseline fails repeatedly, reduce the pilot before attempting all datasets. If GPU access is still absent on October 19, keep a strong CPU methods/learning submission in progress and give the TA a concrete scope/compute issue. Do not wait silently until October 26 or fabricate baseline results.

## Minimum research experiment

Use one frozen VGGT checkpoint, one pinned VGGT-Long version, one resolution, one window size and one overlap. Candidate primary setting: `W=16`, overlap `8`, subject to a fitting-sequence memory pilot. If infeasible, choose a smaller window using fitting/development data and document the decision. A 32-frame test is not the starting requirement.

Fit calibration on KITTI **05/07** as in the proposal. Use **06** for method/feature/threshold selection. Treat **00/02** as the primary held-out evaluation. Other non-fitting/non-development sequences can extend evaluation after freezing. Any numbers on 05/07/06 must be labelled fitting/development results. Leave-one-sequence-out is optional and must rerun the complete selection procedure inside each fold.

Do not randomly split overlapping edges into fitting and test sets. They share frames, outputs, route conditions and errors. Group statistical analysis by sequences or suitably separated temporal blocks. More pixels do not mean more independent scenes.

Start with these comparisons, all using identical edge measurements and candidate sets:

| Method | Purpose |
| --- | --- |
| Upstream equal-information graph | Reproduction anchor. |
| Constant block weights for rotation/translation/scale | Tests whether sensible units and DoF balance explain the gain. |
| Raw confidence-based edge weights with fixed clipping | Tests the native signal without calibration; `mean confidence²` is a heuristic. |
| Geometry-only reliability from registration residual/support/conditioning | Tests whether network confidence adds anything over classical evidence. |
| One-feature calibrated block uncertainties | Candidate method; all fitting and clipping choices disclosed. |

All calibrated and heuristic methods receive comparable fitting/development opportunities. Add the published pixel power law before aggregation only after defining the aggregation and establishing that this comparison is affordable. Refitting it avoids accidentally using published constants derived from our test data.

First run quadratic costs. Then compare a predeclared Huber variant with thresholds chosen using development data. DCS, a wider kernel grid and a large false-loop suite follow only if needed to answer a specific robustness question. The loops-off negative control need not repeat every weighting scheme at large scale once invariance is verified.

The minimum study should answer these questions in order:

1. Does the baseline reproduce under our feasible settings?
2. Can the chosen edge-reference labels be defined stably?
3. Does confidence predict connection error beyond geometric statistics?
4. Do fitted uncertainties transfer to held-out edges, including scale?
5. Does that transfer change the trajectory or robustness after controlling for DoF balance and kernels?

If the answer to question 3 is no, stop expanding the confidence-only method and examine the failure. If calibration improves marginal uncertainty but not ATE, that is a distinct finding. Better calibration and better point estimates are not equivalent objectives.

## Measurement definitions to freeze before the main test

For chunk-to-world transforms `S_i`, let `Z_ij` map chunk j into chunk i. Define

`r_ij = Log(inverse(Z_ij) @ inverse(S_i) @ S_j)`

in the explicit order `[rotation radians (3), Lie translation (3), log-scale (1)]`. The first node is fixed. Changes to the residual convention require corresponding covariance changes.

Define the operational reference chunk gauges using a pose-aware fit to full reference camera poses. This fit and its stability/degeneracy checks are an implementation task still pending real predictions. Labels belong in a separate fitting/evaluation file, never in deployable edge features. For a block-isotropic model, squared-error targets are `||e_rot||²/3`, `||e_trans||²/3` in specified normalized coordinates, and `e_logscale²`.

For held-out connection uncertainty, show predicted second moment against empirical second moment in bins whose boundaries are fixed on fitting data. Report sample counts and temporal/sequence grouping. A possible summary is `sum_b (n_b/N) * abs(mean_b(e²)/mean_b(predicted_variance) - 1)`, separately for each normalized block. Name and define this statistic rather than importing classification ECE without a definition.

For an explicitly assumed Gaussian model, report joint NLL and normalized squared residuals using the fitted matrices before graph-only normalization. Add empirical interval coverage and signed bias; mean NEES alone is insufficient. Very rare loop failures may require a failure detector or mixture model rather than a single Gaussian variance.

For the final trajectory, report one whole-sequence Sim(3)-aligned position ATE, per-sequence values, completion/failure status and official KITTI relative segment errors with scale handling declared. Check the ATE implementation against evo once its environment is available. Do not silently remove failed frames. TUM needs timestamp association and a format-aware evaluator; our current reader does not support TUM input.

For runtime, distinguish backbone inference, retrieval, registration, graph solving, I/O and total runtime. Count overlap and loop-centric inference in total work. Measure actual peak GPU memory on the server. CPU test runtime is not an inference benchmark.

For statistical variability, use multiple offsets/seeds only where meaningful and computationally affordable. Changing the offset requires new backbone outputs. Report independent-sequence results prominently; overlapping-edge bootstrap confidence intervals cannot be treated as independent-scene guarantees.

## Cache and experiment record

Keep dense arrays on server storage. For each window retain the original frame IDs, preprocessing transform, predicted intrinsics, C2W cameras, point/depth outputs with the matching confidence head, masks, dtype, resolution, and source/checkpoint identifiers. Dense outputs are needed for later registration and depth checks; an edge table alone cannot reconstruct them.

For each fixed connection, export a compact record: window IDs, edge type, 4-by-4 measurement, coordinate convention, confidence statistics, residual/support/conditioning features, and the characteristic translation length. Store reference labels separately. Transfer compact graph records to the laptop for reweighting and plotting. Always include sequence/window/overlap/offset/checkpoint/configuration hashes so incompatible runs cannot be mixed.

An illustrative lower-bound storage calculation: at 518-by-154 pixels, five float32 channels per frame occurrence already use about 1.6 MB. A 50% overlap processes nearly twice as many frame occurrences as unique frames; 23,201 frames then give roughly 74 GB for those five channels alone. This excludes RGB, duplicate confidence heads, aligned copies, loops and other arrays. Actual byte counts depend on output shape and dtype. Sample a pilot chunk and calculate the real budget.

## Extensions ranked by value and dependency

1. **Correlation and geometric conditioning:** strongest connection to the scientific question. Compare a naive covariance against a block/frame-resampled and regularized estimate, while testing shared bias. Needs good real labels first.
2. **Scale versus rotation/translation:** separate blockwise ablations identify what actually changes ATE. Uses the same fixed cache.
3. **Loop reliability:** add real rejected/accepted candidates and false-loop stress tests. Needs sufficient examples and possibly additional loop-centric inference.
4. **Transfer to one domain:** TUM can test indoor shift; Waymo tests another driving domain. These are alternative first extensions, not both mandatory immediately. TUM is still evaluated using RGB inputs if claiming monocular operation; depth supplies evaluation labels only.
5. **One window-size/resource study:** requires new forward passes; prioritize after the core effect is understood.
6. **Streaming models, VGGT-Omega, SL(4), all external baseline reruns:** defer. Each introduces new implementation or evaluation questions. A larger menu is not automatically a stronger paper.

Once access is confirmed, request the exact GPU and VRAM, CPU RAM, writable storage, operating system, CUDA/driver availability and queue/time limits. A 16–24 GB NVIDIA GPU and ample server storage would make the pilot easier; this is a planning preference, not a measured minimum. An 8–11 GB device may require smaller windows and separate retrieval batches. Test rather than promise that it will fit.
