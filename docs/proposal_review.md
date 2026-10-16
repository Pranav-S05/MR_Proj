# Proposal review and research decision

Reviewed October 16, 2026 against all seven proposal pages, the mentor context, primary papers, and pinned public source files. This review preserves the accepted proposal and recommends refinements; it does not rewrite history or claim completed experiments.

## Overall assessment

The problem is understandable and worth investigating. A frozen model plus a carefully designed estimator and strong experiments can support a paper. A new large neural model is unnecessary. However, adding an isotonic curve to an existing system is not automatically a research contribution, and a large experiment count cannot substitute for a specific finding.

The broad novelty claim needs revision. A candidate narrower question is:

**On held-out driving sequences, does calibration of the 7-DoF errors of VGGT-Long's chunk connections improve uncertainty quality and loop-constrained trajectories beyond constant DoF balancing, raw confidence, and geometric registration statistics? When does it fail because of shared errors or weak geometry?**

This is a hypothesis, not an established original contribution. A successful course project is realistic; acceptance at a particular conference cannot be predicted from a proposal. Decide a specific submission venue after the pilot and mentor feedback. A workshop or focused empirical paper may suit a limited but well-supported result.

## Existing work changes the novelty boundary

| Work checked | What overlaps | Consequence for us |
| --- | --- | --- |
| [Deng et al., calibration audit, September 10 version](https://arxiv.org/html/2608.29705v3) | Native confidence, error ranking versus magnitude, post-hoc power-law calibration | Pixel calibration on KITTI is a replication component, not the main novelty. |
| [GeoCond, September 16, 2026](https://arxiv.org/html/2609.18465v1) | Pose reliability from frozen geometry and downstream weighting; Appendix C evaluates rotation averaging | We cannot claim that confidence-to-pose reliability or graph weighting is new in general. Compare its scope carefully against chunk-level Sim(3), scale, and long trajectories. |
| [Public VGGT factor-refinement implementation](https://github.com/jashshah999/vggt-factor-refinement/blob/82590a4b1ae96236c1bfbc5092307e681c68967a/src/uncertainty.py) | Confidence-to-pose-noise heuristics and factor construction | A relevant implementation, not evidence that its claimed empirical calibration is independently validated. Our code inspection found preset formulas, not a saved fitting protocol in that file. |
| [VGGT-SLAM 2.0](https://arxiv.org/abs/2601.19887), [UniSim-SLAM](https://arxiv.org/abs/2608.01706) | More recent submap/graph designs | The related-work discussion must extend beyond the 2025 systems. They need not all become expensive rerun baselines. |

This was a targeted literature check, not proof that no other work covers the narrowed question. We should refresh it before making novelty claims in a submission.

The TA's example also needs separating from the lesson: [DROID-SLAM](https://arxiv.org/abs/2108.10869) introduced a learned recurrent system with differentiable dense bundle adjustment. [MASt3R-SLAM](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) is a closer example of constructing a new SLAM system around an existing reconstruction prior. Either way, a paper needs an identifiable technical or empirical contribution.

## What source inspection confirmed

The baseline snapshot is VGGT-Long commit `c160869d1d99c96bb227f414afb3bc68c29c9a76`. Inspection is not a reproduction.

In [`sim3loop.py`](https://github.com/DengKaiCQ/VGGT-Long/blob/c160869d1d99c96bb227f414afb3bc68c29c9a76/loop_utils/sim3loop.py), the exposed optimizer constructs unwhitened residuals and evaluates their squared mean; it provides no per-edge information argument. This supports the proposal's equal-information reading for this code path. It also returns the sequential transforms unchanged when there are no loop constraints.

In [`sim3utils.py`](https://github.com/DengKaiCQ/VGGT-Long/blob/c160869d1d99c96bb227f414afb3bc68c29c9a76/loop_utils/sim3utils.py), `weighted_align_point_maps` combines corresponding confidence scores with their geometric mean and calls robust weighted alignment. The inspected main alignment path uses IRLS, not the RANSAC-plus-Umeyama description on proposal page 3. [`base_model.py`](https://github.com/DengKaiCQ/VGGT-Long/blob/c160869d1d99c96bb227f414afb3bc68c29c9a76/base_models/base_model.py) converts VGGT camera extrinsics to C2W before exporting them. Integration must follow operations and conventions rather than trust potentially confusing comments.

The baseline's backend uses PyTorch/PyPose even on CPU; “Python solver” does not mean “no PyTorch.” Our tiny reference solver instead uses NumPy/SciPy. Before real comparisons, unit-information integration must match the chosen upstream backend on identical graph inputs.

## Mathematical and experimental corrections

**1. “No training” should become “no backbone retraining.”** Isotonic regression fits parameters to labelled data. State which labels and sequences it uses. It is supervised post-hoc calibration.

**2. Loss-optimal confidence is not automatically a standard deviation.** For `C*e - alpha*log(C)`, the unconstrained optimum of the conditional expected loss is `C = alpha / E[e | features]`. The interpretation depends on the actual training residual, normalization, confidence floor, and training distribution. An absolute/norm residual and a squared residual imply different uncertainty relationships. A log-error slope of -1 on deployment data is a testable idealization, not a law. The audit also notes that VGGT's actual alpha is unattested. Calling `C²` a theoretically justified edge precision is too strong; retain it as a heuristic baseline. [Audit, Sections 3–4](https://arxiv.org/html/2608.29705v3)

**3. A “true seam” is not obtained by simply subtracting metric poses.** Each predicted chunk lives in its own arbitrary gauge. Define a reference transform `A_k` from chunk k's predicted cameras to the metric reference using a documented pose-aware fitting objective. With our convention, the reference connection is `Z_ij_ref = inverse(A_i) @ A_j`. Then score `Log(inverse(Z_ij_hat) @ Z_ij_ref)`. Camera positions constrain scale; orientations help constrain rotation. Short or low-motion windows may not determine this reliably. The fits are an operational reference under model distortion, not a uniquely observed physical seam. Report fitting residuals, sensitivity to the anchor frames, and excluded degeneracies. Reusing the same overlap evidence to define and evaluate a seam can make labels optimistic.

**4. The Lie-algebra translation coordinate is not simply the matrix translation.** In a Sim(3) logarithm, translation couples with rotation and scale. Our code explicitly tests this. The new code uses `[rotation, upsilon, log-scale]`; upstream libraries can use a different ordering and residual frame. Permute matrices and, where required, transport covariance with the appropriate Jacobian/adjoint. Matching seven numbers by shape alone is unsafe.

**5. Three learned scalars are a block-isotropic approximation.** A matrix with one rotational, one translational, and one scale variance is 7-by-7 but is not a full learned covariance. It cannot express directional weakness or cross terms. Start with this approximation and describe it honestly.

**6. Fit per-coordinate second moments.** If the target is the squared norm of a 3D rotation error, divide by three before using it as the variance of each rotational coordinate. Do the same for translation. Otherwise the expected normalized error is wrong by a dimensional factor. Systematic bias means the second moment includes both covariance and bias squared. Inspect signed errors and do not call every fitted second moment a calibrated covariance.

**7. Translation uncertainty proportional to seam length is singular near a revisit.** The formula `(sigma_t * ||t_ij||)^-2` diverges when translation is near zero. Use a predeclared positive characteristic length in the residual's coordinate frame, with a floor chosen on fitting data, and document how it changes under a gauge change. Median valid predicted depth in that frame is a candidate, not a validated solution. Scene-dependent normalization must not use test ground truth.

**8. Isotonic regression is one-dimensional in the proposed initial form.** “Mean confidence, residual, inlier fraction, overlap and depth” is not a complete model definition. First fit one curve from one selected scalar to one target. Compare separately with a geometry-only scalar. Any combination or feature selection must be done on fitting/development data and counted as additional fitting.

**9. A tree is a necessary negative control.** With only adjacent edges and one fixed node, every measured transform can be satisfied exactly by composition. Positive weights cannot change that zero-residual solution. Thus the proposed loops-off condition tests correctness, not an expected source of gains, unless other constraints or priors are added. If weights change the chain-only reconstruction, investigate the implementation.

**10. Common weight scaling is harmless only under the right objective.** For pure quadratic costs, multiplying all information matrices by a positive constant preserves the minimizer. With Huber on whitened residual norm, also scale its threshold by the square root of that constant to preserve the equivalent objective. DCS has its own scale parameter, whose convention must be specified. Solver damping and stopping tolerances can also change numerical behavior. Never silently normalize an allegedly calibrated covariance before scoring its statistical calibration.

**11. NEES around seven is a consistency diagnostic, not proof.** Its chi-square interpretation needs a correctly defined, unbiased approximately Gaussian residual with the matching covariance. Mean seven alone can hide wrong shapes, wrong conditional calibration, bias, or badly estimated tails. Report coverage curves, blockwise behavior, sample counts, signed bias and NLL under an explicitly stated model. Overlapping edges are not independent samples for confidence intervals.

**12. “Oracle upper bound” is not justified.** Weighting by an observed true error is a hindsight diagnostic. It is not guaranteed to give the best ATE: graph leverage, direction, bias and conflicting loops matter. It can indicate headroom; failure of one heuristic oracle does not prove that weighting cannot help. Never put it in the deployable-method ranking.

**13. Calibration cannot generally solve false correspondence.** A false loop is a different failure class from noisy registration. A low residual can arise for a wrong match in repeated scenes. Keep geometric verification and robust optimization, and evaluate calibration's incremental benefit. Separate artificial false edges from naturally occurring appearance matches; only the latter test the full candidate-generation failure mode. Report absolute ATE as well as a predeclared survival ratio, especially when the no-corruption ATE is near zero.

**14. Sparse depth evaluation needs actual image conventions.** KITTI reference poses, colour-camera images and LiDAR require the proper inter-camera and sensor transforms. Reference camera 0 and colour camera 2 are not interchangeable. Compare optical-axis depth with depth, not LiDAR range. Apply rectification, resize and crop transforms; resolve occlusion by retaining the nearest valid projected point at a pixel. Ground-truth-fitted scale per window makes this a scale-aligned shape audit, not deployable metric depth. Report the fitting convention and avoid an evaluation subset also fitting its own scale if an independent subset is available.

**15. Ground-truth statistics must be measured rather than copied.** The downloaded KITTI 00 file contains 4,541 poses; the paper table lists 4,542 frames. Our 00–10 files total 23,201 poses. Reference path length for 07 measures about 694.70 m, whereas the table gives 649.70 m. These discrepancies need checking against images once available, not silently truncating to match a table. The current evaluator rejects length mismatches.

**16. Caching does not make all later experiments free of inference.** Changing window size, offset, resolution, model, or adding uncached loop-centric windows requires new forward passes. Adjacent windows reprocess their overlap. A frozen graph can be reweighted cheaply; it is not a universal cache for every experiment. Dense outputs can also exceed local disk/RAM budgets.

**17. Some extensions require new mathematics.** Moving from 7-dimensional Sim(3) to 15-dimensional SL(4) changes the state, uncertainty representation and observability. The page-2 claim that the weighting method carries over unchanged is incorrect. Streaming-model adapters and register-token retrieval also need their own implementation and evaluation.

**18. Rank invariance needs a qualification.** A strictly monotone correction preserves ranks. Isotonic regression can create flat intervals and therefore ties, so Spearman correlation can change numerically. It still cannot reverse the input ordering to discover a previously hidden distinction. Do not interpret tie-induced changes as new ranking information.

**19. Separate loop calibration needs enough independent loop examples.** Fitting on two sequences may provide many adjacent seams but very few distinct correct or incorrect loops. A flexible loop-specific isotonic map can then overfit. Count distinct loop events, show uncertainty, and use a simple pooled/shrunk fallback or leave the claim unmade when support is inadequate. Many candidates around the same revisit do not create many independent examples.

## The highest-value possible upgrade

After a real pilot, compare a naive independent-point covariance with a correlation-aware alternative that resamples spatial blocks or whole shared frames, then shrinks unstable estimates toward a diagonal floor. Test both against held-out 7-DoF errors and final trajectory metrics. This would directly investigate the proposal's independence concern.

That is a research direction, not an implemented or established novel method. Bootstrap variability conditional on one set of network outputs will miss errors shared by the entire reconstruction. Add a synthetic common-bias case and verify against real errors before calling it a network uncertainty estimate. Also measure correlation between adjacent edges; correcting pixels inside one edge does not make graph edges independent.

If confidence has no extra predictive value after geometric statistics, a rigorous result can explain that limitation. It would still need sufficiently broad evidence and differentiation from GeoCond to support publication.
