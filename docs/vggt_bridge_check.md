# VGGT-Bridge: what the paper and its code actually do, against our test (read 15 Nov 2026)

Sources read directly: the PDF `papers/vggt_bridge/vggt_bridge_2610.06594.pdf` (arXiv 2610.06594 v2, 22 Oct 2026,
all 14 pages of text) and the public code https://github.com/BeCow5X5/VGGT-Bridge, cloned to
`papers/vggt_bridge_code/repo` (commit 7bee227, 15 Oct 2026). Nothing from that repository was run.

## Their setup (paper Sec. 4.1, code `run.py`)
- Built on SwiftVGGT, not on VGGT-Long: single Umeyama step per alignment, PCA-whitened DINO tokens for loops.
- Fine chunks B = 75, overlap O = 30 on KITTI and Virtual KITTI (60 / 24 on Waymo). Coarse strides {3, 5}; coarse
  chunk 35 frames at stride 3 (spans about 105 frames), 25 at stride 5. RTX 4090.
- Checkpoint `facebook/VGGT_tracker_fixed/model_tracker_fixed_e20.pt` (we use `facebook/VGGT-1B/model.pt`).
- Geometry: depth head, rescaled by the ratio of predicted focal lengths to a reference intrinsic
  (`align_intrinsics_and_depth`), unprojected with the predicted cameras. We follow VGGT-Long: point head.
- Alignment: Umeyama on pixels with |depth_prev - depth_cur| < 0.2 and confidence > 0.5 x mean. We follow VGGT-Long:
  confidence-weighted Huber IRLS with the spread-ratio scale.
- Pose graph: edge weight = fixed base weight x Cauchy factor (delta 2.0); skip edges have base weight 0.3.

## Their claim and their evidence
- Claim (Sec. 3.3): "VGGT anchors each chunk at its first input frame, and this anchoring biases the reconstructed
  scale to shrink away from that anchor as the input advances."
- Evidence (Sec. 5.2, Fig. 4, KITTI 00-10): pairwise Umeyama log s of every alignment that forms an edge.
  "Fine-sequential alignments are always forward-anchored, whereas each coarse chunk is measured both
  forward-anchored and reverse-anchored."
  - fine sequential edges: mean log s = -0.037 ("roughly 3.6% smaller scale than its predecessor");
  - skip edges through a coarse chunk: -0.032 forward to +0.036 reversed at stride 3; -0.026 to +0.043 at stride 5;
    per-edge shift 0.068 +- 0.037 (positive for 85% of edges) and 0.069 +- 0.029 (83%).
- Reversal in the code (`vggt_bridge.py` line 351): `frame_ids_input = list(reversed(frame_ids))`, outputs re-indexed
  with `[::-1]`; applied only to chunk sets with `stride > 1` (line 116).
- So the reversal was tested on COARSE chunks only. Whether moving the anchor changes the bias of the FINE
  sequential edges was not tested; that part of the claim is an inference from the coarse result.

## Their KITTI numbers (Table 1; ATE RMSE in m, 00 to 10, average)
- VGGT-Long (reproduced by them): 10.81 66.46 56.00 5.56 3.93 10.11 6.90 2.83 57.45 31.17 22.11, avg 24.85
- SwiftVGGT (reproduced by them): 9.92 114.96 55.45 10.38 5.48 12.50 8.38 3.50 69.74 30.92 25.55, avg 31.53
- VGGT-Bridge: 13.74 56.34 48.70 9.33 4.00 11.02 6.93 3.15 47.15 26.24 22.05, avg 22.61
- Ablation (Table 4, averages): SwiftVGGT 31.53; + coarse-stride chunks 26.91; + reversed coarse chunks 22.85;
  + loop-aware policy 22.61. Anchor ablation (Table 5): forward/forward 26.91, reverse/reverse 24.08,
  reverse stride 3 + forward stride 5 22.61. Time (Table 6): VGGT-Long 7611 s, SwiftVGGT 2583 s, VGGT-Bridge 3883 s.
- Note: their reproduced VGGT-Long gives 56.00 m on drive 02, one more independent value near our 52.78 m.

## How our test relates
- Our `study/gpu_run.py --ref last` feeds [last frame, then the others in time order]; theirs feeds the fully
  reversed order. These are the same input for VGGT if the order of the non-first frames does not matter. Reading
  the aggregator we run (`external/VGGT-Long/base_models/vggt/models/aggregator.py`) supports that: the camera and
  register tokens have one version for the first frame and one for all others, and positions are 2D patch positions
  only. Not yet confirmed by a run (shuffle the non-reference frames and compare outputs).
- Our test moves the anchor of the FINE sequential windows (window 32, stride 1): the experiment they did not run.
- Drive 07, log of the in-window early/late step ratio: anchor first +0.053, middle +0.043, last +0.026. Read as
  anchor-dependent part +-0.013 and anchor-independent part +0.039. Their stride-3 skip edges: -0.032 / +0.036, i.e.
  anchor-dependent +-0.034 and almost nothing anchor-independent (in their quantity, which also passes through two
  fine-to-coarse alignments, so the two are not the same measurement).
- So the two results agree in direction: moving the anchor to the end shifts the scale the same way in both. They
  differ in whether a part remains that does not depend on the anchor: in our fine windows it does, and on the drives
  tested so far it is the larger part.

## What can and cannot be said today
- VGGT-Bridge is not shown to be wrong. Their coarse-chunk measurement stands, and our data shows an anchor effect
  with the same direction.
- Our evidence that the fine sequential shrink is mostly independent of the anchor: reference = middle on four
  drives (bias not reduced, trajectories worse), reference = last on one drive (no flip, +0.013 +- 0.008).
- Open explanations for the difference: stride 1 against 3 to 5; window 32 against 75; checkpoint; depth head
  against point head; alignment method; too few drives for `last`; a mistake of ours not yet found.
- Wording for any write-up: we extend their analysis to the fine sequential chain; never "they are wrong".

## Addition, 16 Nov 2026 (full PDF text searched; see `notes/citation_check.md`)
The appendix (not read on 15 Nov) reports the fine-sequential seam scale on five benchmarks: KITTI -0.0370 +- 0.0225,
Virtual KITTI -0.0761 +- 0.0896, Waymo -0.0470 +- 0.0341, EuRoC MAV -0.0387 +- 0.0221, 7-Scenes -0.0167 +- 0.0169. So
the shrink is reported beyond driving data too. "Accepted to ACCV 2026" is in the arXiv listing, not in the PDF.
