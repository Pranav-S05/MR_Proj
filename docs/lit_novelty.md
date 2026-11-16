# Novelty check for the scale-bias finding (search of 14 Nov 2026)

Read through arXiv HTML by a search agent; quotes are short excerpts. VERIFY against the PDFs before citing.

## Verdict
The headline (one-directional per-seam scale shrink from first-frame anchoring, compounding along the chain) is
already published. What was not found anywhere: the oracle component breakdown, the argument that uniform weights are
near-optimal because the error is a constant bias (why confidence weighting does not help), the direct in-window
measurement against ground truth, and a learned scale correction that needs no extra inference.

## Closest prior work
- VGGT-Bridge (Choi et al., KAIST; arXiv 2610.06594, 5 Oct 2026, ACCV 2026). "VGGT anchors each chunk to its first
  frame as a reference"; Umeyama scale on every sequential seam; "a forward-ordered chunk is reconstructed at roughly
  3.6% smaller scale than its predecessor". Mean log s negative on five benchmarks: KITTI 00-10 -0.037, Virtual KITTI
  -0.076, Waymo -0.047, EuRoC -0.039, 7-Scenes -0.017 (KITTI chunk 75, overlap 30). Sign flips when frames are fed
  reversed (-0.032 to +0.036). "the scale drift accumulates without canceling along the sequential edges." Fix:
  reverse-ordered coarse chunks as skip edges. Absent: in-chunk measurement against GT, oracle decomposition,
  middle-frame reference, learned de-biasing, any weighting claim.
- VGGT-Align (Zhang et al.; arXiv 2608.15260, Aug 2026, ACM MM 26). VGGT-Long "suffers progressive scale compression,
  dropping to approximately 0.5" on KITTI 08; "compounding bias rather than zero-mean noise". No cause given. Fix:
  anchor each chunk by camera-to-ground distance and road width (Sim(3) -> 6 DoF). On 02 its ATE is worse (47.04 vs
  37.56), "dominated by rotational drift in long loops, which scale anchoring alone cannot fully address".
- VGGT-Motion (Xiong et al.; arXiv 2602.05508): "we observe a systematic bias at submap boundaries"; uses the midpoint
  of the overlap as alignment anchor; bias not quantified, not framed as scale.
- AMB3R-SLAM (Wang and Agapito; arXiv 2609.19518, 17 Sep 2026): "set the middle frame of each submap as the reference
  view to reduce pose drift at the submap boundaries"; no ablation; base model DA3 or VGGT-Omega, not VGGT.
- VGGT-SLAM 2.0 (arXiv 2601.19887): first frame "is also unequally influential" (limitations).
- pi3 (arXiv 2507.13347): accuracy variance when the first frame is swapped; nothing on scale or chunks.
- LongStream (arXiv 2602.13172): VGGT's fixed reference causes "reference and scale biases"; no measurement.
- Scale Anchoring for VGGT-SLAM 2.0 (Jin et al.; arXiv 2611.06308, 4 Nov 2026): anchors every submap's scale to
  submap 0; indoor only.
- VGGT-Long (arXiv 2507.16443): unweighted pose graph; point-level confidence weighting in the seam alignment helps
  (its Table 6: 8.67 vs 11.28 m on 00). README recommends metric scale + SE(3) (Map-Long).
- Confidence / weighting: VGGT-SLAM uses identity information matrices. "Does VGGT Know What It Does Not Know?"
  (arXiv 2610.35806): confidence is a poor predictor of pose failure; no pose-graph test. "A Calibration Audit of
  Confidence..." (arXiv 2608.29705): confidences overconfident by about 2.4x; no downstream SLAM test. HyVGGT-VO
  (2604.02107) and VGGT-SLAM++ (2604.06830) use confidence-derived noise without ablating it.

## Our four claims
1. Scale dominates, map shrinks 2.5x to 59x: PARTLY KNOWN (VGGT-Bridge per seam, VGGT-Align cumulative). New: oracle
   component replacement, the size per drive.
2. In-window distortion as the cause: LARGELY ANTICIPATED by VGGT-Bridge (asserted, plus reversal control). New:
   direct measurement against GT inside the window (first/second half 1.04 to 1.05), constancy across drives.
3. Uniform weights near-optimal because the error is a constant bias: NO PRIOR REPORT FOUND.
4. Fixes: ground plane (VGGT-Align), metric scale (Map-Long, AMB3R-SLAM), middle reference (AMB3R-SLAM, no ablation,
   not VGGT), reversed windows (VGGT-Bridge), overlap-midpoint anchors (VGGT-Motion): PUBLISHED. Learned per-seam or
   per-frame-position scale correction: no prior report found.

## Other notes
- Not read: MASt3R-SLAM, SLAM3R, CUT3R, TTT3R, StreamVGGT, Stream3R, InfiniteVGGT, FastVGGT, VGGT-X, SAIL-Recon,
  MapAnything, DA3 paper, LONG3R, ViSTA-SLAM, SING3R-SLAM, EC3R-SLAM, SLAM-Former, MUSt3R, Spann3R, GeoCond.
- Teammates' repositories of this project are public: github.com/Pranav-S05/MR_Proj and
  github.com/mehr1sh/mobile_robitics_2026.
- Classical references (from memory, not re-checked): Strasdat, Montiel, Davison, "Scale Drift-Aware Large Scale
  Monocular SLAM", RSS 2010; Song and Chandraker, CVPR 2014 (ground-plane scale for driving); Kitt et al., ECMR 2011.

## Direct check of the three closest papers (second fetch, 14 Nov 2026; all three exist)
- VGGT-Bridge: "VGGT-Bridge: Beyond Sequential Pose Graphs via Coarse-Stride Skip Edges", Choi, Bae, Baek, Kim (KAIST),
  arXiv 2610.06594, v1 5 Oct 2026, v2 22 Oct 2026, "Accepted to ACCV 2026". Verbatim: "VGGT anchors each chunk to its
  first frame as a reference and reconstructs later frames at a progressively smaller scale." Pairwise Umeyama on
  overlaps, B=75, O=30 on KITTI. Edge weights: fixed base weight times a Cauchy kernel factor. No middle-frame
  reference, no learned or constant scale correction, no statement on which Sim(3) component dominates.
  Cost: extra reversed coarse chunks (Seq 08 VGGT inference 197.6 s -> 307.9 s).
  KITTI ATE 00-10 (Table 1): VGGT-Long* 10.81, 66.46, 56.00, 5.56, 3.93, 10.11, 6.90, 2.83, 57.45, 31.17, 22.11
  (avg 24.85); VGGT-Bridge 13.74, 56.34, 48.70, 9.33, 4.00, 11.02, 6.93, 3.15, 47.15, 26.24, 22.05 (avg 22.61).
- VGGT-Align: "VGGT-Align: Bridging Local Reconstruction and Global Consistency for Long-Sequence 3D Reconstruction",
  Zhang, Wu, Li, Wang (NWPU), arXiv 2608.15260, 15 Aug 2026, ACM MM 26. "The baseline suffers progressive scale
  compression, dropping to approximately 0.5 by the end of the sequence." "The root cause is that scale is treated as
  a free variable without independent constraint." Fix: camera-to-ground distance and road width replace the IRLS
  scale; test-time adaptation of normalisation layers.
  KITTI ATE 00-10 (Table 1): VGGT-Long 9.87, 111.06, 37.56, 4.89, 3.75, 9.09, 7.47, 4.02, 62.86, 47.48, 25.49
  (avg 29.41); VGGT-Align 7.74, 65.11, 47.04, 5.12, 2.55, 4.97, 4.21, 4.48, 43.98, 23.11, 11.66 (avg 19.99).
- AMB3R-SLAM: "AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical Backend", Wang and Agapito, arXiv 2609.19518,
  17 Sep 2026. Sec. 3.2 verbatim: "set the middle frame of each submap as the reference view to reduce pose drift at
  the submap boundaries." No ablation of that choice reported.

## Correction after reading the VGGT-Bridge PDF and code directly (15 Nov 2026)
See `notes/vggt_bridge_check.md`. The reversal ("sign flips") was measured on coarse skip edges only (-0.032 to
+0.036 at stride 3, -0.026 to +0.043 at stride 5); the fine sequential value is -0.037 and was never measured with a
moved anchor. The per-dataset values listed above (Virtual KITTI, Waymo, EuRoC, 7-Scenes) were NOT found in the PDF
(it evaluates KITTI, Waymo and Virtual KITTI; Fig. 4 is KITTI 00-10 only): treat them as unverified. Code is public:
https://github.com/BeCow5X5/VGGT-Bridge.

## Checked against the PDFs on 16 Nov 2026: see `notes/citation_check.md`
The "Correction" above was itself wrong on one point: VGGT-Bridge's per-dataset seam scales ARE in its PDF (appendix
table: KITTI -0.0370, Virtual KITTI -0.0761, Waymo -0.0470, EuRoC MAV -0.0387, 7-Scenes -0.0167). Three of the
one-line summaries above are our wording, not the papers' ("Does VGGT Know ...", "Scale Anchoring ...", pi3); use the
wording of `citation_check.md`. Everything else quoted in this file was found as quoted.
