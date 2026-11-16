# Citation check: every quoted number and sentence against its source (16 Nov 2026)

How: the PDFs were downloaded from arXiv (and the RSS 2018 proceedings) into `papers/pdf_check/`, converted with
`pdftotext`, and searched for each number and sentence of `notes/lit_*.md`, `notes/vggt_bridge_check.md` and
`notes/final_PLAN.md`. Two tables were also read as page images (VGGT-Long v2 Table 6, SemanticKITTI page 3).
GitHub issues were read through the public API. "OK" = found as quoted. This file replaces the "verify against the
PDFs" warnings in the three `lit_*.md` notes.

## Corrections to our own notes (three)
1. VGGT-Long's results table (v2) has FOUR chunk rows (30, 60, 90, 120), each with overlap = half the chunk. So
   30/15 and 60/30 are two of its four settings, not "the two". 75/30 is the setting of v1 of the paper and of the
   released `configs/kitti.yaml`. (`final_PLAN.md` revision 2 said "the two chunk settings": wrong wording.)
2. `lit_novelty.md` first listed VGGT-Bridge's per-dataset seam scale (Virtual KITTI -0.076, Waymo -0.047, EuRoC
   -0.039, 7-Scenes -0.017) and then, in its "Correction", said these were not in the PDF. They ARE in the PDF, in
   the appendix table: KITTI -0.0370 +- 0.0225, Virtual KITTI -0.0761 +- 0.0896, Waymo -0.0470 +- 0.0341, EuRoC MAV
   -0.0387 +- 0.0221, 7-Scenes -0.0167 +- 0.0169. The earlier direct read stopped before the appendix.
3. Three summaries in `lit_novelty.md` were our wording, not the papers': "Does VGGT Know What It Does Not Know?"
   does not say "poor predictor"; it reports ROC-AUC 79.7% for VGGT's own point-map confidence against 93.5% for a
   linear probe on its features (pose failure on HardMatch). "Scale Anchoring ..." (2611.06308) says "fixed-root
   scale anchoring" to "the initial submap" and evaluates on nine TUM sequences and three Clio scenes among 17
   sequences; "indoor only" is our reading. pi3 says a fixed reference view "can lead to instability and failures if
   the reference is suboptimal"; the "variance when the first frame is swapped" wording was not checked.

## VGGT-Long, arXiv 2507.16443 (v1 22 Jul 2025; v2 16 Mar 2026; arXiv listing: "ICRA 2026")
- v1 Table 1, one VGGT-Long row: 8.67 121.17 32.08 6.12 4.23 8.31 5.34 4.63 53.10 41.99 18.37, average 27.637. OK.
  v1 text: "we set the chunk size to 75 frames". OK. Overlap 30 is from the author in GitHub issue 27, not the PDF.
- v2 Table 1 (caption: "The overlap was set to half of the chunk"), averages over 00-10 then per drive:
  chunk 30: 44.713 | 9.06 96.20 99.95 19.76 11.36 10.20 10.04 4.00 139.79 50.53 40.94
  chunk 60: 26.358 | 8.06 96.96 34.16 6.83 4.16 9.15 4.68 2.68 63.15 32.24 27.87
  chunk 90: 22.718 | 11.97 40.51 49.85 5.41 2.86 9.88 6.07 3.47 66.27 32.27 21.33
  chunk 120: 25.597 | 16.13 53.43 51.98 4.37 2.15 12.69 11.33 3.603 70.29 34.55 21.05
  Drive 02 = 99.95 / 34.16 / 49.85 / 51.98: OK. "chunk size of 90 or larger ... NVIDIA L20 GPU with 48": OK.
- Our plain upstream runs at 30/15 against the paper's chunk-30 row: average 44.70 against 44.713; per drive within
  1.3% (largest: 03 19.53 against 19.76 and 04 11.22 against 11.36). The baseline of R4 is the paper's own row.
- v2 Table 6 (read as an image): loop closure off 58.69 / 36.01; IRLS off 12.29 / 10.98; confidence weight off
  11.28 / 10.13; all on 8.67 / 8.31 (drives 00 / 05). "8.67 against 11.28 m on drive 00 without the weight": OK.
- GitHub (API): issue 56 (open, 9 Jan 2026): 52.89, 58.07, 73.11, thresholds 0.7 and 0.85: OK. Issue 36 (open, 31 Oct
  2025): RTX 3090, kitti.yaml, drive 02 = 37.465 against the paper's 32.08 (we wrote 37.47): OK. Issue 27: 75, evo
  1.14: OK.

## VGGT-Bridge, arXiv 2610.06594 v2 (22 Oct 2026); "Accepted to ACCV 2026" is in the arXiv listing, not in the PDF
- Sentences: "VGGT anchors each chunk to its first frame as a reference and reconstructs later frames at a
  progressively smaller scale"; "a forward-ordered chunk is reconstructed at roughly 3.6% smaller scale than its
  predecessor"; "the scale drift accumulates without canceling along the sequential edges"; "Fine-sequential
  alignments are always forward-anchored, whereas each coarse chunk is measured both forward-anchored and
  reverse-anchored": all OK.
- Numbers: fine sequential -0.037; skip edges -0.032 / +0.036 (stride 3), -0.026 / +0.043 (stride 5); shifts
  0.068 +- 0.037 (85%) and 0.069 +- 0.029 (83%): OK.
- KITTI rows: VGGT-Long (their rerun) 10.81 66.46 56.00 5.56 3.93 10.11 6.90 2.83 57.45 31.17 22.11, 24.85;
  SwiftVGGT (their rerun) 9.92 114.96 55.45 10.38 5.48 12.50 8.38 3.50 69.74 30.92 25.55, 31.53; VGGT-Bridge 13.74
  56.34 48.70 9.33 4.00 11.02 6.93 3.15 47.15 26.24 22.05, 22.61: OK.
- Table 4 (31.53, 26.91, 22.85, 22.61), Table 5 (26.91, 24.08, 22.61), Table 6 (7,611 s, 2,583 s, 3,883 s), 197.6 s
  to 307.9 s on drive 08: OK.

## VGGT-Align, arXiv 2608.15260 (PDF: ACM MM '26, 10-14 Nov 2026)
- "dropping to approximately 0.5 by the end of the sequence"; "confirming compounding bias rather than zero-mean
  noise"; "The root cause is that scale is treated as a free variable without independent constraint"; "dominated by
  rotational drift in long loops, which scale anchoring alone cannot fully address" (drive 02, 47.04 against 37.56):
  all OK. Fusion weights "default 0.7 ... 0.3": OK.
- Table 1: VGGT-Long 29.41 (9.87 111.06 37.56 4.89 3.75 9.09 7.47 4.02 62.86 47.48 25.49); SwiftVGGT 29.18;
  VGGT-Align 19.99 (7.74 65.11 47.04 5.12 2.55 4.97 4.21 4.48 43.98 23.11 11.66): OK. Its VGGT-Long and SwiftVGGT
  rows are identical to those in the SwiftVGGT paper (copied, not rerun).

## Other reruns of VGGT-Long and smaller claims
- SwiftVGGT, arXiv 2511.18290: VGGT-Long row 9.87 111.06 37.56 ... 29.41; SwiftVGGT 29.18; MASt3R-SLAM "Tracking
  Lost": OK.
- HorizonStream, arXiv 2605.23889: VGGT-Long row 8.64 61.21 52.72 8.78 4.20 9.88 4.67 2.66 72.98 31.84 27.71, 25.94;
  "using the released code and the default settings"; MASt3R-SLAM 530.37, 18.87, 88.98, 159.43, 92.00, 263.75,
  153.07: OK. (AMB3R-SLAM's table carries the same VGGT-Long row.)
- VGGT-Motion, arXiv 2602.05508: "we observe a systematic bias at submap boundaries"; VGGT-Long* ("our local
  re-evaluation under the same parameter settings as its paper") 9.37 93.30 60.14 ... 28.38; VGGT-Motion 24.17: OK.
- So the reruns of VGGT-Long by others average 24.85 (VGGT-Bridge), 25.94 (HorizonStream), 28.38 (VGGT-Motion) and
  29.41 (SwiftVGGT, copied by VGGT-Align). Settings differ between them; do not call them "all 75/30".
- AMB3R-SLAM, arXiv 2609.19518: "set the middle frame of each submap as the reference view to reduce pose drift at
  the submap boundaries": OK; the word "middle" occurs nowhere else, so that choice is not ablated: OK.
- VGGT-SLAM 2.0, arXiv 2601.19887: "the first frame of the submap is also unequally influential": OK.
- LongStream, arXiv 2602.13172: "causing reference and scale biases": OK.
- "A Calibration Audit ...", arXiv 2608.29705: "off by at least 2.4x across all seven models": OK.
- FFVO, arXiv 2609.13733: VGGT-Long row equals the v2 chunk-120 row (51.98 on 02): OK. LoGeR, arXiv 2603.03269:
  carries the v1 values (32.08 on 02): OK.

## KITTI ground truth
- SemanticKITTI (ICCV 2019, arXiv 1904.01416, Sec. 3.1, pages 3-4): "...the provided information of the inertial
  navigation system (INS) often results in map inconsistencies, i.e., streets that are revisited after some time
  have different height. For three sequences, we had to manually add loop closure constraints...": OK.
- SuMa (RSS 2018): "In the training data, we observed some inconsistencies in the height of the recorded GPS-based
  trajectory": OK.
- BoW3D (arXiv 2208.07473): "we do also find that the ground truth pose of sequence 02 and 08 involve large errors
  in experiment. Therefore, we use the more consistent poses refined by ICP": OK.
- LCDNet (arXiv 2103.05056): "not aligned to nearby loop closures"; "groundtruth provided with the SemanticKITTI
  dataset": OK. IMLS-SLAM (arXiv 1802.08633): "big errors in GPS data (used as ground truth) with, for example, more
  than 5 m in the beginning of sequence 8"; 0.22 degrees: OK. DeepCLR (arXiv 2007.11255): "ground truth trajectories
  are partially inaccurate": OK.
- GitHub (API): semantic-kitti-api issue 39 ("for loop closure regions the ground truth trajectory is not
  consistent", sequence 08), SuMa issue 11 ("drift in the z-direction", sequence 00), semantic-kitti-api issue 140
  ("100% good poses"): OK.

## Not checked today (no PDF fetched; do not quote these without checking)
SuMa++ (IROS 2019) sentence; DeepVCP sentence; the KITTI web page note of 2013; Geiger et al. CVPR 2012 sentences;
CT-ICP and MULLS angle values; HyVGGT-VO and VGGT-SLAM++ statements; "VGGT-SLAM uses identity information matrices";
the classical references listed "from memory" in `lit_novelty.md` (Strasdat 2010, Song and Chandraker 2014, Kitt 2011).
