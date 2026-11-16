# VGGT-Long, KITTI 02 at chunk 60: is 34.16 m reproducible? (search of 14 Nov 2026)

Verdict: others report the same problem. Nobody outside the authors' own tables reproduces 34.16 m at chunk 60.
All tables were read from HTML; check cells against the PDFs before citing.

## Paper versions (arXiv 2507.16443)
- v1 (22 Jul 2025): one VGGT-Long row, chunk 75 / overlap 30 (author, issue #27). 02 = 32.08.
  Row 00-10: 8.67 / 121.17 / 32.08 / 6.12 / 4.23 / 8.31 / 5.34 / 4.63 / 53.10 / 41.99 / 18.37, avg 27.637.
- v2 (16 Mar 2026, ICRA 2026): four chunk rows, overlap = half the chunk. 02 = 99.95 / 34.16 / 49.85 / 51.98
  (chunk 30 / 60 / 90 / 120). Chunk 90 and above ran on an NVIDIA L20 48 GB, not the 4090.
- Released config now: chunk 60, overlap 30, SALAD threshold 0.85 (comment: changed from 0.7), 30 iterations, C++ solver.

## GitHub issues (DengKaiCQ/VGGT-Long)
- #56 (open, 9 Jan 2026) https://github.com/DengKaiCQ/VGGT-Long/issues/56: user gets 02 = 52.89, 01 = 58.07,
  08 = 73.11 (evo -a -s). Author: also sees large discrepancies on these sequences; long sequences with few loop
  constraints are unstable and seemingly unrelated parameters change the result; paper used threshold 0.7, code
  uses 0.85; reporting your own reproduced numbers is acceptable.
- #36 (open, 31 Oct 2025) https://github.com/DengKaiCQ/VGGT-Long/issues/36: RTX 3090, kitti.yaml (75/30):
  02 = 37.47 against v1's 32.08. Author: runs on different days differ, suspects nondeterministic PyTorch ops.
- #27 (open, Sep 2025) https://github.com/DengKaiCQ/VGGT-Long/issues/27: v1 used 75/30 and evo 1.14.0 with scale
  correction; bfloat16 causes about 2 cm per chunk of environment-dependent drift.

## Third-party tables
| Paper | 02 | Provenance |
|---|---|---|
| HorizonStream, arXiv 2605.23889 | 52.72 | "released code and the default settings" (chunk 60 rerun; its 06/07/09/10 match the v2 chunk-60 row) |
| VGGT-Motion, arXiv 2602.05508 | 60.14 (VGGT-Long*) | "local re-evaluation under the same parameter settings as its paper" |
| SwiftVGGT, arXiv 2511.18290 | 37.56 | apparently a 75/30 rerun (matches issue #36) |
| VGGT-Align, arXiv 2608.15260 | 37.56 | apparently copied from SwiftVGGT |
| FFVO, arXiv 2609.13733 | 51.98 | copied v2 chunk-120 row |
| LoGeR, arXiv 2603.03269 | 32.08 | copied v1 row |

HorizonStream VGGT-Long row 00-10: 8.64 / 61.21 / 52.72 / 8.78 / 4.20 / 9.88 / 4.67 / 2.66 / 72.98 / 31.84 / 27.71.
Our value at 60/30: 52.78 m.

## MASt3R-SLAM on KITTI
"TL" on all sequences in VGGT-Long v1/v2, VGGT-Motion, VGGT-Align; SwiftVGGT "Tracking Lost". HorizonStream reports
numbers on some drives (01 530.37, 03 18.87, 04 88.98, 05 159.43, 06 92.00, 08 263.75, 10 153.07; dashes on 00, 02,
07, 09).

## Not accessed
Appendices of VGGT-Motion and VGGT-Align; Scal3R per-sequence values; Pi-Long README tables; issues #55 and #58.

## Checked against the PDFs and the GitHub API on 16 Nov 2026: see `notes/citation_check.md`
All table values above were found as quoted (issue 36 gives 37.465 for drive 02). VGGT-Long v2's table has four chunk
rows (30, 60, 90, 120; overlap = half the chunk); its chunk-30 row averages 44.713 m, our upstream plain runs 44.70 m.
