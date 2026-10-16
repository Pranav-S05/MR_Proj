# Sources and verification scope

Checked October 16, 2026. These links are primary papers, author repositories, or official documentation. Reading a paper's abstract is distinguished from checking a method or source file. No literature search proves the absence of prior work.

| Source | Scope checked and reason |
| --- | --- |
| Original `Project_Context.md` and `mobile_robotics_2026.pdf` | Both read in full, including all seven proposal pages and references. |
| [VGGT](https://arxiv.org/abs/2503.11651), [official repository](https://github.com/facebookresearch/vggt) | Abstract, usage/output descriptions and selected source files; establishes what the frozen frontend provides. |
| [VGGT-Long](https://arxiv.org/html/2507.16443v1) | Method Sections 3.1–3.3 and evaluation table; checked against a pinned source snapshot. |
| [Pinned VGGT-Long source](https://github.com/DengKaiCQ/VGGT-Long/tree/c160869d1d99c96bb227f414afb3bc68c29c9a76) | Main pipeline, adapter, alignment utility, config and exposed loop optimizer inspected. C++/Python inner-solver parity is still untested. |
| [Calibration audit, v3](https://arxiv.org/html/2608.29705v3) | Confidence-loss interpretation, calibration scope, uncertainty caveats and baseline implications. |
| [Uncertainty quality on DTU](https://arxiv.org/abs/2606.16479) | Bibliographic record and abstract-level scope checked; not reproduced. |
| [GeoCond](https://arxiv.org/html/2609.18465v1) | Method and downstream rotation-averaging scope, including Appendix C; important novelty overlap. |
| [Public factor-refinement uncertainty module](https://github.com/jashshah999/vggt-factor-refinement/blob/82590a4b1ae96236c1bfbc5092307e681c68967a/src/uncertainty.py) | Entire selected uncertainty source file inspected. Its preset formulas do not independently substantiate the repository's benchmark claims. |
| [DROID-SLAM](https://arxiv.org/abs/2108.10869) | Abstract and architecture description; corrects the claim that it introduced no learned model. |
| [MASt3R-SLAM](https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html) | Official paper abstract and introduction; a closer example of building a system around an existing prior. |
| [VGGT-SLAM](https://arxiv.org/abs/2505.12549), [author explanation of SL(4)](https://gtsam.org/2026/06/24/vggt-slam.html) | Scope of projective ambiguity; motivates treating Sim(3) as an approximation rather than universally sufficient. |
| [VGGT-SLAM 2.0](https://arxiv.org/abs/2601.19887), [UniSim-SLAM](https://arxiv.org/abs/2608.01706) | Abstract/repository-level recent-work check. Not installed or rerun. |
| [VGGT-Omega](https://arxiv.org/abs/2605.15195) | Bibliographic record checked. Drop-in compatibility, memory and proposed token retrieval experiments remain unverified. |
| [Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a.html) | Official abstract. Our squared-error isotonic fit does not inherit a predictive-CDF coverage theorem merely by citing this work. |
| [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | Official download sizes, split and segment-error evaluation description; public pose archive downloaded and inspected. |
| [evo metrics](https://github.com/MichaelGrupp/evo/wiki/Metrics) | Official global-alignment metric documentation; evaluator parity remains a future check. |
| [SciPy least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) | Official solver documentation. The installed CPU version is pinned separately, not claimed to be the latest. |

`source_manifest.json` records exact inspected source versions and hashes, including the pose archive and extracted pose files. `reference_material/` holds the readable downloaded snapshots. They are evidence for inspection, not an installed runnable baseline. Re-running the fetch script can resolve a later upstream commit; preserve the current manifest before changing the research baseline.
