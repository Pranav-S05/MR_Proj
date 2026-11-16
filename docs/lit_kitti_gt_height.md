# KITTI odometry ground truth: height inconsistent at revisits (search of 14 Nov 2026)

Verdict: already documented as a general statement, with height at revisits named specifically. No published
per-sequence vertical numbers for 02 or 00 were found, so our 0.8 to 4.1 m (02) and up to 1.3 m (00) are more
specific than what is in print. Quotes were read from the paper text or GitHub; check against the PDFs before citing.

## Sources
1. SemanticKITTI, Behley et al., ICCV 2019, Sec. 3.1, https://arxiv.org/pdf/1904.01416:
   "This step is needed as the provided information of the inertial navigation system (INS) often results in map
   inconsistencies, i.e., streets that are revisited after some time have different height. For three sequences, we
   had to manually add loop closure constraints..."
2. SuMa, Behley and Stachniss, RSS 2018, https://www.roboticsproceedings.org/rss14/p16.pdf:
   "In the training data, we observed some inconsistencies in the height of the recorded GPS-based trajectory."
3. SuMa++, Chen et al., IROS 2019: "...in most sequences the GPS position is referenced to a base station, which
   makes it quite accurate, but still often only locally consistent."
4. semantic-kitti-api issue #39 (jbehley), https://github.com/PRBonn/semantic-kitti-api/issues/39:
   "We know that for loop closure regions the ground truth trajectory is not consistent. We also know that for some
   sequences the ground truth is obviously wrong due to supposedly a wrong initialization of the IMU (sequence 08)."
5. SuMa issue #11 (jbehley), https://github.com/jbehley/SuMa/issues/11: "...the IMU of the car, which has some drift
   in the z-direction. This drift is visible when one plots the ground truth trajectories of the sequences with loop
   closures (like sequence 00 and even more severe in the beginning of sequence 08)."
6. BoW3D, Cui et al., https://arxiv.org/pdf/2208.07473: "...we do also find that the ground truth pose of sequence 02
   and 08 involve large errors in experiment. Therefore, we use the more consistent poses refined by ICP..."
   (the only sequence-02-specific statement found; no magnitude or axis).
7. LCDNet, Cattaneo et al., https://arxiv.org/pdf/2103.05056: "...the groundtruth for some of these sequences is not
   aligned to nearby loop closures. Therefore, we use the groundtruth provided with the SemanticKITTI dataset..."
8. IMLS-SLAM, Deschaud 2018, https://arxiv.org/pdf/1802.08633: "...we found big errors in GPS data (used as ground
   truth) with, for example, more than 5 m in the beginning of sequence 8." (also the 0.22 degree Velodyne
   vertical-angle correction; CT-ICP 0.205 degrees, MULLS 0.2 degrees).
9. DeepVCP (ICCV 2019): "Some ground truth poses in KITTI involve larger errors, for example, the first 500 frames
   in Sequence 08." DeepCLR, https://arxiv.org/pdf/2007.11255: "the provided ground truth trajectories are partially
   inaccurate".
10. KITTI odometry page: segment lengths changed in 2013 "due to the fact that the GPS/OXTS ground truth error for
    very small sub-sequences was large".
11. Geiger et al., CVPR 2012: "open sky localization errors < 5 cm"; ground truth "is directly given by the output
    of the GPS/IMU localization unit".

## Alternative poses
- SemanticKITTI poses.txt (SuMa-based): http://www.semantic-kitti.org/assets/data_odometry_labels.zip (179 MB,
  CC BY-NC-SA, no registration), dataset/sequences/00..21/poses.txt, same convention as KITTI (camera frame).
  Caveat (issue #140): "For 02, it was still not possible to have 100% good poses".

## Reasons we could be wrong (checked)
- Overpass on 02: nothing documented (absence of evidence only).
- The 0.2 degree Velodyne error distorts both visits the same way (about 7 cm at 20 m): cannot give metres.
- A frame-convention mix-up would affect every sequence, but 05 and 07 are consistent.
- Mechanism: GT height is the raw GNSS/INS altitude (pykitti: tz = packet.alt).

## Checked on 16 Nov 2026: see `notes/citation_check.md`
Sources 1, 2, 4, 5, 6, 7, 8 and DeepCLR of source 9 were found as quoted (PDFs and GitHub API). Not checked: source 3
(SuMa++), DeepVCP of source 9, sources 10 and 11, and the CT-ICP / MULLS angles.
