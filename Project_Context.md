# General Project stuff
The timeline document fixes what you are building, the dataset or robot you are building it on, and who does what by when. Pick a problem where you can show a metric, not a video alone: trajectory error against ground truth, map consistency after loop closure, reprojection error before and after bundle adjustment, disparity error on a stereo benchmark. Qual & Quant are both important.

Aim the final submission at a venue. A semester is enough to get a workshop paper or a short paper out of a good project, and past projects have grown into ICRA, IROS, RA-L and CVPR or ICCV workshop submissions. If that is the target, say so in the timeline document and pick the evaluation protocol the venue expects (a public benchmark, a named metric, ablations against a baseline) from day one rather than adding it in November. The TAs will read drafts.


## Timeline Document
The document should begin with an abstract of less than half a page outlining the project idea.
The abstract should serve as a pitch for the project, neither too brief nor overly complicated.
This should be followed by a more detailed description, within 5 pages (excluding references), covering the motivation, related work (limited strictly to the most relevant works), proposed methodology,
datasets, evaluation plan, and timeline. You must include sections describing the motivation behind
your chosen problem and the contributions your project will make; these sections count toward the
page limit. For clarity, the motivation refers to the shortcomings of existing work that point toward
your chosen direction, and the contributions explain how your project addresses those shortcomings.
Grading will be based solely on the write-up.

## Mid Submission
The mid submission document may be at most 6 pages long (excluding references), with an optional
additional page for appendices. It should include a formal, refined problem statement, a summary
of the progress made so far, and a concrete timeline for the remainder of the project. It should also
include a comprehensive literature review that demonstrates the viability of and motivation for the
project.
If you have used Hugging Face or WandB to train and upload models, include links to the relevant
projects in the repository’s README.
In addition, the remote repository and any Google Drive video links for your project must either
be made public or be shared with your TA mentors. Grading will be based on the write-up, the code,
and a viva. Slides for the viva are optional.

## Final Submission
The final document may be at most 8 pages long (excluding references), with up to two optional
additional pages for appendices. It should be a complete write-up of your project, structured like a
research paper. All claims must be supported by either prior work or empirical results, along with
sufficient analysis, and every research claim should be made as watertight as possible. You may also
discuss challenges, limitations, and potential future work; these will not count toward the page limit,
provided they appear in a separate section immediately before the references. This section must
contain only such discussion and no results from your project, as any results placed there will be
treated as future work or limitations.
If you have used Hugging Face or WandB to train and upload models, include links to the relevant
projects in the repository’s README.
Grading will be based on all aspects of the submission: the write-up, the code (included as a ZIP
archive in the submission), and the final viva, which will require a presentation (preferably included
as a PDF in the submission).

# Project Details given by out mentor before we selected this one
## Title: Feed forward geometry at kilometre scale
GPU, a big one.

VGGT gives you pointmaps and poses for a few dozen frames in one forward pass. A KITTI sequence is 4000 frames. So the obvious move is to chunk it: run the model over overlapping windows, register each window to the next with a Sim(3) (every window has its own arbitrary scale), close loops with the encoder’s global feature, and optimise a pose graph. VGGT-Long does exactly this and gets kilometre scale trajectories out of a monocular camera. Reproducing it is the easy half of this project.

The hard half is the pose graph. What information matrix goes on the edge between two windows? These models output a confidence per pixel, and it’s not obvious that those confidences track the actual error. So check. Compare predicted confidence against measured depth and pose error on sequences with ground truth. Are they calibrated? If they aren’t (our bet), fit a calibration, derive per edge covariances from it, hand those to the backend, and see whether the trajectory improves over uniform weights and over VGGT-Long’s own weighting. Same question for the loop closure edges, which are the ones that fold the map when they’re wrong.

Numbers: ATE against sequence length on KITTI odometry, Waymo and TUM RGB-D, compared with VGGT-Long, MASt3R-SLAM, DROID-SLAM and ORB-SLAM3. Calibration curves (predicted against empirical error) for depth and for relative pose, before and after your calibration. ATE with uniform weights, raw confidence weights and calibrated weights. Peak GPU memory and frames per second, since the whole reason to chunk is that the sequence doesn’t fit.

Bonus: swap VGGT for a streaming model (CUT3R, StreamVGGT) and find out whether a persistent state removes the need for chunking or just hides the drift.

References: Deng et al., VGGT-Long; VGGT; Murai, Dexheimer and Davison, MASt3R-SLAM, CVPR 2025; Wang et al., CUT3R, CVPR 2025; Zhuo et al., StreamVGGT; Structure from Attention, on what these networks attend to; KITTI odometry.
