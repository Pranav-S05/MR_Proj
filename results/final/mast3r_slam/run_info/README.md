MASt3R-SLAM baseline run on the lab server (RTX A4000), 14 Nov 2026.
- Code: github.com/rmurai0610/MASt3R-SLAM at commit e6f4e3d474fad0e11f561482012be864ba8c3f17, unmodified
  (only config/kitti.yaml and the per-sequence intrinsics files config/kitti_SS.yaml were added).
- Environment: setup.sh + pixi.toml (Python 3.11, torch 2.5.1+cu118, its CUDA extensions built against CUDA 11.8).
  Checkpoints: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth plus the retrieval .pth and codebook .pkl
  from download.europe.naverlabs.com (downloaded on a laptop and copied over: the server's link to that host was slow).
- Runs: run_kitti.sh (00, 02), run_more.sh (05, 07, then TUM fr1/desk with the repo's eval_calib / eval_no_calib
  configs and its scoring, `evo_ape tum ... -as`). Logs: kitti.log, more.log, run_SS_MODE.log (includes
  /usr/bin/time -v), mem_SS_MODE.log (nvidia-smi samples of the main process every 30 s; empty for 02 nocalib).
- Trajectories: ../SS_MODE.txt (TUM format, keyframes only, t = frame / 30). Scored with
  `python -m mr_cpu.evaluation --estimate results/final/mast3r_slam/SS_MODE.txt --seq SS --tum-fps 30`;
  table in ../summary.md, figures in ../../figures/mast3r_slam_*.png.
