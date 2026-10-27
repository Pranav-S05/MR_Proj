#!/usr/bin/env bash
# After seq_stage_v2.sh succeeded and its outputs were checked: free the sequence's big inputs.
# Keeps cache/kitti/SEQ/manifest.json (+ descriptors) which the later steps read.
set -euo pipefail
S=$1; M=/scratch2/vpr-team/kavish/mr; R=$M/results/mid/$S
for f in edges.npz labels.npz depth_audit.npz qual/points.npz qual/frames.npz; do
  [ -s $R/$f ] || { echo "missing $R/$f: not freeing $S"; exit 1; }
done
for WO in w30_o15 w60_o30 w32_o16 w75_o30; do
  [ -s $M/results/mid/upstream/${S}_$WO/camera_poses.txt ] || { echo "upstream ${S}_$WO missing: not freeing $S"; exit 1; }
done
rm -rf $M/cache/kitti/$S/windows $M/cache/kitti/$S/loops $M/data/kitti/sequences/$S/image_2 $M/data/kitti/sequences/$S/velodyne
echo "freed $S; /scratch2: $(df -h /scratch2 | tail -1 | awk '{print $4}') free"
