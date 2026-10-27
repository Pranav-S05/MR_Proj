#!/usr/bin/env bash
# Upstream VGGT-Long (its own vggt_long.py, pinned commit, KITTI config) at one chunk size / overlap.
# Copies what we score and plot to results/mid/upstream/<SEQ>_w<W>_o<O>/; `pipeline --steps upstream report` scores it.
#   usage (project root, inside the pixi env): bash scripts/run_upstream.sh SEQ W O
#   EXPS=/dev/shm/... puts VGGT-Long's exps/ (its temporary chunk files) elsewhere, e.g. in RAM.
# VGGT-Long keeps every chunk's temporary files until it exits (10-13 GB per 1000 frames on KITTI). A chunk's
# files are never read again once its PLY is written, so a helper deletes them then (about 6 GB per 1000 frames
# at the peak); the outputs are unchanged.
set -euo pipefail
S=$1; W=$2; O=$3
ROOT=$(pwd); VL=$ROOT/external/VGGT-Long; DST=$ROOT/results/mid/upstream/${S}_w${W}_o${O}
EXPS=${EXPS:-$VL/exps_${S}_w${W}_o${O}}
mkdir -p "$DST" "$EXPS"
printf 'inherit_from: "configs/kitti.yaml"\nModel:\n  chunk_size: %s\n  overlap: %s\n' "$W" "$O" > "$EXPS/config.yaml"
ln -sfn "$EXPS" "$VL/exps"
( while sleep 2; do
    for p in "$EXPS"/*/*/pcd/*_pcd.ply; do
      [ -e "$p" ] || continue
      k=$(basename "$p" _pcd.ply); run=$(dirname "$(dirname "$p")")
      rm -f "$run/_tmp_results_unaligned/chunk_$k.npy" "$run/_tmp_results_aligned/chunk_$k.npy"
    done
  done ) & PRUNE=$!
trap 'kill $PRUNE 2>/dev/null || true' EXIT
t0=$(date +%s)
(cd "$VL" && python vggt_long.py --image_dir "$ROOT/data/kitti/sequences/$S/image_2" --config "$EXPS/config.yaml") > "$DST/run.log" 2>&1
echo $(( $(date +%s) - t0 )) > "$DST/wall_seconds"
OUT=$(dirname "$(find "$EXPS" -name camera_poses.txt | head -1)")
cp "$OUT/camera_poses.txt" "$OUT/intrinsic.txt" "$DST/"
for f in loop_closures.txt sim3_opt_result.png config.yaml; do [ -f "$OUT/$f" ] && cp "$OUT/$f" "$DST/" || true; done
python -m mr_cpu.qualitative ply2npz "$OUT/pcd/combined_pcd.ply" "$DST/pointcloud_sample.npz"
gzip -f "$DST/run.log"
rm -rf "$EXPS" "$VL/exps"
echo "upstream ${S}_w${W}_o${O} done in $(cat "$DST/wall_seconds") s"
