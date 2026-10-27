#!/usr/bin/env bash
# One sequence, everything that needs its images / LiDAR / cache:
#   upstream VGGT-Long at 30/15, 60/30 (paper), 32/16 (ours), 75/30 (released config), temp files in RAM;
#   our full-resolution VGGT run; then edges, labels, audit, qual. Resumable: finished parts are skipped.
# usage: seq_stage_v2.sh SEQ GPU
set -euo pipefail
S=$1; GPU=$2
K=/scratch2/vpr-team/kavish; M=$K/mr
. $K/pixi/env.sh
cd $M
set +u; eval "$(pixi shell-hook --manifest-path $M/pixi.toml)"; set -u
export CUDA_VISIBLE_DEVICES=$GPU
log() { echo "[$(date '+%F %T')] $*"; }

for WO in "30 15" "60 30" "32 16" "75 30"; do
  set -- $WO; W=$1; O=$2; NAME=${S}_w${W}_o${O}
  DST=$M/results/mid/upstream/$NAME
  if [ -f $DST/camera_poses.txt ]; then log "upstream $NAME already done"; continue; fi
  SHM=/dev/shm/kavish_vl/$NAME; rm -rf $SHM; mkdir -p $SHM/exps $DST
  printf 'inherit_from: "configs/kitti.yaml"\nModel:\n  chunk_size: %s\n  overlap: %s\n' $W $O > $SHM/config.yaml
  ln -sfn $SHM/exps $M/external/VGGT-Long/exps
  log "upstream $NAME start (mem available $(awk '/MemAvailable/ {print int($2/1e6)}' /proc/meminfo) GB)"
  # watcher: (1) a chunk's two temp files are never read again once its PLY exists -> delete them early
  # (VGGT-Long itself deletes them only at the very end; outputs are unchanged); (2) record the peak;
  # (3) stop the run if the machine or /dev/shm runs short (other users).
  ( peak=0; while sleep 2; do
      for p in $SHM/exps/*/*/pcd/*_pcd.ply; do
        [ -e "$p" ] || continue
        k=$(basename $p _pcd.ply); run=$(dirname $(dirname $p))
        rm -f $run/_tmp_results_unaligned/chunk_$k.npy $run/_tmp_results_aligned/chunk_$k.npy
      done
      u=$(du -sm $SHM 2>/dev/null | cut -f1); [ "${u:-0}" -gt $peak ] && peak=$u && echo $peak > $DST/peak_tmp_mb
      a=$(awk '/MemAvailable/ {print int($2/1e6)}' /proc/meminfo)
      s=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
      if [ "$a" -lt 15 ] || [ "$s" -lt 6 ]; then
        echo "low memory ($a GB) or /dev/shm ($s GB): stopping" >> $DST/ABORTED
        pkill -f "vggt_long.py --image_dir $M/data/kitti/sequences/$S/" || true
      fi
    done ) & WATCH=$!
  t0=$(date +%s)
  ( cd $M/external/VGGT-Long && python vggt_long.py --image_dir $M/data/kitti/sequences/$S/image_2 --config $SHM/config.yaml ) \
      > $DST/run.log 2>&1 || { kill $WATCH; rm -rf $SHM; log "upstream $NAME FAILED, see $DST/run.log"; exit 1; }
  kill $WATCH || true
  echo $(( $(date +%s) - t0 )) > $DST/wall_seconds
  OUT=$(dirname "$(find $SHM/exps -name camera_poses.txt | head -1)")
  cp $OUT/camera_poses.txt $OUT/intrinsic.txt $DST/
  for f in loop_closures.txt sim3_opt_result.png config.yaml; do [ -f $OUT/$f ] && cp $OUT/$f $DST/ || true; done
  python -m mr_cpu.qualitative ply2npz $OUT/pcd/combined_pcd.ply $DST/pointcloud_sample.npz
  gzip -f $DST/run.log
  rm -rf $SHM $M/external/VGGT-Long/exps
  log "upstream $NAME done ($(cat $DST/wall_seconds) s, peak temp $(cat $DST/peak_tmp_mb 2>/dev/null) MB)"
done

if [ ! -f $M/cache/kitti/$S/manifest.json ] || ! grep -q loop_windows $M/cache/kitti/$S/manifest.json; then
  log "VGGT (ours) $S start"
  python -W ignore::FutureWarning -m mr_gpu.run_vggt run --seq $S --window 32 --overlap 16 --stride 1 --hash \
      --out $M/cache/kitti/$S > $M/logs/gpu_$S.log 2>&1
  log "VGGT (ours) $S done: $(du -sh $M/cache/kitti/$S | cut -f1)"
fi

if [ ! -s $M/results/mid/$S/qual/frames.npz ]; then
  log "CPU edges labels audit qual $S start"
  python -m mr_cpu.pipeline --config configs/kitti_mid.json --fit $S --test $S --steps edges labels audit qual \
      > $M/logs/cpu_$S.log 2>&1
  log "CPU $S done"; tail -n 8 $M/logs/cpu_$S.log
fi
echo STAGE_DONE_$S
