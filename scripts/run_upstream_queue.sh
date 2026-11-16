#!/usr/bin/env bash
# Upstream VGGT-Long (pinned commit, its KITTI config, chunk W / overlap O), unpatched and with the calibrated seam
# scale (vggt_long_calib.py, coefficients from results/final/runs/upstream/coeffs_<variant>.json, fitted on OTHER drives).
#   usage: W=75 O=30 V=first_w75_o30 bash scripts/run_upstream_queue.sh GPU "TAG:SEQ ..."   TAG in plain | C1 | C3
# Output: results/final/runs/upstream/<TAG>_w<W>_o<O>/<SEQ>/{camera_poses.txt, loop_closures.txt, run.log.gz, wall_seconds}
set -uo pipefail
K=/scratch2/vpr-team/kavish; M=$(cd "$(dirname "$0")/.." \&\& pwd); VL=${VL_DIR:-$M/external/VGGT-Long}; OUTR=${OUT_ROOT:-$M/runs/upstream}
. $K/pixi/env.sh
cd $M
set +u; eval "$(pixi shell-hook --manifest-path $M/pixi.toml)"; set -u
GPU=$1; W=${W:-75}; O=${O:-30}; V=${V:-first_w${W}_o${O}}
export CUDA_VISIBLE_DEVICES=$GPU
log() { echo "[$(date '+%F %T')] $*"; }
[ -f $VL/vggt_long_calib.py ] || python scripts/make_upstream_calib.py
for job in $2; do
  TAG=${job%%:*}; S=${job##*:}; DST=$OUTR/${TAG}_w${W}_o${O}/$S
  [ -f $DST/camera_poses.txt ] && { log "$job already done"; continue; }
  for try in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    [ "$used" -lt 1500 ] && break
    log "GPU $GPU has $used MiB in use: waiting"; sleep 60
  done
  [ "$used" -lt 1500 ] || { log "GPU $GPU still busy: stopping"; exit 1; }
  # VGGT-Long's temporary chunk files peak at about 6 GB per 1000 frames (with the pruner): need that plus 15 GB margin
  peak=$(( $(ls $M/data/kitti/sequences/$S/image_2 | wc -l) * 6 / 1000 ))
  if [ "${SHM:-0}" = 1 ]; then
    # temporary files in RAM (as the mid-submission runs did): start only with peak + 30 GB available
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "$avail" -gt $(( peak + 30 )) ] || { log "only $avail GB RAM available, $S needs $(( peak + 30 )) GB: stopping"; exit 1; }
    if [ "${OUTR#/dev/shm}" = "$OUTR" ]; then
      free=$(df -BG --output=avail /scratch2 | tail -1 | tr -dc 0-9)
      [ "$free" -ge 5 ] || { log "only $free GB free on /scratch2 (results need a little): stopping"; exit 1; }
    fi
  else
    free=$(df -BG --output=avail /scratch2 | tail -1 | tr -dc 0-9)
    need=$(( peak + 15 ))
    [ "$free" -gt "$need" ] || { log "only $free GB free on /scratch2, $S needs $need GB: stopping"; exit 1; }
  fi
  if [ "${TAG#plain}" != "$TAG" ]; then unset SEAM_CALIB; SCRIPT=vggt_long.py
  else export SEAM_CALIB=$(python -c "import json;print(json.dumps(json.load(open('results/final/runs/upstream/coeffs_$V.json'))['$TAG']['$S']))"); SCRIPT=vggt_long_calib.py; fi
  if [ "${SHM:-0}" = 1 ]; then EXPS=/dev/shm/kavish_up_${TAG}_${S}; else EXPS=$M/results/final/runs/upstream/tmp_${TAG}_${S}; fi
  rm -rf $EXPS; mkdir -p $DST $EXPS
  trap 'rm -rf $EXPS' EXIT
  printf 'inherit_from: "configs/kitti.yaml"\nModel:\n  chunk_size: %s\n  overlap: %s\n' "$W" "$O" > $EXPS/config.yaml
  ln -sfn $EXPS $VL/exps
  ( while sleep 2; do
      for p in $EXPS/*/*/pcd/*_pcd.ply; do
        [ -e "$p" ] || continue
        k=$(basename "$p" _pcd.ply); run=$(dirname "$(dirname "$p")")
        rm -f "$run/_tmp_results_unaligned/chunk_$k.npy" "$run/_tmp_results_aligned/chunk_$k.npy"
      done
      # watchdog: stop our run if the machine runs low on memory (other users come first)
      if [ "${SHM:-0}" = 1 ] && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 20 ]; then
        echo "low memory: stopped" > $DST/ABORTED; pkill -u $(id -u) -f "[v]ggt_long.*--image_dir $M/data/kitti/sequences/$S/"
      fi
    done ) & PRUNE=$!
  log "$job start ${SEAM_CALIB:-}"
  t0=$(date +%s)
  (cd $VL && python -W ignore $SCRIPT --image_dir $M/data/kitti/sequences/$S/image_2 --config $EXPS/config.yaml) > $DST/run.log 2>&1
  rc=$?
  kill $PRUNE 2>/dev/null
  echo $(( $(date +%s) - t0 )) > $DST/wall_seconds
  OUT=$(dirname "$(find $EXPS -name camera_poses.txt | head -1)" 2>/dev/null)
  if [ $rc -eq 0 ] && [ -n "$OUT" ] && [ -f "$OUT/camera_poses.txt" ]; then
    cp $OUT/camera_poses.txt $DST/; [ -f $OUT/loop_closures.txt ] && cp $OUT/loop_closures.txt $DST/
    [ -f $OUT/pcd/combined_pcd.ply ] && python -W ignore -m mr_cpu.qualitative ply2npz $OUT/pcd/combined_pcd.ply $DST/pointcloud_sample.npz
    log "$job done in $(cat $DST/wall_seconds) s"
  else
    log "$job FAILED rc=$rc: $(tail -n 3 $DST/run.log)"
  fi
  gzip -f $DST/run.log
  rm -rf $EXPS $VL/exps
  unset SEAM_CALIB
done
log QUEUE_DONE
