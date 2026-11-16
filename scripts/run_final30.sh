#!/usr/bin/env bash
# The mid-submission analysis rerun at window 30 / overlap 15 on KITTI 00 02 05 07 (fit 05 07, test 00 02):
# per drive VGGT (GPU 0) -> edges, labels, LiDAR depth audit -> delete the cached maps; then fit, tune, solve, report;
# then the false-loop stress test and the DCS threshold sweeps. Resumes: finished drives / steps are skipped.
set -uo pipefail
K=/scratch2/vpr-team/kavish; M=$(cd "$(dirname "$0")/.." && pwd)
cd $M
log() { echo "[$(date '+%F %T')] $*"; }
. $K/pixi/env.sh
set +u; eval "$(pixi shell-hook --manifest-path $M/pixi.toml)"; set -u
CFG=configs/kitti_final30.json
MIR=https://datasets.rrcx.tk/KITTI/odometry
verify() {  # /scratch2 returns read errors on some freshly written files: delete those and fetch them again
  local S=$1 d zip n
  for d in image_2 velodyne; do
    zip=data_odometry_color.zip; [ $d = velodyne ] && zip=data_odometry_velodyne.zip
    n=0; for f in data/kitti/sequences/$S/$d/*; do cat "$f" > /dev/null 2>&1 || { rm -f "$f"; n=$((n+1)); }; done
    if [ $n -gt 0 ]; then
      python scripts/fetch_from_zip.py $MIR/$zip dataset/sequences/$S/$d/ data/kitti/sequences/$S/$d > logs/refetch_${S}_$d.log 2>&1
      log "$S: $n unreadable $d files fetched again: $(tail -1 logs/refetch_${S}_$d.log)"
    fi
  done
}
RUN=results/final30
for S in 07 05 00 02; do
  if [ ! -f $RUN/$S/depth_audit.npz ]; then
    # one GPU with at least 13 GB free (shared machine); wait as long as needed
    GPU=""
    while [ -z "$GPU" ]; do
      for g in 0 1 2; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
        [ "$used" -lt 3300 ] && { GPU=$g; break; }
      done
      [ -n "$GPU" ] || { log "no GPU with 13 GB free: waiting"; sleep 120; }
    done
    export CUDA_VISIBLE_DEVICES=$GPU
    log "$S: VGGT start on GPU $GPU"
    verify $S
    if ! python -W ignore -m mr_gpu.run_vggt run --seq $S --window 30 --overlap 15 > logs/vggt_$S.log 2>&1; then
      log "$S: VGGT stopped ($(tail -1 logs/vggt_$S.log)); checking the files and retrying once"
      verify $S
      python -W ignore -m mr_gpu.run_vggt run --seq $S --window 30 --overlap 15 > logs/vggt_$S.log 2>&1 \
        || { log "$S: VGGT FAILED: $(tail -3 logs/vggt_$S.log)"; exit 1; }
    fi
    log "$S: VGGT done ($(du -sh cache/kitti/$S | cut -f1) cached)"
    if ! python -m mr_cpu.pipeline --config $CFG --steps edges labels audit --fit $S --test $S > logs/pipe_$S.log 2>&1; then
      log "$S: pipeline stopped; checking the files and retrying once"
      verify $S
      python -m mr_cpu.pipeline --config $CFG --steps edges labels audit --fit $S --test $S > logs/pipe_$S.log 2>&1 \
        || { log "$S: pipeline FAILED: $(tail -3 logs/pipe_$S.log)"; exit 1; }
    fi
    log "$S: edges, labels, audit done"
    find cache/kitti/$S -type f ! -name manifest.json ! -name descriptors.npy -delete
  fi
done
python -m mr_cpu.pipeline --config $CFG --steps fit tune solve report > logs/pipe_all.log 2>&1 \
  && log "fit, tune, solve, report done" || { log "pipeline FAILED: $(tail -3 logs/pipe_all.log)"; exit 1; }
W=8
python -m mr_cpu.stress --config $CFG --out $RUN/stress --workers $W > logs/stress.log 2>&1 && log "stress done"
python -m mr_cpu.stress --config $CFG --out $RUN/stress --summary-only >> logs/stress.log 2>&1
python -m mr_cpu.stress --config $CFG --dcs-sweep wide --out $RUN/stress --workers $W > logs/dcs_wide.log 2>&1 && log "dcs wide done"
python -m mr_cpu.stress --config $CFG --dcs-sweep fine --phis 0.0562,0.1,0.178 --out $RUN/stress --workers $W > logs/dcs_fine.log 2>&1 && log "dcs fine done"
log FINAL30_DONE
