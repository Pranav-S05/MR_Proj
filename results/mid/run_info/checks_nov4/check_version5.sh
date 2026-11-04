#!/usr/bin/env bash
# Is the 02 @ 60/30 gap (paper 34.16 m, pinned code 52.78 m) caused by the code version?
# Runs VGGT-Long at the pre-refactor commit 97bbc20 (Oct 2025, loop threshold 0.7) and the pinned commit c160869.
set -euo pipefail
K=/scratch2/vpr-team/kavish; C=$K/mr_check
. $K/pixi/env.sh
log() { echo "[$(date '+%F %T')] $*"; }
cd $C
if [ ! -d .pixi/envs/default ]; then log "pixi install"; pixi install > logs/pixi.log 2>&1; fi
set +u; eval "$(pixi shell-hook --manifest-path $C/pixi.toml)"; set -u
gpu_idle() {  # no compute processes and under 300 MiB in use
  local u m; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && [ "$m" -lt 300 ]
}
gpu_roomy() {  # at least 13 GB free (our 60-frame runs peak near 11.5 GB); allowed next to a small job (user's OK)
  local t u; t=$(nvidia-smi -i $1 --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)
  u=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
  [ $((t - u)) -ge 13000 ]
}
pick_gpu() {  # prefer a fully idle GPU; else one with >= 13 GB free; each must hold on two checks one minute apart
  while true; do
    for test in gpu_idle gpu_roomy; do
      for g in 0 1 2; do
        if $test $g; then sleep 60; if $test $g; then echo $g; return; fi; fi
      done
    done
    sleep 60
  done
}

if [ ! -d vl_new ]; then
  git clone -q https://github.com/DengKaiCQ/VGGT-Long.git vl_new
  git -C vl_new checkout -q c160869d1d99c96bb227f414afb3bc68c29c9a76
  git -C vl_new worktree add -q ../vl_old 97bbc20
fi
mkdir -p weights
[ "$(stat -c %s weights/model.pt 2>/dev/null)" = 5026874952 ] || cp /scratch2/krish.pandya/plann3r_root/models/vggt/model.pt weights/
[ -s weights/dino_salad.ckpt ] || cp /scratch2/vpr-team/retrieval-transformer/cache/torch_hub/checkpoints/dino_salad.ckpt weights/
[ -s weights/dinov2_vitb14_pretrain.pth ] || curl -sSL -o weights/dinov2_vitb14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
for r in vl_new vl_old; do ln -sfn $C/weights $C/$r/weights; done
echo "commits: new $(git -C vl_new log -1 --format=%h) old $(git -C vl_old log -1 --format='%h %ad' --date=short)"

# images come from fetch_images.py (running separately); it writes data/kitti/sequences/SS.done per sequence
wait_images() {
  until [ -f data/kitti/sequences/$1.done ]; do sleep 60; done
  log "$1 images ready: $(ls data/kitti/sequences/$1/image_2 | wc -l)"
}

stage() {  # copy one sequence's images to RAM (the shared disk is too contended); waits for the fetcher first
  wait_images $1
  if [ ! -f /dev/shm/kavish_imgs/$1.done ]; then
    while tmux has-session -t kavish-stage 2>/dev/null; do sleep 30; done   # a copy started by hand
    if [ ! -f /dev/shm/kavish_imgs/$1.done ]; then
      rm -rf /dev/shm/kavish_imgs/$1; mkdir -p /dev/shm/kavish_imgs/$1
      cp -r data/kitti/sequences/$1/image_2 /dev/shm/kavish_imgs/$1/ && echo STAGED > /dev/shm/kavish_imgs/$1.done
    fi
  fi
  log "$1 images in RAM: $(ls /dev/shm/kavish_imgs/$1/image_2 | wc -l)"
}
unstage() { rm -rf /dev/shm/kavish_imgs/$1 /dev/shm/kavish_imgs/$1.done; }

run() {  # repo seq W O [max_iterations of the loop optimiser]
  local R=$1 S=$2 W=$3 O=$4 IT=${5:-}; local NAME=${R#vl_}_${S}_w${W}_o${O}${IT:+_it$IT}; local DST=$C/results/$NAME
  [ -f $DST/camera_poses.txt ] && { log "$NAME already done"; return; }
  log "$NAME waiting for an idle GPU"
  local GPU; GPU=$(pick_gpu); export CUDA_VISIBLE_DEVICES=$GPU
  log "$NAME got GPU $GPU"
  local SHM=/dev/shm/kavish_check/$NAME; rm -rf $SHM; mkdir -p $SHM/exps $DST
  printf 'inherit_from: "configs/kitti.yaml"\nModel:\n  chunk_size: %s\n  overlap: %s\n' $W $O > $SHM/config.yaml
  [ -n "$IT" ] && printf 'Loop:\n  SIM3_Optimizer:\n    max_iterations: %s\n' $IT >> $SHM/config.yaml
  ln -sfn $SHM/exps $C/$R/exps
  log "$NAME start"
  ( while sleep 2; do for p in $SHM/exps/*/*/pcd/*_pcd.ply; do [ -e "$p" ] || continue
      k=$(basename $p _pcd.ply); d=$(dirname $(dirname $p))
      rm -f $d/_tmp_results_unaligned/chunk_$k.npy $d/_tmp_results_aligned/chunk_$k.npy; done
      a=$(awk '/MemAvailable/ {print int($2/1e6)}' /proc/meminfo); f=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
      if [ "$a" -lt 15 ] || [ "$f" -lt 6 ]; then echo "low memory: stopped" >> $DST/ABORTED; pkill -f "vggt_long.py --image_dir /dev/shm/kavish_imgs" || true; fi
    done ) & local WATCH=$!
  local t0=$(date +%s)
  (cd $C/$R && python vggt_long.py --image_dir /dev/shm/kavish_imgs/$S/image_2 --config $SHM/config.yaml) > $DST/run.log 2>&1 \
    || { kill $WATCH; rm -rf $SHM; log "$NAME FAILED"; return; }
  kill $WATCH || true
  echo $(( $(date +%s) - t0 )) > $DST/wall_seconds
  local OUT; OUT=$(dirname "$(find $SHM/exps -name camera_poses.txt | head -1)")
  cp $OUT/camera_poses.txt $SHM/config.yaml $DST/; [ -f $OUT/loop_closures.txt ] && cp $OUT/loop_closures.txt $DST/
  gzip -f $DST/run.log; rm -rf $SHM $C/$R/exps
  local ATE; ATE=$(python -m mr_cpu.evaluation --estimate $DST/camera_poses.txt --seq $S --kitti data/kitti \
        | python -c 'import json,sys; print(round(json.load(sys.stdin)["ate_rmse_m"], 2))')
  echo "$ATE" > $DST/ate_m
  log "$NAME done: ATE $ATE m, loops $(grep -c '^[0-9]' $DST/loop_closures.txt 2>/dev/null || echo ?), $(cat $DST/wall_seconds) s"
}

stage 02
run vl_old 02 60 30
run vl_new 02 60 30 200
run vl_new 02 90 45
run vl_old 02 30 15
unstage 02; stage 00; run vl_old 00 60 30
unstage 00; stage 05; run vl_old 05 60 30
unstage 05; stage 07; run vl_old 07 60 30
unstage 07
log CHECK_DONE
