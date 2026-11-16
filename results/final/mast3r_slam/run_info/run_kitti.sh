# MASt3R-SLAM baseline on KITTI odometry (resumable). Run in tmux after setup.sh. Usage: SEQS="00 02" bash run_kitti.sh
# Unmodified code and config/base.yaml, headless and single-threaded as in its eval scripts, every frame
# (KITTI is 10 Hz), images from mr_final. Two modes: nocalib (as VGGT-Long ran it) and calib (KITTI P2).
set -euo pipefail
B=/scratch2/vpr-team/kavish/mast3r_slam; M=/scratch2/vpr-team/kavish/mr_final
. /scratch2/vpr-team/kavish/pixi/env.sh
cd $B; set +u; eval "$(pixi shell-hook)"; set -u
export TMPDIR=/scratch2/vpr-team/kavish/.cache/tmp
log() { echo "[$(date "+%F %T")] $*"; }
gpu_idle() { local u m; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && [ "$m" -lt 300 ]; }
gpu_roomy() { local t u; t=$(nvidia-smi -i $1 --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)
  u=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9); [ $((t - u)) -ge 13000 ]; }
pick_gpu() { while true; do for test in gpu_idle gpu_roomy; do for g in 2 1 0; do
  if $test $g; then sleep 60; if $test $g; then echo $g; return; fi; fi; done; done; sleep 60; done; }
cd repo
for S in ${SEQS:-00 02}; do
  P2=$(grep "^P2:" $M/data/kitti/sequences/$S/calib.txt)
  read -r W H < <(python -c "import cv2,sys; i=cv2.imread(sys.argv[1]); print(i.shape[1], i.shape[0])" $M/data/kitti/sequences/$S/image_2/000000.png)
  python - "$P2" $W $H > config/kitti_$S.yaml <<PY
import sys
p = [float(x) for x in sys.argv[1].split()[1:]]
print(f"width: {sys.argv[2]}\nheight: {sys.argv[3]}\ncalibration: [{p[0]}, {p[5]}, {p[2]}, {p[6]}]")
PY
  for MODE in nocalib calib; do
    OUT=logs/kitti/${S}_$MODE
    if [ -s $OUT/image_2.txt ]; then log "$S $MODE already done"; continue; fi
    export CUDA_VISIBLE_DEVICES=$(pick_gpu)
    EXTRA=""; [ $MODE = calib ] && EXTRA="--calib config/kitti_$S.yaml"
    log "$S $MODE start on GPU $CUDA_VISIBLE_DEVICES"
    ( while sleep 30; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -E "^($(pgrep -d"|" -f "main.py --dataset"))," ; done ) > $B/logs/mem_${S}_$MODE.log 2>&1 &
    MON=$!
    /usr/bin/time -v python main.py --dataset $M/data/kitti/sequences/$S/image_2 --config config/kitti.yaml --no-viz \
        --save-as kitti/${S}_$MODE $EXTRA > $B/logs/run_${S}_$MODE.log 2>&1 || log "$S $MODE exited with $?"
    kill $MON 2>/dev/null || true
    log "$S $MODE done: $(wc -l < $OUT/image_2.txt 2>/dev/null || echo 0) keyframes, peak MiB $(sort -t, -k2 -n $B/logs/mem_${S}_$MODE.log | tail -1)"
  done
done
log KITTI_DONE
