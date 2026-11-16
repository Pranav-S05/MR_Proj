# Queued after run_kitti.sh: KITTI 05/07 (once their images are in), then the TUM fr1/desk sanity check with
# the MASt3R-SLAM eval configs (every 2nd frame) and its own scoring (evo_ape tum -as).
set -euo pipefail
B=/scratch2/vpr-team/kavish/mast3r_slam; M=/scratch2/vpr-team/kavish/mr_final
export PYTHONUNBUFFERED=1
log() { echo "[$(date "+%F %T")] $*"; }
while tmux has-session -t kavish-mast3r-run 2>/dev/null; do sleep 60; done
until grep -q IMAGES_0507_DONE $M/logs/get0507.log 2>/dev/null; do
  tmux has-session -t kavish-get0507 2>/dev/null || { log "05/07 images missing"; break; }; sleep 60; done
if grep -q IMAGES_0507_DONE $M/logs/get0507.log; then SEQS="05 07" bash $B/run_kitti.sh; fi
. /scratch2/vpr-team/kavish/pixi/env.sh; cd $B; set +u; eval "$(pixi shell-hook)"; set -u
gpu_idle() { local u m; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
  m=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && [ "$m" -lt 300 ]; }
gpu_roomy() { local t u; t=$(nvidia-smi -i $1 --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)
  u=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9); [ $((t - u)) -ge 13000 ]; }
pick_gpu() { while true; do for test in gpu_idle gpu_roomy; do for g in 2 1 0; do
  if $test $g; then sleep 60; if $test $g; then echo $g; return; fi; fi; done; done; sleep 60; done; }
cd repo
D=../datasets/tum/rgbd_dataset_freiburg1_desk
until [ -f $D/groundtruth.txt ]; do log "waiting for TUM data"; sleep 120; done
for MODE in calib no_calib; do
  OUT=logs/tum/$MODE/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk.txt
  if [ ! -s $OUT ]; then
    export CUDA_VISIBLE_DEVICES=$(pick_gpu)
    log "TUM desk $MODE start on GPU $CUDA_VISIBLE_DEVICES"
    python main.py --dataset $D/ --no-viz --save-as tum/$MODE/rgbd_dataset_freiburg1_desk --config config/eval_$MODE.yaml \
      > $B/logs/tum_desk_$MODE.log 2>&1 || log "TUM $MODE exited with $?"
  fi
  log "TUM desk $MODE: $(evo_ape tum $D/groundtruth.txt $OUT -as 2>&1 | grep -E "rmse" | tr -s " ")"
done
log MORE_DONE
