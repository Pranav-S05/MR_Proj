#!/usr/bin/env bash
# Chain: per sequence stage -> free its inputs; then the global steps.
set -euo pipefail
T=/scratch2/vpr-team/kavish/mr_tools; L=/scratch2/vpr-team/kavish/mr/logs; GPU=${GPU:-1}
for S in 07 05 00 02; do
  bash $T/seq_stage_v2.sh $S $GPU > $L/stage_$S.log 2>&1 || { echo "stage $S failed"; exit 1; }
  bash $T/free_seq.sh $S
done
cd /scratch2/vpr-team/kavish/mr
. /scratch2/vpr-team/kavish/pixi/env.sh
set +u; eval "$(pixi shell-hook --manifest-path pixi.toml)"; set -u
python -m mr_cpu.pipeline --config configs/kitti_mid.json --steps fit tune solve upstream report > $L/final.log 2>&1
echo ALL_DONE
