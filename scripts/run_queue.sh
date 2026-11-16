#!/usr/bin/env bash
# usage: run_queue.sh GPU "REF:SEQ REF:SEQ ..."   (window 32, overlap 16 unless W / O are set)
set -uo pipefail
K=/scratch2/vpr-team/kavish; M=$(cd "$(dirname "$0")/.." \&\& pwd)
. $K/pixi/env.sh
cd $M
set +u; eval "$(pixi shell-hook --manifest-path $M/pixi.toml)"; set -u
GPU=$1; W=${W:-32}; O=${O:-16}
export CUDA_VISIBLE_DEVICES=$GPU
log() { echo "[$(date '+%F %T')] $*"; }
for job in $2; do
  REF=${job%%:*}; S=${job##*:}; OUT=${RUNS_ROOT:-results/final/runs}/${REF}_w${W}_o${O}/$S
  [ -f $OUT/edges.npz ] && { log "$job already done"; continue; }
  # shared machine: start only when the card has at least 13 GB free (checked right before every job)
  for try in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    [ "$used" -lt 3300 ] && break
    log "GPU $GPU has $used MiB in use: waiting"; sleep 60
  done
  [ "$used" -lt 3300 ] || { log "GPU $GPU still busy: stopping the queue"; exit 1; }
  log "$job start"
  python -W ignore::FutureWarning scripts/gpu_run.py --seq $S --ref $REF --window $W --overlap $O --out $OUT ${SUB:+--sub-stride $SUB} \
      > ${LOGS_ROOT:-logs}/${REF}_w${W}_o${O}_$S.log 2>&1 && log "$job done: $(tail -n 1 ${LOGS_ROOT:-logs}/${REF}_w${W}_o${O}_$S.log)" \
      || { log "$job FAILED: $(tail -n 3 ${LOGS_ROOT:-logs}/${REF}_w${W}_o${O}_$S.log)"; }
done
log QUEUE_DONE
