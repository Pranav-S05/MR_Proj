#!/usr/bin/env bash
# GPU part of the mid-submission experiment (run from the project root, inside VGGT-Long's env).
# Every step resumes if interrupted; finished windows are skipped.
#   WINDOW / OVERLAP: pick after the memory sweep (VGGT-Long's own KITTI setting is 75 / 30).
set -euo pipefail
WINDOW=${WINDOW:-32}
OVERLAP=${OVERLAP:-16}

# 1. how much GPU memory each window size needs (writes results/memory_sweep.csv)
python -m mr_gpu.run_vggt memory-sweep --seq 07 --windows 8 16 24 32 48 64

# 2. smoke test: 96 frames of 07, prints PASS/FAIL convention checks
python -m mr_gpu.run_vggt run --seq 07 --window "$WINDOW" --overlap "$OVERLAP" --max-frames 96 \
    --no-loops --smoke --out cache/smoke_07

# 3. the four mid-submission sequences (fitting 05, 07; held-out 00, 02)
for SEQ in 07 05 00 02; do
    python -m mr_gpu.run_vggt run --seq "$SEQ" --window "$WINDOW" --overlap "$OVERLAP" --hash
done

# 4. (optional, recommended) upstream VGGT-Long at the same window size, for the reproduction check:
#    cd external/VGGT-Long && python vggt_long.py --image_dir ../../data/kitti/sequences/07/image_2 --config <yaml with
#    chunk_size=$WINDOW, overlap=$OVERLAP> ; then from the project root:
#    python -m mr_cpu.evaluation --estimate external/VGGT-Long/exps/<run>/camera_poses.txt --seq 07 --evo
echo "GPU part done. Copy cache/kitti/ (and results/memory_sweep.csv) to the CPU machine, then run:"
echo "  python -m mr_cpu.pipeline --config configs/kitti_mid.json"
