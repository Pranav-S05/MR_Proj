#!/usr/bin/env bash
# Why does upstream 02 at chunk 60 give 52.8 m vs the paper's 34.2 m? (1) repeat run, (2) loop threshold 0.7.
set -euo pipefail
K=/scratch2/vpr-team/kavish; M=$K/mr; GPU=$1
. $K/pixi/env.sh; cd $M
set +u; eval "$(pixi shell-hook --manifest-path pixi.toml)"; set -u
export CUDA_VISIBLE_DEVICES=$GPU
if [ ! -d data/kitti/sequences/02/image_2 ] || [ "$(ls data/kitti/sequences/02/image_2 | wc -l)" != 4661 ]; then
  (cd data && unzip -q -o /scratch2/anay/datasets/kitti_odom/data_odometry_color.zip "dataset/sequences/02/image_2/*" \
     && mkdir -p kitti/sequences/02 && rm -rf kitti/sequences/02/image_2 && mv dataset/sequences/02/image_2 kitti/sequences/02/ && rm -rf dataset)
fi
echo "02 images: $(ls data/kitti/sequences/02/image_2 | wc -l)"
for V in repeat loop070; do
  DST=$M/results/mid/upstream_checks/02_w60_o30_$V
  [ -f $DST/camera_poses.txt ] && continue
  SHM=/dev/shm/kavish_vl/check_$V; rm -rf $SHM; mkdir -p $SHM/exps $DST
  printf 'inherit_from: "configs/kitti.yaml"\nModel:\n  chunk_size: 60\n  overlap: 30\n' > $SHM/config.yaml
  [ $V = loop070 ] && printf 'Loop:\n  SALAD:\n    similarity_threshold: 0.7\n' >> $SHM/config.yaml
  ln -sfn $SHM/exps $M/external/VGGT-Long/exps
  ( while sleep 2; do for p in $SHM/exps/*/*/pcd/*_pcd.ply; do [ -e "$p" ] || continue
      k=$(basename $p _pcd.ply); r=$(dirname $(dirname $p))
      rm -f $r/_tmp_results_unaligned/chunk_$k.npy $r/_tmp_results_aligned/chunk_$k.npy; done; done ) & W=$!
  (cd external/VGGT-Long && python vggt_long.py --image_dir $M/data/kitti/sequences/02/image_2 --config $SHM/config.yaml) > $DST/run.log 2>&1
  kill $W || true
  OUT=$(dirname "$(find $SHM/exps -name camera_poses.txt | head -1)")
  cp $OUT/camera_poses.txt $OUT/loop_closures.txt $SHM/config.yaml $DST/
  gzip -f $DST/run.log; rm -rf $SHM external/VGGT-Long/exps
  echo "$V: $(python -m mr_cpu.evaluation --estimate $DST/camera_poses.txt --seq 02 | python -c 'import json,sys; d=json.load(sys.stdin); print(round(d["ate_rmse_m"],2), "m ATE")'), loops $(grep -c "^[0-9]" $DST/loop_closures.txt)"
done
rm -rf data/kitti/sequences/02/image_2
echo CHECK02_DONE
