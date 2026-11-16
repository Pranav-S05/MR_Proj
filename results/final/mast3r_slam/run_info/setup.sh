# Build the MASt3R-SLAM environment (resumable). Run in tmux.
set -euo pipefail
B=/scratch2/vpr-team/kavish/mast3r_slam
. /scratch2/vpr-team/kavish/pixi/env.sh
cd $B
log() { echo "[$(date "+%F %T")] $*"; }
[ -d .pixi/envs/default ] || { log "pixi install"; pixi install; }
set +u; eval "$(pixi shell-hook)"; set -u
export PIP_CACHE_DIR=/scratch2/vpr-team/kavish/.cache/pip
export TMPDIR=/scratch2/vpr-team/kavish/.cache/tmp; mkdir -p $TMPDIR
export CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=8
export CUDA_VISIBLE_DEVICES=2   # setup.py only asks torch.cuda.is_available(); no GPU work during the build
log "torch"
python -c "import torch" 2>/dev/null || pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install "setuptools==70.0.0" wheel "cython<3" ninja "numpy==1.26.4"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
log "lietorch"
python -c "import lietorch" 2>/dev/null || pip install --no-build-isolation "git+https://github.com/princeton-vl/lietorch.git"
cd repo
log "mast3r"; pip install --no-build-isolation -e thirdparty/mast3r
log "in3d";   pip install --no-build-isolation -e thirdparty/in3d
log "mast3r-slam"; pip install --no-build-isolation -e .
python -c "import lietorch, mast3r_slam_backends, curope, mast3r_slam; print(\"IMPORT_OK\")"
log "checkpoints"
cd $B/checkpoints
[ -s MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth ] || cp /scratch2/vpr-team/retrieval-transformer/cache/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth .
U=https://download.europe.naverlabs.com/ComputerVision/MASt3R
for f in MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl; do
  [ -s $f ] || curl -sSL --retry 5 -o $f $U/$f; done
ls -la; ln -sfn $B/checkpoints $B/repo/checkpoints
log SETUP_DONE
