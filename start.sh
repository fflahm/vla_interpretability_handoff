# H-cluster local defaults for vla_interpretability_handoff (reorg/modular-handoff).
# Usage from the repo root:  source start.sh
# Need GitHub/Hugging Face:  source ~/.pjlab_proxy.sh && proxy_on

if [ -z "${CONDA_PREFIX:-}" ]; then
  # shellcheck disable=SC1091
  source /mnt/shared-storage-user/guoshengyu/miniconda3/etc/profile.d/conda.sh
  if [ -d /home/guoshengyu/.conda/envs/vla-interpretability ]; then
    conda activate /home/guoshengyu/.conda/envs/vla-interpretability
  elif [ -d /mnt/shared-storage-user/guoshengyu/envs/vla-interpretability ]; then
    conda activate /mnt/shared-storage-user/guoshengyu/envs/vla-interpretability
  fi
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
fi

unset HF_ENDPOINT

export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export TMPDIR=/tmp

export PI05_PATH=/data/tos/guoshengyu/vla/models/pi05_libero
export PI0_PATH=/data/tos/guoshengyu/vla/models/pi0_libero_finetuned_v044
export HF_HOME=/data/tos/guoshengyu/vla/cache/huggingface
export TORCH_HOME=/data/tos/guoshengyu/vla/cache/torch
export LIBERO_ROOT=/data/tos/guoshengyu/vla/libero/LIBERO
export LIBERO_ASSETS_PATH=/data/tos/guoshengyu/vla/libero/LIBERO/libero/libero/assets
export LIBERO_DATASET_DIR=/data/tos/guoshengyu/vla/libero/libero_spatial
export HF_LEROBOT_HOME=/data/tos/guoshengyu/vla/cache/lerobot

export ROLLOUT_DIR="$PWD/outputs/rollouts/pi0_libero_spatial_task1_full"
export ANALYSIS_DIR="$PWD/outputs/rollouts/pi0_libero_spatial_task1_full/runs/20260628_165717_pool-mean_seed-42"
export ABLATION_DIR="$PWD/outputs/ablation/pi0_ablation_spatial_task1"

echo "PI05_PATH=$PI05_PATH"
echo "PI0_PATH=$PI0_PATH"
echo "HF_HOME=$HF_HOME"
echo "LIBERO_ROOT=$LIBERO_ROOT"
echo "LIBERO_ASSETS_PATH=$LIBERO_ASSETS_PATH"
echo "LIBERO_DATASET_DIR=$LIBERO_DATASET_DIR"
echo "MUJOCO_GL=$MUJOCO_GL"
